//! keel's ACP client.
//!
//! Spawns `keel-core acp` (stdin/stdout = the protocol, stderr = a log file), runs one session in a folder, and
//! turns what the core sends into [`UiEvent`]s for a front end (the TUI, or the plain line mode). The front end
//! sends [`Command`]s back. The delete question arrives as [`UiEvent::Permission`]; the core deletes only when it
//! is answered with an explicit yes. A permission that is dropped unanswered (the front end quit, crashed or was
//! killed) is sent back as "cancelled", which the core treats as no.
//!
//! Session updates are read as JSON by their protocol field names (`sessionUpdate`, `toolCallId`, ...) rather
//! than by matching the crate's Rust types, so a crate upgrade cannot silently break the mapping; replies use the
//! crate's typed constructors.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::v1::ConnectionTo;
use agent_client_protocol::v1::schema::{
    CancelNotification, ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionKind,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, TextContent,
};
use agent_client_protocol::{Agent, ByteStreams};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use tokio::process::Command as Proc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

/// What the front end shows.
#[derive(Debug)]
pub enum UiEvent {
    /// The session is open in `folder`.
    Ready {
        session_id: String,
        folder: String,
        backend: String,
    },
    /// New text from the model (prose only; tool-call markup is never sent).
    Text(String),
    /// A tool call started.
    ToolStart {
        id: String,
        title: String,
        kind: String,
    },
    /// A tool call finished; `diff` is set for file writes.
    ToolEnd {
        id: String,
        ok: bool,
        output: String,
        diff: Option<Diff>,
    },
    /// The delete guard asks. Answer with [`PermissionAsk::answer`].
    Permission(PermissionAsk),
    /// The prompt finished (`end_turn`, `cancelled`, `max_turn_requests`, ...).
    TurnEnd { stop: String },
    /// Something went wrong (shown to the user, the session may continue).
    Error(String),
    /// The connection to keel-core ended.
    Closed,
}

/// A file change: `old` is `None` for a new file.
#[derive(Debug, Clone, PartialEq)]
pub struct Diff {
    pub path: String,
    pub old: Option<String>,
    pub new: String,
}

/// The delete question. Dropping it without answering counts as "no".
#[derive(Debug)]
pub struct PermissionAsk {
    pub title: String,
    pub command: String,
    pub files: Vec<String>,
    reply: oneshot::Sender<bool>,
}

impl PermissionAsk {
    /// A question and the receiver that gets its answer (`Err` = dropped unanswered = no). For tests and mocks.
    pub fn new(
        title: String,
        command: String,
        files: Vec<String>,
    ) -> (Self, oneshot::Receiver<bool>) {
        let (reply, rx) = oneshot::channel();
        (
            Self {
                title,
                command,
                files,
                reply,
            },
            rx,
        )
    }

    /// `true` deletes the listed files (allow once); `false` keeps them.
    pub fn answer(self, allow: bool) {
        let _ = self.reply.send(allow);
    }
}

/// What the front end asks for.
#[derive(Debug)]
pub enum Command {
    Prompt(String),
    Cancel,
    Quit,
}

/// How to start keel-core.
#[derive(Debug, Clone)]
pub struct CoreSpec {
    pub program: String,
    pub args: Vec<String>,
    /// keel-core's stderr goes here (never into the terminal UI).
    pub log: PathBuf,
}

impl CoreSpec {
    /// Development: run the core from a source checkout with uv (`--no-sync`: never change that environment).
    pub fn from_checkout(repo: &Path, extra: &[String], log: PathBuf) -> Self {
        let mut args: Vec<String> = ["run", "--no-sync", "--project"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        args.push(repo.display().to_string());
        args.extend(["keel-core".to_string(), "acp".to_string()]);
        args.extend(extra.iter().cloned());
        Self {
            program: "uv".into(),
            args,
            log,
        }
    }
}

/// Map one `session/update` (as protocol JSON) to UI events.
pub fn map_update(update: &Value) -> Vec<UiEvent> {
    let kind = update
        .get("sessionUpdate")
        .and_then(Value::as_str)
        .unwrap_or("");
    let s = |k: &str| {
        update
            .get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    match kind {
        "agent_message_chunk" => {
            let t = update
                .pointer("/content/text")
                .and_then(Value::as_str)
                .unwrap_or("");
            if t.is_empty() {
                vec![]
            } else {
                vec![UiEvent::Text(t.to_string())]
            }
        }
        "tool_call" => vec![UiEvent::ToolStart {
            id: s("toolCallId"),
            title: s("title"),
            kind: s("kind"),
        }],
        "tool_call_update" => {
            let mut output = String::new();
            let mut diff = None;
            for c in update
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match c.get("type").and_then(Value::as_str) {
                    Some("diff") => {
                        diff = Some(Diff {
                            path: c
                                .get("path")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            old: c.get("oldText").and_then(Value::as_str).map(str::to_string),
                            new: c
                                .get("newText")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        })
                    }
                    Some("content") => {
                        if let Some(t) = c.pointer("/content/text").and_then(Value::as_str) {
                            output.push_str(t);
                        }
                    }
                    _ => {}
                }
            }
            vec![UiEvent::ToolEnd {
                id: s("toolCallId"),
                ok: s("status") != "failed",
                output,
                diff,
            }]
        }
        _ => vec![],
    }
}

fn to_json<T: serde::Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// Run one session: spawn the core, open `folder`, then serve `cmds` until `Quit` or the core goes away.
pub async fn run_session(
    core: CoreSpec,
    folder: PathBuf,
    events: mpsc::UnboundedSender<UiEvent>,
    mut cmds: mpsc::UnboundedReceiver<Command>,
) -> Result<()> {
    if let Some(dir) = core.log.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&core.log)
        .with_context(|| format!("opening the core log {}", core.log.display()))?;
    let log_start = log.metadata().map(|m| m.len()).unwrap_or(0); // this session's part of the log starts here
    let mut child = Proc::new(&core.program)
        .args(&core.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("starting keel-core ({})", core.program))?;
    let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let transport = ByteStreams::new(stdin.compat_write(), stdout.compat());

    let ev_notes = events.clone();
    let ev_perm = events.clone();
    let ev_main = events.clone();
    let result = agent_client_protocol::Client
        .v1()
        .name("keel")
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                for e in map_update(&to_json(&n.update)) {
                    let _ = ev_notes.send(e);
                }
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |req: RequestPermissionRequest, responder, _cx| {
                let j = to_json(&req);
                let raw = j.pointer("/toolCall/rawInput").cloned().unwrap_or(Value::Null);
                let (tx, rx) = oneshot::channel();
                let ask = PermissionAsk {
                    title: j.pointer("/toolCall/title").and_then(Value::as_str).unwrap_or("Delete?").into(),
                    command: raw.get("command").and_then(Value::as_str).unwrap_or("").into(),
                    files: raw.get("files").and_then(Value::as_array).into_iter().flatten()
                        .filter_map(Value::as_str).map(str::to_string).collect(),
                    reply: tx,
                };
                let pick = |k: PermissionOptionKind| {
                    req.options.iter().find(|o| o.kind == k).map(|o| o.option_id.clone())
                };
                let outcome = if ev_perm.send(UiEvent::Permission(ask)).is_err() {
                    RequestPermissionOutcome::Cancelled
                } else {
                    match rx.await {
                        Ok(true) => pick(PermissionOptionKind::AllowOnce)
                            .map(|id| RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id)))
                            .unwrap_or(RequestPermissionOutcome::Cancelled),
                        Ok(false) => pick(PermissionOptionKind::RejectOnce)
                            .map(|id| RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id)))
                            .unwrap_or(RequestPermissionOutcome::Cancelled),
                        Err(_) => RequestPermissionOutcome::Cancelled, // dropped unanswered = no
                    }
                };
                responder.respond(RequestPermissionResponse::new(outcome))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(transport, async move |cx: ConnectionTo<Agent>| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1)).block_task().await?;
            let new = cx.send_request(NewSessionRequest::new(folder.clone())).block_task().await?;
            let meta = to_json(&new.meta);
            let sid = new.session_id.clone();
            let _ = ev_main.send(UiEvent::Ready {
                session_id: sid.to_string(),
                folder: meta.pointer("/keel/folder").and_then(Value::as_str)
                    .unwrap_or(&folder.display().to_string()).to_string(),
                backend: meta.pointer("/keel/backend").and_then(Value::as_str).unwrap_or("?").to_string(),
            });
            while let Some(cmd) = cmds.recv().await {
                match cmd {
                    Command::Quit => break,
                    Command::Cancel => {} // nothing running
                    Command::Prompt(text) => {
                        let req = PromptRequest::new(sid.clone(), vec![ContentBlock::Text(TextContent::new(text))]);
                        let turn = cx.send_request(req).block_task();
                        tokio::pin!(turn);
                        let stop = loop {
                            tokio::select! {
                                r = &mut turn => break match r {
                                    Ok(resp) => to_json(&resp.stop_reason).as_str().unwrap_or("end_turn").to_string(),
                                    Err(e) => { let _ = ev_main.send(UiEvent::Error(e.message.clone())); "error".into() }
                                },
                                c = cmds.recv() => match c {
                                    Some(Command::Cancel) => {
                                        let _ = cx.send_notification(CancelNotification::new(sid.clone()));
                                    }
                                    Some(Command::Quit) | None => return Ok(()),
                                    Some(Command::Prompt(_)) => {
                                        let _ = ev_main.send(UiEvent::Error("still working on the last prompt".into()));
                                    }
                                },
                            }
                        };
                        let _ = ev_main.send(UiEvent::TurnEnd { stop });
                    }
                }
            }
            Ok(())
        })
        .await;
    let _ = events.send(UiEvent::Closed);
    let _ = child.start_kill();
    result.map_err(|e| anyhow!(explain(&e, &core.log, log_start)))
}

/// One readable message for a failed session (first-run problems must say what to do, Ch7 P6): keel-core's own
/// message (it names the fix: `keel download`, start the server, pick a smaller folder), the `details` of an internal
/// error, or, when keel-core stopped before answering, the last lines it wrote to its log.
fn explain(e: &agent_client_protocol::Error, log: &Path, log_start: u64) -> String {
    let data = e.data.as_ref();
    let reason = data
        .and_then(|d| d.get("reason"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if reason == "incoming_transport_closed" {
        let tail = std::fs::read(log)
            .ok()
            .map(|b| String::from_utf8_lossy(&b[(log_start as usize).min(b.len())..]).into_owned())
            .unwrap_or_default();
        let lines: Vec<&str> = tail.lines().filter(|l| !l.trim().is_empty()).collect();
        let last = lines[lines.len().saturating_sub(3)..].join("\n");
        return if last.is_empty() {
            "keel-core stopped before the session started".to_string()
        } else {
            format!("keel-core stopped before the session started:\n{last}")
        };
    }
    let msg = data
        .and_then(|d| d.get("details"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| e.message.clone());
    if matches!(reason, "backend_refused" | "folder_refused") {
        msg // keel-core's own message already says what to do
    } else {
        format!("{msg}\n(keel-core log: {})", log.display())
    }
}
