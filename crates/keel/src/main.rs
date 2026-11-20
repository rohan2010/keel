//! keel: run a task with a local model in a folder. The agent asks before it deletes anything.
//!
//!     keel run "sort my downloads into folders"            in the current folder
//!     keel run --dir ~/Desktop/photos "remove duplicates"
//!     keel run --plain "..."                               line mode (no full UI)
//!     keel run --backend mlx-server "..."                  a running mlx_lm.server instead of in-process MLX
//!     keel run --backend ollama --server-model qwen3.5:4b "..."
//!     keel doctor [--backend mlx-server]                   what this Mac and a model server can do

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use keel_protocol::{Command, CoreSpec, UiEvent, run_session};
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(
    name = "keel",
    version,
    about = "A local-model agent that asks before it deletes anything"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // parsed once per process: boxing the big variant would buy nothing
enum Cmd {
    /// Run a task in a folder
    Run {
        /// What to do, in plain words (leave out to type it in the terminal UI)
        task: Option<String>,
        /// The folder to work in (default: the current folder)
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        /// Line mode instead of the full terminal UI
        #[arg(long)]
        plain: bool,
        /// Model backend: mlx (in-process, the default), mlx-server, ollama, server (any local server),
        /// or scripted (replayed replies, for demos and tests)
        #[arg(long, default_value = "mlx",
              value_parser = ["mlx", "mlx-server", "ollama", "server", "scripted"])]
        backend: String,
        /// Server address (default: the backend's usual one)
        #[arg(long)]
        url: Option<String>,
        /// Server mode: raw (keel renders the prompt; compares with in-process) or chat (the server does)
        #[arg(long)]
        mode: Option<String>,
        /// --backend server: wire format (openai, ollama, llama)
        #[arg(long)]
        flavour: Option<String>,
        /// The model's name on the server (Ollama needs it)
        #[arg(long)]
        server_model: Option<String>,
        /// Scripted backend: JSON file of replies
        #[arg(long)]
        script: Option<PathBuf>,
        /// Scripted backend: seconds per streamed chunk
        #[arg(long)]
        script_delay: Option<f64>,
        /// Where keel keeps session history and logs (default: ~/Library/Application Support/keel)
        #[arg(long)]
        state_dir: Option<PathBuf>,
    },
    /// Check this Mac and (with --backend) a model server; prints a table, exits 1 if something required fails
    Doctor {
        /// Passed to keel-core doctor as given, e.g. `--backend mlx-server` or `--backend server --url ...`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// List or restore the backups keel keeps before it changes or deletes a file
    Backups {
        /// Passed to keel-core backups as given, e.g. `<session>` or `<session> --restore 3`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Download (or `--check`) the model: pinned revision, shows size and free disk, asks first
    Download {
        /// Passed to keel-core download as given, e.g. `--check` or `--model-dir DIR`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Verdict on two runs: real gain, real loss, no detectable effect, or can't compare (and why)
    Compare {
        /// Passed to keel-core compare as given: `<runA> <runB> [--group G] [--pair try] ...`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Run an eval suite as a named arm: `--suite agent|bfcl-mt|when2call --arm NAME [suite flags]`, or `--list`
    Eval {
        /// Passed to keel-core eval as given
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/// keel-core's log, rotated at 5 MB (one old copy kept) so it cannot grow forever.
fn core_log(state: &std::path::Path) -> PathBuf {
    let log = state.join("logs/core.log");
    if std::fs::metadata(&log)
        .map(|m| m.len() > 5 * 1024 * 1024)
        .unwrap_or(false)
    {
        let _ = std::fs::rename(&log, state.join("logs/core.log.1"));
    }
    log
}

fn default_state_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("Library/Application Support/keel")
}

/// How to start keel-core, first match wins:
/// 1. `KEEL_CORE`: the keel-core program to run (any path);
/// 2. a `keel-core` next to this binary (a wheel / `uv tool install` puts both in one bin folder);
/// 3. the source checkout this binary was built from, run with uv (development only: the path is this machine's);
/// 4. `keel-core` on PATH.
///
/// Returns (program, arguments before the keel-core subcommand).
fn core_program() -> Result<(String, Vec<String>)> {
    if let Some(p) = std::env::var_os("KEEL_CORE").filter(|p| !p.is_empty()) {
        return Ok((PathBuf::from(p).display().to_string(), vec![]));
    }
    if let Some(sib) = std::env::current_exe()
        .ok()
        .and_then(|e| e.canonicalize().ok())
        .and_then(|e| e.parent().map(|d| d.join("keel-core")))
        .filter(|p| p.is_file())
    {
        return Ok((sib.display().to_string(), vec![]));
    }
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if checkout.join("pyproject.toml").exists() {
        let repo = checkout.canonicalize()?.display().to_string();
        let args = ["run", "--no-sync", "--project", &repo, "keel-core"];
        return Ok(("uv".into(), args.iter().map(|s| s.to_string()).collect()));
    }
    Ok(("keel-core".into(), vec![]))
}

fn core_spec(extra: Vec<String>, state: &std::path::Path) -> Result<CoreSpec> {
    let log = core_log(state);
    let (program, mut args) = core_program()?;
    args.push("acp".to_string());
    args.extend(extra);
    Ok(CoreSpec { program, args, log })
}

/// `keel doctor` / `compare` / `eval`: run that keel-core command in this terminal (it prints its own output) and
/// exit with its status.
fn forward(sub: &str, args: Vec<String>) -> Result<()> {
    let (program, pre) = core_program()?;
    let status = std::process::Command::new(&program)
        .args(&pre)
        .arg(sub)
        .args(&args)
        .status()
        .with_context(|| format!("could not start keel-core ({program})"))?;
    std::process::exit(status.code().unwrap_or(1));
}

#[tokio::main]
async fn main() {
    // One line a user can act on (`keel: run \`keel download\` ...`), not a Debug dump with "Caused by" chains.
    if let Err(e) = real_main().await {
        eprintln!("keel: {e}");
        std::process::exit(1);
    }
}

/// Bare `keel`: the mascot and where to start.
fn welcome() {
    use std::io::IsTerminal;
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let version = format!("v{}", env!("CARGO_PKG_VERSION"));
    println!("{}\n", keel_tui::logo::text(&version, color));
    println!("Start here:");
    println!(
        "  keel download              get the model (about 5 GB; shows the size and asks first)"
    );
    println!("  keel doctor                check that this Mac is ready");
    println!("  keel run --dir <folder>    work in a folder: type what you want done");
    println!("\nMore: keel --help");
}

async fn real_main() -> Result<()> {
    let Some(cmd) = Cli::parse().cmd else {
        welcome();
        return Ok(());
    };
    let cmd = match cmd {
        Cmd::Doctor { args } => return forward("doctor", args),
        Cmd::Compare { args } => return forward("compare", args),
        Cmd::Eval { args } => return forward("eval", args),
        Cmd::Download { args } => return forward("download", args),
        Cmd::Backups { args } => return forward("backups", args),
        run => run,
    };
    let Cmd::Run {
        task,
        dir,
        plain,
        backend,
        url,
        mode,
        flavour,
        server_model,
        script,
        script_delay,
        state_dir,
    } = cmd
    else {
        unreachable!("doctor returned above")
    };
    let state = state_dir.unwrap_or_else(default_state_dir);
    let folder = dir
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("no such folder: {}", dir.display()))?;
    let task = task.unwrap_or_default();
    if plain && task.trim().is_empty() {
        bail!(
            "line mode needs a task: keel run --plain \"<what to do>\" (or leave out --plain to type it)"
        );
    }
    let mut extra = vec!["--backend".to_string(), backend.clone()];
    extra.extend(server_flags(&url, &mode, &flavour, &server_model));
    if let Some(s) = script {
        extra.extend(["--script".into(), s.canonicalize()?.display().to_string()]);
    }
    if let Some(d) = script_delay {
        extra.extend(["--script-delay".into(), d.to_string()]);
    }
    extra.extend(["--state-dir".into(), state.display().to_string()]);
    let core = core_spec(extra, &state)?;
    let (etx, erx) = mpsc::unbounded_channel();
    let (ctx, crx) = mpsc::unbounded_channel();
    let session = tokio::spawn(run_session(core.clone(), folder, etx, crx));
    let outcome = if plain {
        plain_mode(task, erx, ctx).await
    } else {
        keel_tui::run(task, erx, ctx).await
    };
    let joined = session.await;
    outcome?;
    match joined {
        Ok(Err(e)) => bail!("{e}"),
        Err(e) => bail!("session task failed: {e}"),
        Ok(Ok(())) => Ok(()),
    }
}

/// Server flags passed through to keel-core as given (it checks them and says what's wrong).
fn server_flags(
    url: &Option<String>,
    mode: &Option<String>,
    flavour: &Option<String>,
    server_model: &Option<String>,
) -> Vec<String> {
    let mut out = Vec::new();
    for (flag, value) in [
        ("--url", url),
        ("--mode", mode),
        ("--flavour", flavour),
        ("--server-model", server_model),
    ] {
        if let Some(v) = value {
            out.extend([flag.to_string(), v.clone()]);
        }
    }
    out
}

/// Line mode: stream text, list tool calls, ask y/n on the terminal for deletes.
async fn plain_mode(
    task: String,
    mut events: mpsc::UnboundedReceiver<UiEvent>,
    cmds: mpsc::UnboundedSender<Command>,
) -> Result<()> {
    let ctrl_c = cmds.clone();
    tokio::spawn(async move {
        while tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\n(cancelling...)");
            let _ = ctrl_c.send(Command::Cancel);
        }
    });
    let mut out = std::io::stdout();
    let mut root = String::new();
    let rel = |root: &str, p: &str| {
        p.strip_prefix(root)
            .map(|r| r.trim_start_matches('/').to_string())
            .unwrap_or_else(|| p.to_string())
    };
    while let Some(ev) = events.recv().await {
        match ev {
            UiEvent::Ready {
                folder, backend, ..
            } => {
                println!("keel · {backend} · {folder}\n> {task}\n");
                root = folder;
                cmds.send(Command::Prompt(task.clone()))?;
            }
            UiEvent::Text(t) => {
                print!("{t}");
                out.flush()?;
            }
            UiEvent::ToolStart { title, .. } => println!("\n  ▸ {title}"),
            UiEvent::ToolEnd {
                ok, diff, output, ..
            } => {
                let mark = if ok { "✓" } else { "✗" };
                match diff {
                    Some(d) => {
                        let (old, new) = (
                            d.old.as_deref().unwrap_or("").lines().count(),
                            d.new.lines().count(),
                        );
                        println!("  {mark} {} ({old} -> {new} lines)", rel(&root, &d.path));
                    }
                    None => println!(
                        "  {mark} {}",
                        output
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(100)
                            .collect::<String>()
                    ),
                }
            }
            UiEvent::Permission(ask) => {
                println!("\n  ⚠ {}\n    {}", ask.title, ask.command);
                for f in ask.files.iter().take(30) {
                    println!("      {}", rel(&root, f));
                }
                if ask.files.len() > 30 {
                    println!("      ... and {} more", ask.files.len() - 30);
                }
                let allow = tokio::task::spawn_blocking(ask_yes_no)
                    .await
                    .unwrap_or(false);
                println!("    {}", if allow { "deleting" } else { "kept" });
                ask.answer(allow);
            }
            UiEvent::TurnEnd { stop } => {
                println!("\n\n[{stop}]");
                cmds.send(Command::Quit)?;
            }
            UiEvent::Error(e) => eprintln!("\nerror: {e}"),
            UiEvent::Closed => break,
        }
    }
    Ok(())
}

/// `y` or `n`, nothing else: Enter alone asks again. End of input (closed terminal) = no.
fn ask_yes_no() -> bool {
    let stdin = std::io::stdin();
    loop {
        print!("    Delete these files? [y/n] ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => return false,
            Ok(_) => match line.trim().to_lowercase().as_str() {
                "y" | "yes" => return true,
                "n" | "no" => return false,
                _ => {}
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_flags_pass_through_only_when_given() {
        assert!(server_flags(&None, &None, &None, &None).is_empty());
        let got = server_flags(
            &Some("http://127.0.0.1:9000".into()),
            &Some("chat".into()),
            &None,
            &Some("qwen3.5:4b".into()),
        );
        assert_eq!(
            got,
            [
                "--url",
                "http://127.0.0.1:9000",
                "--mode",
                "chat",
                "--server-model",
                "qwen3.5:4b"
            ]
        );
    }

    #[test]
    fn cli_parses_backend_flags() {
        let cli = Cli::try_parse_from([
            "keel",
            "run",
            "--backend",
            "mlx-server",
            "--mode",
            "raw",
            "tidy up",
        ])
        .expect("parses");
        let Some(Cmd::Run { backend, mode, .. }) = cli.cmd else {
            panic!("expected run")
        };
        assert_eq!(
            (backend.as_str(), mode.as_deref()),
            ("mlx-server", Some("raw"))
        );
    }

    #[test]
    fn doctor_passes_its_flags_through() {
        let cli = Cli::try_parse_from([
            "keel",
            "doctor",
            "--backend",
            "server",
            "--url",
            "http://h:1",
        ])
        .expect("parses");
        let Some(Cmd::Doctor { args }) = cli.cmd else {
            panic!("expected doctor")
        };
        assert_eq!(args, ["--backend", "server", "--url", "http://h:1"]);
    }

    #[test]
    fn compare_and_eval_pass_their_args_through() {
        let cli =
            Cli::try_parse_from(["keel", "compare", "a,b", "c", "--pair", "try"]).expect("parses");
        let Some(Cmd::Compare { args }) = cli.cmd else {
            panic!("expected compare")
        };
        assert_eq!(args, ["a,b", "c", "--pair", "try"]);
        let cli = Cli::try_parse_from([
            "keel", "eval", "--suite", "agent", "--arm", "x", "--split", "dev",
        ])
        .expect("parses");
        let Some(Cmd::Eval { args }) = cli.cmd else {
            panic!("expected eval")
        };
        assert_eq!(args, ["--suite", "agent", "--arm", "x", "--split", "dev"]);
    }
}
