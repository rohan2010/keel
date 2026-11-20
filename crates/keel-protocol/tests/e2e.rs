//! End to end against the real keel-core (scripted backend: no model, no GPU). Needs `uv` and this checkout.

use std::path::PathBuf;

use keel_protocol::{Command, CoreSpec, UiEvent, run_session};
use tokio::sync::mpsc;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

const DELETE: &str = "<tool_call>\n<function=delete>\n<parameter=paths>\n[\"old.log\"]\n</parameter>\n</function>\n</tool_call>";

#[derive(Clone, Copy, Debug)]
enum Answer {
    Yes,
    No,
    Drop,
}

struct Seen {
    text: String,
    kinds: Vec<String>,
    stops: Vec<String>,
    asked: Vec<Vec<String>>,
}

async fn session(answer: Answer) -> (tempfile::TempDir, Seen) {
    let tmp = tempfile::tempdir().expect("tmp");
    let folder = tmp.path().join("proj");
    std::fs::create_dir(&folder).expect("mkdir");
    std::fs::write(folder.join("old.log"), "log").expect("write");
    let script = tmp.path().join("script.json");
    let replies = serde_json::json!([format!("Cleaning up.\n{DELETE}"), "All done."]);
    std::fs::write(&script, replies.to_string()).expect("script");
    let extra = vec![
        "--backend".into(),
        "scripted".into(),
        "--script".into(),
        script.display().to_string(),
        "--state-dir".into(),
        tmp.path().join("state").display().to_string(),
    ];
    let core = CoreSpec::from_checkout(&repo(), &extra, tmp.path().join("core.log"));
    let (etx, mut erx) = mpsc::unbounded_channel();
    let (ctx, crx) = mpsc::unbounded_channel();
    let task = tokio::spawn(run_session(core, folder.clone(), etx, crx));
    let mut seen = Seen {
        text: String::new(),
        kinds: vec![],
        stops: vec![],
        asked: vec![],
    };
    while let Some(ev) = erx.recv().await {
        match ev {
            UiEvent::Ready { .. } => {
                ctx.send(Command::Prompt("clean up".into())).expect("send");
            }
            UiEvent::Text(t) => seen.text.push_str(&t),
            UiEvent::ToolStart { kind, .. } => seen.kinds.push(kind),
            UiEvent::Permission(p) => {
                seen.asked.push(p.files.clone());
                match answer {
                    Answer::Yes => p.answer(true),
                    Answer::No => p.answer(false),
                    Answer::Drop => drop(p),
                }
            }
            UiEvent::TurnEnd { stop } => {
                seen.stops.push(stop);
                ctx.send(Command::Quit).expect("quit");
            }
            UiEvent::Closed => break,
            _ => {}
        }
    }
    task.await.expect("join").expect("session");
    (tmp, seen)
}

#[tokio::test(flavor = "multi_thread")]
async fn yes_deletes() {
    let (tmp, seen) = session(Answer::Yes).await;
    assert!(!tmp.path().join("proj/old.log").exists());
    assert_eq!(seen.kinds, vec!["delete"]);
    assert_eq!(seen.asked.len(), 1);
    assert!(
        seen.asked[0][0].ends_with("proj/old.log"),
        "{:?}",
        seen.asked
    );
    assert!(
        seen.text.contains("Cleaning up.")
            && seen.text.contains("All done.")
            && !seen.text.contains("<tool_call>")
    );
    assert_eq!(seen.stops, vec!["end_turn"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn no_keeps_the_file() {
    let (tmp, seen) = session(Answer::No).await;
    assert_eq!(
        std::fs::read_to_string(tmp.path().join("proj/old.log")).expect("kept"),
        "log"
    );
    assert_eq!(seen.asked.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unanswered_question_keeps_the_file() {
    let (tmp, _) = session(Answer::Drop).await;
    assert_eq!(
        std::fs::read_to_string(tmp.path().join("proj/old.log")).expect("kept"),
        "log"
    );
}
