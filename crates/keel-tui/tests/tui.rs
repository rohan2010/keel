//! The TUI's logic and drawing, without a terminal: app state transitions, key handling (the delete modal takes
//! only y / n / Esc; Enter does nothing), and snapshots of the live area.

use std::time::{Duration, Instant};

use keel_protocol::{Diff, PermissionAsk, UiEvent};
use keel_tui::app::{Action, App, Key, Phase};
use keel_tui::view;
use ratatui::Terminal;
use ratatui::backend::TestBackend;

fn text(lines: &[ratatui::text::Line]) -> String {
    lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn ready(app: &mut App) {
    app.apply(&UiEvent::Ready {
        session_id: "s1".into(),
        folder: "/Users/x/proj".into(),
        backend: "mlx".into(),
    });
}

fn screen(app: &App) -> String {
    let mut t = Terminal::new(TestBackend::new(60, view::HEIGHT)).expect("terminal");
    t.draw(|f| view::draw(f, app, Instant::now(), 0))
        .expect("draw");
    let buf = t.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn streaming_text_goes_to_scrollback_line_by_line() {
    let mut app = App::new("tidy".into());
    ready(&mut app);
    assert!(app.apply(&UiEvent::Text("Hello wor".into())).is_empty());
    assert_eq!(app.live, "Hello wor");
    let out = app.apply(&UiEvent::Text("ld\nNext".into()));
    assert_eq!(text(&out), "Hello world");
    assert_eq!(app.live, "Next");
    let out = app.apply(&UiEvent::ToolStart {
        id: "c1".into(),
        title: "read_file a.txt".into(),
        kind: "read".into(),
    });
    assert_eq!(text(&out), "Next"); // a tool call flushes the partial line
    assert_eq!(app.tool.as_deref(), Some("read_file a.txt"));
}

#[test]
fn tool_end_shows_a_coloured_diff_with_relative_path() {
    let mut app = App::new("t".into());
    ready(&mut app);
    app.apply(&UiEvent::ToolStart {
        id: "c1".into(),
        title: "edit_file notes.md".into(),
        kind: "edit".into(),
    });
    let out = app.apply(&UiEvent::ToolEnd {
        id: "c1".into(),
        ok: true,
        output: "Replaced".into(),
        diff: Some(Diff {
            path: "/Users/x/proj/notes.md".into(),
            old: Some("# Notes\n- old\n".into()),
            new: "# Notes\n- new\n".into(),
        }),
    });
    let t = text(&out);
    assert!(t.contains("✓ edit_file notes.md  +1 -1"), "{t}");
    assert!(
        t.contains("    notes.md") && t.contains("- - old") && t.contains("+ - new"),
        "{t}"
    );
    assert!(app.tool.is_none());
}

#[test]
fn delete_modal_takes_only_y_n_or_esc() {
    let mut app = App::new("t".into());
    ready(&mut app);
    let (ask, _rx) = PermissionAsk::new(
        "Delete 2 files?".into(),
        "rm a.log b.log".into(),
        vec!["/Users/x/proj/a.log".into(), "/Users/x/proj/b.log".into()],
    );
    let out = app.apply(&UiEvent::Permission(ask));
    assert_eq!(app.phase, Phase::Asking);
    assert!(text(&out).contains("a.log") && text(&out).contains("b.log")); // full list also in the scrollback
    let now = Instant::now();
    assert_eq!(app.key(None, true, false, false, false, now), Action::None); // Enter: no default
    assert!(app.hint.as_deref().unwrap_or("").contains("press y"));
    assert_eq!(
        app.key(Some('x'), false, false, false, false, now),
        Action::None
    );
    assert_eq!(
        app.key(Some('n'), false, false, false, false, now),
        Action::Answer(false)
    );
    assert_eq!(
        app.key(None, false, true, false, false, now),
        Action::Answer(false)
    ); // Esc = keep
    assert_eq!(
        app.key(Some('y'), false, false, false, false, now),
        Action::Answer(true)
    );
    let out = app.answered(false);
    assert_eq!(text(&out), "    kept 2 files");
    assert_eq!(app.phase, Phase::Working);
}

#[test]
fn ctrl_c_cancels_then_quits_and_idle_takes_follow_ups() {
    let mut app = App::new("t".into());
    ready(&mut app);
    let t0 = Instant::now();
    assert_eq!(
        app.key(Some('c'), false, false, false, true, t0),
        Action::Cancel
    );
    assert_eq!(
        app.key(
            Some('c'),
            false,
            false,
            false,
            true,
            t0 + Duration::from_millis(500)
        ),
        Action::Quit
    );
    app.apply(&UiEvent::TurnEnd {
        stop: "end_turn".into(),
    });
    assert_eq!(app.phase, Phase::Idle);
    for c in "and now?".chars() {
        app.key(Some(c), false, false, false, false, t0);
    }
    app.key(None, false, false, true, false, t0); // backspace
    assert_eq!(
        app.key(None, true, false, false, false, t0),
        Action::Prompt("and now".into())
    );
    assert_eq!(app.phase, Phase::Working);
}

#[test]
fn screens() {
    let mut app = App::new("tidy my folder".into());
    assert!(screen(&app).contains("starting keel-core"));
    ready(&mut app);
    app.apply(&UiEvent::Text("I'll look at the files".into()));
    assert!(screen(&app).contains("I'll look at the files")); // the line being written
    app.apply(&UiEvent::ToolStart {
        id: "c1".into(),
        title: "list_files .".into(),
        kind: "read".into(),
    });
    let s = screen(&app); // (that line moved to the scrollback)
    assert!(
        !s.contains("I'll look") && s.contains("list_files .") && s.contains("working · step 1"),
        "{s}"
    );

    let (ask, _rx) = PermissionAsk::new(
        "Delete 1 file?".into(),
        "delete old.log".into(),
        vec!["/Users/x/proj/old.log".into()],
    );
    app.apply(&UiEvent::Permission(ask));
    let s = screen(&app);
    assert!(
        s.contains("Delete 1 file?")
            && s.contains("old.log")
            && s.contains(" y  delete")
            && s.contains(" n  keep")
            && s.contains("waiting for you"),
        "{s}"
    );
    println!("--- delete modal ---\n{s}");

    app.answered(true);
    app.apply(&UiEvent::TurnEnd {
        stop: "end_turn".into(),
    });
    let s = screen(&app);
    assert!(
        s.lines().any(|l| l.starts_with('>')) && s.contains("follow-up") && s.contains("ready"),
        "{s}"
    );
    println!("--- idle ---\n{s}");
}

fn bold(l: &ratatui::text::Line, word: &str) -> bool {
    use ratatui::style::Modifier;
    l.spans
        .iter()
        .any(|s| s.content.contains(word) && s.style.add_modifier.contains(Modifier::BOLD))
}

#[test]
fn model_markdown_is_rendered_not_printed() {
    // Lines from a real demo take (Qwen3.5-4B's summary), plus the other markers small models use.
    let mut app = App::new("t".into());
    ready(&mut app);
    let out = app.apply(&UiEvent::Text(
        "**Files sorted by type:**\n- **Images:** IMG_2041 (1).jpg, IMG_2077.jpg\n\
         - `IMG_2041 (1).jpg` (duplicate of Images/IMG_2041.jpg)\n## Done\n\
         1. Created **three** folders\nkeep invoice_march.pdf and my_notes_v2.txt as *they* are\n\
         a * b = 2*3, and **unclosed stays\n```\nrm -rf **x**\n```\n---\n"
            .into(),
    ));
    let t = text(&out);
    assert_eq!(
        t,
        "Files sorted by type:\n• Images: IMG_2041 (1).jpg, IMG_2077.jpg\n\
         • IMG_2041 (1).jpg (duplicate of Images/IMG_2041.jpg)\nDone\n\
         1. Created three folders\nkeep invoice_march.pdf and my_notes_v2.txt as they are\n\
         a * b = 2*3, and **unclosed stays\n  rm -rf **x**\n────────────────────────────────────────"
    );
    assert!(bold(&out[0], "Files sorted by type:"));
    assert!(bold(&out[1], "Images:") && !bold(&out[1], "IMG_2077"));
    assert!(bold(&out[3], "Done"));
    assert!(bold(&out[4], "three") && !bold(&out[4], "Created"));
}

#[test]
fn live_line_is_rendered_and_an_unclosed_fence_ends_with_the_message() {
    let mut app = App::new("t".into());
    ready(&mut app);
    app.apply(&UiEvent::Text("```\nls -la\n".into()));
    app.apply(&UiEvent::Text("**Look".into())); // unfinished: shown as typed until it closes
    assert!(screen(&app).contains("**Look"));
    app.apply(&UiEvent::Text("ing**".into()));
    app.apply(&UiEvent::ToolStart {
        id: "c1".into(),
        title: "list_files .".into(),
        kind: "read".into(),
    });
    let out = app.apply(&UiEvent::Text("**after** the tool\n".into()));
    assert_eq!(text(&out), "after the tool"); // not stuck in code mode
}

// ---- the input line: cursor movement and editing (2026-11-20: arrow keys did nothing) ------------------------------

fn idle() -> App {
    let mut app = App::new(String::new());
    app.phase = Phase::Idle;
    app
}

fn typed(app: &mut App, s: &str) {
    for c in s.chars() {
        app.press(Key::Char(c), false, Instant::now());
    }
}

#[test]
fn arrows_move_the_cursor_and_typing_inserts_there() {
    let mut app = idle();
    typed(&mut app, "hello world");
    for _ in 0..5 {
        app.press(Key::Left, false, Instant::now());
    }
    typed(&mut app, "big ");
    assert_eq!(app.input, "hello big world");
    app.press(Key::Home, false, Instant::now());
    typed(&mut app, "say ");
    app.press(Key::End, false, Instant::now());
    typed(&mut app, "!");
    assert_eq!(app.input, "say hello big world!");
    assert_eq!(app.cursor, app.input.chars().count());
    app.press(Key::Right, false, Instant::now()); // at the end: stays
    assert_eq!(app.cursor, app.input.chars().count());
}

#[test]
fn backspace_and_delete_work_at_the_cursor() {
    let mut app = idle();
    typed(&mut app, "abcd");
    app.press(Key::Left, false, Instant::now());
    app.press(Key::Left, false, Instant::now());
    app.press(Key::Backspace, false, Instant::now()); // removes b
    app.press(Key::Delete, false, Instant::now()); // removes c
    assert_eq!((app.input.as_str(), app.cursor), ("ad", 1));
}

#[test]
fn word_jumps_and_kills() {
    let mut app = idle();
    typed(&mut app, "sort the photos now");
    app.press(Key::WordLeft, false, Instant::now());
    assert_eq!(app.cursor, "sort the photos ".len());
    app.press(Key::WordLeft, false, Instant::now());
    app.press(Key::KillWordBack, false, Instant::now()); // deletes "the "
    assert_eq!(app.input, "sort photos now");
    app.press(Key::WordRight, false, Instant::now());
    app.press(Key::KillToEnd, false, Instant::now());
    assert_eq!(app.input, "sort photos");
    app.press(Key::KillToStart, false, Instant::now());
    assert_eq!((app.input.as_str(), app.cursor), ("", 0));
}

#[test]
fn accented_and_emoji_characters_move_as_one() {
    let mut app = idle();
    typed(&mut app, "café 📁");
    app.press(Key::Left, false, Instant::now());
    app.press(Key::Backspace, false, Instant::now()); // the space before the emoji
    assert_eq!(app.input, "café📁");
}

#[test]
fn up_and_down_recall_earlier_prompts_and_keep_the_draft() {
    let mut app = idle();
    typed(&mut app, "first task");
    assert_eq!(
        app.press(Key::Enter, false, Instant::now()),
        Action::Prompt("first task".into())
    );
    app.phase = Phase::Idle;
    typed(&mut app, "second");
    app.press(Key::Enter, false, Instant::now());
    app.phase = Phase::Idle;
    typed(&mut app, "draft");
    app.press(Key::Up, false, Instant::now());
    assert_eq!(app.input, "second");
    app.press(Key::Up, false, Instant::now());
    assert_eq!(app.input, "first task");
    app.press(Key::Down, false, Instant::now());
    app.press(Key::Down, false, Instant::now());
    assert_eq!(app.input, "draft"); // past the newest: what was being typed comes back
}

#[test]
fn the_input_line_shows_the_cursor_where_it_is() {
    let mut app = idle();
    typed(&mut app, "abc");
    app.press(Key::Left, false, Instant::now());
    let line = view::input_line(&app);
    let parts: Vec<&str> = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert_eq!(parts, vec!["> ", "ab", "c", ""]); // the cursor cell is "c"
}
