//! keel's terminal UI (ratatui, inline): finished output goes into the normal terminal scrollback (so scrolling and
//! copy-paste work); a small live area at the bottom shows what is happening now. The UI only shows what keel-core
//! sends and answers its questions; every rule (the delete guard included) lives in the core.

pub mod app;
pub mod logo;
pub mod md;
pub mod view;

use std::time::{Duration, Instant};

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use std::io::{Stdout, Write, stdout};

use crossterm::cursor::MoveTo;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::style::{
    Attribute, Color as CColor, Print, ResetColor, SetAttribute, SetBackgroundColor,
    SetForegroundColor,
};
use crossterm::terminal::{Clear, ClearType};
use crossterm::{execute, queue};
use keel_protocol::{Command, PermissionAsk, UiEvent};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Size;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::{Terminal, TerminalOptions, Viewport};
use tokio::sync::mpsc;

use app::{Action, App, Key, Phase};

type Term = Terminal<CrosstermBackend<Stdout>>;

/// Run the UI for one task until the user quits or keel-core stops.
///
/// How the screen is handled (inline: finished output stays in the normal scrollback). Two jobs are not left to
/// ratatui, because they broke in real terminals (VS Code, 2026-11-20): finished lines are written here as plain
/// styled text without padding to the terminal width (padded rows were re-wrapped into two rows when the terminal
/// got narrower), and a resize is handled here before ratatui sees it (ratatui clears the whole screen when the width
/// shrinks, which erased the conversation). The live area is a fresh ratatui inline viewport each time; after every
/// frame the hidden cursor is parked at its top-left, so a resize can wipe from there down and start a new one.
pub async fn run(
    task: String,
    events: mpsc::UnboundedReceiver<UiEvent>,
    cmds: mpsc::UnboundedSender<Command>,
) -> Result<()> {
    let mut terminal = ratatui::init_with_options(TerminalOptions {
        viewport: Viewport::Inline(view::HEIGHT),
    });
    let res = event_loop(&mut terminal, task, events, cmds).await;
    let _ = wipe_live_area(&mut terminal);
    ratatui::restore();
    res
}

/// A new live area at the cursor (ratatui asks the terminal where the cursor is, and scrolls if needed).
fn new_live_area() -> Result<Term> {
    Ok(Terminal::with_options(
        CrosstermBackend::new(stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(view::HEIGHT),
        },
    )?)
}

/// Erase the live area (from its top row to the end of the screen) and leave the cursor at its top-left.
fn wipe_live_area(terminal: &mut Term) -> Result<()> {
    let top = terminal.get_frame().area().y;
    execute!(stdout(), MoveTo(0, top), Clear(ClearType::FromCursorDown))?;
    Ok(())
}

/// Keep the (hidden) cursor at the live area's top-left between frames: that row survives the terminal re-wrapping
/// its lines on a resize, so the live area can always be found and wiped.
fn park_cursor(terminal: &mut Term) -> Result<()> {
    let top = terminal.get_frame().area().y;
    execute!(stdout(), MoveTo(0, top))?;
    Ok(())
}

/// The terminal changed size: wipe from the parked cursor down (the old live area, however it was re-wrapped) and
/// start a new live area there. Runs before ratatui's own resize handling can clear the screen.
fn on_resize(terminal: &mut Term, size: &mut Size) -> Result<()> {
    let now = terminal.size()?;
    if now != *size {
        execute!(stdout(), Clear(ClearType::FromCursorDown))?;
        *terminal = new_live_area()?;
        *size = terminal.size()?;
    }
    Ok(())
}

fn ccolor(c: Color) -> CColor {
    match c {
        Color::Reset => CColor::Reset,
        Color::Black => CColor::Black,
        Color::Red => CColor::DarkRed,
        Color::Green => CColor::DarkGreen,
        Color::Yellow => CColor::DarkYellow,
        Color::Blue => CColor::DarkBlue,
        Color::Magenta => CColor::DarkMagenta,
        Color::Cyan => CColor::DarkCyan,
        Color::Gray => CColor::Grey,
        Color::DarkGray => CColor::DarkGrey,
        Color::LightRed => CColor::Red,
        Color::LightGreen => CColor::Green,
        Color::LightYellow => CColor::Yellow,
        Color::LightBlue => CColor::Blue,
        Color::LightMagenta => CColor::Magenta,
        Color::LightCyan => CColor::Cyan,
        Color::White => CColor::White,
        Color::Rgb(r, g, b) => CColor::Rgb { r, g, b },
        Color::Indexed(i) => CColor::AnsiValue(i),
    }
}

/// One styled line as terminal text: no padding after the last character, so a narrower terminal re-wraps it
/// correctly; a line longer than the terminal is soft-wrapped by the terminal itself.
fn write_line(out: &mut impl Write, line: &Line) -> Result<()> {
    for span in &line.spans {
        let st: Style = line.style.patch(span.style);
        queue!(out, SetAttribute(Attribute::Reset), ResetColor)?;
        if let Some(fg) = st.fg {
            queue!(out, SetForegroundColor(ccolor(fg)))?;
        }
        if let Some(bg) = st.bg {
            queue!(out, SetBackgroundColor(ccolor(bg)))?;
        }
        for (m, a) in [
            (Modifier::BOLD, Attribute::Bold),
            (Modifier::DIM, Attribute::Dim),
            (Modifier::ITALIC, Attribute::Italic),
            (Modifier::UNDERLINED, Attribute::Underlined),
            (Modifier::REVERSED, Attribute::Reverse),
            (Modifier::CROSSED_OUT, Attribute::CrossedOut),
        ] {
            if st.add_modifier.contains(m) {
                queue!(out, SetAttribute(a))?;
            }
        }
        queue!(out, Print(span.content.as_ref()))?;
    }
    queue!(
        out,
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\r\n")
    )?;
    Ok(())
}

/// Finished lines into the scrollback: wipe the live area, write the lines where it was, then a new live area below.
fn push(terminal: &mut Term, lines: Vec<Line<'static>>) -> Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    wipe_live_area(terminal)?;
    let mut out = stdout();
    for line in &lines {
        write_line(&mut out, line)?;
    }
    out.flush()?;
    *terminal = new_live_area()?;
    Ok(())
}

/// A terminal key -> the input line's key (and whether Control was held, for Ctrl-C / Ctrl-D). Covers what macOS
/// terminals send: Option-Left / Option-Right arrive as Alt-b / Alt-f in VS Code and Terminal.app.
fn map_key(k: &KeyEvent) -> Option<(Key, bool)> {
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    let alt = k.modifiers.contains(KeyModifiers::ALT);
    Some(match k.code {
        KeyCode::Enter => (Key::Enter, false),
        KeyCode::Esc => (Key::Esc, false),
        KeyCode::Backspace if alt || ctrl => (Key::KillWordBack, false),
        KeyCode::Backspace => (Key::Backspace, false),
        KeyCode::Delete => (Key::Delete, false),
        KeyCode::Left if alt || ctrl => (Key::WordLeft, false),
        KeyCode::Right if alt || ctrl => (Key::WordRight, false),
        KeyCode::Left => (Key::Left, false),
        KeyCode::Right => (Key::Right, false),
        KeyCode::Home => (Key::Home, false),
        KeyCode::End => (Key::End, false),
        KeyCode::Up => (Key::Up, false),
        KeyCode::Down => (Key::Down, false),
        KeyCode::Char(c) if ctrl => match c {
            'a' => (Key::Home, false),
            'e' => (Key::End, false),
            'b' => (Key::Left, false),
            'f' => (Key::Right, false),
            'u' => (Key::KillToStart, false),
            'k' => (Key::KillToEnd, false),
            'w' => (Key::KillWordBack, false),
            'p' => (Key::Up, false),
            'n' => (Key::Down, false),
            _ => (Key::Char(c), true),
        },
        KeyCode::Char('b') if alt => (Key::WordLeft, false),
        KeyCode::Char('f') if alt => (Key::WordRight, false),
        KeyCode::Char(c) => (Key::Char(c), false),
        _ => return None,
    })
}

/// Keys come from a plain thread that polls with a short timeout. (crossterm's async `EventStream` keeps its reader
/// busy and swallows the terminal's reply to the cursor-position query that ratatui's inline viewport sends on
/// resize, which made the UI fail with "cursor position could not be read"; found by tests/test_tui_pty.py.)
fn key_thread(stop: Arc<AtomicBool>) -> mpsc::UnboundedReceiver<KeyEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(30)) {
                Ok(true) => {
                    if let Ok(Event::Key(k)) = event::read()
                        && tx.send(k).is_err()
                    {
                        break;
                    }
                }
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });
    rx
}

async fn event_loop(
    terminal: &mut Term,
    task: String,
    mut events: mpsc::UnboundedReceiver<UiEvent>,
    cmds: mpsc::UnboundedSender<Command>,
) -> Result<()> {
    let mut app = App::new(task.clone());
    let stop = Arc::new(AtomicBool::new(false));
    let mut keys = key_thread(stop.clone());
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let mut n_tick = 0usize;
    let mut pending: Option<PermissionAsk> = None; // the open delete question (dropped = "no")
    let mut size = terminal.size()?;
    loop {
        on_resize(terminal, &mut size)?; // before ratatui's own resize handling (it would clear the whole screen)
        terminal.draw(|f| view::draw(f, &app, Instant::now(), n_tick))?;
        park_cursor(terminal)?;
        tokio::select! {
            ev = events.recv() => {
                let Some(ev) = ev else { break };
                let lines = app.apply(&ev);
                match ev {
                    UiEvent::Ready { .. } if !task.trim().is_empty() => {
                        let _ = cmds.send(Command::Prompt(task.clone()));
                    }
                    UiEvent::Permission(p) => pending = Some(p),
                    UiEvent::Closed => { push(terminal, lines)?; break; }
                    _ => {}
                }
                push(terminal, lines)?;
            }
            key = keys.recv() => {
                let Some(k) = key else { continue };
                if k.kind != KeyEventKind::Press { continue; }
                let Some((key, ctrl)) = map_key(&k) else { continue };
                let action = app.press(key, ctrl, Instant::now());
                match action {
                    Action::None => {}
                    Action::Answer(allow) => {
                        if let Some(p) = pending.take() { p.answer(allow); }
                        let lines = app.answered(allow);
                        push(terminal, lines)?;
                    }
                    Action::Prompt(text) => {
                        push(terminal, vec![Line::from(format!("> {text}")), Line::raw("")])?;
                        let _ = cmds.send(Command::Prompt(text));
                    }
                    Action::Cancel => {
                        if let Some(p) = pending.take() {           // cancelling while asked = keep the files
                            p.answer(false);
                            push(terminal, app.answered(false))?;
                        }
                        let _ = cmds.send(Command::Cancel);
                    }
                    Action::Quit => break,
                }
            }
            _ = tick.tick() => { n_tick += 1; }
        }
        if app.phase == Phase::Closed {
            break;
        }
    }
    stop.store(true, Ordering::Relaxed);
    drop(pending); // an unanswered question becomes "no"
    let _ = cmds.send(Command::Quit);
    terminal.draw(|f| view::draw(f, &app, Instant::now(), n_tick))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(code: KeyCode, mods: KeyModifiers) -> Option<(Key, bool)> {
        map_key(&KeyEvent::new(code, mods))
    }

    #[test]
    fn macos_editing_keys_map_to_line_editing() {
        let none = KeyModifiers::NONE;
        assert_eq!(k(KeyCode::Left, none), Some((Key::Left, false)));
        assert_eq!(
            k(KeyCode::Left, KeyModifiers::ALT),
            Some((Key::WordLeft, false))
        );
        assert_eq!(
            k(KeyCode::Char('b'), KeyModifiers::ALT),
            Some((Key::WordLeft, false))
        ); // Option-Left in VS Code
        assert_eq!(
            k(KeyCode::Char('f'), KeyModifiers::ALT),
            Some((Key::WordRight, false))
        );
        assert_eq!(
            k(KeyCode::Char('a'), KeyModifiers::CONTROL),
            Some((Key::Home, false))
        ); // Cmd-Left in VS Code
        assert_eq!(
            k(KeyCode::Char('e'), KeyModifiers::CONTROL),
            Some((Key::End, false))
        );
        assert_eq!(
            k(KeyCode::Backspace, KeyModifiers::ALT),
            Some((Key::KillWordBack, false))
        );
        assert_eq!(
            k(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some((Key::Char('c'), true))
        ); // still cancels
        assert_eq!(k(KeyCode::Up, none), Some((Key::Up, false)));
    }

    #[test]
    fn history_lines_are_written_without_padding() -> Result<()> {
        let mut out = Vec::new();
        let line = Line::from(vec![
            ratatui::text::Span::raw("keel"),
            ratatui::text::Span::raw(" ready"),
        ]);
        write_line(&mut out, &line)?;
        let s = String::from_utf8(out)?;
        assert!(
            s.ends_with("\r\n") && s.contains("keel") && s.contains(" ready"),
            "{s:?}"
        );
        assert!(!s.contains("ready "), "no spaces after the text: {s:?}");
        Ok(())
    }
}
