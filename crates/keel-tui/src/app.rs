//! The TUI's state, kept pure (no terminal) so it can be tested: `apply` turns a core event into lines for the
//! scrollback, `key` turns a key press into an action.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

use keel_protocol::{Diff, UiEvent};

use crate::md::Md;

pub const MAX_DIFF_LINES: usize = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Connecting,
    Working,
    Asking,
    Idle,
    Closed,
}

/// What the delete modal shows (the answerable `PermissionAsk` itself stays in the event loop).
#[derive(Debug, Clone)]
pub struct Ask {
    pub title: String,
    pub command: String,
    pub files: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Answer(bool),
    Prompt(String),
    Cancel,
    Quit,
}

/// A key, as the input line understands it (the terminal-specific mapping lives in lib.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Enter,
    Esc,
    Backspace,
    Delete,
    Left,
    Right,
    Home,
    End,
    Up,
    Down,
    WordLeft,
    WordRight,
    /// Ctrl-U: delete from the start of the line to the cursor
    KillToStart,
    /// Ctrl-K: delete from the cursor to the end of the line
    KillToEnd,
    /// Ctrl-W / Option-Backspace: delete the word before the cursor
    KillWordBack,
}

pub struct App {
    pub task: String,
    pub folder: String,
    pub backend: String,
    pub phase: Phase,
    pub live: String,
    pub tool: Option<String>,
    pub ask: Option<Ask>,
    pub input: String,
    /// cursor position in `input`, in characters (0 = before the first one)
    pub cursor: usize,
    /// earlier prompts this session, oldest first (Up / Down recall them)
    pub history: Vec<String>,
    hist_pos: Option<usize>,
    draft: String,
    pub steps: usize,
    pub out_chars: usize,
    pub started: Option<Instant>,
    pub hint: Option<String>,
    pub md: Md,
    titles: HashMap<String, String>,
    last_ctrl_c: Option<Instant>,
}

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

impl App {
    pub fn new(task: String) -> Self {
        Self {
            task,
            folder: String::new(),
            backend: String::new(),
            phase: Phase::Connecting,
            live: String::new(),
            tool: None,
            ask: None,
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            hist_pos: None,
            draft: String::new(),
            steps: 0,
            out_chars: 0,
            started: None,
            hint: None,
            md: Md::default(),
            titles: HashMap::new(),
            last_ctrl_c: None,
        }
    }

    fn rel(&self, p: &str) -> String {
        p.strip_prefix(&self.folder)
            .map(|r| r.trim_start_matches('/').to_string())
            .unwrap_or_else(|| p.to_string())
    }

    /// Move finished text out of the live line into the scrollback.
    /// The model's text is markdown: each finished line is rendered (markers hidden, styles applied).
    fn flush_live(&mut self, all: bool, out: &mut Vec<Line<'static>>) {
        while let Some(i) = self.live.find('\n') {
            let line: String = self.live.drain(..=i).collect();
            out.extend(self.md.line(line.trim_end_matches('\n')));
        }
        if all && !self.live.is_empty() {
            let line = std::mem::take(&mut self.live);
            out.extend(self.md.line(&line));
        }
        if all {
            self.md = Md::default(); // a fence the model never closed ends with its message
        }
    }

    /// Apply one core event; returns the lines to add to the scrollback.
    pub fn apply(&mut self, ev: &UiEvent) -> Vec<Line<'static>> {
        let mut out = Vec::new();
        match ev {
            UiEvent::Ready {
                folder, backend, ..
            } => {
                self.folder = folder.clone();
                self.backend = backend.clone();
                let place = crate::logo::short_path(folder, 60); // never wraps beside the bot
                out.extend(crate::logo::banner(&format!("{backend} · {place}")));
                out.push(Line::raw(""));
                if self.task.trim().is_empty() {
                    // `keel run` without a task: wait for one (Ch7 P6)
                    self.phase = Phase::Idle;
                    out.push(Line::styled(
                        "Type what you want done in this folder, e.g. \"sort the photos into folders by year\". \
                         keel asks before it deletes anything.",
                        dim(),
                    ));
                } else {
                    self.phase = Phase::Working;
                    self.started = Some(Instant::now());
                    out.push(Line::from(vec![
                        Span::styled("> ", dim()),
                        Span::raw(self.task.clone()),
                    ]));
                }
                out.push(Line::raw(""));
            }
            UiEvent::Text(t) => {
                self.out_chars += t.chars().count();
                self.live.push_str(t);
                self.flush_live(false, &mut out);
            }
            UiEvent::ToolStart { id, title, .. } => {
                self.flush_live(true, &mut out);
                self.steps += 1;
                self.titles.insert(id.clone(), title.clone());
                self.tool = Some(title.clone());
            }
            UiEvent::ToolEnd {
                id,
                ok,
                output,
                diff,
            } => {
                self.tool = None;
                let title = self.titles.remove(id).unwrap_or_default();
                let (mark, color) = if *ok {
                    ("✓", Color::Green)
                } else {
                    ("✗", Color::Red)
                };
                let mut head = vec![
                    Span::styled(format!("  {mark} "), Style::default().fg(color)),
                    Span::raw(title),
                ];
                if let Some(d) = diff {
                    let (add, del) = diff_counts(d);
                    head.push(Span::styled(format!("  +{add} -{del}"), dim()));
                    out.push(Line::from(head));
                    out.extend(diff_lines(d, &self.rel(&d.path)));
                } else {
                    out.push(Line::from(head));
                    for l in output.lines().filter(|l| !l.trim().is_empty()).take(2) {
                        out.push(Line::styled(format!("    {}", trunc(l, 110)), dim()));
                    }
                }
            }
            UiEvent::Permission(p) => {
                self.flush_live(true, &mut out);
                self.phase = Phase::Asking;
                self.hint = None;
                out.push(Line::styled(
                    format!("  ⚠ {}", p.title),
                    Style::default().fg(Color::Yellow),
                ));
                for f in &p.files {
                    out.push(Line::styled(
                        format!("      {}", self.rel(f)),
                        Style::default().fg(Color::Yellow),
                    ));
                }
                self.ask = Some(Ask {
                    title: p.title.clone(),
                    command: p.command.clone(),
                    files: p.files.iter().map(|f| self.rel(f)).collect(),
                });
            }
            UiEvent::TurnEnd { stop } => {
                self.flush_live(true, &mut out);
                self.tool = None;
                if stop != "end_turn" {
                    out.push(Line::styled(format!("[{}]", stop_text(stop)), dim()));
                }
                out.push(Line::raw(""));
                self.phase = Phase::Idle;
            }
            UiEvent::Error(e) => out.push(Line::styled(
                format!("error: {e}"),
                Style::default().fg(Color::Red),
            )),
            UiEvent::Closed => {
                self.flush_live(true, &mut out);
                self.phase = Phase::Closed;
            }
        }
        out
    }

    /// The user answered the delete modal.
    pub fn answered(&mut self, allow: bool) -> Vec<Line<'static>> {
        let n = self.ask.take().map(|a| a.files.len()).unwrap_or(0);
        self.phase = Phase::Working;
        let (word, color) = if allow {
            ("deleting", Color::Red)
        } else {
            ("kept", Color::Green)
        };
        vec![Line::styled(
            format!("    {word} {n} file{}", if n == 1 { "" } else { "s" }),
            Style::default().fg(color),
        )]
    }

    /// One key press -> what to do (the older interface, kept for callers and tests): a character, or Enter / Esc /
    /// Backspace; `ctrl` = the Control modifier was held.
    pub fn key(
        &mut self,
        ch: Option<char>,
        enter: bool,
        esc: bool,
        backspace: bool,
        ctrl: bool,
        now: Instant,
    ) -> Action {
        let k = if enter {
            Key::Enter
        } else if esc {
            Key::Esc
        } else if backspace {
            Key::Backspace
        } else if let Some(c) = ch {
            Key::Char(c)
        } else {
            return Action::None;
        };
        self.press(k, ctrl, now)
    }

    /// One key press -> what to do. `ctrl` = the Control modifier was held (Ctrl-C, Ctrl-D).
    pub fn press(&mut self, key: Key, ctrl: bool, now: Instant) -> Action {
        if ctrl && key == Key::Char('c') {
            return self.ctrl_c(now);
        }
        match self.phase {
            Phase::Asking => match key {
                Key::Char('y') | Key::Char('Y') => Action::Answer(true),
                Key::Char('n') | Key::Char('N') | Key::Esc => Action::Answer(false),
                _ => {
                    self.hint = Some("press y to delete or n to keep".into()); // Enter does nothing: no default
                    Action::None
                }
            },
            Phase::Idle => {
                if ctrl && key == Key::Char('d') || key == Key::Esc && self.input.is_empty() {
                    return Action::Quit;
                }
                if key == Key::Enter {
                    let text = self.input.trim().to_string();
                    if text.is_empty() {
                        return Action::None;
                    }
                    if self.history.last() != Some(&text) {
                        self.history.push(text.clone());
                    }
                    self.hist_pos = None;
                    self.set_input(String::new());
                    self.phase = Phase::Working;
                    self.started.get_or_insert_with(Instant::now);
                    return Action::Prompt(text);
                }
                if !ctrl
                    || matches!(
                        key,
                        Key::KillToStart
                            | Key::KillToEnd
                            | Key::KillWordBack
                            | Key::Home
                            | Key::End
                    )
                {
                    self.edit(key);
                }
                Action::None
            }
            Phase::Closed => Action::Quit,
            _ => Action::None,
        }
    }

    fn set_input(&mut self, text: String) {
        self.cursor = text.chars().count();
        self.input = text;
    }

    /// Line editing on `input` at `cursor` (all positions in characters, so accents and emoji move as one).
    fn edit(&mut self, key: Key) {
        let mut chars: Vec<char> = self.input.chars().collect();
        let mut at = self.cursor.min(chars.len());
        let word_start = |chars: &[char], mut i: usize| {
            while i > 0 && chars[i - 1].is_whitespace() {
                i -= 1;
            }
            while i > 0 && !chars[i - 1].is_whitespace() {
                i -= 1;
            }
            i
        };
        let word_end = |chars: &[char], mut i: usize| {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
            i
        };
        match key {
            Key::Char(c) => {
                chars.insert(at, c);
                at += 1;
            }
            Key::Backspace if at > 0 => {
                chars.remove(at - 1);
                at -= 1;
            }
            Key::Delete if at < chars.len() => {
                chars.remove(at);
            }
            Key::Left => at = at.saturating_sub(1),
            Key::Right => at = (at + 1).min(chars.len()),
            Key::Home => at = 0,
            Key::End => at = chars.len(),
            Key::WordLeft => at = word_start(&chars, at),
            Key::WordRight => at = word_end(&chars, at),
            Key::KillToStart => {
                chars.drain(..at);
                at = 0;
            }
            Key::KillToEnd => {
                chars.truncate(at);
            }
            Key::KillWordBack => {
                let from = word_start(&chars, at);
                chars.drain(from..at);
                at = from;
            }
            Key::Up | Key::Down => {
                self.recall(key == Key::Up);
                return;
            }
            _ => return,
        }
        self.input = chars.into_iter().collect();
        self.cursor = at;
    }

    /// Up / Down: step through earlier prompts; Down past the newest brings back what was being typed.
    fn recall(&mut self, older: bool) {
        if self.history.is_empty() {
            return;
        }
        let next = match (self.hist_pos, older) {
            (None, true) => {
                self.draft = self.input.clone();
                Some(self.history.len() - 1)
            }
            (None, false) => return,
            (Some(i), true) => Some(i.saturating_sub(1)),
            (Some(i), false) if i + 1 < self.history.len() => Some(i + 1),
            (Some(_), false) => None,
        };
        self.hist_pos = next;
        let text = match next {
            Some(i) => self.history[i].clone(),
            None => std::mem::take(&mut self.draft),
        };
        self.set_input(text);
    }

    fn ctrl_c(&mut self, now: Instant) -> Action {
        let twice = self
            .last_ctrl_c
            .is_some_and(|t| now.duration_since(t) < Duration::from_secs(2));
        self.last_ctrl_c = Some(now);
        match self.phase {
            Phase::Idle | Phase::Closed | Phase::Connecting => Action::Quit,
            _ if twice => Action::Quit,
            _ => {
                self.hint = Some("cancelling (Ctrl-C again to quit)".into());
                Action::Cancel
            }
        }
    }

    /// Rough output speed for the status line (characters / 4 ≈ tokens).
    pub fn tokens_per_s(&self, now: Instant) -> f64 {
        self.started
            .map(|t| (self.out_chars as f64 / 4.0) / now.duration_since(t).as_secs_f64().max(0.1))
            .unwrap_or(0.0)
    }
}

fn stop_text(stop: &str) -> &str {
    match stop {
        "cancelled" => "cancelled",
        "max_turn_requests" => "stopped: step limit reached",
        "max_tokens" => "stopped: the reply or the conversation got too long",
        other => other,
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

pub fn diff_counts(d: &Diff) -> (usize, usize) {
    let old = d.old.clone().unwrap_or_default();
    let td = TextDiff::from_lines(&old, &d.new);
    let (mut add, mut del) = (0, 0);
    for c in td.iter_all_changes() {
        match c.tag() {
            ChangeTag::Insert => add += 1,
            ChangeTag::Delete => del += 1,
            ChangeTag::Equal => {}
        }
    }
    (add, del)
}

/// The changed lines (with a line of context), coloured, at most MAX_DIFF_LINES.
pub fn diff_lines(d: &Diff, path: &str) -> Vec<Line<'static>> {
    let old = d.old.clone().unwrap_or_default();
    let td = TextDiff::from_lines(&old, &d.new);
    let mut out = vec![Line::styled(format!("    {path}"), dim())];
    let mut shown = 0;
    let mut hidden = 0;
    for group in td.grouped_ops(1) {
        for op in group {
            for c in td.iter_changes(&op) {
                if shown >= MAX_DIFF_LINES {
                    hidden += 1;
                    continue;
                }
                let text = c.value().trim_end_matches('\n');
                let (sign, style) = match c.tag() {
                    ChangeTag::Insert => ("+", Style::default().fg(Color::Green)),
                    ChangeTag::Delete => ("-", Style::default().fg(Color::Red)),
                    ChangeTag::Equal => (" ", dim()),
                };
                out.push(Line::styled(
                    format!("    {sign} {}", trunc(text, 110)),
                    style,
                ));
                shown += 1;
            }
        }
    }
    if hidden > 0 {
        out.push(Line::styled(format!("    … {hidden} more lines"), dim()));
    }
    out
}
