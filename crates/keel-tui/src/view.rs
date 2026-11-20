//! Drawing the live area at the bottom of the terminal. Everything finished is already in the scrollback; this
//! area shows the line being written, the tool running, the delete question, the input box and a status line.

use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

use crate::app::{App, Phase};

pub const HEIGHT: u16 = 9;
const SPIN: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn draw(f: &mut Frame, app: &App, now: Instant, tick: usize) {
    let [body, status] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(f.area());
    match app.phase {
        Phase::Asking => ask(f, app, body),
        _ => live(f, app, body, tick),
    }
    f.render_widget(Paragraph::new(status_line(app, now)), status);
}

fn live(f: &mut Frame, app: &App, area: Rect, tick: usize) {
    let mut lines: Vec<Line> = Vec::new();
    if !app.live.is_empty() {
        lines.push(app.md.peek(&app.live));
    }
    let spin = SPIN[tick % SPIN.len()];
    match app.phase {
        Phase::Connecting => lines.push(Line::styled(
            format!("{spin} starting keel-core and the model..."),
            Style::default().add_modifier(Modifier::DIM),
        )),
        Phase::Working => lines.push(match &app.tool {
            Some(t) => Line::from(vec![
                Span::styled(format!("{spin} "), Style::default().fg(Color::Cyan)),
                Span::raw(t.clone()),
            ]),
            None => Line::styled(
                format!("{spin} thinking"),
                Style::default().add_modifier(Modifier::DIM),
            ),
        }),
        Phase::Idle => {
            lines.push(Line::raw(""));
            lines.push(input_line(app));
            let what = if app.started.is_none() {
                "a task"
            } else {
                "a follow-up"
            };
            lines.push(Line::styled(
                format!("type {what} and press Enter · Esc or Ctrl-C to quit"),
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        Phase::Closed => lines.push(Line::styled(
            "keel-core stopped",
            Style::default().fg(Color::Red),
        )),
        Phase::Asking => {}
    }
    // keep the newest lines visible
    let h = area.height as usize;
    let skip = lines.len().saturating_sub(h);
    f.render_widget(
        Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>()).wrap(Wrap { trim: false }),
        area,
    );
}

fn ask(f: &mut Frame, app: &App, area: Rect) {
    let Some(a) = &app.ask else { return };
    let warn = Style::default().fg(Color::Yellow);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(warn)
        .title(Span::styled(
            format!(" {} ", a.title),
            warn.add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    let mut lines = vec![Line::styled(
        a.command.clone(),
        Style::default().add_modifier(Modifier::DIM),
    )];
    let room = (inner.height as usize).saturating_sub(3);
    for p in a.files.iter().take(room) {
        lines.push(Line::raw(format!("  {p}")));
    }
    if a.files.len() > room {
        lines.push(Line::styled(
            format!("  … and {} more (listed above)", a.files.len() - room),
            Style::default().add_modifier(Modifier::DIM),
        ));
    }
    lines.push(Line::from(vec![
        Span::styled(" y ", Style::default().fg(Color::Black).bg(Color::Red)),
        Span::raw(" delete   "),
        Span::styled(" n ", Style::default().fg(Color::Black).bg(Color::Green)),
        Span::raw(" keep"),
    ]));
    if let Some(h) = &app.hint {
        lines.push(Line::styled(h.clone(), warn));
    }
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn status_line(app: &App, now: Instant) -> Line<'static> {
    let dim = Style::default().add_modifier(Modifier::DIM);
    let state = match app.phase {
        Phase::Connecting => "starting",
        Phase::Working => "working",
        Phase::Asking => "waiting for you",
        Phase::Idle => "ready",
        Phase::Closed => "stopped",
    };
    let mut s = format!(
        " keel · {} · {state}",
        if app.backend.is_empty() {
            "…"
        } else {
            &app.backend
        }
    );
    if app.steps > 0 {
        s.push_str(&format!(" · step {}", app.steps));
    }
    if app.phase == Phase::Working {
        // no speed until text streams: some servers return a reply whole (mlx_lm.server raw mode)
        if app.out_chars > 0 {
            s.push_str(&format!(" · ~{:.0} tok/s", app.tokens_per_s(now)));
        }
        s.push_str(" · Ctrl-C cancel");
    }
    if let (Some(h), true) = (&app.hint, app.phase != Phase::Asking) {
        s.push_str(&format!(" · {h}"));
    }
    Line::styled(s, dim)
}

/// The input line with a visible cursor at the edit position: the character under it is shown reversed (a block when
/// the cursor is at the end).
pub fn input_line(app: &App) -> Line<'static> {
    let chars: Vec<char> = app.input.chars().collect();
    let at = app.cursor.min(chars.len());
    let before: String = chars[..at].iter().collect();
    let under: String = chars
        .get(at)
        .map(|c| c.to_string())
        .unwrap_or_else(|| " ".into());
    let after: String = chars
        .get(at + 1..)
        .map(|r| r.iter().collect())
        .unwrap_or_default();
    Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Cyan)),
        Span::raw(before),
        Span::styled(under, Style::default().fg(Color::Black).bg(Color::Cyan)),
        Span::raw(after),
    ])
}
