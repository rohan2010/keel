//! keel's mascot, the block bot (chosen 2026-11-20): shown when a session starts and by a bare `keel`.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// The bot, 9 columns wide, drawn with block characters.
pub const BOT: [&str; 5] = [
    "    ╷    ",
    "  ▄█████▄",
    "  ██▀█▀██",
    "  ███▄███",
    "   ▀▀ ▀▀ ",
];

/// One-line form for tight places (headers).
pub const SMALL: &str = "[▪▪]";

pub const TAGLINE: &str = "a steady harness for small local models";

/// A folder as the banner shows it: the home folder as `~`, and at most `max` characters (the end kept: the end
/// of a path is the part that tells folders apart), so the line beside the bot never wraps.
pub fn short_path(path: &str, max: usize) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let p = match path.strip_prefix(&home) {
        Some(rest) if !home.is_empty() && (rest.is_empty() || rest.starts_with('/')) => {
            format!("~{rest}")
        }
        _ => path.to_string(),
    };
    let n = p.chars().count();
    if n <= max {
        p
    } else {
        format!("…{}", p.chars().skip(n + 1 - max).collect::<String>())
    }
}

/// The bot's colour.
pub const COLOR: Color = Color::Cyan;

/// The bot with up to three lines beside it (name, tagline, one detail line), as TUI lines.
pub fn banner(detail: &str) -> Vec<Line<'static>> {
    let side = [
        Span::styled("keel", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(TAGLINE, Style::default().add_modifier(Modifier::DIM)),
        Span::styled(
            detail.to_string(),
            Style::default().add_modifier(Modifier::DIM),
        ),
    ];
    BOT.iter()
        .enumerate()
        .map(|(i, row)| {
            let mut spans = vec![Span::styled(row.to_string(), Style::default().fg(COLOR))];
            if (2..5).contains(&i) {
                spans.push(Span::raw("    "));
                spans.push(side[i - 2].clone());
            }
            Line::from(spans)
        })
        .collect()
}

/// The same banner as plain terminal text (bare `keel`): cyan bot and bold name when `color`.
pub fn text(detail: &str, color: bool) -> String {
    let (c, b, d, r) = if color {
        ("\x1b[36m", "\x1b[1m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    let side = [
        format!("{b}keel{r}"),
        format!("{d}{TAGLINE}{r}"),
        format!("{d}{detail}{r}"),
    ];
    BOT.iter()
        .enumerate()
        .map(|(i, row)| {
            let beside = if (2..5).contains(&i) {
                format!("    {}", side[i - 2])
            } else {
                String::new()
            };
            format!("{c}{row}{r}{beside}").trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_the_same_width_and_fit_80_columns() {
        let w = BOT[0].chars().count();
        assert!(BOT.iter().all(|r| r.chars().count() == w));
        let longest = text("qwen3.5-4b · a folder with a long name somewhere", false)
            .lines()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0);
        assert!(longest <= 80, "{longest}");
    }

    #[test]
    fn long_folders_are_shortened_from_the_front() {
        let s = short_path(
            "/a/very/long/temporary/folder/name/that/goes/on/and/on/Downloads",
            30,
        );
        assert_eq!(s.chars().count(), 30);
        assert!(s.starts_with('…') && s.ends_with("/on/Downloads"));
        assert_eq!(short_path("/short", 30), "/short");
    }

    #[test]
    fn plain_text_has_no_colour_codes() {
        let t = text("detail", false);
        assert!(
            !t.contains('\x1b')
                && t.contains("keel")
                && t.contains(TAGLINE)
                && t.contains("▄█████▄")
        );
    }
}
