//! A small markdown renderer for the model's replies, one line at a time (text streams in line by line).
//!
//! Covers what small models actually write: `# headings`, `- / * bullets`, `**bold**`, `*italic*`, `` `code` ``,
//! ``` fences, `> quotes`, `---` rules. Markers are hidden and turned into styles. Underscores are left alone on
//! purpose: file names like `invoice_march.pdf` are far more common in this tool than `_italic_`.
//! An unmatched marker stays as typed, so nothing the model wrote is ever lost.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug, Default, Clone)]
pub struct Md {
    in_code: bool,
}

fn code_style() -> Style {
    Style::default().fg(Color::Cyan)
}

impl Md {
    /// Render one finished line; `None` = draw nothing (a ``` fence line).
    pub fn line(&mut self, raw: &str) -> Option<Line<'static>> {
        if raw.trim_start().starts_with("```") {
            self.in_code = !self.in_code;
            return None;
        }
        Some(self.peek(raw))
    }

    /// Render a line without changing the fence state (the live, unfinished line).
    pub fn peek(&self, raw: &str) -> Line<'static> {
        if self.in_code {
            return Line::styled(format!("  {raw}"), code_style());
        }
        let indent_len = raw.len() - raw.trim_start().len();
        let (indent, body) = raw.split_at(indent_len);
        let t = body.trim_end();

        // heading
        let hashes = t.chars().take_while(|&c| c == '#').count();
        if (1..=6).contains(&hashes) && t[hashes..].starts_with(' ') {
            let style = Style::default().add_modifier(Modifier::BOLD);
            return Line::from(inline(t[hashes..].trim_start(), style));
        }
        // horizontal rule
        if t.len() >= 3
            && ["-", "*", "_"]
                .iter()
                .any(|m| t.chars().all(|c| c.to_string() == *m))
        {
            return Line::styled("─".repeat(40), Style::default().add_modifier(Modifier::DIM));
        }
        // quote
        if let Some(rest) = t.strip_prefix("> ") {
            let mut spans = vec![Span::styled(
                format!("{indent}│ "),
                Style::default().add_modifier(Modifier::DIM),
            )];
            spans.extend(inline(rest, Style::default().add_modifier(Modifier::DIM)));
            return Line::from(spans);
        }
        // bullet
        if let Some(rest) = t
            .strip_prefix("- ")
            .or_else(|| t.strip_prefix("* "))
            .or_else(|| t.strip_prefix("+ "))
        {
            let mut spans = vec![Span::raw(format!("{indent}• "))];
            spans.extend(inline(rest, Style::default()));
            return Line::from(spans);
        }
        let mut spans = Vec::new();
        if !indent.is_empty() {
            spans.push(Span::raw(indent.to_string()));
        }
        spans.extend(inline(t, Style::default()));
        Line::from(spans)
    }
}

/// Inline markers -> styled spans. A marker counts only when its closing marker is on the same line.
fn inline(s: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let find = |from: usize, pat: &[char]| -> Option<usize> {
        (from..n.saturating_sub(pat.len() - 1)).find(|&j| chars[j..j + pat.len()] == *pat)
    };
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let (mut bold, mut italic) = (false, false);
    let style = |bold: bool, italic: bool| {
        let mut st = base;
        if bold {
            st = st.add_modifier(Modifier::BOLD);
        }
        if italic {
            st = st.add_modifier(Modifier::ITALIC);
        }
        st
    };
    let flush = |buf: &mut String, out: &mut Vec<Span<'static>>, st: Style| {
        if !buf.is_empty() {
            out.push(Span::styled(std::mem::take(buf), st));
        }
    };
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == '`' {
            if let Some(end) = find(i + 1, &['`']) {
                flush(&mut buf, &mut out, style(bold, italic));
                out.push(Span::styled(
                    chars[i + 1..end].iter().collect::<String>(),
                    code_style(),
                ));
                i = end + 1;
                continue;
            }
        } else if c == '*' && i + 1 < n && chars[i + 1] == '*' {
            // opening needs a closing "**" later; closing is any "**" while bold
            if bold || find(i + 2, &['*', '*']).is_some_and(|e| e > i + 2) {
                flush(&mut buf, &mut out, style(bold, italic));
                bold = !bold;
                i += 2;
                continue;
            }
        } else if c == '*' {
            let opens = !italic
                && i + 1 < n
                && !chars[i + 1].is_whitespace()
                && (i == 0 || !chars[i - 1].is_alphanumeric())
                && find(i + 1, &['*']).is_some_and(|e| e > i + 1 && !chars[e - 1].is_whitespace());
            let closes = italic && i > 0 && !chars[i - 1].is_whitespace();
            if opens || closes {
                flush(&mut buf, &mut out, style(bold, italic));
                italic = !italic;
                i += 1;
                continue;
            }
        }
        buf.push(c);
        i += 1;
    }
    flush(&mut buf, &mut out, style(bold, italic));
    out
}
