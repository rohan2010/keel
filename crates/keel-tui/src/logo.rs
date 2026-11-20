//! keel's mascots (2026-11-20): the screen bot is the main one; nine "moods" (a cat, a pup, an octo alien, a neon
//! visor, a cyber eye, an arcade cabinet, an astronaut, a ringed planet, a starry screen) stand in for it now and then.
//! Shown when a session starts and by a bare `keel`. Each is a pixel grid drawn with half blocks (each terminal cell
//! holds two pixels, one above the other), so the TUI banner, the plain-text banner and the SVGs in
//! `docs-keel/images/mascots/` (`scripts/make_mascots.py`) all come from the same pictures.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// One mascot: `name` is what `KEEL_MASCOT` takes, `title` what the banner shows beside a mood.
///
/// Pixels: `.` empty, `#` body, `h` highlight, `d` shade, `k` screen (left empty without colour, so the face still
/// shows), `a` amber light, `w`/`S` white, `p` pink, `g` green, `m` magenta, `v` purple, `r` ring gold, `R` red,
/// `l` grey.
pub struct Mascot {
    pub name: &'static str,
    pub title: &'static str,
    pub pixels: &'static [&'static str],
}

/// The main mascot: a monitor-headed bot with a smile on its screen.
pub const MAIN: Mascot = Mascot {
    name: "screen",
    title: "screen bot",
    pixels: &[
        ".......a.......",
        ".......#.......",
        "..hhhhhhhhhhh..",
        ".h###########d.",
        ".h#kkkkkkkkk#d.",
        ".h#kkhkkkhkk#d.",
        "#h#kkhkkkhkk#d#",
        "#h#khkkkkkhk#d#",
        ".h#kkhhhhhkk#d.",
        ".h###########d.",
        "..ddddddddddd..",
        "...dd.....dd...",
    ],
};

/// The moods, same screen head drawn from animals, cyberpunk games and outer space.
pub const MOODS: [Mascot; 9] = [
    Mascot {
        name: "cat",
        title: "robo cat",
        pixels: &[
            "..h.........d..",
            "..h#.......#d..",
            "..hhhhhhhhhhh..",
            ".h###########d.",
            ".h#kkkkkkkkk#d.",
            ".h#kkgkkkgkk#d.",
            "#h#kkgkkkgkk#d#",
            "#h#kkkkpkkkk#d#",
            ".h#kkkhkhkkk#d.",
            ".h###########d.",
            "..ddddddddddd..",
            "...dd.....dd...",
        ],
    },
    Mascot {
        name: "pup",
        title: "robo pup",
        pixels: &[
            ".......a.......",
            ".......#.......",
            "ddhhhhhhhhhhhdd",
            "ddh#########ddd",
            "ddh#kkkkkkk#ddd",
            "ddh#khkkkhk#ddd",
            ".dh#khkkkhk#dd.",
            "..h#kkkpkkk#d..",
            "..h#kkhkhkk#d..",
            "..h####p####d..",
            "...ddddddddd...",
            "...dd.....dd...",
        ],
    },
    Mascot {
        name: "octo",
        title: "octo alien",
        pixels: &[
            ".......a.......",
            ".......#.......",
            "..hhhhhhhhhhh..",
            ".h###########d.",
            ".h#kkkkkkkkk#d.",
            ".h#kkgkkkgkk#d.",
            ".h#kkgkkkgkk#d.",
            ".h#kkhkkkhkk#d.",
            ".h#kkkhhhkkk#d.",
            ".h###########d.",
            ".dd.dd.d.dd.dd.",
            "d..d..d.d..d..d",
        ],
    },
    Mascot {
        name: "visor",
        title: "neon visor",
        pixels: &[
            ".........m.....",
            "........m......",
            "..hhhhhhhhhhh..",
            ".h###########d.",
            ".h#kkkkkkkkk#d.",
            "vh#mmmmmmmwm#dv",
            "vh#kkkkkkkkk#dv",
            "vh#kkkkkkkkk#dv",
            ".h#kkkhhhkkk#d.",
            ".h###########d.",
            "..ddddddddddd..",
            "...dd.....dd...",
        ],
    },
    Mascot {
        name: "cyber",
        title: "cyber eye",
        pixels: &[
            ".....m.m.m.....",
            ".....mmmmm.....",
            "..hhhhhhhhhhh..",
            ".h###########d.",
            ".h#kkkkkkkkk#d.",
            ".h#kkhkkRRkk#d.",
            "#h#kkhkkRRkk#d#",
            "#h#kkkkkkkhk#d#",
            ".h#kkkkhhhkk#d.",
            ".h###########d.",
            "..ddddddddddd..",
            "...dd.....dd...",
        ],
    },
    Mascot {
        name: "arcade",
        title: "arcade bot",
        pixels: &[
            ".......a.......",
            ".......#.......",
            "..hhhhhhhhhhh..",
            ".h#kkkkkkkkk#d.",
            ".h#kkhkkkhkk#d.",
            ".h#kkhkkkhkk#d.",
            ".h#khkkkkkhk#d.",
            ".h#kkhhhhhkk#d.",
            ".hhhhhhhhhhhhh.",
            "h##R###m#g#a##d",
            ".ddddddddddddd.",
            "..dd.......dd..",
        ],
    },
    Mascot {
        name: "astronaut",
        title: "astronaut",
        pixels: &[
            ".......R.......",
            ".......w.......",
            "...wwwwwwwww...",
            "..wwwwwwwwwwl..",
            ".wwkwkkkkkkkwl.",
            ".wwkkwkhkhkkwl.",
            ".wwkkkkhkhkkwl.",
            ".wwkkkkkkkkkwl.",
            "..wwwwwwwwwwl..",
            ".##h#######d##.",
            "..ddddddddddd..",
            "...dd.....dd...",
        ],
    },
    Mascot {
        name: "planet",
        title: "ringed planet",
        pixels: &[
            ".........a.........",
            ".S.......#.........",
            "....hhhhhhhhhhh....",
            "...h###########d.S.",
            "...h#kkkkkkkkk#d...",
            "...h#kkhkkkhkk#d...",
            "..#h#kkhkkkhkk#d#..",
            "..#h#khkkkkkhk#d#..",
            "r..h#kkhhhhhkk#d..r",
            ".rrrrrrrrrrrrrrrrr.",
            "....ddddddddddd....",
            ".....dd.....dd.....",
        ],
    },
    Mascot {
        name: "stars",
        title: "starry screen",
        pixels: &[
            ".......a.......",
            "..hhhhhhhhhhh..",
            ".h###########d.",
            ".h#kkkkkkkkS#d.",
            ".h#kkSkkkSkk#d.",
            ".h#kSSSkSSSk#d.",
            "#h#kkSkkkSkk#d#",
            "#h#kakkkkkak#d#",
            ".h#kkaaaaakk#d.",
            ".h###########d.",
            "..ddddddddddd..",
            "...dd.....dd...",
        ],
    },
];

/// How often a mood stands in for the main mascot when `KEEL_MASCOT` is not set: one start in `MOOD_ODDS`.
pub const MOOD_ODDS: u64 = 4;

/// The main mascot and every mood.
pub fn all() -> impl Iterator<Item = &'static Mascot> {
    std::iter::once(&MAIN).chain(MOODS.iter())
}

/// The mascot by name (`screen`, `cat`, ...).
pub fn by_name(name: &str) -> Option<&'static Mascot> {
    all().find(|m| m.name.eq_ignore_ascii_case(name.trim()))
}

/// The mascot for this start. `KEEL_MASCOT` picks one by name, or `random` for any of them; otherwise the main one,
/// with a random mood one start in `MOOD_ODDS`. An unknown name gets the main one.
pub fn pick() -> &'static Mascot {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64).rotate_left(32);
    let r = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11; // mix the low bits up
    pick_with(std::env::var("KEEL_MASCOT").ok().as_deref(), r)
}

fn pick_with(choice: Option<&str>, r: u64) -> &'static Mascot {
    match choice.map(str::trim).filter(|c| !c.is_empty()) {
        Some(c) if c.eq_ignore_ascii_case("random") => all()
            .nth((r % (MOODS.len() as u64 + 1)) as usize)
            .unwrap_or(&MAIN),
        Some(c) => by_name(c).unwrap_or(&MAIN),
        None if r % MOOD_ODDS == 0 => &MOODS[((r / MOOD_ODDS) % MOODS.len() as u64) as usize],
        None => &MAIN,
    }
}

/// 256-colour palette index of a pixel; `None` is empty. Without colour the screen (`k`) is left empty too, so the
/// face drawn on it stays visible.
fn colour(px: u8, color: bool) -> Option<u8> {
    Some(match px {
        b'#' => 80,           // #5fd7d7, the body
        b'h' => 159,          // #afffff
        b'd' => 30,           // #008787
        b'a' => 221,          // #ffd75f
        b'k' if color => 235, // #262626
        b'w' | b'S' => 231,   // #ffffff
        b'p' => 211,          // #ff87af
        b'g' => 84,           // #5fff87
        b'm' => 206,          // #ff5fd7
        b'v' => 135,          // #af5fff
        b'r' => 222,          // #ffd787
        b'R' => 203,          // #ff5f5f
        b'l' => 250,          // #bcbcbc
        _ => return None,
    })
}

/// The main mascot's body colour.
pub const COLOR: Color = Color::Indexed(80);

/// One terminal cell: its character, foreground and background (256-colour indexes).
type Cell = (char, Option<u8>, Option<u8>);

/// A mascot as rows of cells. With `color` off the character alone carries the shape (`█` whenever both pixels are
/// filled); with it on, two different colours in one cell become `▀` on a background.
fn cells(m: &Mascot, color: bool) -> Vec<Vec<Cell>> {
    m.pixels
        .chunks(2)
        .map(|pair| {
            let top = pair[0].as_bytes();
            let bottom = pair.get(1).map(|r| r.as_bytes()).unwrap_or(&[]);
            top.iter()
                .enumerate()
                .map(|(i, &t)| {
                    let b = bottom.get(i).copied().unwrap_or(b'.');
                    match (colour(t, color), colour(b, color)) {
                        (None, None) => (' ', None, None),
                        (Some(t), None) => ('▀', Some(t), None),
                        (None, Some(b)) => ('▄', Some(b), None),
                        (Some(t), Some(b)) if t == b || !color => ('█', Some(t), None),
                        (Some(t), Some(b)) => ('▀', Some(t), Some(b)),
                    }
                })
                .collect()
        })
        .collect()
}

/// A mascot's rows as plain text (no colour), each the same width.
pub fn rows(m: &Mascot) -> Vec<String> {
    cells(m, false)
        .iter()
        .map(|r| r.iter().map(|c| c.0).collect())
        .collect()
}

/// A mascot's rows as TUI spans, runs of the same colours merged into one span.
fn styled_rows(m: &Mascot) -> Vec<Vec<Span<'static>>> {
    cells(m, true)
        .into_iter()
        .map(|row| {
            let mut spans: Vec<(String, Option<u8>, Option<u8>)> = Vec::new();
            for (ch, fg, bg) in row {
                match spans.last_mut() {
                    Some(last)
                        if (last.1, last.2) == (fg, bg)
                            || (ch == ' ' && last.2.is_none() && bg.is_none()) =>
                    {
                        last.0.push(ch)
                    }
                    _ => spans.push((ch.to_string(), fg, bg)),
                }
            }
            spans
                .into_iter()
                .map(|(s, fg, bg)| {
                    let mut st = Style::default();
                    if let Some(f) = fg {
                        st = st.fg(Color::Indexed(f));
                    }
                    if let Some(b) = bg {
                        st = st.bg(Color::Indexed(b));
                    }
                    Span::styled(s, st)
                })
                .collect()
        })
        .collect()
}

/// A mascot's rows with ANSI 256-colour codes, each the same visible width.
fn ansi_rows(m: &Mascot) -> Vec<String> {
    cells(m, true)
        .iter()
        .map(|row| {
            let mut out = String::new();
            for &(ch, fg, bg) in row {
                match (fg, bg) {
                    (None, _) => out.push_str("\x1b[0m "),
                    (Some(f), None) => out.push_str(&format!("\x1b[0;38;5;{f}m{ch}")),
                    (Some(f), Some(b)) => out.push_str(&format!("\x1b[0;38;5;{f};48;5;{b}m{ch}")),
                }
            }
            out.push_str("\x1b[0m");
            out
        })
        .collect()
}

/// The terminal row beside which the three lines of text start (they sit around the middle of the mascot).
fn text_row(m: &Mascot) -> usize {
    m.pixels.len().div_ceil(2).saturating_sub(3) / 2 + 1
}

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

/// The banner for this start (`pick`): a mascot with three lines beside it (name, tagline, one detail line).
pub fn banner(detail: &str) -> Vec<Line<'static>> {
    banner_for(pick(), detail)
}

/// The banner with a given mascot, as TUI lines. A mood's title shows dimmed after the name.
pub fn banner_for(m: &Mascot, detail: &str) -> Vec<Line<'static>> {
    let mut name = vec![Span::styled(
        "keel",
        Style::default().add_modifier(Modifier::BOLD),
    )];
    if m.name != MAIN.name {
        name.push(Span::styled(
            format!(" · {}", m.title),
            Style::default().add_modifier(Modifier::DIM),
        ));
    }
    let side = [
        name,
        vec![Span::styled(
            TAGLINE,
            Style::default().add_modifier(Modifier::DIM),
        )],
        vec![Span::styled(
            detail.to_string(),
            Style::default().add_modifier(Modifier::DIM),
        )],
    ];
    let first = text_row(m);
    styled_rows(m)
        .into_iter()
        .enumerate()
        .map(|(i, mut spans)| {
            if let Some(s) = i.checked_sub(first).and_then(|k| side.get(k)) {
                spans.push(Span::raw("    "));
                spans.extend(s.iter().cloned());
            }
            Line::from(spans)
        })
        .collect()
}

/// The banner for this start as plain terminal text (bare `keel`): coloured mascot and bold name when `color`.
pub fn text(detail: &str, color: bool) -> String {
    text_for(pick(), detail, color)
}

/// The banner with a given mascot as plain terminal text.
pub fn text_for(m: &Mascot, detail: &str, color: bool) -> String {
    let (b, d, r) = if color {
        ("\x1b[1m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let mood = if m.name != MAIN.name {
        format!("{d} · {}{r}", m.title)
    } else {
        String::new()
    };
    let side = [
        format!("{b}keel{r}{mood}"),
        format!("{d}{TAGLINE}{r}"),
        format!("{d}{detail}{r}"),
    ];
    let first = text_row(m);
    let bot = if color { ansi_rows(m) } else { rows(m) };
    bot.iter()
        .enumerate()
        .map(|(i, row)| {
            let beside = i
                .checked_sub(first)
                .and_then(|k| side.get(k))
                .map(|s| format!("    {s}"))
                .unwrap_or_default();
            format!("{row}{beside}").trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(t: &str) -> String {
        let mut plain = String::new();
        let mut esc = false;
        for ch in t.chars() {
            match (esc, ch) {
                (false, '\x1b') => esc = true,
                (true, 'm') => esc = false,
                (true, _) => {}
                (false, c) => plain.push(c),
            }
        }
        plain
    }

    #[test]
    fn every_mascot_is_a_clean_grid_that_fits_80_columns() {
        let mut names = std::collections::HashSet::new();
        for m in all() {
            assert!(names.insert(m.name), "duplicate name {}", m.name);
            let w = m.pixels[0].len();
            assert!(
                m.pixels.iter().all(|r| r.len() == w),
                "{} rows differ in width",
                m.name
            );
            assert!(
                m.pixels
                    .iter()
                    .all(|r| r.bytes().all(|b| b == b'.' || colour(b, true).is_some())),
                "{}",
                m.name
            );
            assert!(rows(m).iter().all(|r| r.chars().count() == w));
            for color in [false, true] {
                let t = strip(&text_for(
                    m,
                    "qwen3.5-4b · a folder with a long name somewhere",
                    color,
                ));
                let longest = t.lines().map(|l| l.chars().count()).max().unwrap_or(0);
                assert!(longest <= 80, "{} {longest}", m.name);
                assert!(
                    t.contains("keel") && t.contains(TAGLINE) && t.contains("long name"),
                    "{}",
                    m.name
                );
            }
        }
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
    fn plain_text_has_no_colour_codes_and_keeps_the_face() {
        let t = text_for(&MAIN, "detail", false);
        // the screen is left empty without colour, so the eyes show as blocks on it
        assert!(!t.contains('\x1b') && t.contains("██  ▄   ▄  ██"), "{t}");
        assert!(!t.contains(" · "), "the main mascot has no mood title");
    }

    #[test]
    fn coloured_text_draws_the_same_outline() {
        let plain = strip(&text_for(&MAIN, "detail", true));
        assert!(plain.contains("▄▀▀▀▀▀▀▀▀▀▀▀▄"), "{plain}");
        assert_eq!(text_for(&MAIN, "detail", false).lines().count(), 6);
    }

    #[test]
    fn moods_show_their_title() {
        let cat = by_name("CAT").expect("the cat mood exists");
        assert!(text_for(cat, "d", false).contains("keel · robo cat"));
        let lines = banner_for(cat, "d");
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
            .collect();
        assert!(text.contains(" · robo cat"));
    }

    #[test]
    fn the_main_mascot_is_the_usual_pick() {
        assert_eq!(pick_with(Some("astronaut"), 0).name, "astronaut");
        assert_eq!(pick_with(Some("nope"), 0).name, "screen");
        assert_eq!(pick_with(Some(" Screen "), 0).name, "screen");
        let moods = (0..4000u64)
            .filter(|&r| pick_with(None, r).name != "screen")
            .count();
        assert_eq!(moods, 1000, "one start in {MOOD_ODDS} gets a mood");
        let seen: std::collections::HashSet<_> =
            (0..4000u64).map(|r| pick_with(None, r).name).collect();
        assert_eq!(seen.len(), MOODS.len() + 1, "every mood turns up");
        let any: std::collections::HashSet<_> = (0..100u64)
            .map(|r| pick_with(Some("random"), r).name)
            .collect();
        assert_eq!(any.len(), MOODS.len() + 1);
    }
}
