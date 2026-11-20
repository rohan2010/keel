# keel agent task cards

Twenty-five in-house tasks on a sandboxed folder (written by us; not a public benchmark): the original 13 and 12 added in Chapter 6 (2026-11-19, scoring fixed before any model ran on them). Each card says what the task asks, how it is scored, and which seeds are dev / test / holdout, with links to the generator, scorer and oracle, so our agent numbers can be checked. Generated from the task code by `scripts/make_task_cards.py`.

| task | title | kind | useful | false success | set |
|---|---|---|---|---|---|
| [A1](A1.md) | Sort files into folders | state | yes | yes | original |
| [A3](A3.md) | Find a file and its date | answer | yes | yes | original |
| [A4](A4.md) | Delete duplicates | state | yes | yes | original |
| [A5](A5.md) | Flatten nested folders | state | yes | yes | Ch6 |
| [A6](A6.md) | Archive files older than a date | state | yes | yes | Ch6 |
| [A7](A7.md) | Largest file and its size | answer | yes | yes | Ch6 |
| [B1](B1.md) | Notes to to-do list | state | yes | yes | original |
| [B2](B2.md) | Merge dated meeting notes | state | yes | yes | Ch6 |
| [C1](C1.md) | Invoices to CSV | state | yes | yes | original |
| [C3](C3.md) | Spending question on a CSV | answer | yes | yes | original |
| [C4](C4.md) | Remove duplicate contacts | state | yes | yes | Ch6 |
| [C5](C5.md) | Orders question on a CSV | answer | yes | yes | Ch6 |
| [D1](D1.md) | Rename a CSS class | state | yes | yes | original |
| [D2](D2.md) | Fix broken links | state | yes | yes | original |
| [D3](D3.md) | Rename a Python function | state | yes | yes | Ch6 |
| [D4](D4.md) | Bump the version everywhere | state | yes | yes | Ch6 |
| [E1](E1.md) | Set up a project | state | yes | yes | original |
| [E3](E3.md) | Organise a photo dump (long) | state | yes | yes | original |
| [E4](E4.md) | Folder structure from a spec | state | yes | yes | Ch6 |
| [E5](E5.md) | Split a log by day (long) | state | yes | yes | Ch6 |
| [F1](F1.md) | Refuse: send an email | limits | yes | no | original |
| [F2](F2.md) | Refuse: missing file | limits | yes | no | original |
| [F3](F3.md) | Safety: 'delete everything' | safety | no | no | original |
| [F4](F4.md) | Refuse: book a meeting room | limits | yes | no | Ch6 |
| [F5](F5.md) | Ask: rename with no name given | limits | yes | no | Ch6 |

Scorer versions matter when quoting numbers: v1 = up to the 2026-11-09 fixes (B1 headings, D1 BEM modifier); v2 = 2026-11-10 (F1 send attempt fails, A4 reported backup copy accepted).
