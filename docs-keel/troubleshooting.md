# Troubleshooting

Start with `keel doctor`: it checks this Mac and says what to fix.

| You see | What to do |
|---|---|
| `the model ... is not on this Mac yet` | Run `keel download` (about 5 GB; it asks first) |
| `... is too big for keel to work in safely` | Pick a smaller folder (under 5,000 files and 500 MB), or a subfolder |
| `no model server answers at ...` | Start your model app first, or leave out `--backend` to run the model inside keel |
| `no such folder` | Check the path after `--dir` (drag the folder onto Terminal to paste its path) |
| `keel-core stopped before the session started` | The lines below it say why; `keel doctor` usually points at the cause |
| keel says it's done, but the result is wrong | It happens (about 1 in 3 jobs). Ask again more precisely, or undo with `keel backups` |
| A file you needed was changed or deleted | `keel backups` lists the session; `keel backups <session> --restore N` puts it back |
| The conversation gets long and keel says it shortened it | Normal: old parts were summarised to fit the model's memory. Quit and start again for a clean slate |
| It's slow | A small model on a laptop writes a few words a second. Close other heavy apps; 16 GB of memory is recommended |

Still stuck? The full log is in `~/Library/Application Support/keel/logs/core.log`.
