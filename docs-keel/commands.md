# Commands

## `keel` (what most people use)

| Command | What it does |
|---|---|
| `keel` | Shows the mascot and where to start |
| `keel download` | Gets the AI model (about 5 GB): shows the size and free disk, asks first, checks every file. `--check` only verifies the model you have |
| `keel doctor` | Checks this Mac (memory, the model, Python and MLX, the state folder, disk) and says what to fix. `--backend mlx-server` also tests a model server |
| `keel run --dir <folder> ["task"]` | Works in a folder. Without a task it opens and waits for you to type one. `--plain` gives simple line output instead of the full screen |
| `keel backups [<session>] [--restore N]` | Lists sessions with backups, what each kept, and puts file N back |
| `keel compare <runA> <runB>` | Verdict on two eval runs (see [How we measure](how-we-measure.md)) |
| `keel eval ...` | Runs an eval suite (same as `keel-core eval`) |

`keel run` options: `--backend mlx` (default: the model inside keel), `mlx-server`, `ollama`, `server` (any
OpenAI-compatible server, with `--url`, `--mode raw|chat`, `--flavour openai|ollama|llama`), or `scripted` (replays
replies, for demos and tests).

## `keel-core` (evals and setup)

| Command | What it does |
|---|---|
| `run` | The agent tasks as one eval run (`--tasks`, `--split dev\|test\|holdout`, `--per-task`, the layer flags) |
| `eval` | Any suite as a named arm with tries and repeats; `--list` shows suites, tiers and item counts |
| `compare` | Verdict on two runs; `--prereg <name>` uses a frozen plan |
| `prereg` | Freezes a pre-registration plan before the runs exist |
| `noise` | Records how far two identical runs disagree |
| `bfcl-mt`, `when2call` | Public benchmarks (BFCL multi-turn, When2Call) |
| `baseline` | Runs other agent tools (goose, opencode) on keel's tasks, sandboxed |
| `install-bfcl` | Adds the BFCL checker (needed for `bfcl-mt` / `when2call`) |
| `download`, `doctor`, `backups` | The same as the `keel` commands |
| `acp` | Serves keel over ACP on stdin/stdout (used by the terminal app and editors) |

Every command has `--help`.

## Where keel keeps things

- **Your sessions, backups and the log:** `~/Library/Application Support/keel/` (in `sessions/`, `backups/`,
  `logs/core.log`)
- **The AI model:** `~/.cache/huggingface/hub/`, after `keel download`
- **Eval runs, frozen plans, comparison reports:** `experiments/` in a source checkout

Settings, all optional: `KEEL_STATE_DIR`, `KEEL_EXPERIMENTS`, `KEEL_MODELS`, `KEEL_DATA` and `KEEL_BASELINES` move
those folders; `KEEL_CORE` points `keel` at a specific keel-core; `HF_HOME` moves the model cache; `NO_COLOR` turns off
colour.
