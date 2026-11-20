# Changelog

## 0.1.0 (not released yet)

The first version for other people to install.

**Using keel**
- `keel run --dir <folder>` works in a folder with a small AI model running on your Mac (Qwen3.5-4B, 8-bit). Leave
  out the task to type it in the app.
- Asks before every delete (no way to turn it off), keeps a backup before changing or deleting a file
  (`keel backups` to restore), refuses to send email or messages, and can only change the folder you give it.
- `keel download` gets the model (shows the size, asks first, checks every file); `keel doctor` checks the Mac.
- Long conversations are shortened instead of stopping when they fill the model's memory.
- Clear one-line messages for the usual first-run problems (no model yet, folder too big, model server not running).
- Other local models through Ollama, LM Studio or any OpenAI-compatible server (`--backend`).

**Measuring keel**
- 24 everyday file tasks with checkers; eval suites for BFCL multi-turn and When2Call.
- `keel compare`: paired and task-clustered tests, pre-registration, fixed-order hypotheses, and a verdict.
- Headline: keel's default stack passed 68% of 96 tries against 53% for the same model alone (pre-registered,
  task-clustered one-sided p = 0.013). It did not reduce false "done" claims. Re-check it with
  `evidence/ch6-headline/verify.py`.

Not yet: Windows / Linux, a graphical app, a release on PyPI.
