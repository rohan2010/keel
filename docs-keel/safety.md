# How keel keeps your files safe

Every safety check lives in keel-core and is always on. There is no setting that turns the delete guard off.

## The layers of protection

| Protection | What it does |
|---|---|
| **Sandbox** | Every command runs under the macOS sandbox. It can write only in your folder (and a scratch folder), has no network, can't signal other programs, and can read only system folders plus your folder. Your folder itself can't be deleted. (`agent/sandbox.py`) |
| **Delete guard** | Before a command that might delete, keel runs it on a scratch copy of your folder, sees exactly which files would disappear, and asks you. Cancelling, an error, or quitting all count as "no". (`agent/guard.py`) |
| **Backups** | Before a file is changed or deleted, a copy is kept. `keel backups` lists them and puts files back. The newest 50 sessions are kept. (`agent/backups.py`) |
| **Send guard** | Commands that would send email or messages (`mail`, `sendmail`, `osascript`, `smtplib`, ...) are refused before they start; the model is told keel can't send. (`agent/guard.py`) |
| **Folder limits** | Folders over 5,000 files or 500 MB are refused, so the copies above stay quick. (`agent/sandbox.py`) |
| **Time and output limits** | A command is stopped after 30 seconds; output over 100 MB stops it too. (`agent/sandbox.py`) |

## What is not protected

- **A wrong change inside your folder.** If you ask for a rename and the model renames the wrong files, the sandbox
  allows it: it is your folder. The backups are the undo. Check results that matter.
- **Reading your folder.** The model reads the files you point it at. That stays on your Mac, but don't point keel at a
  folder you wouldn't want it to read.
- **Other operating systems.** The sandbox is a macOS feature. keel does not run on Windows or Linux.
- **A bigger model on a server.** With `--backend server`, keel's safety still applies to the tools, but the model runs
  in whatever app you started (Ollama, LM Studio); what that app does is up to it.

## How it is tested

`tests/checks/check_sandbox.py` tries to escape (write outside, read your home folder, use the network, delete the
scratch tools, signal other programs) and every attempt must fail, while normal work (Python, node, pipes, temp files)
must still work. `tests/checks/check_guard.py`, `tests/test_send_guard.py`, `tests/test_guard_home.py` and
`tests/test_p5_safety.py` cover the delete guard, the send guard and the audit's reproductions. They run with
`just test`.
