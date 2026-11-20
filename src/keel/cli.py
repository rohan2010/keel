"""keel-core: command-line entry point for the Python core.

  keel-core eval --suite agent --arm base --split dev        any suite as a named arm (tries, repeats)
  keel-core eval --list                                      suites, splits, item counts, sealed tiers
  keel-core compare <runA> <runB>                            verdict: gain / loss / no effect / can't compare
  keel-core run --name smoke --tasks A3 F1 --per-task 1      agent tasks
  keel-core bfcl-mt --name smoke --split dev --limit 3       BFCL multi-turn
  keel-core when2call --name w2c --split dev [--probe-gate]  When2Call (call / no-call decisions)
  keel-core acp                                              ACP agent on stdio (the `keel` TUI, Zed, ...)
  keel-core backups [<session>] [--restore N]                silent backups of changed / deleted files
  keel-core doctor [--backend mlx-server]                    what this Mac and a model server can do
  keel-core download [--check]                               get (or verify) the model: pinned, asks first
"""

import argparse
import importlib
import sys

# name -> (module, help). Modules are imported only when their command runs, so a missing optional dependency (the BFCL
# checker for bfcl-mt / when2call) never stops `doctor`, `download`, `run` or the TUI from starting.
COMMANDS = {
    "eval": ("keel.commands.run_eval", "run a suite as a named arm (tries, repeats) and write one record shape"),
    "compare": ("keel.commands.run_compare", "verdict on two arms: real gain / loss / no detectable effect / can't compare"),
    "noise": ("keel.commands.run_noise", "record or list noise bands (how far two identical runs disagree)"),
    "prereg": ("keel.commands.run_prereg", "freeze a pre-registration before the runs exist"),
    "run": ("keel.commands.run_agent", "run the model through the agent tasks"),
    "bfcl-mt": ("keel.commands.run_bfcl_mt", "BFCL v4 multi-turn eval, scored with the official checker"),
    "when2call": ("keel.commands.run_when2call", "When2Call call / ask / can't-answer eval, gate off or on (P8)"),
    "acp": ("keel.acp_server", "serve the agent over ACP on stdio (for the keel TUI and other ACP clients)"),
    "baseline": ("keel.commands.run_baseline", "run an off-the-shelf agent (goose, opencode) on keel's tasks, sandboxed"),
    "backups": ("keel.commands.run_backups", "list or restore the silent backups kept before a file changes or is deleted"),
    "doctor": ("keel.doctor", "check this Mac and a model server; writes the server's capabilities record"),
    "install-bfcl": ("keel.commands.run_install_bfcl", "add the BFCL checker (bfcl-eval, --no-deps) to this environment, for bfcl-mt"),
    "download": ("keel.commands.run_download", "download or verify the model (pinned revision; shows size and free disk; asks first)"),
}
NEEDS_BFCL = {"bfcl-mt", "when2call"}


def load(name: str):
    """The command's module (a module object put in COMMANDS by a test is used as is)."""
    mod = COMMANDS[name][0]
    if not isinstance(mod, str):
        return mod
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as err:
        if err.name and err.name.startswith("bfcl_eval"):
            raise SystemExit(f"keel-core {name} needs the BFCL checker: run `keel-core install-bfcl` first") from None
        raise


def _chosen(argv: list[str]) -> str | None:
    """The subcommand named in argv (the first word that is one), without importing anything."""
    return next((t for t in argv if t in COMMANDS), None)


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    """Every command is listed; only the chosen one's module is imported for its arguments (argv None: all)."""
    ap = argparse.ArgumentParser(prog="keel-core", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    chosen = _chosen(argv) if argv is not None else None
    for name, (_, help_) in COMMANDS.items():
        if argv is not None and name != chosen:
            sub.add_parser(name, help=help_, allow_abbrev=name != "eval")
            continue
        mod = load(name)
        p = sub.add_parser(name, help=help_, description=mod.__doc__.splitlines()[0],
                           allow_abbrev=name != "eval")   # eval passes unknown flags on: no prefix guessing
        mod.add_arguments(p)
        p.set_defaults(_run=mod.run)
    return ap


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser(argv)
    a, extra = ap.parse_known_args(argv)
    if extra and a.command != "eval":                  # only eval passes unknown flags on (to the suite's command)
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    if a.command == "eval":
        a._extra = extra
    run = a._run
    del a._run, a.command
    rc = run(a)
    if type(rc) is int and rc != 0:                   # commands that return an exit code (download)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
