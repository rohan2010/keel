# Adding a task

A task is three functions and a name, registered in a dictionary:

```python
def x1(seed: int) -> Item:                 # build: a fresh folder (files as bytes) and the request text
    ...
    return Item(task="X1", seed=seed, kind="files", request="Sort the receipts by month.",
                files={"r1.pdf": b"...", ...}, gold={"expected": {...}})

def x1_score(item, work: Path, answer: str, commands: list[str]) -> dict:
    ...                                    # look at the folder (and the answer); never trust what the model says
    return {"correct": ok == n, "score": ok / n, "lost": [...], "wrong": [...]}

def x1_oracle(item, work: Path) -> str:    # solve it the intended way (used to prove the task is solvable)
    ...
    return "Sorted 12 receipts into 3 month folders."

X_TASKS = {"X1": (x1, x1_score, x1_oracle, "Receipts by month")}
```

Look at `src/keel/evals/tasks_ch6.py` for twelve complete examples.

## Steps

1. Write the three functions in a new module, e.g. `src/keel/evals/tasks_extra.py`, and add its dictionary to
   `_all_tasks()` in `src/keel/evals/tasks.py`.
2. Check it: `uv run python tests/checks/check_agent_tasks.py` must show the oracle at 100% and "do nothing" at 0% over
   20 seeds. Add a pytest test for anything special (a refusal, a safety case).
3. Write its card: `uv run python scripts/make_task_cards.py`, then explain in plain words what the checker accepts.
4. Run it on dev seeds only (`keel-core run --split dev --tasks X1 --per-task 4`), read the transcripts, and fix the
   checker if it marks a right answer wrong.
5. **Freeze it before measuring.** Record a sha256 of the task files (see `plans/tasks-freeze-v2.sha256`) before any run on
   test seeds. After that, the task does not change.

## Rules that keep results honest

- The checker reads the folder, not the model's claim. A task that can be passed by saying "done" is a broken task.
- Never tune a task on test seeds. Test seeds are for one comparison per change, and every use is logged.
- `tasks.py` and `tasks_ch6.py` are hash-frozen because the Chapter 6 result was measured on them. Adding a line to
  `_all_tasks()` changes that hash: record it as a new freeze, and keep the old file for re-checking the old result.
