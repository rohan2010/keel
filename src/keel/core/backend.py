"""The one interface between keel and a model. Nothing above this file knows which model runs.

A backend serves episodes. An episode is any object with the scheduler fields listed in `EpisodeLike`; the
backend asks it for its next prompt, generates a reply, and hands the reply back, until the episode says it is done.
This keeps the research scheduler's shape (continuous batching, per-episode cache reuse) while letting a server or
cloud backend implement the same loop with plain requests.

No MLX import here: `keel.backends.mlx` implements this for in-process MLX, `keel.backends.server` for local servers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class Capabilities:
    """What a backend can do. Recorded in every run folder, so a result says which features it really had."""
    name: str
    forbid_tools_next_turn: bool = False   # can force a text-only reply (loop breaker)
    hidden_states: bool = False            # can expose layer activations (probe gate)
    logit_bias: bool = False
    seeded: bool = False                   # per-episode random streams (paired evals)
    state_save: bool = False               # can save / fork model state at a position (prefix + turn caches)
    batched: bool = False                  # serves many episodes in one batch
    validated: bool = False                # keel's published numbers come from this backend

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RequestStats:
    """Per request, so evals can say where time and tokens went."""
    prompt_tokens: int = 0                 # tokens in the prompt
    reused_tokens: int = 0                 # of those, served from saved state (not re-read)
    output_tokens: int = 0
    prompt_seconds: float = 0.0
    generate_seconds: float = 0.0


@dataclass
class Price:
    """Cost per million tokens, in USD. Local backends are 0; time is reported separately."""
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class EpisodeLike(Protocol):
    done: bool
    stop: str
    prompt_tokens: int
    processed_tokens: int
    output_tokens: int
    t_start: float
    seconds: float
    ban_tool_call: bool                    # read before each request: this reply must be text (forbid_tools)


class RunHandle:
    """Episodes joining a run in progress (plumbing for the swarm experiment, nothing more).

    The caller makes one, passes it in `RunHooks.handle`, and code running inside an episode (its `apply_fn`, e.g. a
    future spawn tool) calls:
        handle.submit(child)        add an episode to the run that is going on now; returns it
        handle.wait(a, b, ...)      block until those episodes are finished (their `finish_fn` has run)
    A waiting episode gives its place back to the backend while it waits (MLX: it doesn't count against the batch;
    server: its slot is free), so a parent waiting on children can't deadlock a run with one slot. Submitting after the
    run has returned raises. Without a handle, every backend runs exactly as before.
    """

    def __init__(self):
        import threading
        self._cv = threading.Condition()
        self._inbox: list = []
        self._finished: set[int] = set()
        self._closed = False
        self.waiting = 0                             # episodes blocked in wait() right now
        self.submitted = 0
        self._wake: Callable[[], None] | None = None
        self._before_wait: Callable[[], None] | None = None
        self._after_wait: Callable[[], None] | None = None
        self._drive: Callable[[Any], None] | None = None

    # -- for backends ---------------------------------------------------------------------------------------------
    def attach(self, *, wake=None, before_wait=None, after_wait=None, drive=None) -> None:
        self._wake, self._before_wait, self._after_wait, self._drive = wake, before_wait, after_wait, drive
        with self._cv:
            self._closed = False

    def take(self) -> list:
        """Episodes submitted since the last call (in submission order)."""
        with self._cv:
            out, self._inbox = self._inbox, []
            return out

    def has_new(self) -> bool:
        with self._cv:
            return bool(self._inbox)

    def finished(self, e: Any) -> None:
        with self._cv:
            self._finished.add(id(e))
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    # -- for episodes ---------------------------------------------------------------------------------------------
    def is_finished(self, e: Any) -> bool:
        with self._cv:
            return id(e) in self._finished

    def submit(self, e: Any) -> Any:
        with self._cv:
            if self._closed:
                raise RuntimeError("this run has finished; an episode can only join a run in progress")
            self._inbox.append(e)
            self.submitted += 1
        if self._wake is not None:
            self._wake()
        return e

    def wait(self, *eps: Any, timeout: float | None = None) -> list:
        if self._drive is not None:                  # single-threaded backend: run the episode right here
            for e in eps:
                if not self.is_finished(e):
                    self._drive(e)
        with self._cv:
            self.waiting += 1
        if self._before_wait is not None:
            self._before_wait()
        try:
            with self._cv:
                ok = self._cv.wait_for(lambda: all(id(e) in self._finished for e in eps) or self._closed, timeout)
                ok = ok and all(id(e) in self._finished for e in eps)
        finally:
            with self._cv:
                self.waiting -= 1
            if self._after_wait is not None:
                self._after_wait()
        if not ok:
            raise TimeoutError("waited episodes did not finish" + ("" if timeout is None else f" in {timeout}s"))
        return list(eps)


@dataclass
class RunHooks:
    """How a backend talks to the episodes it serves.

    prompt_fn(e)          -> the model input for e's next reply (token ids for MLX; messages for API backends)
    apply_fn(e, text, fr) -> record a reply (runs tools; may set e.done). Called in a worker thread.
    finish_fn(e)          -> record a finished episode
    seed_fn(e)            -> a stable episode id for per-episode random streams, or None
    label_fn(e)           -> short name for progress lines, or None
    token_fn(e, text)     -> new reply text as it is generated (live view), or None; observes only
    """
    prompt_fn: Callable[[Any], Any]
    apply_fn: Callable[[Any, str, str], None]
    finish_fn: Callable[[Any], None]
    seed_fn: Callable[[Any], str] | None = None
    label_fn: Callable[[Any], str] | None = None
    token_fn: Callable[[Any, str], None] | None = None
    handle: RunHandle | None = None               # episodes may submit / wait for episodes during the run


@dataclass
class RunLimits:
    max_context: int = 16000
    batch_size: int = 6
    context_budget: int = 48000            # (episodes generating) x (longest context) token slots
    progress_every: float = 60.0
    extra: dict = field(default_factory=dict)   # backend-specific knobs (prefill_step, prefill_batch, ...)


@runtime_checkable
class Backend(Protocol):
    def capabilities(self) -> Capabilities: ...

    def price(self) -> Price: ...

    def run(self, episodes: Iterable[Any], hooks: RunHooks, limits: RunLimits) -> dict:
        """Serve every episode until done. Returns run stats (reused / fresh requests, peak memory, seconds...)."""
        ...

    def complete(self, prompt: Any, *, forbid_tools: bool = False) -> tuple[str, str, RequestStats]:
        """One reply for one prompt: (text, finish_reason, stats). For `keel run` and simple tools."""
        ...
