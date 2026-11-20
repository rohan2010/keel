# Adding a model backend

Most models need no code: if your app speaks the OpenAI API (Ollama, LM Studio, llama.cpp's server,
mlx_lm.server), use it as it is:

```
keel run --backend server --url http://127.0.0.1:1234 --mode chat --dir <folder>
keel doctor --backend server --url http://127.0.0.1:1234      # what that server can and can't do
```

`--mode raw` sends keel's own prompt (it needs the model's tokenizer files, which keel finds after `keel download`);
`--mode chat` lets the server apply its own chat template and tool format. `keel doctor` records what the server supports (for example, whether
it can force a text-only reply when the model loops) so keel uses the best method it has.

## Writing a new backend

Everything above the backend talks to models through one small interface (`src/keel/core/backend.py`):

```python
class Backend(Protocol):
    def capabilities(self) -> Capabilities: ...     # what it can do (recorded in every run)
    def price(self) -> Price: ...                   # cost per token, 0 for local models
    def run(self, episodes, hooks: RunHooks, limits: RunLimits) -> dict: ...
    def complete(self, prompt, *, forbid_tools: bool = False) -> tuple[str, str, RequestStats]: ...
```

`run` serves a batch of episodes: for each one it asks `hooks.prompt_fn(e)` for the input, generates a reply, and hands
it back with `hooks.apply_fn(e, text, finish_reason)` until the episode is done, then calls `hooks.finish_fn(e)`. The
simplest complete example is `src/keel/backends/scripted.py` (replays fixed replies); `server.py` is the real one for
HTTP servers and `mlx.py` for in-process MLX (continuous batching, per-episode seeds).

Then wire it into the backend choice (`src/keel/backends/choose.py`) and the terminal session (`acp_server.py`).

## Before trusting numbers from it

keel's published results come from the in-process MLX backend (`validated: true` in its capabilities). A new backend's
numbers are only comparable to them after a same-results check: run the golden tasks (`scripts/golden.sh`) on both and
compare, and use `keel-core compare`, which lists a backend difference and gives no verdict unless you accept it with
`--allow backend`.
