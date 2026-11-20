# Third-party data and tools

keel's own code is MIT-licensed (see `LICENSE`). It downloads or uses the following at run time; none of it is
bundled in this repository or the wheel.

| What | Licence | How keel uses it |
|---|---|---|
| Qwen3.5-4B (MLX 8-bit conversion, `mlx-community/Qwen3.5-4B-8bit`) | Apache-2.0 (base model `Qwen/Qwen3.5-4B`; the MLX conversion states no licence of its own) | Downloaded by `keel download` into the Hugging Face cache |
| BFCL v4 data and checker (`bfcl-eval`, Gorilla / UC Berkeley) | Apache-2.0 | Optional, for the BFCL eval: `keel-core install-bfcl` |
| When2Call test set (NVIDIA, `nvidia/When2Call`) | CC-BY-4.0 | Optional, fetched by `scripts/fetch_when2call.py`; attribution printed with results |
| goose, opencode | Apache-2.0, MIT | Optional baselines for `keel-core baseline`, installed separately |
