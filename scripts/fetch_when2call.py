"""Fetch the When2Call test set (NVIDIA, CC-BY-4.0) into <data>/external/when2call/ (keel.paths.data_dir(): repo data/ in a checkout; git-ignored).

    uv run python scripts/fetch_when2call.py

Pinned to one dataset revision so every run reads the same items. Test files only (MCQ 3,652 items + the 300-item
LLM-judge subset, ~22 MB); the training sets are not needed (keel does no fine-tuning).
Source: https://huggingface.co/datasets/nvidia/When2Call  Licence: https://creativecommons.org/licenses/by/4.0/
"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from keel import paths

REPO = "nvidia/When2Call"
REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
FILES = ["README.md", "test/when2call_test_mcq.jsonl", "test/when2call_test_llm_judge.jsonl"]
OUT = paths.data_dir() / "external" / "when2call"      # repo data/ in a checkout; KEEL_DATA moves it


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": REPO, "revision": REVISION, "license": "cc-by-4.0", "files": {}}
    for f in FILES:
        p = Path(hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=OUT))
        data = p.read_bytes()
        rows = data.count(b"\n") if f.endswith(".jsonl") else None
        manifest["files"][f] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "rows": rows}
        print(f"  {f}: {len(data) / 1e6:.1f} MB" + (f", {rows} rows" if rows is not None else ""))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
