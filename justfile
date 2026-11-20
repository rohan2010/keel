# keel dev commands (https://github.com/casey/just). Rust: crates/; Python core: src/keel (uv).

# build the keel binary
build:
    cargo build

# build the installable wheel (Python core + the Rust binary, macOS arm64) into dist/
wheel:
    uv build --wheel -o dist

# all no-model checks: Rust (incl. the end-to-end test against keel-core) + Python
test:
    cargo test
    uv run pytest -q tests

# lints
lint:
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check
    uv run ruff check .

# run a task in a folder with the local model, e.g. `just run ~/Desktop/demo "remove duplicates"`
run dir task:
    cargo run -q -- run --dir {{dir}} "{{task}}"

# remove build output and caches (never your runs in experiments/, the model, or the venv)
clean:
    cargo clean
    rm -rf dist .pytest_cache .ruff_cache
    find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
