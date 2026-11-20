"""Chapter 5 E0: thinking mode plumbing (flag, sampler preset, reasoning kept for the template, unfinished thinking,
compare treats it as a listed difference). No model."""

import argparse
import json
from pathlib import Path

import pytest

from keel.agent.loop import run_episodes_on
from keel.backends.scripted import ScriptedBackend
from keel.commands import run_agent
from keel.evals import tasks as T
from keel.evals.compare import compare_arms, load_arm
from keel.specs import PLAIN_SAMPLER, QWEN35_4B_8BIT, THINKING_SAMPLER, run_spec_and_sampling


def args(*argv):
    ap = argparse.ArgumentParser()
    run_agent.add_arguments(ap)
    return ap.parse_args(["--name", "x", *argv])


def test_without_thinking_the_spec_and_sampler_are_the_old_ones():
    spec, s = run_spec_and_sampling(QWEN35_4B_8BIT, args())
    assert spec.chat_template_kwargs == {"enable_thinking": False}
    assert (s.top_p, s.top_k, s.presence_penalty) == (0.8, 20, 0.0) == tuple(PLAIN_SAMPLER.values())


def test_thinking_preset_and_overrides():
    spec, s = run_spec_and_sampling(QWEN35_4B_8BIT, args("--thinking", "--temperature", "0.6"))
    assert spec.chat_template_kwargs == {"enable_thinking": True} and s.temperature == 0.6
    assert (s.top_p, s.top_k, s.presence_penalty) == tuple(THINKING_SAMPLER.values())
    _, s = run_spec_and_sampling(QWEN35_4B_8BIT, args("--thinking", "--top-p", "0.8", "--presence-penalty", "0"))
    assert (s.top_p, s.presence_penalty) == (0.8, 0.0)                     # the same-sampler secondary arm


def test_thinking_refused_on_servers():
    with pytest.raises(SystemExit, match="in-process MLX only"):
        run_agent.run(args("--thinking", "--backend", "mlx-server"))


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_reasoning_is_kept_for_the_template_and_the_call_runs(tmp_path):
    it = T.build("C1", 0)
    reply = "I should look first.\n</think>\n\nLet me look.\n" + call("list_files")
    be = ScriptedBackend({it.id: [reply, "The folder is listed.\n</think>\n\nDone."]})
    res = run_episodes_on(be, [it], tmp_path, thinking=True)
    msgs = json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"]
    a = [m for m in msgs if m["role"] == "assistant"]
    assert a[0]["reasoning_content"] == "I should look first." and a[0]["content"].startswith("Let me look.")
    assert "</think>" not in a[0]["content"] and res[0]["answer"] == "Done."
    assert any(m["role"] == "tool" for m in msgs)                           # the call ran


def test_unfinished_thinking_is_no_answer_and_no_call(tmp_path):
    it = T.build("C1", 0)
    be = ScriptedBackend({it.id: [("Hmm, maybe I should <tool_call>\n<function=list_files>", "length")]})
    res = run_episodes_on(be, [it], tmp_path, thinking=True)
    r = res[0]
    assert r["stop"] == "length" and r["answer"] == "" and "thinking_unfinished@1" in r["notes"]
    assert r["commands"] == []                                              # nothing inside the thinking ran


def test_thinking_off_messages_have_no_reasoning_key(tmp_path):
    it = T.build("C1", 0)
    run_episodes_on(ScriptedBackend({it.id: ["Done."]}), [it], tmp_path)
    msgs = json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"]
    assert all("reasoning_content" not in m for m in msgs)                  # flags off = the old messages


def test_template_renders_kept_reasoning_after_the_last_user_turn():
    from transformers import AutoTokenizer
    d = Path(QWEN35_4B_8BIT.path)
    if not (d / "tokenizer_config.json").is_file():
        pytest.skip("local model folder missing")
    tok = AutoTokenizer.from_pretrained(str(d))
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello.", "reasoning_content": "The user greets me."}]
    on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert "<think>\nThe user greets me.\n</think>\n\nHello." in on and on.endswith("<think>\n")


def test_compare_lists_thinking_instead_of_refusing(tmp_path):
    def run(name, thinking, outcomes):
        d = tmp_path / name
        d.mkdir()
        ids = [f"T{i}_0" for i in range(len(outcomes))]
        (d / "config.json").write_text(json.dumps({
            "kind": "agent", "split": "dev", "items": ids, "batch_size": 1, "sampling": {"temperature": 0.6},
            "model": {"name": "m", "chat_template_kwargs": {"enable_thinking": thinking}}}))
        (d / "outputs.jsonl").write_text("".join(json.dumps({"id": i, "task": "T", "kind": "state", "seed": 0,
                                                             "correct": ok, "score": float(ok), "stop": "answer",
                                                             "answer": "ok", "steps": 1}) + "\n"
                                                 for i, ok in zip(ids, outcomes, strict=True)))
        return str(d)
    c = compare_arms(load_arm(run("off", False, [False] * 12)), load_arm(run("on", True, [True] * 12)))
    assert c.verdict == "real gain"
    d = {x["key"]: x for x in c.differences}
    assert d["template"]["blocking"] is False and "model" not in d
