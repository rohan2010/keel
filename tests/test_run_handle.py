"""Chapter 3, P5b: episodes can join a running run (swarm groundwork; no swarm here). Scripted and server backends,
no model. The MLX scheduler's path is checked on the real model by doc/product/spikes/p5b-join/mlx_join.py."""

import threading
import time

import pytest
from test_server_backend import Fake

from keel.backends.scripted import ScriptedBackend
from keel.backends.server import ServerBackend
from keel.core.backend import RunHandle, RunHooks, RunLimits


class Ep:
    """The smallest episode a backend can serve. `on_reply(ep, text)` decides what a reply does."""

    def __init__(self, name, on_reply=None):
        self.session_id = name
        self.name = name
        self.on_reply = on_reply or (lambda ep, text: ep.end(text))
        self.done, self.stop, self.result = False, "", None
        self.prompt_tokens = self.processed_tokens = self.output_tokens = 0
        self.t_start = self.seconds = 0.0
        self.ban_tool_call = False
        self.steps = 0
        self.notes = []

    def end(self, text):
        self.done, self.stop, self.result = True, "answer", text


def hooks(order, handle):
    def apply(e, text, fr):
        e.steps += 1
        e.on_reply(e, text)
    return RunHooks(prompt_fn=lambda e: e.name, apply_fn=apply, finish_fn=lambda e: order.append(e.name),
                    handle=handle)


def parent_spawning(handle, children, at_step=1, timeout=10):
    """A parent that, on its `at_step`-th reply, submits `children`, waits for them, then answers with their results."""
    def on_reply(ep, text):
        if ep.steps == at_step:
            for c in children:
                handle.submit(c)
            handle.wait(*children, timeout=timeout)
            ep.end("parent got: " + " + ".join(c.result for c in children))
    return on_reply


def scripted():
    return ScriptedBackend(lambda e, k: f"{e.name} reply {k}")


def server(fake, delay=0.0):
    f = fake(lambda p, b, n: {"choices": [{"text": f"{b['prompt']} reply", "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 2}}, delay=delay)
    return ServerBackend(f.url, "openai"), f


@pytest.fixture
def fake():
    made = []

    def make(respond, **kw):
        f = Fake(respond, **kw)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


BACKENDS = ["scripted", "server-1slot", "server-3slots"]


def make_backend(kind, fake):
    if kind == "scripted":
        return scripted(), RunLimits(batch_size=1)
    be, _ = server(fake, delay=0.05)
    return be, RunLimits(batch_size=1 if kind == "server-1slot" else 3)


@pytest.mark.parametrize("kind", BACKENDS)
def test_parent_submits_two_children_mid_run_and_gets_both_results(kind, fake):
    handle, order = RunHandle(), []
    c1, c2 = Ep("c1"), Ep("c2")
    parent = Ep("parent")
    parent.on_reply = parent_spawning(handle, [c1, c2])
    other = Ep("other")
    be, limits = make_backend(kind, fake)
    stats = be.run([parent, other], hooks(order, handle), limits)
    assert parent.result == ("parent got: c1 reply 0 + c2 reply 0" if kind == "scripted"
                             else "parent got: c1 reply + c2 reply")
    assert sorted(order) == ["c1", "c2", "other", "parent"]
    assert order.index("parent") > max(order.index("c1"), order.index("c2"))    # children finish first
    assert stats["submitted"] == 2 and handle.is_finished(c1) and handle.is_finished(c2)


@pytest.mark.parametrize("kind", BACKENDS)
def test_episode_submitted_after_the_others_finished_still_runs(kind, fake):
    handle, order = RunHandle(), []
    late = Ep("late")

    def last_step(ep, text):                                 # the last episode alive submits on its way out
        handle.submit(late)
        ep.end(text)
    first, last = Ep("first"), Ep("last", last_step)
    be, limits = make_backend(kind, fake)
    if kind != "scripted":
        last.on_reply = lambda ep, text: (time.sleep(0.3), last_step(ep, text))   # `first` is long finished by then
    be.run([first, last], hooks(order, handle), limits)
    assert order == ["first", "last", "late"] and late.done
    with pytest.raises(RuntimeError, match="run has finished"):
        handle.submit(Ep("too late"))


@pytest.mark.parametrize("kind", BACKENDS)
def test_nested_spawn_and_waiting_parents_do_not_block_slots(kind, fake):
    """A grandparent waits on a parent that waits on its children: with one slot, only freed slots make this work."""
    handle, order = RunHandle(), []
    kids = [Ep("k1"), Ep("k2")]
    parent = Ep("parent")
    parent.on_reply = parent_spawning(handle, kids)
    grand = Ep("grand")
    grand.on_reply = parent_spawning(handle, [parent])
    be, limits = make_backend(kind, fake)
    be.run([grand], hooks(order, handle), limits)
    assert order[-1] == "grand" and set(order) == {"k1", "k2", "parent", "grand"}
    assert grand.result.startswith("parent got: parent got:")


def test_server_slots_still_cap_concurrent_requests_with_a_handle(fake):
    be, f = server(fake, delay=0.2)
    handle, order = RunHandle(), []
    eps = [Ep(f"e{i}") for i in range(5)]
    be.run(eps, hooks(order, handle), RunLimits(batch_size=2))
    assert len(order) == 5 and f.peak == 2


def test_no_handle_means_the_old_path(fake):
    order = []
    be, _ = server(fake)
    st = be.run([Ep("a"), Ep("b")], hooks(order, None), RunLimits(batch_size=2))
    assert "submitted" not in st and sorted(order) == ["a", "b"]
    st = scripted().run([Ep("a")], hooks(order, None), RunLimits())
    assert "submitted" not in st


def test_scripted_refuses_a_wait_cycle():
    handle, order = RunHandle(), []
    a = Ep("a")
    a.on_reply = lambda ep, text: handle.wait(ep)            # waits on itself
    with pytest.raises(RuntimeError, match="cycle"):
        scripted().run([a], hooks(order, handle), RunLimits())


def test_wait_timeout_and_waiting_count():
    h = RunHandle()
    h.attach()
    e = Ep("never")
    seen = []
    t = threading.Thread(target=lambda: seen.append(pytest.raises(TimeoutError, h.wait, e, timeout=0.3)))
    t.start()
    time.sleep(0.1)
    assert h.waiting == 1
    t.join()
    assert h.waiting == 0 and seen
    h.finished(e)
    assert h.wait(e, timeout=0.1) == [e]


def test_agent_loop_passes_the_handle(tmp_path):
    from keel.agent.loop import run_episodes_on
    from keel.evals import tasks as T
    handle = RunHandle()
    be = ScriptedBackend(["Nothing to do."])
    res = run_episodes_on(be, [T.build("F3", 0)], tmp_path, handle=handle)
    assert res[0]["stop"] == "answer" and handle._closed
