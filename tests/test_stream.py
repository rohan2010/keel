"""P2: the live view of a streaming reply hides tool-call markup and think blocks, whatever the chunking."""

import random

import pytest

from keel.core.stream import VisibleText

CASES = [
    ("Let me look at the files.\n\n<tool_call>\n<function=list_files>\n</function>\n</tool_call>",
     "Let me look at the files.\n\n"),
    ("\n\nDone: I moved 3 files.", "Done: I moved 3 files."),
    ("<think>plan it</think>The answer is 4.", "The answer is 4."),
    ("Use a < b and <b>bold</b> here.", "Use a < b and <b>bold</b> here."),
    ("Ends with a partial <tool", "Ends with a partial <tool"),        # never became a tag: shown at the end
    ("<tool_call>\n<function=read_file>\n</function>\n</tool_call>", ""),
    ("<thinking about it> is fine", "<thinking about it> is fine"),
]


@pytest.mark.parametrize("text,want", CASES)
def test_visible_text_any_chunking(text, want):
    rng = random.Random(0)
    for trial in range(50):                      # one char at a time, whole, and random splits
        cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(0, 6)))) if trial else []
        if trial == 1:
            cuts = list(range(1, len(text)))
        parts = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)], strict=True)]
        v = VisibleText()
        got = "".join(v.feed(p) for p in parts) + v.flush()
        assert got == want, (parts, got)


def test_nothing_after_a_tool_call():
    v = VisibleText()
    assert v.feed("Ok <tool_call>x") == "Ok "
    assert v.feed("more prose later") == "" and v.flush() == ""
