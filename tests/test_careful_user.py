"""Careful simulated user judges by content, not path (E3 P-split smoke 2: moved originals were approved for rm -rf)."""

from keel.agent.guard import _files, careful_user


def _make(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_moved_original_is_still_original(tmp_path):
    _make(tmp_path, {"IMG_1.jpg": "photo1", "IMG_2.jpg": "photo2"})
    conf = careful_user(tmp_path, _files(tmp_path))
    (tmp_path / "2021" / "03").mkdir(parents=True)
    (tmp_path / "IMG_1.jpg").rename(tmp_path / "2021" / "03" / "2021-03-01_1200.jpg")
    assert conf("rm -rf 2021", ["2021/03/2021-03-01_1200.jpg"]) is False


def test_agent_made_content_and_duplicates_approved(tmp_path):
    _make(tmp_path, {"a.jpg": "A", "copy of a.jpg": "A"})
    conf = careful_user(tmp_path, _files(tmp_path))
    (tmp_path / "notes.txt").write_text("agent notes")
    assert conf("rm notes.txt", ["notes.txt"]) is True
    assert conf("rm 'copy of a.jpg'", ["copy of a.jpg"]) is True
    assert conf("rm a.jpg 'copy of a.jpg'", ["a.jpg", "copy of a.jpg"]) is False


def test_copied_original_then_delete_one(tmp_path):
    _make(tmp_path, {"a.jpg": "A"})
    conf = careful_user(tmp_path, _files(tmp_path))
    (tmp_path / "2021").mkdir()
    (tmp_path / "2021" / "a.jpg").write_text("A")       # sorted copy
    assert conf("rm a.jpg", ["a.jpg"]) is True           # the copy stays
    assert conf("rm -rf 2021 a.jpg", ["2021/a.jpg", "a.jpg"]) is False
