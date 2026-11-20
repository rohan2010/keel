"""Chapter 6 B1: every argument has a source (src/keel/core/arg_source.py)."""

from keel.core.arg_source import ArgSourceCheck, Sources, unsourced

POST = {"name": "post_comment", "parameters": {"properties": {
    "tweet_id": {"type": "integer"}, "comment_content": {"type": "string"}}}}
BUY = {"name": "place_order", "parameters": {"properties": {
    "symbol": {"type": "string"}, "amount": {"type": "integer"},
    "order_type": {"type": "string", "enum": ["Buy", "Sell"]}, "price": {"type": "number", "default": 0.0}}}}


def test_invented_comment_is_held_given_one_runs():
    ck = ArgSourceCheck()
    srcs = ["Comment on my latest tweet.", "{'tweet_id': 10, 'content': 'Shipping today'}"]
    call = [{"post_comment": {"tweet_id": 10, "comment_content": "Great job, congrats!"}}]
    msg = ck.check(call, [POST], srcs)
    assert msg and "comment_content" in msg and "ask the user" in msg and "tweet_id" not in msg
    ok = [{"post_comment": {"tweet_id": 10, "comment_content": "Congrats on shipping"}}]
    assert ck.check(ok, [POST], srcs + ["Say: congrats on shipping"]) is None


def test_held_once_then_the_same_call_runs():
    ck = ArgSourceCheck()
    call = [{"place_order": {"symbol": "AAPL", "amount": 100, "order_type": "Buy"}}]
    srcs = ["Buy some Apple stock, symbol AAPL."]
    assert ck.check(call, [BUY], srcs) is not None          # 100 shares: nobody said 100
    assert ck.check(call, [BUY], srcs) is None              # the model insists: it runs
    assert ck.holds == 1


def test_enum_default_bool_and_numbers():
    src = Sources(["Buy 1,000 shares of AAPL at 150.5"])
    assert unsourced({"symbol": "AAPL", "amount": 1000, "order_type": "Buy", "price": 150.5}, src,
                     BUY["parameters"]) == []
    assert unsourced({"price": 0.0}, Sources(["buy AAPL"]), BUY["parameters"]) == []   # the default
    assert unsourced({"amount": 15}, src, BUY["parameters"]) == [("amount", 15)]          # not "150.5" or "1,000"
    assert unsourced({"flag": True}, Sources([]), {}) == []


def test_paths_file_stems_and_case():
    src = Sources(["Move the Report into the archive folder.", "ls: documents  archive  report.txt"])
    assert unsourced({"source": "report.txt", "destination": "archive/report.txt"}, src, {}) == []
    assert unsourced({"file_name": "notes.md"}, Sources(["create a file called notes"]), {}) == []
    assert unsourced({"dir": "Documents"}, src, {}) == []                                   # case-insensitive
    assert unsourced({"dir": "photos"}, src, {}) == [("dir", "photos")]


def test_long_text_needs_half_its_words():
    src = Sources(["Email Sarah that the quarterly budget review moves to Friday afternoon"])
    assert unsourced({"body": "Hi Sarah, the quarterly budget review moves to Friday afternoon."}, src, {}) == []
    assert unsourced({"body": "Please remember to bring snacks and your laptop charger tomorrow"}, src, {}) != []


def test_lists_and_nested_objects():
    src = Sources(["Invite alice@example.com and bob@example.com"])
    assert unsourced({"to": ["alice@example.com", "bob@example.com"]}, src, {}) == []
    assert unsourced({"to": ["alice@example.com", "carol@example.com"]}, src, {}) == [("to", "carol@example.com")]
    assert unsourced({"meta": {"city": "Paris"}}, Sources(["weather in paris"]), {}) == []


def test_whole_reply_is_held_when_one_call_lacks_a_source():
    ck = ArgSourceCheck()
    calls = [{"place_order": {"symbol": "AAPL", "order_type": "Buy", "amount": 5}},
             {"post_comment": {"tweet_id": 3, "comment_content": "Bought it!"}}]
    msg = ck.check(calls, [BUY, POST], ["Buy 5 AAPL and tell everyone on tweet 3"])
    assert msg is not None and "comment_content" in msg and "amount" not in msg
