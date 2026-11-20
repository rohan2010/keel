"""keel: an agent harness that makes small local models dependable at tool use, plus an eval runner that
says when a change really helped.

`keel.core` (records, tools, layers, events, backend interface) imports no ML framework; backends live in
`keel.backends` and load their framework only when used.
"""
