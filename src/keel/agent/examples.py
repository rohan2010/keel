"""Short worked examples appended to the system prompt (`--examples`).

Target: judgment failures that no concrete check can catch: renaming a longer name that merely contains
the old one (D1), and file-type judgment such as putting videos with images (A1). The examples teach the general
rule with different names and situations from the task set; a test checks they contain no task id and no task file
name. They were written knowing which failure types they target, so only unseen dev seeds and validation runs can
show whether they generalize (said in the write-up).
"""

EXAMPLES = """Examples of careful work:

1. Renaming a name everywhere. Asked to rename the CSS class `card` to `tile`: first search for `card`. Change it
   where it is the whole name (`class="card big"`, `.card {`, `querySelector('.card')`), but leave longer names that
   only contain it, such as `card-header` or `cardholder`: those are different names, so they stay as they are.

2. Sorting files by type. Decide by the extension. Images: .jpg .jpeg .png .gif .webp .heic .svg. Videos: .mp4 .mov
   .avi .mkv. Audio: .mp3 .wav .m4a .flac. Documents: .pdf .doc .docx .txt .md .rtf .ppt .pptx .xls .xlsx .csv.
   Code: .py .js .ts .html .css .json .sh .java .go. Archives and installers: .zip .tar .gz .dmg .iso .pkg. A video is
   not an image. If the user's folders have no place for a type, it goes in the folder meant for everything else.

3. Before you say you are done, look at what you changed (read a changed file back, or list the folder) and compare it
   with what the user asked."""
