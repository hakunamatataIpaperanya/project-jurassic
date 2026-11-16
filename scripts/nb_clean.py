# Collapse tqdm stderr spam in executed notebooks: each progress bar keeps only its final state.
# usage: .venv/bin/python scripts/nb_clean.py <notebook.ipynb> [...]   (also called by run_notebook.py)
import re
import sys

import nbformat

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BAR = re.compile(r"^(.*?):\s*\d+%\|")


def _collapse(texts):
    lines = []
    for raw in ANSI.sub("", "".join(texts)).split("\n"):
        seg = [x for x in raw.split("\r") if x.strip()]
        if seg:
            lines.append(seg[-1].rstrip())
    last, order = {}, []
    for ln in lines:
        m = BAR.match(ln)
        key = ("bar", m.group(1)) if m else ("txt", ln)
        if key not in last:
            order.append(key)
        last[key] = ln
    return "\n".join(last[k] for k in order)


def _flush(buf, outs):
    # stderr ที่สะสมไว้ -> output เดียวที่ย่อแล้ว (ถ้ายังเหลือข้อความ)
    if buf:
        text = _collapse(buf)
        if text:
            outs.append(nbformat.v4.new_output("stream", name="stderr", text=text + "\n"))
        buf.clear()


def clean_notebook(nb):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        outs, buf = [], []
        for o in cell.get("outputs", []):
            if o.output_type == "stream" and o.name == "stderr":
                buf.append(o.text)
            else:
                _flush(buf, outs)
                outs.append(o)
        _flush(buf, outs)
        cell.outputs = outs
    return nb


if __name__ == "__main__":
    for path in sys.argv[1:]:
        nb = nbformat.read(path, as_version=4)
        nbformat.write(clean_notebook(nb), path)
        print("cleaned", path)
