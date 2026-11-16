# Numbers quoted in the notebook summaries must be traceable to an executed output (catches summaries left stale after a rerun).
# Every 0.xxx value in a summary must appear (as printed, or rounded to 3 decimals) in the outputs of the SAME notebook or of a
# notebook the summary names ("notebook 15"). Differences / spreads (prefixed by +, −, -, ±), percentages and map resolutions
# (0.002°) are not metrics and are not checked here.
import glob
import html
import re

import nbformat
import pytest

from conftest import ROOT

NUM = re.compile(r"(?<![+−\-±\d.])0\.\d{3}(?![\d°])")
LONG = re.compile(r"(?<![\d.])0\.\d{4,}")


def _pool(path):
    pool = set()
    for c in nbformat.read(path, 4).cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            texts = [o.get("text", "")] + [str(v) for k, v in o.get("data", {}).items() if k in ("text/plain", "text/html")]
            for t in texts:
                t = html.unescape(t)
                pool |= set(re.findall(r"0\.\d{3}", t))
                pool |= {f"{float(x):.3f}" for x in LONG.findall(t)}
    return pool


NBS = {p.split("/")[-1][:-6]: p for p in glob.glob(str(ROOT / "notebooks" / "*.ipynb"))}
POOLS = {}


def pool_of(name):
    if name not in POOLS:
        POOLS[name] = _pool(NBS[name])
    return POOLS[name]


@pytest.mark.parametrize("name", sorted(NBS))
def test_summary_numbers_traceable(name):
    nb = nbformat.read(NBS[name], 4)
    summary = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown" and "## สรุป" in c.source)
    refs = {n for n in NBS for num in re.findall(r"notebook (\d+b?)", summary) if n.startswith(num + "_")}
    pool = set().union(*(pool_of(n) for n in refs | {name}))
    summary = re.sub(r"[±+−\-]\s*0\.\d+", " ", summary)          # ผลต่าง / ช่วงความแปรปรวน (เช่น "± 0.006") เป็นค่าที่คำนวณต่อ ไม่ตรวจ
    missing = sorted({x for x in NUM.findall(summary) if x not in pool})
    assert not missing, f"numbers in summary not found in this notebook or the notebooks it names ({sorted(refs)}): {missing}"


def _pct_pool(name):
    # ค่าร้อยละที่ output แสดง: "48%" หรือสัดส่วน 0.479 (→ 48)
    pool = set()
    for c in nbformat.read(NBS[name], 4).cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            t = html.unescape("\n".join([o.get("text", "")] + [str(v) for k, v in o.get("data", {}).items() if k in ("text/plain", "text/html")]))
            pool |= {str(round(float(x))) for x in re.findall(r"(\d+(?:\.\d+)?)\s?%", t)}
            pool |= {str(f(100 * float(x))) for x in re.findall(r"(?<![\d.])0\.\d+", t) for f in (round, lambda v: int(v + 0.5))}
    return pool


@pytest.mark.parametrize("name", sorted(NBS))
def test_summary_percentages_traceable(name):
    # ร้อยละในสรุปต้องมาจาก output ของ notebook นี้หรือ notebook ที่สรุปอ้างถึง (ผลต่าง ±/+/− ไม่ตรวจ)
    nb = nbformat.read(NBS[name], 4)
    summary = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown" and "## สรุป" in c.source)
    if not summary:
        pytest.skip("ไม่มีสรุป")
    refs = {n for n in NBS for num in re.findall(r"notebook (\d+b?)", summary) if n.startswith(num + "_")} | {name}
    pool = set().union(*(_pct_pool(n) for n in refs))
    summary = re.sub(r"[±+−\-]\s*\d+(?:\.\d+)?\s?%", " ", summary)
    missing = sorted({p for p in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%", summary) if str(round(float(p))) not in pool}, key=float)
    assert not missing, f"percentages in summary not found in outputs of {sorted(refs)}: {missing}"
