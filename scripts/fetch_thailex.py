# Scrape the Thai Stratigraphic Lexicon (thailex.geolex.org): one row per formation with Thai synonym, period, age interval,
# province/region, lithology text and main lithology pattern. The "Fossils" section is deliberately NOT stored: it names the
# genera found in each formation, which would leak the task-1 target.
# usage: .venv/bin/python scripts/fetch_thailex.py
import html
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "thailex"
BASE = "https://thailex.geolex.org"


def lines_of(page):
    t = re.sub(r"<(style|script).*?</\1>", "", page, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
    return [ln.strip() for ln in t.split("\n") if ln.strip()]


def after(lines, key, n=1):
    # ค่าหลังหัวข้อ "<key>:" (มีโคลอน) — เมนูด้านบนของเว็บมีคำว่า "Period" ไม่มีโคลอน จึงต้องตรงตัว
    i = next((i for i, ln in enumerate(lines) if ln == key + ":"), None)
    return " ".join(lines[i + 1:i + 1 + n]) if i is not None else None


def section(lines, start, stops):
    i = next((i for i, ln in enumerate(lines) if ln == start), None)
    if i is None:
        return None
    out = []
    for ln in lines[i + 1:]:
        if ln in stops:
            break
        if not ln.startswith("[Figure"):
            out.append(ln)
    return " ".join(out)


def parse(name, page):
    L = lines_of(page)
    syn = None
    for i, ln in enumerate(L):
        if ln.startswith("Synonym"):
            syn = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if not syn and i + 1 < len(L):                    # ชื่อไทยอยู่บรรทัดถัดไป
                syn = L[i + 1]
            break
    syn = syn.strip(" ;.") or None if syn else None
    # ชื่อไทย (หมวดหิน.../กลุ่มหิน...) ตัวแรกในบรรทัด Synonym หรือส่วนต้นของหน้า ใช้จับคู่กับแผนที่หน่วยหินและ site_FORMATION ของ ทธ.
    start = next((i for i, ln in enumerate(L) if ln == "Type Locality and Naming"), 0)
    th = re.findall(r"(?:หมวดหิน|กลุ่มหิน)[\u0E00-\u0E7F]+", " ".join([syn or ""] + L[start:start + 40]))
    return {
        "name_en": name, "synonym": syn, "name_th": th[0] if th else None, "period": after(L, "Period"), "age_interval": after(L, "Age Interval"),
        "province": after(L, "Province"),
        "lithology": section(L, "Lithology and Thickness", {"Lithology Pattern:", "Relationships and Distribution"}),
        "lithology_pattern": after(L, "Lithology Pattern"),
        "lower_contact": after(L, "Lower contact"), "upper_contact": after(L, "Upper contact"),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    idx = requests.get(f"{BASE}/all-formations", timeout=60).text
    names = sorted({urllib.parse.unquote(m) for m in re.findall(r'href="/?formations/([^"]+)"', idx)})
    rows = []
    for i, n in enumerate(names):
        cache = OUT / "pages" / f"{re.sub(r'[^A-Za-z0-9]+', '_', n)}.html"
        cache.parent.mkdir(exist_ok=True)
        if not cache.exists():
            r = requests.get(f"{BASE}/formations/{urllib.parse.quote(n)}", timeout=60)
            r.raise_for_status()
            cache.write_text(r.text, encoding="utf-8")
            time.sleep(0.5)                                   # สุภาพกับเซิร์ฟเวอร์
        rows.append(parse(n, cache.read_text(encoding="utf-8")))
        print(f"{i + 1}/{len(names)} {n}", end="\r")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "formations.csv", index=False, encoding="utf-8-sig")
    print(f"\nบันทึก {OUT / 'formations.csv'}: {len(df)} หน่วย | มีชื่อไทย {df.synonym.notna().sum()} | มีหินวิทยา {df.lithology.notna().sum()}")
