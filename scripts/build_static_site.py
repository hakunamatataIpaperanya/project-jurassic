# สร้างเว็บแบบไม่มีเซิร์ฟเวอร์ใน dist/ สำหรับอัปขึ้น Cloudflare Pages (หน้าเว็บชุดเดียวกับ webapp/static ที่ server ใช้)
#   ก่อนรัน: scripts/web_models.py (โมเดล ONNX) และ scripts/web_static.py (ข้อมูล JSON + ตารางทายสกุล)
#   รัน:     .venv/bin/python scripts/build_static_site.py [โฟลเดอร์ปลายทาง]      ดูก่อนอัป: .venv/bin/python -m http.server -d dist 8060
#   อัปขึ้น: npx wrangler@4.156.1 deploy      (ตาม wrangler.jsonc: Cloudflare Workers static assets ชื่อ project-jurassic; ครั้งแรก: npx wrangler@4.156.1 login)
# ข้อมูล / โมเดลอยู่ในโฟลเดอร์ที่ตั้งชื่อตาม hash ของเนื้อหา (data-xxxxxxxx/ ...) จึงให้เบราว์เซอร์เก็บไว้ได้นาน และได้ไฟล์ใหม่ทันทีเมื่อ build ใหม่
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "webapp" / "static"
DATA = ROOT / "webapp" / "data"
WEB = ROOT / "models" / "web"
SITE_URL = os.environ.get("SITE_URL", "https://project-jurassic.roject--urassic.workers.dev")   # URL เต็มของเว็บ (ภาพตัวอย่างลิงก์ต้องเป็น URL เต็ม)
MAX_FILE = 25 * 1024 * 1024                         # ขีดจำกัดของ Cloudflare Pages: ไฟล์ละ 25 MiB, ไม่เกิน 20,000 ไฟล์
MAX_FILES = 20000
DATA_FILES = ["meta", "grid", "pbdb", "options", "showcase", "period_profile", "provinces", "geology", "cells"]
MODEL_FILES = ["genus_known.onnx", "genus_new.onnx", "specimen.json", "specimen.bin"]


def _hash(files):
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:10]


def _copy_set(files, dist, prefix):
    d = dist / f"{prefix}-{_hash(files)}"
    d.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, d / f.name)
    return d


def build(dist):
    data = [DATA / f"{n}.json" for n in DATA_FILES]
    models = [WEB / n for n in MODEL_FILES]
    man = json.loads((WEB / "bert" / "manifest.json").read_text(encoding="utf-8"))
    bert = [WEB / "bert" / n for n in ["manifest.json", "tokenizer.json", "tokenizer_config.json", *man["parts"]]]
    missing = [f.relative_to(ROOT) for f in data + models + bert if not f.exists()]
    if missing:
        sys.exit(f"ไม่มีไฟล์ {missing} — รัน scripts/web_models.py และ scripts/web_static.py ก่อน")
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    shutil.copytree(STATIC, dist / "static", ignore=shutil.ignore_patterns("index.html", "busy.html", ".DS_Store"))
    dirs = {"data": _copy_set(data, dist, "data"), "models": _copy_set(models, dist, "models"), "bert": _copy_set(bert, dist, "bert")}
    # หน้าแรก: index.html ของ server + ที่อยู่ข้อมูล (app.js เห็นแล้วเปลี่ยนเป็นโหมด static) + เลขเวอร์ชันของ css / js ตามเนื้อหา
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in re.findall(r'"/static/([\w.\-]+\.(?:css|js))"', html):
        html = html.replace(f'"/static/{name}"', f'"/static/{name}?v={_hash([STATIC / name])}"')
    cfg = json.dumps({**{k: f"/{d.name}/" for k, d in dirs.items()}, "built": date.today().isoformat()})   # built = วันที่ build (หน้าแรกบอก "ปรับปรุงเว็บ")
    html = html.replace('<meta charset="utf-8">', f'<meta charset="utf-8">\n  <meta name="jurassic-static" content=\'{cfg}\'>', 1)
    # ภาพตัวอย่างลิงก์ (Open Graph): URL เต็ม + เลขเวอร์ชันตามเนื้อหาภาพ (แอปแชทจำภาพเก่าตาม URL)
    og = STATIC / "img" / "og.jpg"
    html = html.replace('content="/static/img/og.jpg"', f'content="{SITE_URL}/static/img/og.jpg?v={_hash([og])}"')
    html = html.replace('<meta property="og:type"', f'<meta property="og:url" content="{SITE_URL}/">\n  <meta property="og:type"', 1)
    (dist / "index.html").write_text(html, encoding="utf-8")
    # เก็บในเบราว์เซอร์: โฟลเดอร์ข้อมูล / โมเดล (ชื่อตาม hash) เก็บได้ 1 ปี, หน้าแรกตรวจใหม่ทุกครั้ง
    rules = ["/\n  Cache-Control: no-cache", "/index.html\n  Cache-Control: no-cache", "/static/*\n  Cache-Control: public, max-age=3600"]
    rules += [f"/{d.name}/*\n  Cache-Control: public, max-age=31536000, immutable" for d in dirs.values()]
    (dist / "_headers").write_text("\n".join(rules) + "\n", encoding="utf-8")
    files = [p for p in dist.rglob("*") if p.is_file()]
    big = [(p.relative_to(dist), p.stat().st_size) for p in files if p.stat().st_size > MAX_FILE]
    if big or len(files) > MAX_FILES:
        sys.exit(f"เกินขีดจำกัดของ Cloudflare Pages: {big} / {len(files)} ไฟล์")
    return dirs, files


def _mb(files):
    return sum(p.stat().st_size for p in files) / 1e6


def main():
    dist = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "dist"
    dirs, files = build(dist)
    print(f"{dist.relative_to(ROOT) if dist.is_relative_to(ROOT) else dist}: {len(files)} ไฟล์ {_mb(files):.0f} MB")
    for d in dirs.values():
        print(f"  {d.name}/  {_mb(d.iterdir()):.1f} MB")
    print(f"  ไฟล์ใหญ่สุด {max(p.stat().st_size for p in files) / 2**20:.1f} MiB (ขีดจำกัด 25 MiB)")


if __name__ == "__main__":
    main()
