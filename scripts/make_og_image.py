# ภาพตัวอย่างลิงก์ (Open Graph: ภาพที่ LINE / Facebook / iMessage แสดงเมื่อแชร์ลิงก์เว็บ) = ภาพหน้าแรกของเว็บ 1200 × 630
#   ต้องมีเว็บเปิดอยู่ เช่น  .venv/bin/python -m http.server -d dist 8060   (หลัง scripts/build_static_site.py)
#   รัน:  uv run --no-project --with playwright python scripts/make_og_image.py [URL] [--lang en] [--out ไฟล์]   -> webapp/static/img/og.jpg
#         (README ภาษาอังกฤษใช้ --lang en --out assets/readme/web_en.jpg)
#   แล้ว build ใหม่ (scripts/build_static_site.py) ให้ภาพเข้าไปใน dist/
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARGS = sys.argv[1:]


def opt(name, default):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


POS = [a for i, a in enumerate(ARGS) if not a.startswith("--") and (i == 0 or not ARGS[i - 1].startswith("--"))]
URL = POS[0] if POS else "http://127.0.0.1:8060/"
LANG = opt("--lang", "th")
OUT = ROOT / opt("--out", "webapp/static/img/og.jpg")
SHOW = "Phuwiangosaurus"                         # ซากที่แสดงในภาพหน้าแรก (ไดโนเสาร์ไทย)

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    ctx = b.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
    ctx.add_init_script(f"localStorage.setItem('pj-lang','{LANG}'); localStorage.setItem('pj-theme','dark'); localStorage.setItem('pj-intro-hidden','1');")
    pg = ctx.new_page()
    pg.goto(URL + "#index")
    pg.wait_for_timeout(5000)
    for _ in range(8):                           # เลื่อนไปที่ซากที่ต้องการ แล้วรอภาพประกอบเสร็จ
        if SHOW in pg.inner_text("#hero-scan"):
            break
        pg.click('.sc-btn[data-d="1"]')
        pg.wait_for_timeout(3500)
    pg.mouse.move(5, 5)                          # ไม่ให้เมาส์อยู่บนภาพ (ภาพหยุดเปลี่ยนตัวตอนชี้)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT), type="jpeg", quality=88)
    b.close()
print(OUT.relative_to(ROOT), f"{OUT.stat().st_size / 1e3:.0f} kB")
