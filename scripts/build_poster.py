# สร้างโปสเตอร์ A2 แนวตั้ง (42 × 59.4 ซม.) เป็น JPG 300 DPI 2 ฉบับ: ภาษาอังกฤษ (docs/poster/poster_en.html) และภาษาไทย (poster_th.html)
#   หน้าตาร่วมกันใน docs/poster/poster.css (โทนสว่างตามเว็บโหมดสว่าง) · แผนที่: docs/poster/make_poster_maps.py · ภาพหน้าจอ: docs/poster/shoot_web.py
#   แถบหัว: docs/poster/Header_Senior Project Showcase 2023_42x4.6cm_outline.ai (เวกเตอร์ แปลงเป็นภาพ 300 DPI ด้วย PyMuPDF)
#   ฟอนต์: IBM Plex Sans / IBM Plex Sans Thai / Oswald / Rajdhani / Kanit ใน docs/poster/fonts (ไม่ต้องใช้อินเทอร์เน็ตตอนสร้าง)
#   ถ่ายหน้า HTML ด้วย Chrome ที่ 300 / 96 = 3.125 พิกเซลต่อ CSS px แล้วปรับเป็น 4961 × 7016 พิกเซลพอดี
#   และตรวจว่าเนื้อหาในแต่ละคอลัมน์ไม่ล้นกรอบ (พิมพ์ที่ว่างที่เหลือของแต่ละคอลัมน์เป็นเซนติเมตร)
# รัน:  <python ที่มี playwright, pymupdf, pillow> scripts/build_poster.py [en|th|v2en|v2th|v3en|v3th]   (ไม่ระบุ = ทุกฉบับ)
import sys
from pathlib import Path

import pymupdf
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
POSTER = ROOT / "docs" / "poster"
HEADER_AI = POSTER / "Header_Senior Project Showcase 2023_42x4.6cm_outline.ai"
VERSIONS = {"en": (POSTER / "poster_en.html", POSTER / "Innovative_Paleontological_Data_Management_Prototype_Thailand.jpg"),
            "th": (POSTER / "poster_th.html", POSTER / "ต้นแบบนวัตกรรมการจัดการข้อมูลบรรพชีวินวิทยา.jpg"),
            # v2: แบบอินโฟกราฟิก (ตัวอักษรน้อย ไอคอน ตัวเลขใหญ่) — poster_v2.css + icons.js
            "v2en": (POSTER / "poster_v2_en.html", POSTER / "Innovative_Paleontological_Data_Management_Prototype_Thailand_v2.jpg"),
            "v2th": (POSTER / "poster_v2_th.html", POSTER / "ต้นแบบนวัตกรรมการจัดการข้อมูลบรรพชีวินวิทยา_v2.jpg"),
            # v3: v1 + ตัวเลขเด่นในหัวข้อ + แผนภาพแบบภาพประกอบ — poster_v3.css + icons.js
            "v3en": (POSTER / "poster_v3_en.html", POSTER / "Innovative_Paleontological_Data_Management_Prototype_Thailand_v3.jpg"),
            "v3th": (POSTER / "poster_v3_th.html", POSTER / "ต้นแบบนวัตกรรมการจัดการข้อมูลบรรพชีวินวิทยา_v3.jpg")}
DPI = 300
W_CM, H_CM = 42.0, 59.4
W_PX, H_PX = round(W_CM / 2.54 * DPI), round(H_CM / 2.54 * DPI)       # 4961 × 7016
SCALE = DPI / 96                                                    # CSS px → พิกเซลภาพ


def render_header():
    out = POSTER / "_build" / "header.png"
    out.parent.mkdir(exist_ok=True)
    page = pymupdf.open(str(HEADER_AI))[0]
    page.get_pixmap(dpi=DPI, alpha=False).save(str(out))
    return out


def build(html, out):
    css_w, css_h = W_CM / 2.54 * 96, H_CM / 2.54 * 96
    shot = POSTER / "_build" / f"{html.stem}.png"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--allow-file-access-from-files"])
        pg = b.new_page(viewport={"width": int(css_w) + 1, "height": int(css_h) + 1}, device_scale_factor=SCALE)
        pg.goto(html.as_uri())
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(800)
        cm = 96 / 2.54
        free = pg.evaluate("""() => [...document.querySelectorAll('.col')].map(c => {
            const kids = [...c.children]; const last = kids[kids.length - 1].getBoundingClientRect();
            return [c.getBoundingClientRect().bottom - last.bottom, c.scrollHeight - c.clientHeight];
        })""")
        fonts = pg.evaluate("() => [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family + ' ' + f.weight)")
        pg.screenshot(path=str(shot), clip={"x": 0, "y": 0, "width": css_w, "height": css_h})
        b.close()
    im = Image.open(shot).convert("RGB")
    if im.size != (W_PX, H_PX):
        im = im.resize((W_PX, H_PX), Image.LANCZOS)
    im.save(out, "JPEG", quality=95, subsampling=0, dpi=(DPI, DPI), optimize=True)
    print(f"saved {out.name} | {im.size[0]} × {im.size[1]} px @ {DPI} DPI = {W_CM} × {H_CM} cm | ฟอนต์ที่โหลด {len(set(fonts))}")
    for k, (gap, over) in enumerate(free, 1):
        state = f"ล้น {over / cm:.1f} ซม." if over > 1 else f"ว่างท้ายคอลัมน์ {gap / cm:.1f} ซม."
        print(f"  คอลัมน์ {k}: {state}")


def main():
    render_header()
    for lang in (sys.argv[1:] or list(VERSIONS)):
        build(*VERSIONS[lang])


if __name__ == "__main__":
    sys.exit(main())
