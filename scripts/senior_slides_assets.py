# ภาพประกอบสไลด์นำเสนอ senior project (scripts/build_senior_slides.py) → docs/slides/assets/
#   1) ไอคอน Lucide (POSTER_ICONS) และภาพเงาซาก (POSTER_SIL) จาก docs/poster/icons.js เป็น PNG พื้นโปร่งใส ตามสีของธีม (icons/<ชื่อ>-<สี>.png)
#   2) ภาพหน้าจอเว็บต้นแบบโหมดสว่าง ภาษาไทย (web_*.png) — ต้องมีเว็บรันที่ 127.0.0.1:8050 (scripts/run_webapp.sh)
#   3) แผนที่โทนสว่างของโปสเตอร์ (docs/poster/_build/maps_light) และรูปแผนที่จากรายงาน
#   4) รูปกราฟที่มีอยู่แล้ว (charts/): ภาพผลลัพธ์ใน notebooks/*.ipynb (ตัดเฉพาะกราฟย่อยที่ใช้) และรูปกราฟของรายงาน
# รัน:  <python ที่มี playwright> scripts/senior_slides_assets.py [icons] [web] [maps] [charts]   (ไม่ใส่ = ทำทุกอย่าง)
import base64
import io
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "slides" / "assets"
POSTER = ROOT / "docs" / "poster"
WEB = "http://127.0.0.1:8050/"
COLORS = {"ink": "#2A251F", "green": "#3D7A3A", "rust": "#C4581C", "white": "#FFFFFF", "dim": "#8B8273"}


def icons(p):
    d = OUT / "icons"
    d.mkdir(parents=True, exist_ok=True)
    js = (POSTER / "icons.js").read_text()
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(device_scale_factor=1)
    pg.set_content("<html><body style='margin:0;background:transparent'><div id='box'></div></body></html>")
    pg.add_script_tag(content=js)
    names = pg.evaluate("Object.keys(POSTER_ICONS).concat(Object.keys(POSTER_SIL))")
    for name in names:
        svg = pg.evaluate("(n) => POSTER_ICONS[n] || POSTER_SIL[n]", name)
        vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
        h = 256
        w = max(1, round(h * vb[2] / vb[3]))
        for key, col in COLORS.items():
            pg.evaluate("""([svg, w, h, col]) => { const b = document.getElementById('box');
                b.style.cssText = `width:${w}px;height:${h}px;color:${col};line-height:0`;
                b.innerHTML = svg; const s = b.querySelector('svg'); s.setAttribute('width', w); s.setAttribute('height', h); }""",
                        [svg, w, h, col])
            pg.locator("#box").screenshot(path=str(d / f"{name}-{key}.png"), omit_background=True)
    b.close()
    print("icons:", len(names), "×", len(COLORS))


def web(p):
    b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    init = "localStorage.setItem('pj-lang','th'); localStorage.setItem('pj-theme','light'); localStorage.setItem('pj-intro-hidden','1');"
    # หน้าแรก: ลดภาพเคลื่อนไหวเพื่อให้ได้ภาพนิ่งที่อ่านง่าย (แบบเดียวกับรูปในรายงาน)
    c = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, reduced_motion="reduce")
    c.add_init_script(init)
    pg = c.new_page()
    pg.goto(WEB + "#index")
    pg.wait_for_function("document.querySelector('.sh-genus') && document.querySelector('.sh-genus').textContent.length > 3", timeout=120000)
    pg.wait_for_timeout(5000)
    pg.screenshot(path=str(OUT / "web_index.png"))
    print("saved web_index")
    c.close()
    c = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    c.add_init_script(init)
    pg = c.new_page()
    pg.goto(WEB + "#map")
    pg.wait_for_selector("#loading.hide", state="attached", timeout=180000)
    pg.wait_for_timeout(4000)
    pg.screenshot(path=str(OUT / "web_map.png"))
    print("saved web_map")
    # วิเคราะห์จุดที่แหล่งภูเวียง พร้อมดึงภาพ Sentinel-2 จริง (ไม่มีคำเตือนว่าไม่ได้ใช้ภาพดาวเทียม)
    pg.evaluate("() => { document.querySelector('#with-s2').checked = true; analysePoint(16.6803, 102.2357); }")
    pg.wait_for_selector("#use-point", timeout=300000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(OUT / "web_point.png"))
    print("saved web_point")
    pg.check("#show-geo")
    pg.wait_for_timeout(4000)
    pg.evaluate("() => map.jumpTo({center: [102.2, 16.6], zoom: 8.6})")
    pg.wait_for_timeout(3000)
    box = pg.locator("#mapcanvas").bounding_box()
    x, y = pg.evaluate("() => { const p = map.project([102.25, 16.68]); return [p.x, p.y]; }")
    pg.mouse.move(box["x"] + x, box["y"] + y)
    pg.wait_for_timeout(900)
    pg.screenshot(path=str(OUT / "web_geology.png"))
    print("saved web_geology")
    pg.uncheck("#show-geo")
    pg.click('.nav a[data-go="identify"]')
    pg.wait_for_timeout(1500)
    pg.click('.look-card[data-card="tooth"]')
    pg.wait_for_timeout(300)
    pg.click('.look-opt[data-opt="cone"]')
    pg.click('input[name="loc"][value="site"]')
    pg.fill("#site-input", pg.evaluate("() => [...state.siteByLabel.keys()].find(k => k.includes('ภูเวียง'))"))
    pg.dispatch_event("#site-input", "change")
    pg.click("#spec-form button[type=submit]")
    pg.wait_for_selector("#holo-big canvas, #holo-big .holo-svg", timeout=180000)
    pg.wait_for_timeout(3500)
    pg.evaluate("() => document.querySelector('#spec-result').scrollIntoView({block: 'start'})")
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(OUT / "web_identify.png"))
    print("saved web_identify")
    c.close()
    b.close()


def maps():
    d = OUT / "maps"
    d.mkdir(parents=True, exist_ok=True)
    for f in (POSTER / "_build" / "maps_light").glob("*.png"):
        shutil.copy2(f, d / f.name)
    for name in ("map_cv", "map_points"):
        shutil.copy2(ROOT / "docs" / "report_src" / "figures" / f"{name}.png", d / f"{name}.png")
    print("maps:", sorted(x.name for x in d.glob("*.png")))


# รูปกราฟใน notebook: (ไฟล์ปลายทาง, notebook, ลำดับรูปใน notebook นับจาก 1, กรอบตัด (ซ้าย, บน, ขวา, ล่าง) เป็นพิกเซล หรือ None = ทั้งรูป)
NB_PLOTS = [("nb_period_final.png", "05_final_period", 2, (0, 0, 1160, 539)),      # ยุคสุดท้ายของแหล่ง แยกตามที่มา (กราฟซ้าย)
            ("nb_ftype.png", "01_eda_fossils", 1, (508, 0, 1415, 594)),             # 20 ประเภทที่พบมากสุด (f_type) สีตามกลุ่ม (กราฟขวา)
            ("nb_period_recorded.png", "01_eda_fossils", 10, (0, 0, 557, 594)),     # ยุคที่ระบุ (เก่าสุด) n=166 จาก 518 (กราฟซ้าย)
            ("nb_stageB.png", "10_task2_models", 2, None)]                         # ขั้น B: macro AUC + top-3 ของแต่ละแบบจำลอง × วิธีแบ่ง
REPORT_PLOTS = ["chart_stageA.png", "chart_genus.png", "cm_period_spatial_block.png"]   # docs/report_src/figures


def charts():
    d = OUT / "charts"
    d.mkdir(parents=True, exist_ok=True)
    for name, nb, idx, box in NB_PLOTS:
        cells = json.loads((ROOT / "notebooks" / f"{nb}.ipynb").read_text())["cells"]
        pngs = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code"
                for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        data = pngs[idx - 1]
        im = Image.open(io.BytesIO(base64.b64decode(data if isinstance(data, str) else "".join(data))))
        (im.crop(box) if box else im).save(d / name)
    for name in REPORT_PLOTS:
        shutil.copy2(ROOT / "docs" / "report_src" / "figures" / name, d / name)
    print("charts:", sorted(x.name for x in d.glob("*.png")))


if __name__ == "__main__":
    want = sys.argv[1:] or ["icons", "web", "maps", "charts"]
    OUT.mkdir(parents=True, exist_ok=True)
    if "maps" in want:
        maps()
    if "charts" in want:
        charts()
    if "icons" in want or "web" in want:
        with sync_playwright() as p:
            if "icons" in want:
                icons(p)
            if "web" in want:
                web(p)
