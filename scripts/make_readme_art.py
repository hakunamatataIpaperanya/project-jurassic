# ภาพ pixel art ของหน้า README บน GitHub -> assets/readme/
#   banner.svg  ป้ายหัวเรื่อง (เคลื่อนไหว: ดาวกะพริบ, ดาวตก, ไดโนเสาร์ hologram เดินอยู่กับที่บนชั้นหิน, แถบสแกน)
#   map.png     แผนที่โอกาสพบซากทั้งประเทศ: 1 pixel = 10 กม. จากผลโมเดลจริง (webapp/data/grid.json, เปอร์เซ็นไทล์เทียบทั้งประเทศ)
#   typing.svg  แถบพิมพ์คำถามแบบเทอร์มินัล (สลับ 3 คำถามวนไป)
#   stats.svg   การ์ดตัวเลขสำคัญ (ขอบเรืองแสงเป็นจังหวะ)
#   divider_night.svg / divider_sea.svg / divider_dusk.svg  แถบคั่นหัวข้อ 3 ฉาก สัตว์ต่างกัน:
#               กลางคืน: เทอโรพอดเดินใต้ดาว · ทะเลโบราณ: แอมโมไนต์ ปลา ไทรโลไบต์ · ยามเย็น: เทอโรซอร์บินผ่านภูเขาไฟ
#   results.svg ผลของโมเดลเป็นแท่ง pixel (ตัวเลขชุดเดียวกับตาราง "See every number" ใน README.md)
#   contributor.svg  แถบสั้นท้าย README: ผู้ร่วมพัฒนาพิเศษ @E27-25
# รัน:  .venv/bin/python scripts/make_readme_art.py
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "readme"

# ---------------------------------------------------------------- ฟอนต์ pixel 5 × 7
FONT = {
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"], "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "], "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"], "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"], "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "], "J": ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"], "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"], "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "], "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"], "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "], "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "], "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "# # #", " # # "], "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "], "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "], "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"], "3": ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "], "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "], "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "], "9": [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
    ".": ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "], ",": ["     ", "     ", "     ", "     ", " ##  ", "  #  ", " #   "],
    ":": ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "], "/": ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "],
    "-": ["     ", "     ", "     ", "#####", "     ", "     ", "     "], "+": ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "],
    "%": ["##   ", "##  #", "   # ", "  #  ", " #   ", "#  ##", "   ##"], "*": ["     ", "  #  ", "# # #", " ### ", "# # #", "  #  ", "     "],
    "(": ["   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # "], ")": [" #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   "],
    "=": ["     ", "     ", "#####", "     ", "#####", "     ", "     "], "?": [" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "],
    ">": ["#    ", " #   ", "  #  ", "   # ", "  #  ", " #   ", "#    "], "@": [" ### ", "#   #", "# ###", "# # #", "# ###", "#    ", " ####"],
    " ": ["     "] * 7,
}


def text_pixels(s, scale=1):
    # ข้อความ -> รายการ (x, y, แถวของตัวอักษร) ขนาดตัวละ 6 × 7 pixel (คูณ scale)
    out = []
    for i, ch in enumerate(s.upper()):
        g = FONT.get(ch, FONT[" "])
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v == "#":
                    for dy in range(scale):
                        for dx in range(scale):
                            out.append((i * 6 * scale + c * scale + dx, r * scale + dy, r))
    return out


def text_width(s, scale=1):
    return len(s) * 6 * scale - scale


# ---------------------------------------------------------------- SVG จากตาราง pixel
def runs(grid):
    # ตาราง [y][x] ของสี (None = โปร่ง) -> สี่เหลี่ยมต่อเนื่องแนวนอนต่อสี (ไฟล์เล็กกว่าสี่เหลี่ยมทีละ pixel มาก)
    by = {}
    for y, row in enumerate(grid):
        x = 0
        while x < len(row):
            c = row[x]
            if c is None:
                x += 1
                continue
            x0 = x
            while x < len(row) and row[x] == c:
                x += 1
            by.setdefault(c, []).append(f"M{x0} {y}h{x - x0}v1h{-(x - x0)}z")
    return "".join(f'<path fill="{c}" d="{"".join(d)}"/>' for c, d in by.items())


def blank(w, h):
    return [[None] * w for _ in range(h)]


def put(grid, x, y, c):
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        grid[y][x] = c


def mix(a, b, t):
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]


# ---------------------------------------------------------------- ป้ายหัวเรื่อง
def sauropod(frame):
    # ไดโนเสาร์ซอโรพอด (ภูเวียงโกซอรัส) วาดด้วยรูปทรงง่าย ๆ บนตาราง 58 × 36: หน้ากาก + ขาสองท่า (เดิน)
    W, H = 58, 36
    m = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]

    def disc(cx, cy, r):
        m[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True

    def bez(p0, p1, p2, r0, r1):
        for t in np.linspace(0, 1, 60):
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
            disc(x, y, r0 + (r1 - r0) * t)

    m[((xx - 26) / 13.5) ** 2 + ((yy - 18.5) / 7.2) ** 2 <= 1] = True          # ลำตัว
    bez((35, 16), (45, 13), (48, 4), 4.2, 2.1)                                   # คอ
    m[((xx - 50.5) / 3.6) ** 2 + ((yy - 3.6) / 2.3) ** 2 <= 1] = True          # หัว
    bez((14, 18), (5, 21), (0, 29), 4.0, 0.6)                                    # หาง
    step = [(-1, 0, 1, 0), (1, -1, -1, 1)][frame]                               # ขาหน้าใกล้ / ไกล / หลังใกล้ / ไกล: เลื่อนไปมา
    for (x, d), lift in zip([(33, step[0]), (37, step[1]), (16, step[2]), (20, step[3])], (frame, 1 - frame, 1 - frame, frame)):
        m[22:H - lift, x + d:x + d + 3] = True                                   # ขาเป็นเสา (ยกขึ้น 1 pixel ตอนก้าว)
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    g = blank(W, H)
    for y in range(H):
        for x in range(W):
            if not m[y, x]:
                continue
            if edge[y, x]:
                c = "#c8f6ff"                                                     # ขอบสว่าง (แบบ hologram)
            elif y < 15:
                c = "#5ee0ff"
            elif y < 21:
                c = "#18c6f0"
            else:
                c = "#0f8fbf"
            if y % 3 == 0 and not edge[y, x]:
                c = mix(c, "#ffffff", 0.25)                                       # เส้นสแกน
            g[y][x] = c
    g[3][52] = "#0b0c29"                                                          # ตา
    return g


FOSSILS = {   # ซากเล็ก ๆ ในชั้นหิน (5 × 5)
    "ammonite": [" ### ", "#   #", "# # #", "#  ##", " ### "],
    "bone": ["#   #", " ### ", "  #  ", " ### ", "#   #"],
    "trilobite": [" ### ", "#####", " # # ", "#####", " ### "],
    "shell": ["  #  ", " ### ", "## ##", "# # #", " ### "],
    "fish": ["#  # ", " ####", "#### ", " ####", "#  # "],
}
PERIODS = [("#f9f97f", 3), ("#fd9a52", 4), ("#7fc64e", 4), ("#34b2c9", 4), ("#812b92", 4), ("#f04028", 4), ("#67a599", 5)]   # ใหม่ -> เก่า (สี ICS)


def banner():
    W, H, GROUND = 256, 96, 70
    rnd = random.Random(7)
    sky = blank(W, H)
    top, low = "#070820", "#1d2156"
    for y in range(GROUND):                                   # ฟ้าไล่สีแบบขั้น + dither
        t = y / GROUND
        for x in range(W):
            k = (t * 6) % 1
            lvl = int(t * 6) + (1 if k * 16 > BAYER[y % 4][x % 4] else 0)
            sky[y][x] = mix(top, low, min(lvl, 6) / 6)
    for cx, cy, r, c in [(229, 15, 8, "#ffe7a3")]:            # ดวงจันทร์เสี้ยว
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r and (x - cx + 4) ** 2 + (y - cy + 2) ** 2 > (r - 1) ** 2:
                    put(sky, x, y, c)
        put(sky, cx + 4, cy + 3, "#e8c97a")
        put(sky, cx + 2, cy + 5, "#e8c97a")
    for layer, (base, amp, c) in enumerate([(50, 9, "#1a1c4a"), (58, 7, "#13143a")]):    # ภูเขาไกล / ใกล้
        h = base
        for x in range(W):
            h += rnd.choice([-1, 0, 0, 1]) if x % 2 else 0
            h = max(base - amp, min(base + 2, h))
            peak = base - amp + abs(((x + layer * 37) % 64) - 32) * amp / 32
            top_y = int(max(h, peak) if layer else min(h, peak + 3))
            for y in range(top_y, GROUND):
                sky[y][x] = c
    ground = blank(W, H)
    y0 = GROUND
    for x in range(W):                                         # ผิวดิน (หญ้ากลางคืน)
        ground[y0][x] = "#2e6b4f" if x % 3 else "#3f8a62"
        ground[y0 + 1][x] = "#24533e"
    # ชั้นหิน = ยุค (บนใหม่ ล่างเก่า, สี ICS หรี่ลงให้เข้ากับกลางคืน): เส้นแบ่งชั้นเป็นคลื่นต่อเนื่อง ไม่มีช่องว่างระหว่างชั้น
    bounds, yb = [], y0 + 2
    for _, th in PERIODS:
        bounds.append(yb)
        yb += th
    for x in range(W):
        cuts = [b + (round(math.sin(x / 17 + i) * 0.8) if i else 0) for i, b in enumerate(bounds)] + [H]
        for i, (c, _) in enumerate(PERIODS):
            cc = mix(c, "#0b0c29", 0.45)
            for yy in range(cuts[i], cuts[i + 1]):
                speck = rnd.random() < 0.06
                ground[yy][x] = mix(cc, "#ffffff", 0.18) if speck else cc
    fossils = []
    for k, (name, fx, fy) in enumerate([("ammonite", 30, 82), ("bone", 88, 77), ("trilobite", 132, 90), ("shell", 196, 81), ("fish", 60, 92), ("ammonite", 236, 89), ("bone", 160, 86)]):
        for r, row in enumerate(FOSSILS[name]):
            for c, v in enumerate(row):
                if v == "#":
                    put(ground, fx + c, fy + r, "#efe4c6")
        fossils.append((fx + 2, fy + 2, k))
    # ชื่อเรื่อง: เงาสามมิติ + ไล่สีจากเหลืองถึงส้ม
    title = blank(W, H)
    tx, ty = 12, 12
    for x, yv, r in text_pixels("PROJECT JURASSIC", 2):
        put(title, tx + x, ty + yv + 2, "#5a2410")
    for x, yv, r in text_pixels("PROJECT JURASSIC", 2):
        put(title, tx + x, ty + yv, mix("#fff1a3", "#f08a2c", r / 6))
    for x, yv, r in text_pixels("THAILAND FOSSIL ATLAS", 1):
        put(title, tx + x, 36 + yv, "#9fe9ff")
    for x, yv, r in text_pixels("MACHINE LEARNING * GEOSCIENCE", 1):
        put(title, tx + x, 46 + yv, "#7b86c9")
    stars = [(rnd.randrange(W), rnd.randrange(54)) for _ in range(70)]
    stars = [(x, y) for x, y in stars if not (8 <= x <= 210 and 8 <= y <= 56) or rnd.random() < 0.25]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="1024" height="{H * 4}" shape-rendering="crispEdges">',
           "<title>Project Jurassic: Thailand Fossil Atlas (pixel art)</title>", runs(sky)]
    for i, (x, y) in enumerate(stars):                          # ดาวกะพริบ
        big = i % 9 == 0
        col = ["#ffffff", "#ffe7a3", "#9fe9ff"][i % 3]
        dur, begin = 1.6 + (i % 7) * 0.45, (i * 0.37) % 3
        if big:
            shape = f'<path fill="{col}" d="M{x} {y - 1}h1v3h-1zM{x - 1} {y}h3v1h-3z"/>'
        else:
            shape = f'<rect x="{x}" y="{y}" width="1" height="1" fill="{col}"/>'
        svg.append(f'<g opacity="1">{shape}<animate attributeName="opacity" values="1;0.15;1" dur="{dur:.2f}s" begin="{begin:.2f}s" repeatCount="indefinite"/></g>')
    # ดาวตก
    svg.append('<g opacity="0"><path fill="#ffffff" d="M0 0h2v1h-2z"/><path fill="#9fe9ff" d="M-3 -1h3v1h-3z"/><path fill="#5b6bd1" d="M-6 -2h3v1h-3z"/>'
               '<animateTransform attributeName="transform" type="translate" values="120 4;196 30" dur="7s" keyTimes="0;1" repeatCount="indefinite" calcMode="spline" keySplines="0.3 0 0.7 1"/>'
               '<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.55;0.6;0.72;0.78;1" dur="7s" repeatCount="indefinite"/></g>')
    svg.append(runs(ground))
    for fx, fy, k in fossils:                                   # ซากในชั้นหินเรืองแสงเป็นจังหวะ
        svg.append(f'<rect x="{fx}" y="{fy}" width="1" height="1" fill="#ffffff" opacity="0"><animate attributeName="opacity" values="0;1;0" dur="3s" begin="{k * 0.6:.1f}s" repeatCount="indefinite"/></rect>')
    # แท่นฉาย + ไดโนเสาร์ hologram (เดินอยู่กับที่ 2 ท่า, ลอยขึ้นลง, กะพริบ) + แถบสแกน
    dx, dy = 192, GROUND - 35
    ring = blank(W, H)
    for a in range(0, 360, 6):
        for rx, ry, c in [(30, 4, "#18d1ff"), (20, 2.6, "#5ee0ff")]:
            put(ring, round(dx + 28 + rx * math.cos(math.radians(a))), round(GROUND + 1 + ry * math.sin(math.radians(a))), c)
    svg.append(f'<g opacity="0.9">{runs(ring)}<animate attributeName="opacity" values="0.5;0.95;0.5" dur="2.4s" repeatCount="indefinite"/></g>')
    glow = blank(W, H)                                           # แสงเรืองรอบภาพฉาย: จุด dither หนาแน่นตรงกลาง บางลงที่ขอบ
    for y in range(GROUND - 40, GROUND):
        for x in range(dx - 8, dx + 64):
            d = ((x - dx - 28) / 36) ** 2 + ((y - GROUND + 16) / 26) ** 2
            if d < 1 and (1 - d) * 16 > BAYER[y % 4][x % 4] + 6:
                put(glow, x, y, "#1b5f9a")
    svg.append(f'<g opacity="0.55">{runs(glow)}</g>')
    frames = [runs(sauropod(f)) for f in (0, 1)]
    svg.append(f'<g transform="translate({dx} {dy})"><g>'
               f'<g>{frames[0]}<animate attributeName="opacity" values="1;0" dur="0.9s" calcMode="discrete" repeatCount="indefinite"/></g>'
               f'<g opacity="0">{frames[1]}<animate attributeName="opacity" values="0;1" dur="0.9s" calcMode="discrete" repeatCount="indefinite"/></g>'
               '<animateTransform attributeName="transform" type="translate" values="0 0;0 -1;0 0" dur="1.8s" calcMode="discrete" repeatCount="indefinite"/>'
               '<animate attributeName="opacity" values="1;1;0.55;1;1;0.8;1" keyTimes="0;0.4;0.42;0.44;0.8;0.81;1" dur="4.3s" repeatCount="indefinite"/></g>'
               '<rect x="-2" width="62" height="1" fill="#e9fdff" opacity="0.7"><animate attributeName="y" values="0;36" dur="2.6s" repeatCount="indefinite"/></rect></g>')
    svg.append(runs(title))
    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------- แผนที่ pixel จากผลโมเดลจริง
LEVEL_COL = {"vhigh": (255, 216, 107), "high": (242, 154, 63), "mid": (123, 90, 166), "low": (35, 37, 90), "none": (110, 107, 146)}
BG = (11, 12, 41)


def draw_text(img, s, x, y, col, scale=1):
    for px, py, _ in text_pixels(s, scale):
        if 0 <= x + px < img.shape[1] and 0 <= y + py < img.shape[0]:
            img[y + py, x + px] = col


def potential_map(step=2, cell=4):
    g = json.loads((ROOT / "webapp" / "data" / "grid.json").read_text(encoding="utf-8"))
    ix, iy = np.array(g["ix"]) // step, np.array(g["iy"]) // step
    rank = np.array(g["rank"]["bg_bias"]) / 1000
    out = np.array(g["inside"]["bg_bias"]) == 0
    x0, y0, x1, y1 = ix.min(), iy.min(), ix.max(), iy.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    s, n, o = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
    np.add.at(s, (y1 - iy, ix - x0), rank)
    np.add.at(n, (y1 - iy, ix - x0), 1)
    np.add.at(o, (y1 - iy, ix - x0), out)
    pad, legend_w = 6, 150
    W, H = w * cell + 2 * pad * cell + legend_w, h * cell + 2 * pad * cell
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BG
    rng = np.random.default_rng(3)
    for _ in range(140):                                          # ดาวเล็ก ๆ บนพื้นหลัง
        img[rng.integers(0, H), rng.integers(0, W)] = (60, 66, 130)
    for r in range(h):
        for c in range(w):
            if not n[r, c]:
                continue
            v = s[r, c] / n[r, c]
            lv = "none" if o[r, c] / n[r, c] > 0.5 else "vhigh" if v >= 0.9 else "high" if v >= 0.75 else "mid" if v >= 0.5 else "low"
            y, x = (r + pad) * cell, (c + pad) * cell
            img[y:y + cell, x:x + cell] = LEVEL_COL[lv]
            img[y + cell - 1, x:x + cell] = (np.array(LEVEL_COL[lv]) * 0.8).astype(np.uint8)    # ร่องเล็ก ๆ แบบ tile
    lx = w * cell + 2 * pad * cell - 10
    draw_text(img, "FOSSIL", lx, 30, (255, 216, 107), 2)
    draw_text(img, "POTENTIAL", lx, 50, (255, 216, 107), 2)
    draw_text(img, "1 PIXEL = 10 KM", lx, 74, (159, 233, 255))
    draw_text(img, "VS WHOLE COUNTRY", lx, 84, (159, 233, 255))
    for k, (lv, label) in enumerate([("vhigh", "TOP 10%"), ("high", "TOP 10-25%"), ("mid", "TOP 25-50%"), ("low", "LOWER HALF"), ("none", "NOT ENOUGH DATA")]):
        y = 108 + k * 18
        img[y:y + 10, lx:lx + 10] = LEVEL_COL[lv]
        draw_text(img, label, lx + 16, y + 2, (220, 214, 234))
    draw_text(img, "MODEL: XGBOOST", lx, 210, (123, 134, 201))
    draw_text(img, "5 KM GRID, SURVEY BG", lx, 220, (123, 134, 201))
    Image.fromarray(img).resize((W * 2, H * 2), Image.NEAREST).save(OUT / "map.png", optimize=True)


# ---------------------------------------------------------------- ผลของโมเดลเป็นแท่ง pixel
RESULTS = [   # (ชื่อ, ค่า 0–1, งาน) — ตรงกับตาราง "See every number" ใน README.md และหน้าเว็บ
    ("WHERE: POTENTIAL AUC, UNSEEN AREAS", 0.887, 2),
    ("WHERE: INDEPENDENT PBDB TEST AUC", 0.797, 2),
    ("WHERE: NEW-AREA HIT RATE (RANDOM .25)", 0.57, 2),
    ("WHAT TYPE: REAL TYPE IN TOP 3", 0.776, 2),
    ("GENUS IN TOP 3: KNOWN SITE", 0.863, 1),
    ("GENUS IN TOP 3: NEW SITE", 0.587, 1),
    ("SITE PERIOD: WITHIN 1 PERIOD", 0.900, 1),
]


def results():
    W, row = 256, 19
    H = 26 + row * len(RESULTS) + 26
    g = blank(W, H)
    for y in range(H):
        for x in range(W):
            g[y][x] = "#0b0c29"
    for x in range(W):
        g[0][x] = g[H - 1][x] = "#3a3f8f"
    for y in range(H):
        g[y][0] = g[y][W - 1] = "#3a3f8f"
    for x, y, r in text_pixels("MODEL RESULTS", 2):
        put(g, 8 + x, 6 + y, mix("#fff1a3", "#f08a2c", r / 6))
    legend = [(H - 22, "#18d1ff", "TASK 1: FOSSIL NAME + SITE PERIOD"), (H - 12, "#f29a3f", "TASK 2: WHERE + WHAT TYPE")]
    for ly, col, lab in legend:
        for yy in range(ly + 1, ly + 6):
            for xx in range(8, 13):
                put(g, xx, yy, col)
        for x, y, r in text_pixels(lab, 1):
            put(g, 17 + x, ly + y, "#9ea3d6")
    for i, (label, v, task) in enumerate(RESULTS):
        y0 = 26 + i * row
        col = "#18d1ff" if task == 1 else "#f29a3f"
        for x, y, r in text_pixels(label, 1):
            put(g, 8 + x, y0 + y, "#dcd6ea")
        bar_y, n_blocks = y0 + 9, 40
        for b in range(n_blocks):                                 # แท่งแบบช่อง ๆ (40 ช่อง = 1.0)
            filled = b < round(v * n_blocks)
            c = (mix(col, "#ffffff", 0.35) if b % 2 == 0 else col) if filled else "#1d2156"
            for yy in range(bar_y, bar_y + 6):
                for xx in range(8 + b * 5, 8 + b * 5 + 4):
                    put(g, xx, yy, c if yy > bar_y else mix(c, "#ffffff", 0.3) if filled else c)
        if "RANDOM" in label:
            for yy in range(bar_y - 1, bar_y + 7):
                put(g, 8 + round(0.25 * n_blocks) * 5 - 1, yy, "#ff6b8a")
        txt = f"{v:.3f}".lstrip("0") if v < 1 else "1.000"
        for x, y, r in text_pixels(txt, 1):
            put(g, 214 + x, bar_y + y - 1, "#fff1a3")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="1024" height="{H * 4}" shape-rendering="crispEdges">'
            f"<title>Model results (pixel art)</title>{runs(g)}</svg>")


def svg_open(W, H, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * 4}" height="{H * 4}" shape-rendering="crispEdges">'
            f"<title>{title}</title>")


def frame_box(g, W, H, bg="#0b0c29", line="#3a3f8f"):
    for y in range(H):
        for x in range(W):
            g[y][x] = bg
    for x in range(W):
        g[0][x] = g[H - 1][x] = line
    for y in range(H):
        g[y][0] = g[y][W - 1] = line


# ---------------------------------------------------------------- แถบพิมพ์คำถาม (เทอร์มินัล)
PHRASES = ["WHERE SHOULD I DIG?", "WHAT FOSSIL IS THIS?", "HOW OLD IS THIS ROCK?"]


def typing():
    S, P, STEP = 2, 4.0, 0.085                                     # ขนาดตัวอักษร, เวลาต่อคำถาม (วินาที), เวลาต่อตัวอักษร
    W, H = 18 + max(map(len, PHRASES)) * 6 * S + 10, 24
    T = P * len(PHRASES)
    g = blank(W, H)
    frame_box(g, W, H)
    for x, y, r in text_pixels(">", S):
        put(g, 5 + x, 5 + y, "#18d1ff")
    svg = [svg_open(W, H, "Where should I dig? What fossil is this? How old is this rock?"), runs(g)]
    x0 = 18
    cursor = []                                                     # (เวลา, x) ของเคอร์เซอร์
    for k, phrase in enumerate(PHRASES):
        start, end = k * P, k * P + P - 0.45
        cursor.append((start, x0))
        for i, ch in enumerate(phrase):
            on = start + 0.2 + i * STEP
            cursor.append((on, x0 + (i + 1) * 6 * S))
            if ch == " ":
                continue
            glyph = blank(W, H)
            for x, y, r in text_pixels(ch, S):
                put(glyph, x0 + i * 6 * S + x, 5 + y, mix("#fff1a3", "#f08a2c", r / 6))
            svg.append(f'<g opacity="0">{runs(glyph)}<animate attributeName="opacity" values="0;1;0;0" '
                       f'keyTimes="0;{on / T:.4f};{end / T:.4f};1" dur="{T}s" calcMode="discrete" repeatCount="indefinite"/></g>')
        cursor.append((end, x0))
    times = ";".join(f"{t / T:.4f}" for t, _ in cursor) + ";1"
    xs = ";".join(str(x) for _, x in cursor) + f";{x0}"
    svg.append(f'<rect x="{x0}" y="5" width="{S * 5}" height="{7 * S}" fill="#18d1ff">'
               f'<animate attributeName="x" values="{xs}" keyTimes="0;{times[times.index(";") + 1:]}" dur="{T}s" calcMode="discrete" repeatCount="indefinite"/>'
               '<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.5;0.5;1" dur="0.9s" repeatCount="indefinite"/></rect>')
    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------- การ์ดตัวเลข
STATS = [("518", "FOSSIL SITES", "REGISTERED", "#ffd86b"), ("8,521", "SPECIMENS", "IN REGISTER", "#f29a3f"),
         ("20.7K", "MAP CELLS", "5 KM EACH", "#18d1ff"), (".887", "AUC ON", "UNSEEN AREAS", "#7fc64e")]


def stats():
    TW, gap, H = 78, 4, 40
    W = len(STATS) * TW + (len(STATS) - 1) * gap
    svg = [svg_open(W, H, "518 fossil sites, 8,521 specimens, 20,706 map cells, AUC 0.887 on unseen areas")]
    for k, (num, l1, l2, col) in enumerate(STATS):
        x0 = k * (TW + gap)
        g = blank(W, H)
        for y in range(H):
            for x in range(x0, x0 + TW):
                g[y][x] = "#0b0c29"
        nw = text_width(num, 2)
        for x, y, r in text_pixels(num, 2):
            put(g, x0 + (TW - nw) // 2 + x, 6 + y, mix("#ffffff", col, r / 6))
        for row, lab, c in ((24, l1, "#dcd6ea"), (32, l2, "#7b86c9")):
            lw = text_width(lab)
            for x, y, r in text_pixels(lab):
                put(g, x0 + (TW - lw) // 2 + x, row + y - 1, c)
        svg.append(runs(g))
        edge = blank(W, H)
        for x in range(x0, x0 + TW):
            edge[0][x] = edge[H - 1][x] = col
        for y in range(H):
            edge[y][x0] = edge[y][x0 + TW - 1] = col
        for cx, cy in ((x0, 0), (x0 + TW - 1, 0), (x0, H - 1), (x0 + TW - 1, H - 1)):
            put(edge, cx, cy, None)                                   # มุมมน
        svg.append(f'<g opacity="0.5">{runs(edge)}<animate attributeName="opacity" values="0.35;1;0.35" dur="2.4s" begin="{k * 0.6:.1f}s" repeatCount="indefinite"/></g>')
    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------- แถบคั่นหัวข้อ 3 ฉาก (แต่ละฉากมีสัตว์และพื้นหลังต่างกัน)
class Sprite:
    # วาดรูปทรงที่ความละเอียด s เท่าแล้วย่อลง (pixel ที่ถูกปิดเกินครึ่ง = ทึบ) ขอบจึงเรียบกว่าวาดทีละ pixel
    def __init__(self, W, H, s=6):
        self.W, self.H, self.s, self.im = W, H, s, {}

    def _draw(self, layer):
        if layer not in self.im:
            self.im[layer] = Image.new("L", (self.W * self.s, self.H * self.s), 0)
        return ImageDraw.Draw(self.im[layer])

    def ellipse(self, cx, cy, rx, ry, layer="near"):
        s = self.s
        self._draw(layer).ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=255)

    def poly(self, pts, layer="near"):
        self._draw(layer).polygon([(x * self.s, y * self.s) for x, y in pts], fill=255)

    def curve(self, p0, p1, p2, r0, r1, layer="near"):
        for t in np.linspace(0, 1, 48):
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
            self.ellipse(x, y, r0 + (r1 - r0) * t, r0 + (r1 - r0) * t, layer)

    def mask(self, layer="near"):
        if layer not in self.im:
            return np.zeros((self.H, self.W), bool)
        return np.asarray(self.im[layer].resize((self.W, self.H), Image.BOX)) >= 128


def holo(sp, pal, dots=(), scan=True):
    # สีแบบ hologram เหมือนหน้าเว็บ: ขอบสว่าง ไล่สีบนลงล่าง เส้นสแกนทุก 3 แถว; ส่วนที่อยู่ไกล (ขา/ปีกอีกข้าง) สีเข้ม
    near = sp.mask("near")
    far = sp.mask("far") & ~near
    edge = near & ~(np.roll(near, 1, 0) & np.roll(near, -1, 0) & np.roll(near, 1, 1) & np.roll(near, -1, 1))
    rows = np.where(near.any(1))[0]
    top, bot = rows.min(), rows.max()
    g = blank(sp.W, sp.H)
    for y in range(sp.H):
        for x in range(sp.W):
            if near[y, x]:
                t = (y - top) / max(1, bot - top)
                c = pal["edge"] if edge[y, x] else pal["hi"] if t < 0.4 else pal["mid"] if t < 0.75 else pal["lo"]
                if scan and y % 3 == 0 and not edge[y, x]:
                    c = mix(c, "#ffffff", 0.22)
                g[y][x] = c
            elif far[y, x]:
                g[y][x] = pal["far"]
    for x, y, key in dots:
        if near[y, x]:
            g[y][x] = pal[key]
    return g


AMBER = {"edge": "#fff1c9", "hi": "#ffd86b", "mid": "#f2a43f", "lo": "#c76f22", "far": "#7a4416", "eye": "#2a1405"}
CORAL = {"edge": "#ffe6d2", "hi": "#ffb48f", "mid": "#f0855e", "lo": "#c2563d", "far": "#7d2f2a", "eye": "#2b0d0a", "line": "#8a3326"}
VIOLET = {"edge": "#f1e6ff", "hi": "#cba9ff", "mid": "#9d6ff0", "lo": "#6d47c4", "far": "#4a2f8f", "eye": "#1a0f33"}


def theropod(frame):
    # เทอโรพอด (ไดโนเสาร์กินเนื้อเดินสองขา) หันขวา 30 × 24: ขาสลับหน้า-หลังสองท่า
    sp = Sprite(30, 24)
    hip = (13.5, 13)
    fwd, back = ((17, 17.5), (18, 21.8)), ((11, 18), (8.5, 21.8))
    for (knee, ankle), layer in zip((back, fwd) if frame == 0 else (fwd, back), ("far", "near")):
        sp.curve(hip, knee, knee, 2.7, 1.5, layer)
        sp.curve(knee, ankle, ankle, 1.2, 0.9, layer)
        sp.poly([(ankle[0] - 1.2, 21.4), (ankle[0] + 2.8, 21.4), (ankle[0] + 2.8, 23), (ankle[0] - 1.2, 23)], layer)
    sp.ellipse(14, 11, 7.5, 4.3)                                   # ลำตัว
    sp.curve((8, 10), (3, 8.5), (0.6, 7), 3.0, 0.6)                # หาง
    sp.curve((19, 9.5), (21.5, 7.5), (22.5, 4.8), 2.8, 2.4)        # คอ
    sp.ellipse(24.6, 3.5, 4.2, 2.5)                                # หัว
    sp.ellipse(25.2, 5.6, 3.6, 1.1)                                # ขากรรไกร
    sp.curve((20, 11), (22, 12), (22.6, 13.6), 0.9, 0.6)           # แขนเล็ก
    return holo(sp, AMBER, [(26, 2, "eye")] + [(x, 5, "far") for x in range(25, 29)])


def ammonite(frame):
    # แอมโมไนต์ (หมึกโบราณมีเปลือกขด) ว่ายไปทางขวา 24 × 17: หนวดสะบัดสองท่า, ลายขดบนเปลือก
    sp = Sprite(24, 17)
    cx, cy, R = 15.5, 8.5, 7.6
    sp.ellipse(cx, cy, R, R)
    sp.ellipse(9.4, 9.8, 2.2, 2.3)                                 # หัวโผล่จากปากเปลือก
    ends = [[(1, 6.5), (0.4, 10), (1.6, 13.4)], [(1.8, 8), (0.3, 10.8), (1.2, 12.4)]][frame]
    for (ex, ey), sy in zip(ends, (8.8, 10, 11.2)):
        sp.curve((8.6, sy), ((8.6 + ex) / 2, sy + (1.3 if frame else -1.3)), (ex, ey), 0.8, 0.6)
    q, Rs = 0.42, R - 1.4
    dots = [(9, 9, "eye")]
    for y in range(17):
        for x in range(8, 24):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            r, th = math.hypot(dx, dy), math.atan2(dy, dx) % (2 * math.pi)
            if r < 1 or any(abs(r - Rs * q ** (th / (2 * math.pi) + k)) < 0.5 for k in range(4)):
                dots.append((x, y, "line"))
            elif r > Rs + 0.3 and r < R - 0.6 and (th * 12 / (2 * math.pi)) % 1 < 0.22:
                dots.append((x, y, "lo"))                           # ลายบั้งรอบนอก
    return holo(sp, CORAL, dots, scan=False)


def pterosaur(frame):
    # เทอโรซอร์ (สัตว์เลื้อยคลานบินได้ มีหงอนบนหัว) บินไปทางขวา 30 × 17: กระพือปีกขึ้น-ลง
    sp = Sprite(30, 17)
    if frame == 0:
        sp.poly([(16.4, 8), (15, 3), (9, 0.3), (13, 8.4)], "far")
        sp.poly([(15.8, 8.6), (12.4, 3.6), (2.6, 0.4), (6.6, 3.9), (10.6, 9.6)])
    else:
        sp.poly([(16.4, 9.6), (15, 13.4), (9, 16.6), (13, 9.8)], "far")
        sp.poly([(15.8, 9), (12.4, 12.6), (2.6, 16.4), (6.6, 13), (10.6, 9.8)])
    sp.ellipse(14, 9.2, 3.9, 1.9)                                  # ลำตัว
    sp.curve((17, 8.8), (19.5, 7.8), (21, 7.2), 1.5, 1.3)          # คอ
    sp.ellipse(22.2, 7, 2.2, 1.5)                                  # หัว
    sp.poly([(23.2, 6.2), (29.8, 7.9), (23.2, 8.2)])               # จะงอยปาก
    sp.poly([(21.2, 6.5), (15.6, 3.4), (16.4, 3.2), (22.6, 5.8)])  # หงอน
    sp.curve((10.4, 9.6), (9, 10), (7.4, 10.4), 0.8, 0.55)         # ขาหลัง/หาง
    return holo(sp, VIOLET, [(22, 6, "eye")])


def sky(g, y0, y1, stops):
    # ไล่สีท้องฟ้า/น้ำแบบ dither (ลาย Bayer 4 × 4) ให้ดูเป็น pixel art
    n = len(stops) - 1
    for y in range(y0, y1):
        t = (y - y0) / max(1, y1 - y0 - 1) * n
        i = min(int(t), n - 1)
        for x in range(len(g[0])):
            g[y][x] = stops[i + 1] if (t - i) * 16 > BAYER[y % 4][x % 4] + 0.5 else stops[i]


def cut_corners(g):
    H, W = len(g), len(g[0])
    for x, y in ((0, 0), (1, 0), (0, 1), (W - 1, 0), (W - 2, 0), (W - 1, 1), (0, H - 1), (1, H - 1), (0, H - 2), (W - 1, H - 1), (W - 2, H - 1), (W - 1, H - 2)):
        g[y][x] = None


def strata(g, y0, rnd, fossils, shade="#0b0c29", amount=0.35, grass=("#3f8a62", "#2e6b4f")):
    H, W = len(g), len(g[0])
    for x in range(W):
        if grass:
            g[y0][x] = grass[0] if x % 3 else grass[1]
        for y in range(y0 + (1 if grass else 0), H):
            c = PERIODS[((y - y0 - 1) // 2) % len(PERIODS)][0]
            g[y][x] = mix(c, shade, amount) if rnd.random() > 0.05 else "#efe4c6"
    for name, fx in fossils:
        for r, row in enumerate(FOSSILS[name]):
            for c, v in enumerate(row):
                if v == "#":
                    put(g, fx + c, y0 + 2 + r, "#efe4c6")


def stamp(g, art, x0, y0, col):
    for r, row in enumerate(art):
        for c, v in enumerate(row):
            if v != " ":
                put(g, x0 + c, y0 + r, col[v] if isinstance(col, dict) else col)


def flip(frames, period):
    # สลับภาพทีละท่า (เดิน / กระพือปีก / หนวดสะบัด)
    n = len(frames)
    return "".join(f'<g opacity="{1 if i == 0 else 0}">{runs(f)}<animate attributeName="opacity" values="{";".join("1" if j == i else "0" for j in range(n))}" '
                   f'dur="{period}s" calcMode="discrete" repeatCount="indefinite"/></g>' for i, f in enumerate(frames))


def move(inner, pts, dur, begin=0.0, extra=""):
    vals = ";".join(f"{x:g} {y:g}" for x, y in pts)
    return (f'<g>{inner}<animateTransform attributeName="transform" type="translate" values="{vals}" dur="{dur}s" begin="{begin:g}s" '
            f'repeatCount="indefinite"{extra}/></g>')


def twinkle(x, y, c, dur, begin, w=1, h=1, lo=0.15):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}"><animate attributeName="opacity" values="1;{lo};1" dur="{dur:.1f}s" '
            f'begin="{begin:.1f}s" repeatCount="indefinite"/></rect>')


ARAUCARIA = ["   #   ", "  ###  ", "   #   ", " ##### ", "   #   ", "#######", "   #   ", "   #   "]
CYCAD = ["#  #  #", " # # # ", "  ###  "]
FISH = [" ### #", "######", " ### #"]
TRILOBITE = [["  #####  ", " ####### ", "#########", " # # # # "], ["  #####  ", " ####### ", "#########", "# # # # #"]]
CLOUD = ["   ####     ", " ########## ", "############"]
FLOCK = [["## ##", "  #  "], ["  #  ", "## ##"]]


def divider_night():
    # ฉากกลางคืนยุคจูแรสซิก: ดาวกะพริบ ดาวตก จันทร์เสี้ยว หิ่งห้อย ต้นอะรอคาเรีย และเทอโรพอดเดินผ่าน
    W, H, G = 256, 36, 28
    rnd = random.Random(21)
    g = blank(W, H)
    sky(g, 0, G, ["#06071c", "#0b0c29", "#141845", "#1f2560"])
    for _ in range(70):
        put(g, rnd.randrange(W), rnd.randrange(G - 6), rnd.choice(["#3d4486", "#5b63a8", "#8088c8"]))
    for y in range(16):
        for x in range(214, 236):
            if (x - 224.5) ** 2 + (y - 8.5) ** 2 <= 30 and (x - 226.8) ** 2 + (y - 7) ** 2 > 22:
                g[y][x] = "#fff1a3"
    for x in range(W):
        h = int(4 + 2.5 * math.sin(x / 17) + 1.5 * math.sin(x / 7 + 1))
        for y in range(G - h, G):
            g[y][x] = "#161a4a"
    for tx in (18, 70, 131, 168, 244):
        stamp(g, ARAUCARIA, tx, G - 8 - rnd.randrange(0, 4), "#0e1238")
    strata(g, G, rnd, [("ammonite", 34), ("bone", 108), ("trilobite", 176), ("shell", 228)])
    for cx in (52, 96, 150, 205):
        stamp(g, CYCAD, cx, G - 3, {"#": "#2e6b4f"})
    cut_corners(g)
    out = [svg_open(W, H, "Night in the Jurassic: a theropod walks under twinkling stars"), runs(g)]
    for _ in range(16):
        out.append(twinkle(rnd.randrange(4, W - 4), rnd.randrange(1, G - 8), rnd.choice(["#ffffff", "#c8f6ff", "#fff1a3"]), rnd.uniform(1.6, 3.6), rnd.uniform(0, 3)))
    for sx, sy in ((44, 4), (120, 2), (188, 6)):                   # ดาวดวงใหญ่รูป +
        out.append(f'<g>{runs([[None, "#ffffff", None], ["#ffffff", "#ffffff", "#ffffff"], [None, "#ffffff", None]])}'
                   f'<animate attributeName="opacity" values="1;0.25;1" dur="2.8s" begin="{sx % 3}s" repeatCount="indefinite"/></g>'.replace("<path", f'<path transform="translate({sx} {sy})"'))
    trail = blank(12, 6)
    for i in range(6):
        put(trail, 2 * i, i, mix("#ffffff", "#1f2560", i / 6))
        put(trail, 2 * i + 1, i, mix("#ffffff", "#1f2560", i / 6 + 0.08))
    out.append(f'<g opacity="0">{move(runs(trail), [(150, -2), (100, 23), (100, 23)], 7, extra=' keyTimes="0;0.12;1"')}'
               '<animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;0.02;0.1;0.12;1" dur="7s" repeatCount="indefinite"/></g>')
    for fx, fy, d in ((60, G - 6, 3.1), (112, G - 9, 2.4), (160, G - 5, 3.6), (214, G - 8, 2.8)):
        out.append(move(twinkle(0, 0, "#ffe98a", d / 3, fx % 2, lo=0.0), [(fx, fy), (fx + 4, fy - 2), (fx + 1, fy - 4), (fx, fy)], d * 2))
    out.append(move(flip([theropod(0), theropod(1)], 0.7), [(-32, G - 23), (W + 2, G - 23)], 18))
    out.append("</svg>")
    return "".join(out)


def divider_sea():
    # ฉากทะเลโบราณ: แสงส่องลงน้ำ ฟองอากาศ สาหร่ายไหว แอมโมไนต์ ฝูงปลา และไทรโลไบต์คลานบนพื้นทะเล
    W, H, S = 256, 36, 30
    rnd = random.Random(5)
    g = blank(W, H)
    sky(g, 0, S, ["#1d6f9f", "#145a86", "#0e456d", "#0a3354", "#072540"])
    for x in range(W):
        top = S + round(0.8 * math.sin(x / 11) + 0.6 * math.sin(x / 5.3))
        for y in range(top, H):
            g[y][x] = ("#d8b97a" if y == top else "#a4844f" if y < top + 3 else "#7d6440") if rnd.random() > 0.06 else "#efe4c6"
    for name, fx in (("shell", 40), ("fish", 128), ("ammonite", 200)):
        stamp(g, FOSSILS[name], fx, S + 1, "#efe4c6")
    for rx in (22, 90, 172, 238):
        stamp(g, ["  ###  ", " ##### ", "#######"], rx, S - 2, {"#": "#2b3a52"})
    cut_corners(g)
    out = [svg_open(W, H, "An ancient sea: an ammonite, a school of fish and a trilobite"), runs(g)]
    for x0, d, b in ((30, 5, 0), (110, 6, 1.5), (190, 4.5, 3)):    # ลำแสงส่องลงน้ำ
        out.append(f'<polygon points="{x0},0 {x0 + 14},0 {x0 - 8},{S} {x0 - 20},{S}" fill="#d9f6ff" opacity="0.05">'
                   f'<animate attributeName="opacity" values="0.03;0.11;0.03" dur="{d}s" begin="{b}s" repeatCount="indefinite"/></polygon>')
    wave = blank(2 * W, 2)
    for x in range(2 * W):
        if x % 16 < 5:
            wave[0][x] = "#8fd6ff"
        elif 8 <= x % 16 < 10:
            wave[1][x] = "#5fb6e6"
    out.append(move(runs(wave), [(0, 0), (-W, 0)], 16))
    for sx in (12, 60, 104, 146, 222, 250):                        # สาหร่าย: สองท่าไหวไปมา
        frames = []
        for f in (0, 1):
            fr = blank(W, H)
            for k, h in enumerate((rnd.randrange(8, 14), rnd.randrange(6, 11))):
                for d in range(h):
                    x = sx + 2 * k + round((d / h) * 1.6 * math.sin(d * 0.55 + f * math.pi + k))
                    put(fr, x, S - 1 - d, "#46c08a" if d % 3 == 0 else "#2f9b6e")
            frames.append(fr)
        out.append(flip(frames, 1.8))
    for _ in range(12):                                            # ฟองอากาศลอยขึ้น
        x, sz, d = rnd.randrange(6, W - 6), rnd.choice((1, 1, 2)), rnd.uniform(4, 7)
        out.append(f'<rect x="{x}" y="{S - 2}" width="{sz}" height="{sz}" fill="#bfefff" opacity="0.7">'
                   f'<animate attributeName="y" values="{S - 2};1" dur="{d:.1f}s" begin="{rnd.uniform(0, 5):.1f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0.8;0.8;0" keyTimes="0;0.8;1" dur="{d:.1f}s" begin="{rnd.uniform(0, 5):.1f}s" repeatCount="indefinite"/></rect>')
    fish = blank(6, 3)
    stamp(fish, FISH, 0, 0, "#9fe7ff")
    for x in range(6):
        if fish[2][x]:
            fish[2][x] = "#5bbfe0"
    fish[1][1] = "#072540"
    school = "".join(f'<g transform="translate({dx} {dy})">{runs(fish)}</g>' for dx, dy in ((0, 0), (9, -4), (10, 4), (19, 1)))
    out.append(move(school, [(W + 4, 8), (-30, 11)], 13, begin=2))
    out.append(move(move(flip([ammonite(0), ammonite(1)], 0.8), [(0, 0), (0, -2), (0, 0)], 2.4), [(-26, 9), (W + 2, 6)], 22))
    tri = [blank(9, 4), blank(9, 4)]
    for f in (0, 1):
        stamp(tri[f], TRILOBITE[f], 0, 0, "#9aa8c7")
        for x in range(9):
            if tri[f][0][x]:
                tri[f][0][x] = "#d4def3"
            if tri[f][3][x]:
                tri[f][3][x] = "#5d6b8a"
    out.append(move(flip(tri, 0.5), [(W + 4, S - 4), (-12, S - 4)], 40, begin=0))
    out.append("</svg>")
    return "".join(out)


def divider_dusk():
    # ฉากยามเย็น: ดวงอาทิตย์ลายแถบ ภูเขาไฟมีควัน เมฆลอย และเทอโรซอร์บินผ่าน
    W, H, G = 256, 36, 29
    rnd = random.Random(9)
    g = blank(W, H)
    sky(g, 0, G, ["#1a1240", "#3b1f5c", "#6e2f6b", "#b34a5f", "#ee7f4a", "#ffb35c"])
    for y in range(G):
        for x in range(56, 86):
            if (x - 70.5) ** 2 + (y - (G - 10)) ** 2 <= 110 and (y % 3 != 1 or y < G - 11):     # ดวงอาทิตย์ลายแถบครึ่งล่าง
                g[y][x] = "#ffe08a" if y < G - 13 else "#ffc35c"
    for x in range(W):
        h = int(5 + 2 * math.sin(x / 13 + 2) + 1.5 * math.sin(x / 5.7))
        for y in range(G - h, G):
            g[y][x] = "#4a2357"
    vx = 190
    for x in range(vx - 26, vx + 27):
        top = G - 17 + max(0, abs(x - vx) - 3) * 17 // 24
        for y in range(top, G):
            g[y][x] = "#2d1640"
        if abs(x - vx) <= 3:
            g[G - 17][x] = "#ff7a3c"
            g[G - 16][x] = "#c94a3a"
    strata(g, G, rnd, [("bone", 26), ("fish", 98), ("shell", 150), ("ammonite", 226)], shade="#1a0f2e", amount=0.5, grass=("#1d0f2e", "#170c26"))
    cut_corners(g)
    out = [svg_open(W, H, "Dusk: a pterosaur flies past a smoking volcano"), runs(g)]
    for i in range(5):                                             # ควันภูเขาไฟ
        out.append(f'<rect x="{vx - 1}" y="{G - 19}" width="2" height="2" fill="#8a6a8f" opacity="0">'
                   f'<animate attributeName="y" values="{G - 19};0" dur="5s" begin="{i}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="x" values="{vx - 1};{vx + 9}" dur="5s" begin="{i}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0.8;0.4;0" dur="5s" begin="{i}s" repeatCount="indefinite"/></rect>')
    cloud = blank(12, 3)
    stamp(cloud, CLOUD, 0, 0, "#f6b38f")
    for x in range(12):
        cloud[2][x] = "#d27a8a"
    for (cx, cy), d in (((40, 4), 60), ((150, 8), 80), ((230, 2), 70)):
        out.append(move(runs(cloud), [(cx, cy), (cx - W - 14, cy)], d))           # สองก้อนต่อกัน: ก้อนหนึ่งออกซ้าย อีกก้อนเข้าทางขวา
        out.append(move(runs(cloud), [(cx + W + 14, cy), (cx, cy)], d))
    flock = [blank(5, 2), blank(5, 2)]
    for f in (0, 1):
        stamp(flock[f], FLOCK[f], 0, 0, "#2a1440")
    for (bx, by), d in (((0, 6), 30), ((14, 3), 30), ((26, 8), 30)):
        out.append(move(flip(flock, 0.6), [(bx - 40, by), (bx + W + 10, by - 2)], d, begin=3))
    out.append(move(flip([pterosaur(0), pterosaur(1)], 0.5), [(-32, 7), (60, 3), (140, 9), (220, 4), (W + 2, 7)], 14))
    out.append("</svg>")
    return "".join(out)


E27_TITLES = [   # ฉายายาวเหยียดของ @E27-25 (เขียนครบทุกชื่อในป้าย contributor.svg)
    "KING OF ALL DINOS", "EMPEROR OF THE JURASSIC", "LORD OF THE 8,521 FOSSILS", "GUARDIAN OF THE 518 SITES",
    "RULER OF ALL 20,706 MAP CELLS", "SUPREME OVERLORD OF THE KHORAT PLATEAU", "WHISPERER OF PHUWIANGOSAURUS",
    "BEST FRIEND OF SIAMOTYRANNUS", "CHIROPRACTOR OF SIAMOSAURUS", "TAMER OF XGBOOST", "SHEPHERD OF 10,000 DECISION TREES",
    "SLAYER OF OVERFITTING", "BREAKER OF NAN", "BANISHER OF DATA LEAKAGE", "KEEPER OF THE GOLDEN ROCK HAMMER",
    "HIGH PRIEST OF SPATIAL CROSS-VALIDATION", "MASTER OF THE ONNX HOLOGRAM", "SURVIVOR OF THE ASTEROID (IT MISSED HIM)",
    "DIGGER OF HOLES WITH 0.887 AUC", "THE ONE WHO MAKES TRILOBITES BOW", "SOLE REASON THE 227 TESTS PASS", "FATHER OF PANGEA",
    "FIRST OF HIS COMMIT", "THE UN-REVERTED", "PROTECTOR OF THE REALM OF SAO KHUA", "EATER OF BUGS (THE CODE KIND)",
    "LORD OF THE DINO NUGGETS", "AND SO ON AND SO ON AND SO ON...",
]


def wrap_titles(titles, max_chars):
    # ต่อฉายาด้วย " * " แล้วตัดบรรทัดโดยไม่หั่นกลางฉายา
    lines, cur = [], ""
    for t in titles:
        nxt = f"{cur} * {t}" if cur else t
        if len(nxt) > max_chars and cur:
            lines.append(cur)
            cur = t
        else:
            cur = nxt
    return lines + [cur]


def contributor():
    # ป้ายท้าย README: ผู้ร่วมพัฒนาพิเศษ @E27-25 พร้อมฉายาครบทุกชื่อ (กรอบทอง ดาวกะพริบ แถบสแกน เทอโรพอดเดินผ่าน)
    W, LH, T0 = 320, 10, 38
    lines = wrap_titles(E27_TITLES, (W - 16) // 6)
    G = T0 + len(lines) * LH + 28
    H = G + 8
    rnd = random.Random(27)
    g = blank(W, H)
    sky(g, 0, G, ["#06071c", "#0b0c29", "#141845", "#1f2560", "#2a2a6e"])
    for x in range(W):
        h = int(4 + 2 * math.sin(x / 15) + math.sin(x / 6 + 2))
        for y in range(G - h, G):
            g[y][x] = "#161a4a"
    for tx in (14, 92, 210, 296):
        stamp(g, ARAUCARIA, tx, G - 8 - rnd.randrange(0, 3), "#0e1238")
    strata(g, G, rnd, [("ammonite", 30), ("bone", 110), ("trilobite", 190), ("shell", 270)])
    cut_corners(g)
    label, name = "SPECIAL CONTRIBUTOR", "@E27-25"
    lw, nw = text_width(label), text_width(name, 2)
    txt = blank(W, H)
    for x, y, r in text_pixels(label):
        put(txt, (W - lw) // 2 + x, 5 + y, "#9fe9ff")
    nx = (W - nw) // 2
    for x, y, r in text_pixels(name, 2):
        put(txt, nx + x, 16 + y + 1, "#5a2410")
    for x, y, r in text_pixels(name, 2):
        put(txt, nx + x, 16 + y, mix("#fff1a3", "#f08a2c", r / 6))
    for sx in (nx - 12, nx + nw + 7):                               # ดาวทองสองข้างชื่อ
        stamp(txt, [" # ", "###", " # "], sx, 20, "#ffd86b")
    for x in range(W // 2 - 60, W // 2 + 61):                       # เส้นคั่นใต้ชื่อ
        if x % 4 != 3:
            put(txt, x, T0 - 5, "#3a3f8f")
    cols = ["#fff1a3", "#9fe9ff"]
    for i, line in enumerate(lines):                                # ฉายาทั้งหมด: สลับสีทอง / ฟ้า ทีละบรรทัด, ดาวคั่นสีส้ม
        x0, y0 = (W - text_width(line)) // 2, T0 + i * LH
        for k, ch in enumerate(line):
            c = "#f29a3f" if ch == "*" else cols[i % 2]
            for x, y, r in text_pixels(ch):
                put(txt, x0 + k * 6 + x, y0 + y, mix(c, "#c76f22", r / 10) if ch != "*" and i % 2 == 0 else c)
    edge = blank(W, H)
    for x in range(2, W - 2):
        edge[0][x] = edge[H - 1][x] = "#ffd86b"
    for y in range(2, H - 2):
        edge[y][0] = edge[y][W - 1] = "#ffd86b"
    out = [svg_open(W, H, "Special contributor: E27-25, " + ", ".join(t.title() for t in E27_TITLES)), runs(g)]
    for _ in range(18):
        x, y = rnd.randrange(4, W - 4), rnd.randrange(2, 30)
        if (W - lw) // 2 - 16 <= x <= (W + lw) // 2 + 16:
            continue
        out.append(twinkle(x, y, rnd.choice(["#ffffff", "#c8f6ff", "#fff1a3"]), rnd.uniform(1.6, 3.4), rnd.uniform(0, 3)))
    out.append(f'<g transform="translate(0 {G - 23})">{move(flip([theropod(0), theropod(1)], 0.7), [(-32, 0), (W + 2, 0)], 18)}</g>')
    out.append(runs(txt))
    out.append(f'<rect x="4" width="{W - 8}" height="1" fill="#e9fdff" opacity="0.18">'      # แถบสแกนไล่ลงบนรายชื่อฉายา
               f'<animate attributeName="y" values="{T0 - 2};{T0 + len(lines) * LH}" dur="4s" repeatCount="indefinite"/></rect>')
    out.append(f'<g>{runs(edge)}<animate attributeName="opacity" values="0.45;1;0.45" dur="2.4s" repeatCount="indefinite"/></g>')
    out.append("</svg>")
    return "".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contributor.svg").write_text(contributor(), encoding="utf-8")
    (OUT / "banner.svg").write_text(banner(), encoding="utf-8")
    potential_map()
    (OUT / "results.svg").write_text(results(), encoding="utf-8")
    (OUT / "typing.svg").write_text(typing(), encoding="utf-8")
    (OUT / "stats.svg").write_text(stats(), encoding="utf-8")
    (OUT / "divider_night.svg").write_text(divider_night(), encoding="utf-8")
    (OUT / "divider_sea.svg").write_text(divider_sea(), encoding="utf-8")
    (OUT / "divider_dusk.svg").write_text(divider_dusk(), encoding="utf-8")
    for f in sorted(OUT.iterdir()):
        print(f.relative_to(ROOT), f"{f.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
