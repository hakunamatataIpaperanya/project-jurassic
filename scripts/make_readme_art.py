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


def _edge(m):
    return m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))


def mirrored(inner, w):
    # กลับซ้าย-ขวา (ให้สัตว์ว่าย/คลานไปทางซ้าย)
    return f'<g transform="translate({w} 0) scale(-1 1)">{inner}</g>'


# ---------------------------------------------------------------- ทากทะเล (nudibranch / sea slug)
SLUGS = {   # mode: ติ่งบนหลัง (cer) / ใบไม้ (leaf) / พู่เหงือก (gill) / ขอบหยัก (ruffle) / ขนฟู (fuzz)
    "flab": dict(mode="cer", body=("#c39bff", "#9d6ff0"), edge="#f6eeff", base="#ff6fae", tip="#fff36b", rh="#ff9a3c", spots=None),
    "aeolid": dict(mode="cer", body=("#ffffff", "#e3ecff"), edge="#ffffff", base="#ff8a3d", tip="#ffffff", rh="#ff8a3d", spots=None),
    "leaf": dict(mode="leaf", body=("#a8f57f", "#62d24c"), edge="#efffe0", base="#3fcf5a", tip="#eaff6a", rh="#f4fff0", spots=None),
    "chromo": dict(mode="gill", body=("#3f7bff", "#2a55e0"), edge="#ffd23f", base="#ff8a3d", tip="#ffd0a0", rh="#ff8a3d", spots="#ffffff"),
    "pink": dict(mode="gill", body=("#ff8fd0", "#ff5fb4"), edge="#ffffff", base="#ffd23f", tip="#fff3b0", rh="#ffd23f", spots="#ffe1f2"),
    "tiger": dict(mode="gill", body=("#ffa24a", "#ff7a1f"), edge="#fff3d6", base="#ffffff", tip="#ffffff", rh="#3a1a08", spots="#3a1a08"),
    "nembrotha": dict(mode="gill", body=("#22223a", "#15152a"), edge="#3dff8a", base="#ff8a3d", tip="#ffd23f", rh="#ff5a2a", spots="#3dff8a"),
    "lettuce": dict(mode="ruffle", body=("#8ff07a", "#55c84a"), edge="#f2fff0", base="#6fe05f", tip="#ffffff", rh="#c9ffc0", spots=None),
    "bunny": dict(mode="fuzz", body=("#ffffff", "#f1f1f6"), edge="#ffffff", base="#ffffff", tip="#ffffff", rh="#202024", spots="#2a2a30"),
}


def nudibranch(frame, kind):
    # ทากทะเลคลานไปทางขวา 26 × 13 หลายชนิด (ดู SLUGS)
    st = SLUGS[kind]
    sp = Sprite(26, 13)
    fat = st["mode"] == "fuzz"
    sp.ellipse(12.5, 8.4 if fat else 9, 9.5 if fat else 10, 3.4 if fat else 2.4)    # ลำตัว (กระต่ายทะเลตัวอ้วนกลม)
    sp.ellipse(12.5, 10.6, 10.8, 1.2)                              # เท้า
    sp.curve((4, 9.5), (1.6, 10.6), (0.3, 11.2), 1.6, 0.5)         # หาง
    w = 0.5 if frame else -0.5
    if fat:                                                        # หูกระต่าย (rhinophore สีดำ)
        sp.curve((18.5, 5.6), (18.8, 3.4), (19 + w, 1.4), 1.0, 0.7, "rh")
        sp.curve((20.8, 6), (21.6, 3.8), (22.4 + w, 2), 1.0, 0.7, "rh")
        sp.curve((5, 6), (4, 4.6), (3.4 + w, 3.6), 0.7, 0.5, "cer")   # พู่เหงือกเล็ก ๆ ด้านหลัง
    else:
        sp.curve((20.5, 7.6), (21, 5), (21.4 + w, 2.6), 0.8, 0.5, "rh")    # หนวดรับกลิ่น (rhinophore)
        sp.curve((22.5, 7.8), (23.6, 5.2), (24.6 + w, 3.2), 0.8, 0.5, "rh")
        sp.curve((22, 9.6), (24, 10), (25.6, 11 + w), 0.6, 0.45, "rh")       # หนวดปาก
    if st["mode"] == "cer":
        for i, x in enumerate(np.arange(5, 19.5, 2.7)):
            sway = 0.6 if (i + frame) % 2 else -0.6
            sp.curve((x, 7.6), (x + sway * 0.5, 5.4), (x + sway, 2.4 + (i % 2) * 1.2), 0.62, 0.42, "cer")
    elif st["mode"] == "leaf":
        for i, x in enumerate(np.arange(4.5, 19, 2.5)):
            sp.ellipse(x + (0.3 if (i + frame) % 2 else -0.3), 5.6 + (i % 2) * 0.8, 1.05, 1.9, "cer")
    elif st["mode"] == "ruffle":
        for x in np.arange(3.5, 20, 0.5):
            sp.ellipse(x, 6.4 + 1.1 * math.sin(x * 1.3 + frame * math.pi), 0.6, 1.5, "cer")
    elif st["mode"] == "gill":
        for ex, ey in [(2.6, 4.2), (4.6, 2.6), (6.8, 2.6), (8.8, 4.2)]:
            sp.curve((5.7, 7.6), ((5.7 + ex) / 2 + w * 0.3, 5.6), (ex + w * 0.4, ey), 0.55, 0.4, "cer")
    near, cer, rhm = sp.mask("near"), sp.mask("cer"), sp.mask("rh")
    e = _edge(near)
    rows = np.where(cer.any(1))[0]
    g = blank(26, 13)
    for y in range(13):
        for x in range(26):
            if rhm[y, x]:
                g[y][x] = st["rh"] if (y > 3 or fat) else "#fff6dc"
            if near[y, x]:
                if e[y, x]:
                    g[y][x] = st["edge"] if not fat else "#d9d9e6"
                elif st["spots"] and (x * 7 + y * 3) % (6 if fat else 11) == 0:
                    g[y][x] = st["spots"]
                else:
                    g[y][x] = st["body"][0] if y < 9 else st["body"][1]
            if cer[y, x] and not (near[y, x] and y >= 8):
                g[y][x] = st["tip"] if y <= rows.min() + 1 else st["base"]
    if kind == "leaf":                                              # ตาดำ ๆ ของ "แกะใบไม้" (leaf sheep)
        g[8][21] = g[8][23] = "#101010"
    if fat:
        g[7][22] = "#202024"
    return g


def blue_dragon(frame):
    # มังกรทะเลสีน้ำเงิน (Glaucus) ลอยใกล้ผิวน้ำ 22 × 11: กิ่งนิ้วสีฟ้าแผ่ออกสามคู่
    sp = Sprite(22, 11)
    sp.curve((1, 5.5), (11, 4.6), (20.5, 5.4), 1.0, 1.4)
    w = 0.5 if frame else -0.5
    for x in (6.5, 11, 15):
        for k in (-1, 0, 1):
            sp.curve((x, 5), (x + k * 1.2, 3), (x + k * 2 + w, 0.8), 0.45, 0.35, "cer")
            sp.curve((x, 5.6), (x + k * 1.2, 7.6), (x + k * 2 - w, 9.8), 0.45, 0.35, "cer")
    near, cer = sp.mask("near"), sp.mask("cer")
    g = blank(22, 11)
    for y in range(11):
        for x in range(22):
            if cer[y, x]:
                g[y][x] = "#bfe0ff" if y in (0, 1, 9, 10) else "#3f7bff"
            if near[y, x]:
                g[y][x] = "#1f4fd8" if y == 5 else "#e6efff"
    g[5][20] = "#ffffff"
    return g


def spanish_dancer(frame):
    # ทากทะเล "นักเต้นสเปน" (Spanish dancer) ว่ายกลางน้ำ 28 × 15: ขอบแมนเทิลสีแดงขอบขาวโบกเป็นคลื่น
    sp = Sprite(28, 15)
    ph = frame * math.pi
    top = [(x, 7 - 2.6 - 1.8 * math.sin(x * 0.55 + ph)) for x in np.arange(3, 24.5, 0.5)]
    bot = [(x, 7.5 + 2.6 + 1.8 * math.sin(x * 0.55 + ph + 1.6)) for x in np.arange(3, 24.5, 0.5)]
    sp.poly(top + [(24, 7.2), (3, 7.2)], "mantle")
    sp.poly(bot + [(24, 7.4), (3, 7.4)], "mantle")
    sp.ellipse(14, 7.3, 10.5, 2)
    sp.curve((24, 6.2), (25.5, 4.6), (26.5, 3.6), 0.6, 0.45, "rh")
    sp.curve((24, 8.2), (25.5, 9.8), (26.5, 10.8), 0.6, 0.45, "rh")
    sp.ellipse(3, 7.3, 1.8, 1.8, "rh")                             # พู่เหงือกขาวด้านท้าย
    near, man, rh = sp.mask("near"), sp.mask("mantle"), sp.mask("rh")
    me = _edge(man | near)
    g = blank(28, 15)
    for y in range(15):
        for x in range(28):
            if man[y, x]:
                g[y][x] = "#ffffff" if me[y, x] else ("#e8234a" if (x + y) % 4 else "#ff4d6d")
            if near[y, x]:
                g[y][x] = "#ff8fa3"
            if rh[y, x]:
                g[y][x] = "#fff0f3"
    return g


# ---------------------------------------------------------------- ไดโนเสาร์และสัตว์ดึกดำบรรพ์
TEAL = {"edge": "#e6fff6", "hi": "#7ff5cf", "mid": "#33cfa6", "lo": "#1d9a82", "far": "#147563", "eye": "#062a24"}
SUNSET = {"edge": "#fff0e0", "hi": "#ffb070", "mid": "#ff7f50", "lo": "#e0583a", "far": "#a53a2a", "eye": "#2a0c05"}
SKYBLUE = {"edge": "#f1f5ff", "hi": "#b3ccff", "mid": "#7c9cff", "lo": "#5670e0", "far": "#3c4fb0", "eye": "#0b1240"}
LIME = {"edge": "#fbffe0", "hi": "#d8f56a", "mid": "#a8d93a", "lo": "#6fae2a", "far": "#4c8020", "eye": "#1a2a05"}
SLATE = {"edge": "#f4f8ff", "hi": "#c9d8ee", "mid": "#8fa9cc", "lo": "#5f7fa8", "far": "#41608a", "eye": "#0b1530"}
PLUM = {"edge": "#f6e9ff", "hi": "#d6a8ff", "mid": "#a873f0", "lo": "#7a4cc8", "far": "#56349a", "eye": "#1a0f33"}


def plesiosaur(frame):
    # พลีซิโอซอร์ (สัตว์เลื้อยคลานทะเลคอยาว) ว่ายไปทางขวา 44 × 20: พายครีบสองท่า
    sp = Sprite(44, 20)
    down = lambda x: [(x + 2.5, 12), (x - 1, 19.6), (x - 3.6, 19.2), (x - 2.5, 12)]   # ครีบพายลง / ยกขึ้น
    up = lambda x: [(x + 2.5, 10), (x - 1, 2.4), (x - 3.6, 2.8), (x - 2.5, 10)]
    near, far = (down, up) if frame == 0 else (up, down)
    for x in (22, 12):
        sp.poly(far(x + 1), "far")
        sp.poly(near(x), "near")
    sp.ellipse(17, 11, 9, 3.4)                                     # ลำตัว
    sp.curve((9, 11), (4, 11.5), (0.5, 12.5), 2.6, 0.5)            # หาง
    sp.curve((25, 10), (32, 7.5), (37.5, 5), 2.0, 1.2)             # คอยาว
    sp.ellipse(39.8, 4.6, 2.9, 1.8)                                # หัว
    sp.poly([(41.5, 3.8), (43.8, 5), (41.5, 5.8)])                 # ปาก
    return holo(sp, TEAL, [(40, 3, "eye")], scan=False)


def spinosaurus(frame):
    # สไปโนซอรัส (ไดโนเสาร์ที่ว่ายน้ำเก่ง) 46 × 24: ใบเรือลายบนหลัง หางพาย ขาตีน้ำสองท่า
    sp = Sprite(46, 24)
    sp.poly([(12.5, 13), (14, 6), (18, 2.6), (23, 3), (27, 6.5), (28.5, 13)], "sail")
    kick = [(15, 23), (25, 20)] if frame == 0 else [(19, 20), (22, 23)]
    for (hx, (fx, fy)), layer in (((17, kick[0]), "far"), ((25, kick[1]), "far"), ((18, (kick[1][0] - 7, kick[1][1])), "near"), ((26, (kick[0][0] + 9, kick[0][1])), "near")):
        sp.curve((hx, 16), (hx - 1, 19), (fx, fy), 1.5, 0.8, layer)
    sp.ellipse(20, 15, 9, 3.8)                                     # ลำตัว
    sp.curve((11, 15), (5, 16), (0.5, 14.5), 2.8, 0.9)             # หาง
    sp.poly([(1, 14.5), (3.5, 10.5), (8, 13.5), (8, 16.5), (3.5, 19 if frame else 18), (1, 15.5)])   # พายหาง
    sp.curve((28, 13), (31.5, 11), (34, 10), 2.4, 1.9)             # คอ
    sp.ellipse(36, 9.8, 3, 2)                                      # หัว
    sp.poly([(37, 8.6), (45.5, 10), (45.5, 11.2), (37, 11.8)])     # ปากยาวแบบจระเข้
    g = holo(sp, SUNSET, [(37, 9, "eye")], scan=False)
    sail, near = sp.mask("sail"), sp.mask("near")
    se = _edge(sail)
    for y in range(24):
        for x in range(46):
            if sail[y, x] and not near[y, x]:
                g[y][x] = "#fff3c4" if se[y, x] else ("#ffd23f" if (x // 2) % 2 else "#ff5f7a")
    return g


def ichthyosaur(frame):
    # อิกทิโอซอร์ (สัตว์เลื้อยคลานทะเลรูปร่างคล้ายโลมา) 36 × 16: ปากแหลมยาว ครีบหลัง หางรูปพระจันทร์เสี้ยว
    sp = Sprite(36, 16)
    sp.ellipse(16, 8, 9, 3.4)
    sp.ellipse(24, 7.8, 2.8, 2.4)
    sp.poly([(24, 6.8), (35.5, 8), (24, 9.4)])                     # ปากแหลม
    sp.poly([(12.5, 5), (15.5, 0.8), (19, 4.8)])                   # ครีบหลัง
    sp.curve((8, 8), (5, 8), (3, 8), 2.0, 0.9)
    u = 1.2 if frame else -1.2
    sp.poly([(3.5, 8), (0.3, 1.2 + u), (1.8, 1 + u), (5, 7)])      # หางบน
    sp.poly([(3.5, 8), (0.3, 14.8 + u), (1.8, 15 + u), (5, 9)])    # หางล่าง
    fl = [(19, 10), (15.5, 14.8), (14, 14.2), (16.5, 10)] if frame else [(19, 10), (14, 13), (13, 12), (16.5, 9.6)]
    sp.poly(fl, "far")
    sp.poly([(x + 2, y) for x, y in fl])
    g = holo(sp, SKYBLUE, [(25, 7, "eye")], scan=False)
    g[7][26] = "#ffffff"
    return g


def mosasaur(frame):
    # โมซาซอร์ (กิ้งก่าทะเลยักษ์) 50 × 17: ขากรรไกรมีฟันขาว ครีบพาย หางมีแพนล่าง
    sp = Sprite(50, 17)
    w = 1 if frame else -1
    sp.ellipse(22, 8, 12, 3.3)
    sp.curve((11, 8.5), (4, 9.5 + w), (0.5, 9), 2.6, 0.8)
    sp.poly([(4, 9.4), (1, 14 + w * 0.6), (3.4, 14 + w * 0.6), (7, 10)])
    sp.ellipse(37, 7.4, 4.4, 2.3)
    sp.poly([(38, 5.9), (49.5, 7.6), (38, 8.3)])                   # ปากบน
    sp.poly([(37, 8.4), (48, 8.6), (48, 9.6), (36, 10.2)])         # ปากล่าง
    for x, ph in ((29, 0), (15, 1)):
        d = 4.5 if (frame + ph) % 2 else 3
        sp.poly([(x + 2, 10), (x - 1.5, 10 + d + 1), (x - 3.5, 10 + d), (x - 1, 10)])
        sp.poly([(x + 3, 10), (x, 10 + 8 - d), (x - 2, 10 + 7.5 - d), (x, 10)], "far")
    g = holo(sp, LIME, [(39, 6, "eye")], scan=False)
    for x in range(40, 48, 2):                                     # ฟัน
        g[8][x] = "#ffffff"
    return g


def megalodon(frame):
    # เมกาโลดอน (ฉลามยักษ์ดึกดำบรรพ์) 48 × 18: ท้องขาว ปากเต็มไปด้วยฟัน หางสะบัด
    sp = Sprite(48, 18)
    sp.ellipse(22, 9, 14, 4.2)
    sp.poly([(33, 5.6), (47.5, 8.6), (44, 10.8), (33, 12.4)])     # จมูก
    sp.poly([(17, 5.6), (21.5, 0.4), (26, 5.2)])                   # ครีบหลัง
    sp.curve((9, 9), (5, 9), (3, 9), 2.4, 1.0)
    u = 1.2 if frame else -1.2
    sp.poly([(4, 9), (0.3, 1.5 + u), (2.2, 1.3 + u), (6.5, 8)])
    sp.poly([(4, 9.5), (1, 15.5 + u), (2.8, 15.6 + u), (6.5, 10.5)])
    pec = [(28, 11.5), (23, 17.4), (21.4, 16.8), (25, 11.5)] if frame else [(28, 11.5), (22, 16.2), (20.6, 15.4), (25, 11.2)]
    sp.poly([(x - 2, y) for x, y in pec], "far")
    sp.poly(pec)
    g = holo(sp, SLATE, [(40, 7, "eye")], scan=False)
    near, e = sp.mask("near"), _edge(sp.mask("near"))
    for y in range(10, 14):                                        # ท้องขาว
        for x in range(10, 45):
            if near[y, x] and not e[y, x]:
                g[y][x] = "#eef4ff"
    for x in range(36, 45):                                        # ปาก + ฟัน
        g[10][x] = "#ffffff" if x % 2 else "#7a2335"
    for x in (30, 32, 34):                                         # ช่องเหงือก
        for y in (7, 8, 9):
            g[y][x] = "#5f7fa8"
    return g


def archelon(frame):
    # อาร์คีลอน (เต่าทะเลยักษ์ดึกดำบรรพ์) 36 × 18: กระดองลายเหลืองทอง กระพือครีบ
    sp = Sprite(36, 18)
    sp.ellipse(16, 8.6, 10.5, 4.6, "shell")
    sp.curve((25, 9.4), (28, 9), (30, 9), 1.6, 1.4)               # คอ
    sp.ellipse(31.5, 8.8, 2.6, 1.9)                                # หัว
    sp.poly([(33, 8), (35.6, 9.2), (33, 10)])                      # จะงอยปาก
    dn = [(21, 11), (16, 17.6), (14, 17), (18, 10.5)]
    upf = [(21, 7), (16, 0.4), (14, 1), (18, 7.5)]
    near, far = (dn, upf) if frame == 0 else (upf, dn)
    sp.poly(far, "far")
    sp.poly(near)
    sp.poly([(9, 11.5), (5, 14.6), (4, 13.6), (7.5, 11)])          # ครีบหลัง
    sp.poly([(6, 9), (2.5, 10), (6, 10.6)])                        # หาง
    g = holo(sp, TEAL, [(32, 8, "eye")], scan=False)
    sh = sp.mask("shell")
    she = _edge(sh)
    for y in range(18):
        for x in range(36):
            if sh[y, x]:
                plate = (x + (y // 3) * 3) % 6 == 0 or y % 3 == 0
                g[y][x] = "#fff1b0" if she[y, x] else ("#c98a1e" if plate else ("#ffd25a" if y < 8 else "#ffb43a"))
    return g


def dunkleosteus(frame):
    # ดังเคิลออสเตียส (ปลาเกราะยักษ์ยุคดีโวเนียน) 40 × 18: หัวเป็นเกราะกระดูก อ้า-หุบขากรรไกรใบมีด
    sp = Sprite(40, 18)
    sp.ellipse(17, 9, 13, 4.4)
    sp.poly([(12, 5.2), (15, 1.4), (18.5, 5)])                     # ครีบหลัง
    sp.curve((5, 9), (2.5, 8.5), (1, 8), 2, 0.7)
    u = 1 if frame else -1
    sp.poly([(3, 8.6), (0.3, 2.5 + u), (2, 2.4 + u), (6, 7.6)])
    sp.poly([(3, 9.4), (0.8, 13.5 + u), (2.4, 13.6 + u), (6, 10.4)])
    sp.ellipse(29, 8.2, 8.4, 5.2, "armor")
    if frame:
        sp.poly([(30, 10.5), (39, 14), (37.6, 15), (28.5, 12.6)], "armor")   # อ้าปาก
    else:
        sp.poly([(30, 10.5), (39.2, 11.2), (38.6, 12.6), (28.5, 12.6)], "armor")
    g = holo(sp, PLUM, [], scan=False)
    ar = sp.mask("armor")
    ae = _edge(ar)
    for y in range(18):
        for x in range(40):
            if ar[y, x]:
                seam = x in (25, 31) or (y == 8 and x < 34)
                g[y][x] = "#fff7e6" if ae[y, x] else ("#b08a58" if seam else ("#efdcb4" if y < 9 else "#d9bf8c"))
    g[6][33], g[6][32], g[5][33] = "#101010", "#ffd23f", "#ffd23f"
    return g


def anomalocaris(frame):
    # อะโนมาโลคาริส (นักล่ายุคแคมเบรียน) 32 × 15: ครีบข้างลำตัวโบกเป็นคลื่น แขนหน้าโค้งงอ ตาบนก้าน
    sp = Sprite(32, 15)
    for i, x in enumerate(range(5, 24, 2)):
        sp.ellipse(x, 7.5, 1.3, 3.0 + 1.3 * math.sin(i * 0.9 + frame * math.pi), "lobe")
    sp.ellipse(14, 7.5, 10, 2.1)
    sp.ellipse(25, 7.5, 3.2, 2.5)
    sp.poly([(4, 7.5), (0.3, 3.6), (1.6, 7.5), (0.3, 11.4)])       # หางพัด
    c = 0.8 if frame else 0
    for sgn in (-1, 1):
        sp.curve((27, 7.5 + sgn * 1.2), (31, 7.5 + sgn * 2.4), (30 - c, 7.5 + sgn * (5 - c)), 0.8, 0.5, "app")   # แขนหน้า
        sp.ellipse(24, 7.5 + sgn * 4, 1.2, 1.2, "eye")                                                       # ตาบนก้าน
    near, lobe, app, eye = sp.mask("near"), sp.mask("lobe"), sp.mask("app"), sp.mask("eye")
    ne, le = _edge(near), _edge(lobe)
    g = blank(32, 15)
    for y in range(15):
        for x in range(32):
            if lobe[y, x]:
                g[y][x] = "#ffc2d1" if le[y, x] else "#ff6f91"
            if near[y, x]:
                g[y][x] = "#fff0f3" if ne[y, x] else ("#ff9fb0" if x % 2 else "#ff8aa0")
            if app[y, x]:
                g[y][x] = "#ffcf6b" if x % 2 else "#ff9a3c"
            if eye[y, x]:
                g[y][x] = "#202040"
    g[3][24] = g[11][24] = "#ffffff"
    return g


def sauropod_bright(frame):
    # ซอโรพอดตัวเดียวกับในป้ายหัวเรื่อง เปลี่ยนจากสี hologram เป็นสีเขียวสดใส
    base = {"#c8f6ff": "#f4ffd9", "#5ee0ff": "#a8ec6a", "#18c6f0": "#76d34a", "#0f8fbf": "#4fae3a"}
    lighter = {mix(k, "#ffffff", 0.25): mix(v, "#ffffff", 0.2) for k, v in base.items()}
    g = sauropod(frame)
    for y, row in enumerate(g):
        for x, c in enumerate(row):
            if c in base:
                row[x] = base[c]
            elif c in lighter:
                row[x] = lighter[c]
            if row[x] == base["#18c6f0"] and (x * 3 + y * 5) % 13 == 0:
                row[x] = "#3f8f2e"                                   # ลายจุดบนหลัง
    return g


def fish_school(col, stripe, eye="#13325c", n=5):
    fish = blank(6, 3)
    stamp(fish, FISH, 0, 0, col)
    fish[1][2] = fish[1][3] = stripe
    fish[1][1] = eye
    pos = ((0, 0), (9, -4), (10, 4), (19, 1), (27, -3), (28, 5), (37, 0))[:n]
    return "".join(f'<g transform="translate({dx} {dy})">{runs(fish)}</g>' for dx, dy in pos)


CORAL_BRANCH = [" #   # ", " # # # ", "## # ##", " ##### ", "  ###  ", "   #   ", "   #   "]
CORAL_FAN = ["  # #  ", " # # # ", "# # # #", " ##### ", "  ###  ", "   #   ", "   #   "]
CORAL_BRAIN = ["  ####  ", " ###### ", "########"]
CORAL_TUBE = ["# # #", "# # #", "#####", " ### "]
CRINOID = ["#  #  #", " # # # ", "  ###  ", "   #   "]
STARFISH = ["  #  ", "#####", " ### ", " # # "]
ANEMONE = ["# # # #", " # # # ", "#######", " ##### "]


def outlined(g, s, x0, y0, scale, colf, line):
    pix = text_pixels(s, scale)
    for x, y, r in pix:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (0, 2), (1, 2)):
            put(g, x0 + x + dx, y0 + y + dy, line)
    for x, y, r in pix:
        put(g, x0 + x, y0 + y, colf(r))


def loop_x(inner, x_start, x_min, x_max, dur, rev=False):
    # วิ่งวนต่อเนื่องโดยเริ่มที่ตำแหน่ง x_start (ใช้ translate ภายนอกด้วย เพื่อให้ภาพนิ่งและเบราว์เซอร์กระจายทั่วฉากเหมือนกัน)
    span = x_max - x_min
    if rev:
        f = max(0.01, min(0.99, (x_start - x_min) / span))
        pts = [(0, 0), (x_min - x_start, 0), (x_max - x_start, 0), (0, 0)]
    else:
        f = max(0.01, min(0.99, (x_max - x_start) / span))
        pts = [(0, 0), (x_max - x_start, 0), (x_min - x_start, 0), (0, 0)]
    return f'<g transform="translate({x_start} 0)">{move(inner, pts, dur, extra=f' keyTimes="0;{f:.4f};{f + 0.0001:.4f};1"')}</g>' 


def at(y, inner):
    return f'<g transform="translate(0 {y})">{inner}</g>'


def contributor():
    # ป้ายท้าย README: ผู้ร่วมพัฒนาพิเศษ @E27-25 พร้อมฉายาครบทุกชื่อ บนฉากสีสดใส
    #   บนฟ้า: เกาะมีซอโรพอด + เทอโรพอด, เทอโรซอร์บิน, ภูเขาไฟ
    #   ใต้น้ำ 4 ช่องว่าย: สไปโนซอรัส อิกทิโอซอร์ พลีซิโอซอร์ โมซาซอร์ เมกาโลดอน อาร์คีลอน ดังเคิลออสเตียส อะโนมาโลคาริส
    #   ทากทะเล 9 ชนิดคลานบนพื้นทราย 2 แถว + นักเต้นสเปน + มังกรทะเลสีน้ำเงิน
    W, LH, SKY = 320, 10, 58
    T0 = SKY + 46
    lines = wrap_titles(E27_TITLES, (W - 24) // 6)
    cy0, cy1 = T0 - 6, T0 + len(lines) * LH + 1
    L1 = cy1 + 6                                                    # สไปโนซอรัส + อิกทิโอซอร์
    L2 = L1 + 30                                                    # พลีซิโอซอร์ + โมซาซอร์
    L3 = L2 + 32                                                    # เมกาโลดอน + อาร์คีลอน
    L4 = L3 + 28                                                    # ดังเคิลออสเตียส + อะโนมาโลคาริส + นักเต้นสเปน
    S = L4 + 30                                                     # พื้นทราย
    H = S + 26
    rnd = random.Random(27)
    g = blank(W, H)
    # ---- ฟ้า ดวงอาทิตย์ ภูเขาไฟ เกาะ
    sky(g, 0, SKY, ["#62c8ff", "#86d8ff", "#aee7ff", "#d4f5ff"])
    for y in range(4, 28):
        for x in range(22, 48):
            d = (x - 34.5) ** 2 + (y - 15.5) ** 2
            if d <= 90:
                g[y][x] = "#fff27a" if d <= 56 else "#ffd23f"
    for x in range(62, 132):
        top = max(SKY - 26 + abs(x - 97) * 0.85, SKY - 26)
        for y in range(int(top), SKY):
            g[y][x] = "#9b6bb8" if y < SKY - 14 else "#7f56a0"
        if abs(x - 97) <= 3:
            g[SKY - 26][x] = "#ff5a2a"
            g[SKY - 25][x] = "#ff9a3c"
    for x in range(128, 302):
        top = int(SKY - 9 + max(0, abs(x - 214) - 62) * 0.7)
        for y in range(top, SKY):
            g[y][x] = "#5fd35a" if y == top else "#3fb84a" if y == top + 1 else "#ffd486"
    for tx in (160, 202, 266):
        stamp(g, ARAUCARIA, tx, SKY - 17, "#2f9b4a")
    # ---- ทะเล พื้นทราย ปะการัง
    sky(g, SKY, S + 2, ["#8af0ff", "#5fdcff", "#36bff2", "#1f9fe0", "#1784cc", "#136dba", "#0f5aa6"])
    for x in range(W):
        top = S + round(0.9 * math.sin(x / 13) + 0.6 * math.sin(x / 5.1))
        for y in range(top, H):
            c = "#ffe7a8" if y == top else "#ffd486" if y < top + 6 else "#f2b86a" if y < top + 16 else "#e5a55a"
            g[y][x] = c if rnd.random() > 0.07 else "#fff7e0"
    for name, fx in (("ammonite", 30), ("bone", 92), ("trilobite", 150), ("fish", 206), ("shell", 262), ("ammonite", 304)):
        stamp(g, FOSSILS[name], fx, S + 20, "#fffaf0")
    for art, x, col in ((CORAL_BRANCH, 6, "#ff5fa2"), (CORAL_FAN, 44, "#b06bff"), (CORAL_BRAIN, 84, "#ffcf4a"), (ANEMONE, 112, "#ff7a45"),
                        (CORAL_BRANCH, 140, "#ff9a3c"), (CORAL_TUBE, 170, "#5ad8ff"), (CORAL_FAN, 196, "#ff5fa2"), (CORAL_BRAIN, 226, "#7fe36b"),
                        (ANEMONE, 252, "#d26bff"), (CORAL_BRANCH, 282, "#ffd23f"), (CORAL_TUBE, 308, "#ff6f91")):
        stamp(g, art, x, S - len(art) + 1, col)
    for cx, h in ((26, 12), (124, 16), (238, 14)):                  # พลับพลึงทะเล (crinoid) ก้านยาว
        for y in range(S - h, S + 1):
            g[y][cx + 3] = "#e0568f"
        stamp(g, CRINOID, cx, S - h - 4, "#ff8fc4")
    for sx in (66, 186, 296):
        stamp(g, STARFISH, sx, S - 1, "#ff7a45")
    cut_corners(g)

    out = [svg_open(W, H, "Special contributor: E27-25, " + ", ".join(t.title() for t in E27_TITLES)), runs(g)]
    cloud = blank(12, 3)
    stamp(cloud, CLOUD, 0, 0, "#ffffff")
    for x in range(12):
        cloud[2][x] = "#d8f3ff"
    for (cx, cy), d in (((60, 6), 50), ((180, 12), 70), ((270, 4), 60)):
        out.append(move(runs(cloud), [(cx, cy), (cx - W - 14, cy)], d))
        out.append(move(runs(cloud), [(cx + W + 14, cy), (cx, cy)], d))
    for i in range(5):                                             # ควันภูเขาไฟ
        out.append(f'<rect x="96" y="{SKY - 28}" width="2" height="2" fill="#ffffff" opacity="0">'
                   f'<animate attributeName="y" values="{SKY - 28};2" dur="5s" begin="{i}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="x" values="96;108" dur="5s" begin="{i}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0.9;0.5;0" dur="5s" begin="{i}s" repeatCount="indefinite"/></rect>')
    # ไดโนเสาร์บนเกาะ (เดินอยู่กับที่) + เทอโรซอร์บินผ่าน
    out.append(f'<g transform="translate(150 {SKY - 45})">{flip([sauropod_bright(0), sauropod_bright(1)], 0.9)}</g>')
    out.append(f'<g transform="translate(222 {SKY - 32})">{flip([theropod(0), theropod(1)], 0.7)}</g>')
    out.append(move(flip([pterosaur(0), pterosaur(1)], 0.5), [(-32, 8), (100, 3), (200, 10), (W + 2, 5)], 16))
    out.append(at(20, move(mirrored(flip([pterosaur(1), pterosaur(0)], 0.6), 30), [(W + 2, 0), (-32, 0)], 22, begin=-9)))
    # ใต้น้ำ: แสงแดด คลื่นผิวน้ำ สาหร่าย
    for x0, d, b in ((40, 5, 0), (130, 6, 1.5), (220, 4.5, 3), (300, 5.5, 2)):
        out.append(f'<polygon points="{x0},{SKY} {x0 + 18},{SKY} {x0 - 20},{S} {x0 - 38},{S}" fill="#ffffff" opacity="0.1">'
                   f'<animate attributeName="opacity" values="0.04;0.18;0.04" dur="{d}s" begin="{b}s" repeatCount="indefinite"/></polygon>')
    wave = blank(2 * W, 2)
    for x in range(2 * W):
        if x % 16 < 6:
            wave[0][x] = "#ffffff"
        elif 8 <= x % 16 < 11:
            wave[1][x] = "#c9f7ff"
    out.append(at(SKY, move(runs(wave), [(0, 0), (-W, 0)], 14)))
    for sx in (18, 58, 100, 132, 162, 214, 244, 274, 316):
        frames = []
        for f in (0, 1):
            fr = blank(W, H)
            for k, hh in enumerate((rnd.randrange(10, 20), rnd.randrange(7, 14))):
                for d in range(hh):
                    x = sx + 2 * k + round((d / hh) * 1.8 * math.sin(d * 0.5 + f * math.pi + k))
                    put(fr, x, S - d, "#5cf29a" if d % 3 == 0 else "#2fcf7a")
            frames.append(fr)
        out.append(flip(frames, 1.8))
    # ช่วงหัวเรื่อง: มังกรทะเลสีน้ำเงินลอยใกล้ผิวน้ำ แอมโมไนต์ ฝูงปลา
    for y, dur, x0 in ((SKY + 4, 34, 24), (SKY + 28, 40, 268)):
        bd = move(flip([blue_dragon(0), blue_dragon(1)], 1.2), [(0, 0), (0, 1), (0, 0)], 3)
        out.append(at(y, loop_x(mirrored(bd, 22), x0, -24, W + 2, dur, rev=True)))
    out.append(at(SKY + 12, loop_x(move(flip([ammonite(0), ammonite(1)], 0.8), [(0, 0), (0, -2), (0, 0)], 2.4), 12, -26, W + 2, 24)))
    out.append(at(SKY + 18, loop_x(fish_school("#ffe14a", "#ff9a3c"), 264, -40, W + 4, 15, rev=True)))
    # การ์ดรายชื่อฉายา
    out.append(f'<rect x="8" y="{cy0}" width="{W - 16}" height="{cy1 - cy0}" fill="#0b2a5c" opacity="0.8"/>')
    card = blank(W, H)
    for x in range(9, W - 9):
        card[cy0][x] = card[cy1 - 1][x] = "#9fe9ff"
    for y in range(cy0 + 1, cy1 - 1):
        card[y][8] = card[y][W - 9] = "#9fe9ff"
    txt = blank(W, H)
    label, name = "SPECIAL CONTRIBUTOR", "@E27-25"
    lw, nw = text_width(label), text_width(name, 2)
    outlined(txt, label, (W - lw) // 2, SKY + 6, 1, lambda r: "#ffffff", "#0b3a6e")
    nx = (W - nw) // 2
    outlined(txt, name, nx, SKY + 17, 2, lambda r: mix("#fff6b0", "#ff9a3c", r / 6), "#0b2a5c")
    for sx in (nx - 13, nx + nw + 8):
        stamp(txt, [" # ", "###", " # "], sx, SKY + 21, "#ffffff")
    cols = ["#fff1a3", "#9fe9ff"]
    for i, line in enumerate(lines):                                # ฉายาทั้งหมด สลับสีทอง / ฟ้า, ดาวคั่นสีชมพู
        x0, y0 = (W - text_width(line)) // 2, T0 + i * LH
        for k, ch in enumerate(line):
            c = "#ff8fc4" if ch == "*" else cols[i % 2]
            for x, y, r in text_pixels(ch):
                put(txt, x0 + k * 6 + x, y0 + y, mix(c, "#ffb04a", r / 10) if ch != "*" and i % 2 == 0 else c)
    out.append(runs(card) + runs(txt))
    out.append(f'<rect x="10" width="{W - 20}" height="1" fill="#ffffff" opacity="0.15">'
               f'<animate attributeName="y" values="{cy0 + 1};{cy1 - 2}" dur="4s" repeatCount="indefinite"/></rect>')
    # ---- 4 ช่องว่ายน้ำของสัตว์ดึกดำบรรพ์ (ว่ายสวนกัน กระจายเต็มฉาก)
    out.append(at(L1, loop_x(move(flip([spinosaurus(0), spinosaurus(1)], 0.8), [(0, 0), (0, 1), (0, 0)], 1.6), 36, -48, W + 2, 26)))
    out.append(at(L1 + 7, loop_x(mirrored(flip([ichthyosaur(0), ichthyosaur(1)], 0.5), 36), 210, -38, W + 2, 13, rev=True)))
    out.append(at(L1 + 3, loop_x(fish_school("#ff8fd0", "#ffffff", n=6), 130, -44, W + 4, 17)))
    out.append(at(L2, loop_x(move(flip([plesiosaur(0), plesiosaur(1)], 1.0), [(0, 0), (0, 2), (0, 0)], 2), 185, -46, W + 2, 20)))
    out.append(at(L2 + 13, loop_x(mirrored(flip([mosasaur(0), mosasaur(1)], 0.7), 50), 65, -52, W + 2, 18, rev=True)))
    out.append(at(L3, loop_x(mirrored(flip([megalodon(0), megalodon(1)], 0.6), 48), 235, -50, W + 2, 16, rev=True)))
    out.append(at(L3 + 8, loop_x(move(flip([archelon(0), archelon(1)], 1.1), [(0, 0), (0, 2), (0, 0)], 2.2), 110, -38, W + 2, 28)))
    out.append(at(L3 + 4, loop_x(fish_school("#7fffd4", "#ffd23f", n=4), 25, -36, W + 4, 12, rev=True)))
    out.append(at(L4, loop_x(flip([dunkleosteus(0), dunkleosteus(1)], 0.9), 20, -42, W + 2, 22)))
    out.append(at(L4 + 6, loop_x(mirrored(flip([anomalocaris(0), anomalocaris(1)], 0.6), 32), 160, -34, W + 2, 19, rev=True)))
    out.append(at(L4 + 2, loop_x(move(mirrored(flip([spanish_dancer(0), spanish_dancer(1)], 0.7), 28), [(0, 0), (0, -3), (0, 0)], 2.6),
                                  260, -30, W + 2, 24, rev=True)))
    # ---- ทากทะเล 9 ชนิดคลานบนพื้นทราย 2 แถว + ไทรโลไบต์
    tri = [blank(9, 4), blank(9, 4)]
    for f in (0, 1):
        stamp(tri[f], TRILOBITE[f], 0, 0, "#b9a3f0")
        for x in range(9):
            if tri[f][0][x]:
                tri[f][0][x] = "#efe6ff"
            if tri[f][3][x]:
                tri[f][3][x] = "#7b5fc0"
    out.append(at(S - 3, loop_x(flip(tri, 0.5), 146, -12, W + 4, 36, rev=True)))
    for row_y, slugs in ((S - 11, (("flab", 46, 14, False), ("chromo", 46, 78, False), ("leaf", 60, 152, True), ("pink", 52, 220, True), ("tiger", 40, 282, False))),
                         (S + 8, (("nembrotha", 50, 22, True), ("bunny", 64, 96, False), ("lettuce", 56, 178, True), ("aeolid", 44, 256, False)))):
        for kind, dur, x0, rev in slugs:
            fr = flip([nudibranch(0, kind), nudibranch(1, kind)], 0.9)
            out.append(at(row_y, loop_x(mirrored(fr, 26) if rev else fr, x0, -28, W + 2, dur, rev=rev)))
    out.append(at(S + 15, loop_x(mirrored(flip(tri, 0.5), 9), 218, -12, W + 4, 42)))
    for _ in range(18):                                             # ฟองอากาศ
        x, sz, d = rnd.randrange(6, W - 6), rnd.choice((1, 1, 2)), rnd.uniform(6, 11)
        b = rnd.uniform(0, 8)
        out.append(f'<rect x="{x}" y="{S - 2}" width="{sz}" height="{sz}" fill="#ffffff" opacity="0.8">'
                   f'<animate attributeName="y" values="{S - 2};{SKY + 2}" dur="{d:.1f}s" begin="{b:.1f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0.9;0.9;0" keyTimes="0;0.85;1" dur="{d:.1f}s" begin="{b:.1f}s" repeatCount="indefinite"/></rect>')
    edge = blank(W, H)
    for x in range(2, W - 2):
        edge[0][x] = edge[H - 1][x] = "#ffd86b"
    for y in range(2, H - 2):
        edge[y][0] = edge[y][W - 1] = "#ffd86b"
    out.append(f'<g>{runs(edge)}<animate attributeName="opacity" values="0.55;1;0.55" dur="2.4s" repeatCount="indefinite"/></g>')
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
