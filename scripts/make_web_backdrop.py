# แถบภาพทิวทัศน์ยุคจูราสสิกตามขอบล่างของหน้าแรกเว็บ (โหมดมืด / สว่าง) -> webapp/static/img/backdrop-{dark,light}.webp
#   ภูเขาในหมอก + เนินสองลูกที่มีฝูงไดโนเสาร์คอยาว (ไททันโนซอร์) ยืนบนยอด + เนินข้างซ้าย / ขวาสูงเท่ากัน + เนินแถวหน้าเตี้ย ๆ
#   จัดให้สมดุลซ้าย-ขวา; เทอโรซอร์บนฟ้า; ท้องฟ้าโปร่งใส (ดาว / ดวงอาทิตย์วาดด้วย CSS)
#   โหมดสว่าง = เขียวในหมอก, โหมดมืด = เนินสีคราม / ม่วงเข้มมีขอบแสงจันทร์
# รัน: .venv/bin/python scripts/make_web_backdrop.py   (สุ่มแบบกำหนด seed: ได้ภาพเดิมทุกครั้ง)
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parents[1] / "webapp" / "static" / "img"
W, H, SS = 3000, 420, 2                     # แถบกว้าง (ภาพจริง) และตัวคูณวาดละเอียด (ย่อลงตอนจบให้เส้นเรียบ)
w, h = W * SS, H * SS
U = 3.75 * h                                # หน่วยขนาดของไดโนเสาร์ / เทอโรซอร์ (อิงความสูงแถบ)
X = np.arange(w)


def hexrgb(s):
    return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))


def smooth1d(rng, n, sigma):
    v = gaussian_filter(rng.standard_normal(n), sigma, mode="wrap")
    return v / (np.abs(v).max() + 1e-9)


def ridge(rng, base, amp, bumps, peaks=()):
    # สันเขา y(x) (ค่ามาก = ต่ำ): ระดับฐาน + คลื่นสุ่มหลายขนาด + ยอดเขาแบบเกาส์เซียน
    y = base * h + sum(a * h * smooth1d(rng, w, s * SS) for a, s in zip(amp, bumps))
    for px, ph, pw in peaks:
        y -= ph * h * np.exp(-0.5 * ((X - px * w) / (pw * w)) ** 2)
    return y


def mask_below(y):
    return np.arange(h)[:, None] >= y[None, :]


def field2d(rng, sigma):
    v = gaussian_filter(rng.standard_normal((h // 4, w // 4)), sigma / 4)
    v = (v - v.min()) / (v.max() - v.min())
    return np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), float) / 255


def watercolor(img, mask, top, bottom, rng, blur=18, alpha=235, fade=0.5):
    # ภูเขาในหมอก: ไล่สีบน -> ล่าง ขอบนุ่ม เนื้อกระดาษ และจางลงด้านล่าง
    yy = np.arange(h)[:, None] / h
    c0, c1 = np.array(hexrgb(top), float), np.array(hexrgb(bottom), float)
    tex = field2d(rng, 30)[..., None]
    col = c0 + (c1 - c0) * np.clip((yy[..., None] - 0.15) / 0.6, 0, 1) + (tex - 0.5) * 18
    m = gaussian_filter(mask.astype(float), blur * SS)
    a = m * alpha * (1 - fade * np.clip((yy - 0.35) / 0.65, 0, 1)) * (0.85 + 0.3 * (tex[..., 0] - 0.5))
    layer = Image.fromarray(np.dstack([np.clip(col, 0, 255), np.clip(a, 0, 255)]).astype(np.uint8), "RGBA")
    img.alpha_composite(layer)


def tube(center, widths):
    # รูปหลายเหลี่ยมรอบเส้นกลาง (ความหนาเปลี่ยนตามจุด) ใช้วาดคอ / หาง
    c = np.asarray(center, float)
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1) * np.asarray(widths, float)[:, None] / 2
    return [tuple(p) for p in c + nrm] + [tuple(p) for p in (c - nrm)[::-1]]


def bezier(p0, p1, p2, p3, n=48):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(q, float) for q in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def sauropod(d, x, y, s, color, flip=False):
    # ไดโนเสาร์คอยาวกลุ่มไททันโนซอร์ (แบบภูเวียงโกซอรัส) หันขวา (flip = หันซ้าย), เท้าอยู่ที่ระดับ y, s = หน่วยขนาด
    # ลำตัวทรงถังสูงที่ไหล่ คอโค้งตัว S ยกไปข้างหน้า หัวเล็ก หางยาวเรียวลู่ลง ขาเป็นเสา 4 ข้าง (ขาไกลเยื้องจากขาใกล้ให้เห็นครบ)
    f = -1 if flip else 1
    def T(pts):
        return [(x + f * px * s, y + py * s) for px, py in pts]
    a = np.linspace(0, 2 * np.pi, 72)
    d.polygon(T(zip(0.54 * np.cos(a), -0.66 + 0.26 * np.sin(a))), fill=color)                  # ลำตัว (ท้องสูงจากพื้นราว 0.4)
    d.polygon(T(zip(0.22 + 0.30 * np.cos(a), -0.74 + 0.23 * np.sin(a))), fill=color)           # ไหล่
    u = np.linspace(0, 1, 48)
    neck = bezier((0.38, -0.80), (0.60, -0.95), (0.66, -1.35), (0.86, -1.55))
    d.polygon(T(tube(neck, 0.27 - 0.18 * u ** 0.8)), fill=color)
    d.polygon(T(zip(0.92 + 0.11 * np.cos(a), -1.565 + 0.058 * np.sin(a))), fill=color)         # หัว
    tail = bezier((-0.30, -0.70), (-0.78, -0.72), (-1.10, -0.58), (-1.55, -0.46))               # โคนหางเริ่มในลำตัว ไม่เป็นรอยหัก
    d.polygon(T(tube(tail, 0.30 * (1 - u) ** 1.15 + 0.012)), fill=color)
    for lx, lw in ((-0.40, 0.13), (-0.22, 0.14), (0.16, 0.12), (0.32, 0.13)):                   # หลังไกล หลังใกล้ หน้าไกล หน้าใกล้
        d.polygon(T([(lx, -0.55), (lx + lw, -0.55), (lx + lw, -0.05), (lx + lw + 0.02, 0), (lx - 0.015, 0), (lx, -0.05)]), fill=color)


def pterosaur(d, x, y, s, color):
    pts = [(x - s, y - 0.25 * s), (x - 0.45 * s, y - 0.05 * s), (x - 0.12 * s, y + 0.08 * s), (x + 0.25 * s, y - 0.02 * s),
           (x + 0.45 * s, y - 0.12 * s), (x + 0.2 * s, y + 0.02 * s), (x + 0.55 * s, y - 0.3 * s), (x + 0.1 * s, y + 0.12 * s),
           (x - 0.12 * s, y + 0.18 * s), (x - 0.5 * s, y + 0.02 * s)]
    d.polygon(pts, fill=color)


def ridge_line(d, y, color, width):
    # ขอบแสงบาง ๆ ตามสันเขา (แสงจันทร์ในโหมดมืด)
    step = 6 * SS
    d.line([(float(x), float(y[x])) for x in range(0, w, step)] + [(w - 1.0, float(y[-1]))], fill=color, width=width)


def build(theme):
    rng = np.random.default_rng(7)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    light = theme == "light"
    rim = (150, 140, 230, 90)
    # ---- ชั้นไกล: ภูเขาในหมอก ยอดกระจายเท่า ๆ กันทั้งสองฝั่ง
    far = ridge(rng, 0.50, (0.02, 0.008), (300, 70), [(0.08, 0.17, 0.07), (0.30, 0.22, 0.08), (0.50, 0.13, 0.09), (0.70, 0.22, 0.08), (0.92, 0.17, 0.07)])
    if light:
        watercolor(img, mask_below(far), "#c3cac5", "#eef0ea", rng, blur=20, alpha=225)
    else:
        watercolor(img, mask_below(far), "#1c1e4e", "#0b0c29", rng, blur=10, alpha=240, fade=0.0)
    # ---- ชั้นกลาง: เนินสองลูก ยอดราบ (ฝูงไดโนเสาร์ยืนบนยอด)
    mid = ridge(rng, 0.70, (0.006,), (200,), [(0.27, 0.13, 0.10), (0.73, 0.13, 0.10)])
    if light:
        watercolor(img, mask_below(mid), "#b5cb8a", "#d8e3b6", rng, blur=2.5, alpha=250, fade=0.15)
    else:
        watercolor(img, mask_below(mid), "#272163", "#16154a", rng, blur=3, alpha=255, fade=0.0)
        ridge_line(d, mid, rim, int(1.4 * SS))
    herd = "#4a6b3a" if light else "#07081c"
    # ฝูงซ้ายหันขวา ฝูงขวาหันซ้าย (หันเข้ากลาง); วางบนยอดเนินซึ่งเกือบราบ เท้าทุกข้างแตะพื้น
    for px, sc, fl in [(0.245, 0.024, False), (0.292, 0.017, False), (0.705, 0.024, True), (0.748, 0.018, True), (0.776, 0.012, True)]:
        xi = int(px * w)
        span = slice(max(0, xi - int(0.03 * w)), xi + int(0.03 * w))
        ground = float(mid[span].max()) + 0.014 * h                  # จุดต่ำสุดใต้ลำตัว + จมลงเล็กน้อย: เท้าแตะพื้น ไม่ลอย
        if not light:
            sauropod(d, xi - 1.2 * SS, ground - 1.2 * SS, sc * U, (90, 86, 170, 255), fl)
        sauropod(d, xi, ground, sc * U, herd, fl)
    # ---- เนินข้างซ้าย / ขวา สูงและกว้างเท่ากัน ขอบเป็นพุ่มไม้
    side = ridge(rng, 1.10, (0.008, 0.006), (60, 5), [(0.0, 0.62, 0.12), (1.0, 0.62, 0.12)])
    if light:
        watercolor(img, mask_below(side), "#46733a", "#6a984f", rng, blur=2, alpha=250, fade=0.0)
    else:
        watercolor(img, mask_below(side), "#1f1a52", "#110e33", rng, blur=2, alpha=255, fade=0.0)
        ridge_line(d, side, rim, int(1.2 * SS))
    # ---- เนินแถวหน้าเตี้ย ๆ เต็มความกว้าง ให้ภาพมีฐาน
    front = ridge(rng, 0.92, (0.02, 0.006), (160, 30), [(0.5, 0.05, 0.22)])
    if light:
        watercolor(img, mask_below(front), "#5b8746", "#4d7a3d", rng, blur=2, alpha=250, fade=0.0)
    else:
        watercolor(img, mask_below(front), "#141142", "#0b0a26", rng, blur=2, alpha=255, fade=0.0)
        ridge_line(d, front, rim, int(1.0 * SS))
    # ---- เทอโรซอร์: กลุ่มละฝั่ง
    flyer = "#6d746a" if light else "#3a3c78"
    for px, py, sc in [(0.36, 0.12, 0.016), (0.39, 0.19, 0.011), (0.64, 0.10, 0.014)]:
        pterosaur(d, px * w, py * h, sc * U, flyer)
    out = img.resize((W, H), Image.LANCZOS)
    path = OUT / f"backdrop-{theme}.webp"
    out.save(path, "WEBP", quality=84, method=6)
    print("saved", path.name, out.size, f"{path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    for t in ("dark", "light"):
        build(t)
