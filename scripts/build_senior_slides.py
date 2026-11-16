# สไลด์นำเสนอ senior project (Project Jurassic) — สร้างจากสเปกชุดเดียวเป็น
#   1) docs/slides/Project_Jurassic_Senior_Project.pptx  ข้อความ กล่อง และตารางแก้ได้ใน PowerPoint + บทพูดใน Speaker notes
#      กราฟเป็นรูปที่มีอยู่แล้วจาก notebook และรายงาน (scripts/senior_slides_assets.py charts → docs/slides/assets/charts/)
#   2) docs/slides/preview/slide_NN.png  ภาพพรีวิวจาก HTML ที่วางตำแหน่งเหมือนไฟล์ .pptx (ใช้ตรวจการตัดบรรทัดและข้อความล้น)
#   3) docs/slides/บทพูด_สไลด์_senior.md  บทพูดทุกสไลด์
# ธีม: โทนสว่างแบบโปสเตอร์ v3 (docs/poster/poster.css) · หัวข้อ Kanit · เนื้อหา IBM Plex Sans Thai · โค้ด IBM Plex Mono
#   (ไฟล์ฟอนต์ใน docs/slides/fonts/ — ติดตั้งก่อนเปิดใน PowerPoint)
# สองงานของโครงงาน: งาน 1 (ทายชื่อสกุลและเติมยุค) สีเขียว อยู่ซ้าย · งาน 2 (ทายโอกาสพบซากจากพิกัด) สีส้มอิฐ อยู่ขวา และขึ้นเมื่อคลิก
# ตัวเลขทุกตัวมาจากรายงาน (docs/report_src/parts) ภาพประกอบจาก scripts/senior_slides_assets.py (docs/slides/assets/)
# รัน: <venv ที่มี python-pptx + playwright + pillow> python scripts/build_senior_slides.py
#   ข้อความ: "**หนา**", "{i}เอียง{/i}", "{c:RRGGBB}สี{/c}", "\n" = ขึ้นย่อหน้าใหม่ · พิกัดทุกค่าเป็นนิ้ว (สไลด์ 13.333 × 7.5)
import html as H
import re
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SL = ROOT / "docs" / "slides"
A = SL / "assets"
C = A / "charts"                           # รูปกราฟเดิมจาก notebook (nb_*) และรายงาน (chart_*, cm_*)
FONTS = SL / "fonts"
OUT = SL / "Project_Jurassic_Senior_Project.pptx"

# ---------------------------------------------------------------- ธีม (สีจากโปสเตอร์ v3)
BG, SURF, PANEL, LINE = "F4F5EF", "FBFBF6", "EBEBE6", "D9D3C8"
INK, INK2, MUTED, DIM = "2A251F", "3A342B", "5D554A", "8B8273"
ACC, ACCHI, GREEN, SAGE, PILL, BASE = "C4581C", "E07A30", "3D7A3A", "8FB07A", "F8E7B5", "BDB6A8"
T1, T2 = GREEN, ACC                       # งาน 1 ทายชื่อ/ยุค · งาน 2 พิกัด
T1S, T2S = "E5EEE1", "F8E7DC"             # สีอ่อนของสองงาน
HEAD, HEADM, KANIT = "Kanit SemiBold", "Kanit Medium", "Kanit"
BODY, BODYM = "IBM Plex Sans Thai", "IBM Plex Sans Thai Medium"
MONO = "IBM Plex Mono"                     # โค้ด / กล่อง Key Features (ภาษาไทยใช้ IBM Plex Sans Thai แทน)
W_IN, H_IN = 13.333, 7.5
X0, XW = 0.6, 12.133
Y0 = 1.62                                  # บนสุดของพื้นที่เนื้อหา
LX, RX, HW = 0.6, 6.98, 5.75               # ครึ่งซ้าย / ครึ่งขวา ของสไลด์สองงาน
SECTIONS = {1: "FOSSIL 101", 2: "WHY THIS PROJECT", 3: "SYSTEM OVERVIEW", 4: "DATA UNDERSTANDING", 5: "DATA PREPARATION",
            6: "MODELING", 7: "EVALUATION", 8: "DEPLOYMENT"}
FONT_FILES = {(HEAD, False): "Kanit-SemiBold", (HEAD, True): "Kanit-Bold", (HEADM, False): "Kanit-Medium", (HEADM, True): "Kanit-Bold",
              (KANIT, False): "Kanit-Regular", (KANIT, True): "Kanit-Bold", (BODY, False): "IBMPlexSansThai-Regular",
              (BODY, True): "IBMPlexSansThai-Bold", (BODYM, False): "IBMPlexSansThai-Medium", (BODYM, True): "IBMPlexSansThai-Bold",
              (MONO, False): "IBMPlexMono-Regular", (MONO, True): "IBMPlexMono-SemiBold"}
_FONT_CACHE = {}


def tw(text, size, font=BODY, bold=False):
    # ความกว้างข้อความ (นิ้ว) จากไฟล์ฟอนต์จริง — ใช้กำหนดความกว้างป้าย
    key = (font, bold, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(str(FONTS / f"{FONT_FILES[(font, bold)]}.ttf"), int(size * 10))
    return _FONT_CACHE[key].getlength(text) / 10 / 72


# ---------------------------------------------------------------- ข้อความ
TOK = re.compile(r"(\*\*|\{i\}|\{/i\}|\{c:[0-9A-Fa-f]{6}\}|\{/c\})")


def runs_of(s, color=INK, bold_color=None):
    out, b, it, cs = [], False, False, [color]
    for part in TOK.split(s):
        if part == "**":
            b = not b
        elif part == "{i}":
            it = True
        elif part == "{/i}":
            it = False
        elif part.startswith("{c:"):
            cs.append(part[3:9].upper())
        elif part == "{/c}":
            cs.pop()
        elif part:
            out.append((part, b, bold_color if (b and bold_color and len(cs) == 1) else cs[-1], it))
    return out


def P(text, size=14, color=INK2, font=BODY, align="l", bold_color=None, bullet=None, space=4, line=1.12, spacing=0):
    items = text if isinstance(text, list) else text.split("\n")
    return [{"runs": runs_of(t, color, bold_color), "size": size, "font": font, "align": align, "bullet": bullet,
             "space": space, "line": line, "spacing": spacing} for t in items]


class Slide:
    def __init__(self, sec=None, title="", tag="", notes="", kind="content"):
        self.sec, self.title, self.tag, self.notes, self.kind = sec, title, tag, notes, kind
        self.items, self.step = [], 0

    def add(self, **it):
        it["step"] = self.step
        self.items.append(it)

    @contextmanager
    def reveal(self):
        # ทุกชิ้นที่วางในบล็อกนี้ขึ้นเมื่อคลิก (แอนิเมชันค่อย ๆ ปรากฏ)
        self.step = 1
        yield
        self.step = 0

    def text(self, x, y, w, h, ps, valign="t", inset=0.03):
        self.add(k="text", x=x, y=y, w=w, h=h, ps=ps, valign=valign, inset=inset)

    def rect(self, x, y, w, h, fill=None, line=None, lw=1.0, radius=0.0, dash=False, ps=None, valign="m", inset=0.1, shape="rect"):
        self.add(k="rect", x=x, y=y, w=w, h=h, fill=fill, line=line, lw=lw, radius=radius, dash=dash, ps=ps or [], valign=valign,
                 inset=inset, shape=shape)

    def line(self, x1, y1, x2, y2, color=DIM, lw=1.25, dash=False, head=False):
        self.add(k="line", x1=x1, y1=y1, x2=x2, y2=y2, color=color, lw=lw, dash=dash, head=head)

    def img(self, path, x, y, w, h, align="c", valign="m"):
        path = Path(path)
        iw, ih = Image.open(path).size
        sc = min(w / iw, h / ih)
        dw, dh = iw * sc, ih * sc
        dx = {"l": 0, "c": (w - dw) / 2, "r": w - dw}[align]
        dy = {"t": 0, "m": (h - dh) / 2, "b": h - dh}[valign]
        self.add(k="img", path=path, x=x + dx, y=y + dy, w=dw, h=dh)
        return x + dx, y + dy, dw, dh

    def icon(self, name, color, x, y, w, h, align="c", valign="m"):
        return self.img(A / "icons" / f"{name}-{color}.png", x, y, w, h, align, valign)

    def table(self, x, y, colw, rows, size=12, rowh=0.36, head=INK, align=None, hl=None, first_bold=True, rowhs=None):
        self.add(k="table", x=x, y=y, colw=colw, rows=rows, size=size, rowh=rowh, head=head, align=align or ["l"] * len(colw),
                 hl=hl or {}, first_bold=first_bold, rowhs=rowhs)

    def chart(self, kind, x, y, w, h, cats, series, fmt="0.000", vmax=None, size=11, legend=False, gap=55, point_colors=None,
              label_w=None):
        self.add(k="chart", kind=kind, x=x, y=y, w=w, h=h, cats=cats, series=series, fmt=fmt, vmax=vmax, size=size, legend=legend,
                 gap=gap, point_colors=point_colors, label_w=label_w)


# ---------------------------------------------------------------- ชิ้นส่วนประกอบ
def pill(s, x, y, text, fill=PILL, color=INK, size=12, font=BODYM, h=0.32, padx=0.13, line=None, bold=False):
    w = tw(text, size, font, bold) + 2 * padx + 0.02
    s.rect(x, y, w, h, fill, line, 1.0, h / 2, ps=P(("**" + text + "**") if bold else text, size, color, font, "c", space=0, line=1.0),
           inset=0.02)
    return w


def task_head(s, x, y, n, name):
    col = T1 if n == 1 else T2
    w = pill(s, x, y, f"TASK {n}", col, "FFFFFF", 12, HEAD, 0.38, 0.16)
    s.text(x + w + 0.12, y - 0.04, HW - w - 0.12, 0.46, P(name, 17, col, HEADM, space=0, line=1.0), "m")


def two(s, name1, name2, y=Y0, bottom=6.9):
    # สไลด์สองงาน: เส้นแบ่งกลาง + หัวงาน 1 (หัวงาน 2 วางใน s.reveal() พร้อมเนื้อหาครึ่งขวา)
    s.line(6.665, y + 0.02, 6.665, bottom, LINE, 1.0, dash=True)
    task_head(s, LX, y, 1, name1)
    s._name2 = name2


def right_head(s, y=Y0):
    task_head(s, RX, y, 2, s._name2)


def card(s, x, y, w, h, title=None, body=None, icon=None, icolor="ink", fill=SURF, line=LINE, tsize=15, bsize=13,
         tcolor=INK, bcolor=INK2, pad=0.16, isize=0.42, lw=1.0, radius=0.12, bold_color=None, dash=False):
    s.rect(x, y, w, h, fill, line, lw, radius, dash)
    tx, ty = x + pad, y + pad - 0.02
    if icon:
        s.icon(icon, icolor, x + pad, y + pad, isize, isize, "l", "t")
        tx = x + pad + isize + 0.12
    if title:
        th = 0.36 if tsize <= 16 else 0.44
        s.text(tx, ty, x + w - pad - tx, th, P(title, tsize, tcolor, HEADM, space=0, line=1.05), "m" if icon else "t")
        ty += max(th, isize if icon else 0) + 0.06
    if body:
        s.text(x + pad, ty, w - 2 * pad, y + h - ty - 0.08, P(body, bsize, bcolor, bold_color=bold_color, space=3) if isinstance(body, str)
               else body)


def stat(s, x, y, w, value, label, color, vsize=34, lsize=12.5, lcolor=INK2, h=None):
    vh = vsize / 72 * 1.25
    s.text(x, y, w, vh, P(value, vsize, color, HEAD, space=0, line=1.0), "b")
    s.text(x, y + vh + 0.02, w, (h or 1.0) - vh, P(label, lsize, lcolor, space=1, line=1.08))


def bullets(s, x, y, w, h, items, size=14, color=INK2, bcolor=None, space=7, bold_color=None):
    s.text(x, y, w, h, P(items, size, color, bullet=bcolor or DIM, space=space, bold_color=bold_color))


def badge(s, x, y, kind):
    lab, fill, col = {"good": ("ได้ผล", T1S, T1), "bad": ("ไม่ได้ผล", T2S, T2), "same": ("เท่าเดิม", PANEL, MUTED)}[kind]
    s.rect(x, y, 0.92, 0.3, fill, None, 0, 0.15, ps=P(lab, 11, col, HEADM, "c", space=0, line=1.0), inset=0.02)


def arrow(s, x1, y1, x2, y2, color=DIM, lw=1.5, dash=True):
    s.line(x1, y1, x2, y2, color, lw, dash, head=True)


def box(s, x, y, w, h, text, fill=SURF, line=LINE, size=13, color=INK, font=BODY, lw=1.0, radius=0.1, align="c", dash=False):
    s.rect(x, y, w, h, fill, line, lw, radius, dash, ps=P(text, size, color, font, align, space=1, line=1.05), inset=0.08)


def frame_img(s, path, x, y, w, h, pad=0.08, fill="FFFFFF", line=LINE):
    s.rect(x, y, w, h, fill, line, 1.0, 0.1)
    return s.img(path, x + pad, y + pad, w - 2 * pad, h - 2 * pad)


# ---------------------------------------------------------------- แผงโค้ด + กล่อง Key Features (หน้าเชื่อมข้อมูล)
CODE_BG, CODE_FG, CODE_NUM = "221E19", "EDE6DA", "6E665A"
CODE_COL = {"c": "9C9384", "s": "EBCB8B", "k": "E07A30", "n": "8FB07A", "f": "9CC79A"}
CODE_TOK = re.compile(r"""(?P<s>"[^"]*"|'[^']*')|(?P<c>#.*)|(?P<k>\b(?:def|return|for|in|if|else|not|is|and|or|None|True|False|lambda)\b)"""
                      r"|(?P<n>\b\d+(?:\.\d+)?\b)|(?P<f>(?<=def )\w+)")
THAI = re.compile(r"[\u0e00-\u0e7f]")


def code_width(line, size):
    # ความกว้างบรรทัดโค้ด (นิ้ว): ตัวอักษรไทยวัดด้วยฟอนต์เนื้อหา เพราะฟอนต์โค้ดไม่มีอักษรไทย
    return sum(tw(ch, size, BODY if THAI.match(ch) else MONO) for ch in line)


def code_ps(src, size):
    # โค้ด → ย่อหน้าละบรรทัด ระบายสีแบบง่าย (คอมเมนต์ สตริง คำสงวน ตัวเลข ชื่อฟังก์ชัน) · ช่องว่างนำหน้าเป็น NBSP
    out = []
    for line in src.strip("\n").split("\n"):
        lead = len(line) - len(line.lstrip(" "))
        body, runs, pos = line[lead:], [], 0
        if lead:
            runs.append(("\u00a0" * lead, False, CODE_FG, False))
        for m in CODE_TOK.finditer(body):
            if m.start() > pos:
                runs.append((body[pos:m.start()], False, CODE_FG, False))
            runs.append((m.group(), False, CODE_COL[m.lastgroup], False))
            pos = m.end()
        if pos < len(body):
            runs.append((body[pos:], False, CODE_FG, False))
        out.append({"runs": runs or [("\u00a0", False, CODE_FG, False)], "size": size, "font": MONO, "align": "l", "bullet": None,
                    "space": 0, "line": 1.0, "spacing": 0})
    return out


def code_panel(s, x, y, w, h, src, size=10, label=None, source=None):
    s.rect(x, y, w, h, CODE_BG, None, 0, 0.14)
    ps = code_ps(src, size)
    num_w = 0.5
    s.text(x + 0.08, y + 0.16, num_w, h - 0.3, [dict(q, runs=[(str(i + 1), False, CODE_NUM, False)], align="r") for i, q in enumerate(ps)],
           "t", 0.02)
    s.text(x + num_w + 0.22, y + 0.16, w - num_w - 0.32, h - 0.3, ps, "t", 0.02)
    avail = w - num_w - 0.36
    for q in ps:   # บรรทัดโค้ดต้องไม่ตัดขึ้นบรรทัดใหม่ ไม่อย่างนั้นเลขบรรทัดจะเลื่อน
        line = "".join(r[0] for r in q["runs"])
        assert code_width(line, size) <= avail, (round(code_width(line, size), 2), avail, line)
    assert len(ps) * size * 1.2 / 72 <= h - 0.3, ("โค้ดยาวเกินแผง", len(ps))
    if source:
        s.text(x + w - 3.2, y + h - 0.36, 3.05, 0.3, P(source, 9.5, CODE_NUM, MONO, "r", space=0), "m")
    if label:
        lw = tw(f"[ {label} ]", 17, HEADM) + 0.1
        s.icon("database", "green", x + w - lw - 0.55, y + h + 0.12, 0.42, 0.42)
        s.text(x + w - lw, y + h + 0.08, lw, 0.5, P(f"[ {label} ]", 17, INK, HEADM, "r", space=0), "m")


def keybox(s, x, y, w, h, lines, size=11.5, head="Key Features:", fill=PANEL):
    # กล่องสรุปแบบโค้ด: หัว "Key Features:" + บรรทัดขึ้นต้นด้วย "- " (ใช้ **หนา** ได้)
    s.rect(x, y, w, h, fill, None, 0, 0.14)
    ps = (P(head, size, INK, MONO, space=3, line=1.05) if head else []) + P(
        [("- " + t) if not t.startswith(" ") else t for t in lines], size, INK2, MONO, bold_color=ACC, space=2, line=1.08)
    s.text(x + 0.2, y + 0.1, w - 0.36, h - 0.2, ps, "m", 0.02)


# ---------------------------------------------------------------- เนื้อหา
def build_deck():
    D = []

    # ===== 1 ปก
    s = Slide(kind="title", notes=(
        "สวัสดีค่ะ วันนี้จะนำเสนอโครงงาน ต้นแบบนวัตกรรมการจัดการข้อมูลบรรพชีวินวิทยา เพื่อคาดการณ์แหล่งขุดค้น ระบุชื่อ และประเมินยุคของซากดึกดำบรรพ์ในประเทศไทย "
        "โครงงานนี้ใช้ข้อมูลเปิดของกรมทรัพยากรธรณี ได้แก่ทะเบียนตัวอย่าง 8,521 ชิ้นและแหล่งซาก 518 แหล่ง มาเชื่อมกับข้อมูลเชิงพื้นที่ "
        "แล้วสร้างแบบจำลองและแผนที่โอกาสพบซากครอบคลุมทั้งประเทศ 20,706 ช่อง"))
    s.rect(8.55, 0.55, 4.18, 6.4, "FFFFFF", LINE, 1.0, 0.16)
    s.img(A / "maps" / "potential_bg_bias.png", 8.7, 0.7, 3.88, 5.7)
    s.text(8.7, 6.42, 3.88, 0.4, P("แผนที่โอกาสพบซาก · กริด 5 กม. 20,706 ช่อง", 11, DIM, align="c", space=0))
    s.rect(0.63, 0.98, 0.17, 0.17, None, ACC, 1.6, shape="diamond")
    s.text(0.9, 0.88, 6.8, 0.36, P("PROJECT JURASSIC · SENIOR PROJECT 2569", 12, ACC, HEADM, space=0, spacing=2), "m")
    s.text(0.6, 1.4, 7.75, 1.5, P(["ต้นแบบนวัตกรรมการจัดการข้อมูลบรรพชีวินวิทยา", "เพื่อคาดการณ์แหล่งขุดค้น ระบุชื่อ",
                                   "และประเมินยุคของซากดึกดำบรรพ์ในประเทศไทย"], 26, INK, HEAD, space=0, line=1.08))
    s.text(0.6, 3.05, 7.4, 0.75, P("Innovative Paleontological Data Management Prototype for Predicting Excavation Sites, "
                                   "Identifying Taxa, and Estimating Geological Periods of Fossils in Thailand", 13, MUTED, space=0, line=1.15))
    for k, (v, lab, col) in enumerate([("8,521", "ตัวอย่างในทะเบียน ทธ.", GREEN), ("518", "แหล่งซากดึกดำบรรพ์", ACC),
                                       ("20,706", "ช่องกริดทั่วประเทศ", INK)]):
        stat(s, 0.6 + k * 2.35, 4.1, 2.2, v, lab, col, 30, 12, h=1.05)
    s.text(0.6, 5.55, 7.6, 1.2, P(["**สิดาพร จิตระวีวรรณ** · 66070203", "อาจารย์ที่ปรึกษา **อาจารย์เฉลิมพล ศิริกายน**",
                                   "สาขาวิชาวิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ · คณะเทคโนโลยีสารสนเทศ สจล."], 13, MUTED, bold_color=INK,
                                  space=2))
    D.append(s)

    # ===== 2 วาระ
    s = Slide(None, "Agenda · ลำดับการนำเสนอ", "// AGENDA", (
        "การนำเสนอแบ่งเป็น 8 ส่วน เริ่มจากพื้นฐานเรื่องซากดึกดำบรรพ์ที่ต้องรู้ก่อน แล้วเป็นเหตุผลและจุดแข็งของงาน ภาพรวมระบบ "
        "จากนั้นไล่ตามขั้นของ CRISP-DM ตั้งแต่ทำความเข้าใจข้อมูล เตรียมข้อมูล สร้างแบบจำลอง วัดผล และเว็บต้นแบบ "
        "โครงงานมี 2 งานที่เชื่อมกัน คืองาน 1 ทายชื่อสกุลและเติมยุค สีเขียว และงาน 2 ทายโอกาสพบซากจากพิกัด สีส้ม "
        "ในแต่ละขั้นจะอธิบายงาน 1 ทางซ้ายก่อน แล้วคลิกให้งาน 2 ขึ้นทางขวา"))
    ag = [("FOSSIL 101", "พื้นฐานซากดึกดำบรรพ์ที่ต้องรู้ก่อน"), ("WHY THIS PROJECT", "ปัญหา เป้าหมาย และจุดแข็ง"),
          ("SYSTEM OVERVIEW", "ภาพรวมระบบทั้งหมด"), ("DATA UNDERSTANDING", "ข้อมูลและ Target Label"),
          ("DATA PREPARATION", "เชื่อมข้อมูล เติมยุค สร้างคุณลักษณะ"), ("MODELING", "แบบจำลองของ 2 งาน"),
          ("EVALUATION", "วัดผลแยกตามประเภท"), ("DEPLOYMENT", "เว็บต้นแบบ")]
    for k, (en, th) in enumerate(ag):
        x, y = 0.6 + (k % 4) * 3.09, 1.75 + (k // 4) * 2.0
        s.rect(x, y, 2.86, 1.78, SURF, LINE, 1.0, 0.14)
        s.text(x + 0.2, y + 0.14, 1.2, 0.6, P(f"{k + 1:02d}", 28, ACC, HEAD, space=0, line=1.0), "m")
        s.text(x + 0.2, y + 0.78, 2.5, 0.4, P(en, 13, INK, HEADM, space=0, spacing=1), "m")
        s.text(x + 0.2, y + 1.16, 2.5, 0.55, P(th, 12.5, MUTED, space=0))
    s.text(0.6, 5.98, 3.0, 0.4, P("โครงงานมี 2 งานที่เชื่อมกัน", 14, INK, HEADM, space=0), "m")
    w = pill(s, 3.55, 6.02, "TASK 1", T1, "FFFFFF", 12, HEAD, 0.34)
    s.text(3.55 + w + 0.1, 5.98, 3.2, 0.42, P("ทายชื่อสกุลและเติมยุคที่ขาดหาย", 14, T1, HEADM, space=0), "m")
    w = pill(s, 7.75, 6.02, "TASK 2", T2, "FFFFFF", 12, HEAD, 0.34)
    s.text(7.75 + w + 0.1, 5.98, 4.0, 0.42, P("ทายโอกาสพบซากจากพิกัด (lat, long)", 14, T2, HEADM, space=0), "m")
    D.append(s)

    # ===== 3 ซากดึกดำบรรพ์คืออะไร
    s = Slide(1, "ซากดึกดำบรรพ์คืออะไร", "// WHAT IS A FOSSIL", (
        "ซากดึกดำบรรพ์คือซากหรือร่องรอยของสิ่งมีชีวิตในอดีตที่ถูกเก็บรักษาไว้ในหิน มีทั้งซากร่างกาย เช่น กระดูก ฟัน เปลือก และร่องรอย เช่น รอยตีนไดโนเสาร์ "
        "ทะเบียนของกรมทรัพยากรธรณีแบ่งตัวอย่าง 8,521 ชิ้นเป็น 5 กลุ่ม ส่วนใหญ่เป็นสัตว์มีกระดูกสันหลัง 5,676 ชิ้น รองลงมาคือสัตว์ไม่มีกระดูกสันหลัง 2,492 ชิ้น "
        "ซากทั้งหมดได้รับความคุ้มครองตามพระราชบัญญัติคุ้มครองซากดึกดำบรรพ์ พ.ศ. 2551 และ ทธ. เป็นหน่วยงานหลักที่ดูแลข้อมูล"))
    s.text(X0, Y0, XW, 0.5, P("ซากหรือร่องรอยของสิ่งมีชีวิตในอดีตที่ถูกเก็บรักษาไว้ในหิน · ทะเบียนของกรมทรัพยากรธรณี (ทธ.) แบ่งตัวอย่าง "
                              "**8,521 ชิ้น** เป็น 5 กลุ่ม", 15, INK2, bold_color=INK), "m")
    groups = [("phuwiangosaurus", "5,676", "สัตว์มีกระดูกสันหลัง", "Vertebrate", "ไดโนเสาร์ ปลา จระเข้ เต่า สัตว์เลี้ยงลูกด้วยนม"),
              ("brachiopod", "2,492", "สัตว์ไม่มีกระดูกสันหลัง", "Invertebrate", "แบรคิโอพอด หอย ไทรโลไบต์ ปะการัง"),
              ("question", "149", "อื่น ๆ", "Other", "จุลซาก เช่น เรดิโอลาเรีย ฟิวซูลินิด สาหร่าย"),
              ("leaf", "131", "พืช", "Plant", "ซากใบ ลำต้น และเนื้อไม้"),
              ("footprint", "73", "ร่องรอย", "Trace fossil", "รอยตีนไดโนเสาร์ — สิ่งที่สัตว์ทิ้งไว้ ไม่ใช่ตัวซาก")]
    for k, (ic, n, th, en, ex) in enumerate(groups):
        x = 0.6 + k * 2.47                 # ไม่มีกรอบการ์ด (ผู้จัดทำลบออกใน PowerPoint)
        s.icon(ic, "ink", x + 0.2, 2.45, 1.9, 1.0)
        s.text(x + 0.18, 3.55, 1.95, 0.6, P(n, 30, ACC if k else GREEN, HEAD, space=0, line=1.0), "m")
        s.text(x + 0.18, 4.15, 1.95, 0.42, P(th, 14.5, INK, HEADM, space=0, line=1.0), "m")
        s.text(x + 0.18, 4.52, 1.95, 0.3, P(en, 11, DIM, space=0), "m")
        s.text(x + 0.18, 4.85, 1.95, 0.75, P(ex, 12, MUTED, space=0, line=1.08))
    s.rect(0.6, 5.92, XW, 0.8, None, ACC, 1.2, 0.12)
    s.icon("shield-check", "rust", 0.8, 6.08, 0.48, 0.48)
    s.text(1.42, 5.95, 11.1, 0.74, P("ได้รับความคุ้มครองตาม **พ.ร.บ.คุ้มครองซากดึกดำบรรพ์ พ.ศ. 2551** — ทธ. เป็นหน่วยงานหลักที่สำรวจ ขึ้นทะเบียน "
                                     "และดูแลข้อมูลซากของประเทศ", 14, INK2, bold_color=INK), "m")
    D.append(s)

    # ===== 4 ธรณีกาล
    s = Slide(1, "อายุของซาก = อายุของชั้นหินที่ซากฝังอยู่", "// GEOLOGIC TIME", (
        "นักธรณีวิทยาบอกอายุของซากจากชั้นหินที่ซากฝังอยู่ โดยใช้มาตราธรณีกาลสากล กราฟซ้ายคือจำนวนแหล่งของ ทธ. ในแต่ละยุค "
        "กราฟนี้มาจาก notebook ขั้นยุคสุดท้าย สีน้ำเงินคือยุคที่ ทธ. บันทึกไว้ สีอื่นคือยุคที่โครงงานเติมจากหลักฐาน PBDB และชื่อซากในแหล่ง "
        "ยุคเพอร์เมียนมีมากที่สุด 159 แหล่ง เป็นหินปูนทะเลทางภาคเหนือและตะวันตก รองลงมาคือครีเทเชียสและจูแรสซิกบนที่ราบสูงโคราช "
        "ด้านขวาคือซากตัวอย่าง 7 สกุลข้ามยุคที่ใช้บนหน้าแรกของเว็บ ปัญหาคือแหล่ง 352 จาก 518 แหล่งไม่มีข้อมูลยุค โครงงานเติมจนเหลือไม่ทราบ 129 แหล่ง"))
    s.text(X0, Y0, 6.6, 0.4, P("จำนวนแหล่ง ทธ. ในแต่ละยุค (เก่า → ใหม่)", 14, INK, HEADM, space=0), "m")
    frame_img(s, C / "nb_period_final.png", X0, 2.05, 6.6, 3.15)
    s.text(X0, 5.35, 6.6, 1.1, P(["**น้ำเงิน** DMR record = ทธ. บันทึกไว้ · **น้ำเงินเข้ม** = หลายวิธีตรงกัน",
                                  "**เขียว** = PBDB ใกล้แหล่ง · **ส้ม** = ชื่อซากในแหล่ง · ยังไม่ทราบยุค 129 แหล่ง"],
                                 12.5, MUTED, bold_color=INK2, bullet=DIM, space=4))
    s.text(7.65, Y0, 5.1, 0.4, P("ซากตัวอย่างข้ามยุค (หน้าแรกของเว็บ)", 14, INK, HEADM, space=0), "m")
    show = [("ammonoid", "Armenoceras", "นอติลอยด์", "ออร์โดวิเชียน", "นครศรีธรรมราช"),
            ("trilobite", "Plagiolaria", "ไทรโลไบต์", "คาร์บอนิเฟอรัส", "พัทลุง"),
            ("brachiopod", "Reticulatia", "แบรคิโอพอด", "เพอร์เมียน", "เลย"),
            ("phuwiangosaurus", "Isanosaurus", "ซอโรพอด", "ไทรแอสซิก", "ชัยภูมิ"),
            ("fish", "Thaiichthys", "ปลากินพืชพุทธบุตร", "จูแรสซิก", "กาฬสินธุ์"),
            ("phuwiangosaurus", "Phuwiangosaurus", "ไดโนเสาร์", "ครีเทเชียส", "กาฬสินธุ์"),
            ("elephant", "Stegolophodon", "ช้างสี่งา", "เทอร์เชียรี", "ลำปาง")]
    for k, (ic, g, com, p, _) in enumerate(show):
        y = 2.08 + k * 0.62
        s.rect(7.65, y, 5.08, 0.54, SURF, LINE, 1.0, 0.1)
        s.icon(ic, "ink", 7.75, y + 0.06, 0.75, 0.42)
        s.text(8.62, y + 0.02, 2.0, 0.5, P(f"{{i}}{g}{{/i}}", 14, INK, HEADM, space=0), "m")
        s.text(10.55, y + 0.02, 2.12, 0.5, P(f"{com} · **{p}**", 11.5, MUTED, bold_color=GREEN if p in ("ไทรแอสซิก", "จูแรสซิก", "ครีเทเชียส")
                                             else INK2, space=0, line=1.0), "m")
    s.rect(7.65, 6.48, 5.08, 0.42, T2S, None, 0, 0.1, ps=P("**352 จาก 518 แหล่ง** ไม่มียุค → เติมแล้วเหลือ **129**", 12.5, INK2,
                                                            bold_color=ACC, align="c", space=0), inset=0.05)
    D.append(s)

    # ===== 5 การตั้งชื่อ
    s = Slide(1, "ชื่อวิทยาศาสตร์: สกุล (Genus) + ชนิด (Species)", "// TAXONOMY", (
        "ชื่อวิทยาศาสตร์ของสิ่งมีชีวิตมี 2 ส่วน คือชื่อสกุลกับชื่อชนิด เช่น Phuwiangosaurus sirindhornae "
        "แต่ในทะเบียนจริงมีหลายรูปแบบ บางชิ้นรู้แค่สกุลแล้วตามด้วย sp. บางชิ้นมี cf. แปลว่ายังไม่แน่ใจ และ 3,314 ชิ้นเขียนเป็น N/A ตามด้วยกลุ่ม คือยังไม่ระบุสกุล "
        "หลังแยกชื่อแล้ว มี 3,944 ชิ้นที่ระบุถึงสกุลได้ ซึ่งใช้เป็นป้ายกำกับ ส่วนอีก 4,577 ชิ้นยังไม่รู้สกุล งาน 1 จึงทายสกุลเป็นหลัก"))
    gx = 0.6
    gw = tw("Phuwiangosaurus", 38, HEAD)
    s.text(gx, 1.62, 9.0, 0.8, P("{i}{c:3D7A3A}Phuwiangosaurus{/c} sirindhornae{/i}", 38, INK, HEAD, space=0, line=1.0), "m")
    s.line(gx + 0.05, 2.5, gx + gw - 0.05, 2.5, GREEN, 2.0)
    sw = tw(" sirindhornae", 38, HEAD)
    s.line(gx + gw + 0.15, 2.5, gx + gw + sw - 0.05, 2.5, DIM, 2.0)
    s.text(gx, 2.56, gw, 0.5, P("**สกุล (Genus)** — สิ่งที่งาน 1 ทาย", 13, T1, bold_color=T1, space=0), "t")
    s.text(gx + gw + 0.15, 2.56, sw + 0.4, 0.5, P("**ชนิด (Species)**", 13, MUTED, bold_color=INK2, space=0), "t")
    s.text(X0, 3.25, 8.0, 0.4, P("ชื่อในทะเบียนมี 5 รูปแบบ", 14, INK, HEADM, space=0), "m")
    s.table(X0, 3.7, [3.05, 3.15, 1.6], [["รูปแบบ", "ตัวอย่าง", "จำนวนชิ้น"],
                                         ["สกุล + ชนิด", "Phuwiangosaurus sirindhornae", "2,613"],
                                         ["สกุล + sp. (ยังไม่ระบุชนิด)", "Linoproductus sp.", "1,262"],
                                         ["มี cf. (ยังไม่แน่ใจ)", "cf. + ชื่อ", "327"],
                                         ["ชื่อกลุ่มที่สูงกว่าสกุล", "Sauropod", "512"],
                                         ["N/A ตามด้วยกลุ่ม (ยังไม่ระบุ)", "N/A (Dinosaur)", "3,314"]],
            size=12.5, rowh=0.44, head=INK, align=["l", "l", "r"])
    s.rect(8.75, 3.25, 3.98, 1.62, T1S, None, 0, 0.14)
    stat(s, 8.95, 3.33, 3.6, "3,944", "ชิ้นที่ระบุถึงสกุลได้ → ใช้เป็นป้ายกำกับ (131 สกุล)", T1, 34, 12.5, h=1.45)
    s.rect(8.75, 5.05, 3.98, 1.62, T2S, None, 0, 0.14)
    stat(s, 8.95, 5.13, 3.6, "4,577", "ชิ้นยังไม่รู้สกุล → งาน 1 ทายสกุล 3 อันดับให้", T2, 34, 12.5, h=1.45)
    D.append(s)

    # ===== 6 ธรณีวิทยาไทย
    s = Slide(1, "ทำไมไทยมีซากหลากหลาย", "// GEOLOGY OF THAILAND", (
        "ประเทศไทยมีซากหลากหลายเพราะดินแดนไทยประกอบจากแผ่นเปลือกโลกย่อยอย่างน้อย 2 แผ่น คือฉาน-ไทยทางตะวันตกกับอินโดจีนทางตะวันออก "
        "หินปูนยุคเพอร์เมียนของกลุ่มหินราชบุรีและสระบุรีมีซากสัตว์ทะเลจำนวนมาก ส่วนที่ราบสูงโคราชเป็นแอ่งตะกอนบนบกยุคจูแรสซิกถึงครีเทเชียส จึงพบไดโนเสาร์ "
        "แผนที่ด้านซ้ายคือหน่วยหินตามยุคจากแผนที่ธรณีวิทยา 1:250,000 ของ ทธ. จะเห็นว่าแหล่งซากกระจุกบนหินบางยุค "
        "ข้อนี้คือเหตุผลที่แผนที่ธรณีวิทยาเป็นคุณลักษณะสำคัญของงาน 2"))
    frame_img(s, A / "maps" / "geology_periods.png", X0, Y0, 3.45, 4.95)
    s.text(X0, 6.6, 3.45, 0.32, P("หน่วยหินตามยุค (แผนที่ 1:250,000 ทธ.)", 10.5, DIM, align="c", space=0))
    pts = [("layers", "ink", "แผ่นเปลือกโลกย่อย 2 แผ่น", "ฉาน-ไทย (Sibumasu) ทางตะวันตก และอินโดจีน (Indochina) ทางตะวันออก "
            "มาชนกันเมื่อมหาสมุทรพาลีโอเททิสปิดตัว แต่ละแผ่นจึงมีลำดับชั้นหินและกลุ่มซากต่างกัน"),
           ("brachiopod", "ink", "ทะเลยุคเพอร์เมียน", "หินปูนกลุ่มหินราชบุรีและกลุ่มหินสระบุรี มีแบรคิโอพอด ปะการัง ไครนอยด์ และฟิวซูลินิดจำนวนมาก"),
           ("phuwiangosaurus", "ink", "ที่ราบสูงโคราช (จูแรสซิก–ครีเทเชียส)", "แอ่งตะกอนบนบกของกลุ่มหินโคราช พบไดโนเสาร์ จระเข้ เต่า "
            "และปลาน้ำจืด เช่น หมวดหินเสาขัว และสกุลใหม่ที่ยังค้นพบต่อเนื่อง เช่น สยามแรปเตอร์")]
    for k, (ic, col, t, b) in enumerate(pts):
        y = Y0 + k * 1.42
        card(s, 4.35, y, 8.38, 1.28, t, b, ic, col, tsize=16, bsize=13, isize=0.5)
    s.rect(4.35, 6.0, 8.38, 0.86, None, ACC, 1.2, 0.12)
    s.text(4.55, 6.02, 8.0, 0.82, P("แหล่งซากกระจุกตัวบนหินบางยุค → แผนที่ธรณีวิทยาช่วยบอกได้ว่า **“ที่ไหนน่าจะพบซาก”** "
                                    "(เป็นคุณลักษณะหลักของงาน 2)", 14, INK2, bold_color=ACC), "m")
    D.append(s)

    # ===== 7 ปัญหา
    s = Slide(2, "ข้อมูลมีมาก แต่ใช้ร่วมกันไม่ได้", "// PROBLEM", (
        "แม้ ทธ. จะเปิดข้อมูลไว้มาก แต่ใช้วิเคราะห์ร่วมกันไม่ได้ ข้อแรก ทะเบียนตัวอย่างไม่มีพิกัดและไม่มีรหัสแหล่ง มีแค่ชื่อเขตการปกครองที่เขียนด้วยมือ "
        "ข้อสอง แหล่งร้อยละ 68 ไม่มีข้อมูลยุค ข้อสาม ตัวอย่าง 4,577 ชิ้นยังไม่ได้ระบุสกุล และข้อสี่ ข้อมูลเป็นข้อความภาษาไทยอิสระ "
        "ปัญหาเหล่านี้นำไปสู่ 2 คำถามของโครงงาน คือพื้นที่ใดน่าจะพบซาก และซากที่พบน่าจะเป็นอะไร ยุคไหน"))
    probs = [("0", "คีย์ร่วม", "ทะเบียน 8,521 ชิ้นไม่มีพิกัดและรหัสแหล่ง มีแค่ชื่อจังหวัด อำเภอ ตำบลที่เขียนด้วยมือ"),
             ("68%", "แหล่งไม่มียุค", "352 จาก 518 แหล่งไม่มีข้อมูลยุคทางธรณีวิทยา"),
             ("4,577", "ชิ้นยังไม่รู้สกุล", "ต้องรอผู้เชี่ยวชาญระบุ ส่วนประชาชนที่พบซากมักไม่รู้แม้แต่ประเภท"),
             ("991", "แบบของ “ส่วนของซาก”", "ข้อความภาษาไทยอิสระ ชื่อสถานที่ปนชื่อสัตว์ เช่น “ถ้ำค้างคาว”")]
    for k, (v, t, b) in enumerate(probs):
        x = 0.6 + k * 3.09
        s.rect(x, Y0, 2.9, 2.85, SURF, LINE, 1.0, 0.14)
        s.text(x + 0.2, Y0 + 0.15, 2.5, 0.8, P(v, 38, ACC, HEAD, space=0, line=1.0), "m")
        s.text(x + 0.2, Y0 + 0.98, 2.5, 0.42, P(t, 15, INK, HEADM, space=0), "m")
        s.text(x + 0.2, Y0 + 1.45, 2.5, 1.3, P(b, 13, MUTED, space=0, line=1.1))
    s.text(X0, 4.62, XW, 0.45, P("+ ข้อมูลเชิงพื้นที่ (แผนที่ธรณี ความสูง ภาพดาวเทียม แผ่นเปลือกโลกโบราณ) อยู่คนละแหล่ง คนละความละเอียด คนละระบบพิกัด",
                                 13, MUTED, space=0), "m")
    for k, (q, ic, col, tint, tag) in enumerate([("พื้นที่ใดน่าจะพบซาก?", "map-pin", "rust", T2S, ("TASK 2", T2)),
                                                 ("ซากที่พบน่าจะเป็นอะไร ยุคไหน?", "search", "green", T1S, ("TASK 1", T1))]):
        x = 0.6 + k * 6.17
        s.rect(x, 5.3, 5.96, 1.5, tint, None, 0, 0.14)
        s.icon(ic, col, x + 0.28, 5.62, 0.62, 0.62)
        s.text(x + 1.1, 5.4, 4.6, 0.85, P(q, 21, INK, HEAD, space=0), "m")
        pill(s, x + 1.1, 6.25, "→ " + tag[0], tag[1], "FFFFFF", 12, HEAD, 0.34)
    D.append(s)

    # ===== 8 เป้าหมาย
    s = Slide(2, "โครงงานนี้ทำเพื่ออะไร", "// PURPOSE", (
        "โครงงานมีเป้าหมาย 3 ด้าน ด้านแรกคือจัดการข้อมูลบรรพชีวินวิทยา เชื่อมทะเบียนที่ไม่มีพิกัดเข้ากับแหล่งซาก เติมยุคที่ขาด "
        "และรวมข้อมูลเชิงพื้นที่ทั้งประเทศเป็นชุดเดียว ด้านที่สองคือช่วยผู้เชี่ยวชาญและกรมทรัพยากรธรณี ด้วยแผนที่โอกาสพบซากและรายชื่อสกุลที่น่าจะเป็น "
        "ด้านที่สามคือทำให้คนทั่วไปหันมาสนใจซากดึกดำบรรพ์ ผ่านเว็บที่ใช้ง่าย ทายชื่อได้จากหน้าตาที่คนทั่วไปบอกได้ และมีภาพสามมิติของซากข้ามยุค"))
    pil = [("database", "จัดการข้อมูลบรรพชีวินวิทยา", ["เชื่อมทะเบียนที่ไม่มีพิกัดเข้ากับแหล่งซาก", "เติมยุคที่ขาดของแหล่ง",
                                                       "รวมแผนที่ธรณี ความสูง ภาพดาวเทียม เป็นชุดเดียวทั้งประเทศ"]),
           ("target", "ช่วยผู้เชี่ยวชาญและ ทธ.", ["แผนที่โอกาสพบซากทั้งประเทศ สำหรับวางแผนสำรวจ", "เสนอชื่อสกุล 3 อันดับให้ผู้เชี่ยวชาญตรวจ",
                                                   "ยุคที่เติมมีระดับความเชื่อมั่นกำกับ"]),
           ("globe", "ให้คนทั่วไปสนใจซากดึกดำบรรพ์", ["เว็บไทย / อังกฤษ คลิกดูโอกาสพบซากได้ทุกจุด", "ทายชื่อจากหน้าตาที่คนทั่วไปบอกได้",
                                                         "ภาพสามมิติของซากข้ามยุค"])]
    for k, (ic, t, items) in enumerate(pil):
        x = 0.6 + k * 4.11
        s.rect(x, Y0, 3.91, 4.85, SURF, LINE, 1.0, 0.14)
        s.rect(x + 0.28, Y0 + 0.3, 1.0, 1.0, PILL, None, 0, shape="ellipse")
        s.icon(ic, "ink", x + 0.48, Y0 + 0.5, 0.6, 0.6)
        s.text(x + 0.28, Y0 + 1.5, 3.4, 0.9, P(t, 19, INK, HEAD, space=0, line=1.05), "t")
        bullets(s, x + 0.28, Y0 + 2.5, 3.4, 2.5, items, 14, space=10)
    D.append(s)

    # ===== 8b จุดแข็งที่สร้าง impact
    s = Slide(2, "จุดแข็งที่สร้าง Impact", "// IMPACT", (
        "จุดแข็งของงานที่สร้างผลจริง ข้อแรก ทธ. และนักสำรวจวางแผนได้คุ้มขึ้น ถ้าสำรวจแค่ร้อยละ 10 ของประเทศที่คะแนนสูงสุด จะครอบคลุมจุดพบซากอิสระได้ร้อยละ 48 "
        "ราว 5 เท่าของการเลือกแบบสุ่ม และใช้กำหนดพื้นที่เฝ้าระวังเพื่อคุ้มครองซากได้ ข้อสอง ผู้เชี่ยวชาญได้ชื่อสกุลที่เป็นไปได้ 3 อันดับสำหรับ 4,577 ชิ้นที่ยังไม่ระบุ "
        "ซึ่งช่วยลดเวลาจัดการทะเบียน ข้อสาม ตัวอย่างร้อยละ 92 ได้ตำแหน่งและบริบททางธรณี และแหล่งที่ไม่รู้ยุคลดเหลือ 129 แหล่ง "
        "ข้อสี่ ประชาชนที่พบซากบอกแค่หน้าตาก็ได้ชื่อที่น่าจะเป็น พร้อมคำแนะนำให้แจ้ง ทธ. ตามกฎหมาย ซึ่งช่วยทั้งการเรียนรู้และการอนุรักษ์ "
        "ข้อห้า แผนที่และเว็บเข้าถึงได้ทุกคน และข้อหก วิธีนี้นำไปใช้กับข้อมูลมรดกทางธรรมชาติอื่นที่คล้ายกันได้"))
    imp = [("ทธ. · นักสำรวจ", ACC, "≈ 5×", "วางแผนสำรวจคุ้มกว่า",
            "สำรวจแค่ 10% ของประเทศที่คะแนนสูงสุด เจอจุดพบซากอิสระ (PBDB) **48%** (สุ่มได้ ≈ 10%) · ใช้กำหนดพื้นที่เฝ้าระวังเพื่อคุ้มครองซาก"),
           ("ผู้เชี่ยวชาญ", GREEN, "4,577", "ชิ้นที่ยังไม่รู้สกุล ได้ชื่อเสนอ 3 อันดับ",
            "ชื่อจริงอยู่ใน 3 อันดับ **86.3%** (แหล่งเดิม) → ช่วยลดเวลาจัดการทะเบียน"),
           ("ทธ. · นักวิจัย", GREEN, "92%", "ตัวอย่างได้ตำแหน่งและบริบทธรณี",
            "จากทะเบียนที่ไม่มีพิกัด → ได้พิกัด หมวดหิน ยุค · แหล่งไม่รู้ยุคลด **352 → 129** พร้อมระดับความเชื่อมั่น"),
           ("ประชาชน", ACC, "72.3%", "บอกแค่หน้าตาก็ได้ชื่อที่น่าจะเป็น",
            "ไม่ต้องรู้ประเภทซาก (ชื่อจริงอยู่ใน 3 อันดับ แหล่งเดิม) + แนะนำให้**แจ้ง ทธ. ตามกฎหมาย** → ส่งเสริมการเรียนรู้และการอนุรักษ์"),
           ("ทุกคน · การศึกษา", ACC, "20,706", "ช่องแผนที่ครอบคลุมทั้งประเทศ",
            "เว็บไทย / อังกฤษ คลิกดูได้ทุกจุด + ภาพสามมิติของซากข้ามยุค → ซากดึกดำบรรพ์เข้าถึงง่ายขึ้น"),
           ("หน่วยงานอื่น", GREEN, "ต่อยอดได้", "ใช้กับข้อมูลมรดกทางธรรมชาติอื่น",
            "วิธีเชื่อมข้อมูลภาษาไทยที่ไม่มีพิกัดเข้ากับข้อมูลเชิงพื้นที่ ใช้กับข้อมูลทรัพยากรธรณีอื่นที่คล้ายกันได้")]
    for k, (who, col, v, head, body) in enumerate(imp):
        x, y = 0.6 + (k % 3) * 4.11, Y0 + (k // 3) * 2.66
        s.rect(x, y, 3.91, 2.5, SURF, LINE, 1.0, 0.14)
        pill(s, x + 0.2, y + 0.18, who, T2S if col == ACC else T1S, col, 11.5, HEADM, 0.32)
        s.text(x + 0.2, y + 0.55, 3.5, 0.66, P(v, 30, col, HEAD, space=0, line=1.0), "m")
        s.text(x + 0.2, y + 1.2, 3.5, 0.4, P(head, 14, INK, HEADM, space=0, line=1.0), "m")
        s.text(x + 0.2, y + 1.6, 3.55, 0.85, P(body, 11.5, INK2, bold_color=col, space=0, line=1.08))
    D.append(s)

    # ===== 9 สองงานที่เชื่อมกัน
    s = Slide(2, "2 งานที่ทำเชื่อมต่อกัน", "// TWO CONNECTED TASKS", (
        "โครงงานแบ่งเป็น 2 งาน งาน 1 รับข้อมูลของตัวอย่าง เช่น ชื่อสามัญ ประเภท ส่วนของซาก และสถานที่พบ หรือแค่หน้าตาที่คนทั่วไปบอกได้ "
        "แล้วทายสกุล 3 อันดับ ชนิดที่น่าจะเป็น และยุคของแหล่ง ใช้เติมข้อมูลที่ขาดในทะเบียน "
        "[คลิก] งาน 2 รับพิกัดใดก็ได้ในไทย แล้วบอกโอกาสพบซาก ประเภทซาก 3 อันดับ และบอกว่าจุดนั้นอยู่ในขอบเขตที่แบบจำลองเชื่อถือได้หรือไม่ "
        "สองงานใช้ข้อมูลที่เชื่อมแล้วและคุณลักษณะเชิงพื้นที่ชุดเดียวกัน และผลรวมกันอยู่บนเว็บเดียว"))
    two(s, "ทายชื่อและยุคของซาก", "คาดการณ์จากพิกัด (lat, long)")

    def io_col(x, col, tint, inp, out, use, ic_in, ic_out):
        s.rect(x, 2.2, HW, 1.25, SURF, LINE, 1.0, 0.12)
        s.icon(ic_in, "ink", x + 0.2, 2.35, 0.42, 0.42)
        s.text(x + 0.75, 2.27, 1.5, 0.5, P("ข้อมูลเข้า", 13, DIM, HEADM, space=0), "m")
        s.text(x + 0.2, 2.78, HW - 0.4, 0.66, P(inp, 13.5, INK2, space=0, line=1.08))
        arrow(s, x + HW / 2, 3.5, x + HW / 2, 3.86, col)
        s.rect(x, 3.92, HW, 1.62, tint, None, 0, 0.12)
        s.icon(ic_out, "green" if col == T1 else "rust", x + 0.2, 4.07, 0.42, 0.42)
        s.text(x + 0.75, 3.99, 1.5, 0.5, P("ผลลัพธ์", 13, col, HEADM, space=0), "m")
        s.text(x + 0.2, 4.5, HW - 0.4, 1.0, P(out, 13.5, INK, bold_color=col, space=2, line=1.1))
        s.text(x, 5.65, HW, 0.62, P(use, 13, MUTED, bold_color=INK, space=0, line=1.1))

    io_col(LX, T1, T1S, "ชื่อสามัญ ประเภท ส่วนของซาก จังหวัด / แหล่ง — หรือแค่ “หน้าตา” ที่คนทั่วไปบอกได้",
           "**สกุล 3 อันดับ** (131 สกุล) · ชนิดที่น่าจะเป็น\n**ยุคของแหล่ง** (11 ยุค)", "ใช้เติมข้อมูลที่ขาด: **4,577 ชิ้น**ที่ไม่รู้สกุล · **352 แหล่ง**ที่ไม่รู้ยุค",
           "file-text", "search")
    with s.reveal():
        right_head(s)
        io_col(RX, T2, T2S, "พิกัดใดก็ได้ในประเทศไทย", "**ขั้น A** โอกาสพบซาก + อันดับเทียบทั้งประเทศ\n**ขั้น B** ประเภทซาก 3 อันดับจาก 16 · **AOA** บอกว่าเชื่อถือได้ไหม",
               "ใช้ทำ **แผนที่โอกาสพบซาก 20,706 ช่อง** ทั่วประเทศ", "map-pin", "map")
        s.rect(X0, 6.36, XW, 0.5, PANEL, None, 0, 0.12, ps=P("เชื่อมกันด้วย **ข้อมูลที่เชื่อมแล้ว** และ **คุณลักษณะเชิงพื้นที่ชุดเดียวกัน** · ผลรวมอยู่บน **เว็บเดียว**",
                                                              13, INK2, bold_color=INK, align="c", space=0), inset=0.05)
    D.append(s)

    # ===== 10 ภาพรวมระบบ
    s = Slide(3, "ภาพรวมระบบทั้งหมด", "// SYSTEM DIAGRAM", (
        "นี่คือภาพรวมของระบบตั้งแต่ข้อมูลจนถึงเว็บ เริ่มจากข้อมูลต้นทาง 5 กลุ่ม เข้าสู่ขั้นจัดการข้อมูล คือเชื่อมตัวอย่างกับแหล่ง เติมยุค และกันข้อมูลรั่ว "
        "จากนั้นสร้างคุณลักษณะเชิงพื้นที่ 152 ตัวต่อจุดและคุณลักษณะเชิงภาษา แล้วป้อนให้แบบจำลองของงาน 1 สีเขียว และงาน 2 สีส้ม "
        "ผลทั้งหมดรวมเป็นไฟล์ผลลัพธ์สุดท้ายและกริด 5 กิโลเมตร แล้วเว็บต้นแบบเรียกใช้ผ่าน API"))
    cols = [("ข้อมูลต้นทาง", "database", [("ทะเบียน ทธ.\n8,521 ชิ้น", None), ("แหล่ง ทธ.\n518 แหล่ง", None), ("แผนที่ธรณี · DEM\nSentinel-2", None),
                                           ("แผ่นเปลือกโลกโบราณ", None), ("PBDB 290 · ThaiLex", None)]),
            ("จัดการข้อมูล", "link", [("เชื่อมตัวอย่าง → แหล่ง\n(fuzzy) 92%", None), ("เติมยุค\n352 → 129 แหล่ง", None), ("ตรวจคุณภาพ\n+ กันข้อมูลรั่ว", None)]),
            ("คุณลักษณะ", "layers", [("เชิงพื้นที่\n152 ตัว / จุด", None), ("เชิงภาษา: TF-IDF\nWangchanBERTa · ThaiLex", None)]),
            ("แบบจำลอง", "cog", [("TASK 1\nสกุล: XGBoost / LightGBM\nยุค: LightGBM", T1), ("TASK 2\nขั้น A: XGBoost\nขั้น B: Random Forest + AOA", T2)]),
            ("ผลลัพธ์รวม", "file-text", [("ไฟล์ผลลัพธ์สุดท้าย\n5 ไฟล์", None), ("กริด 5 กม.\n20,706 ช่อง", None)]),
            ("เว็บต้นแบบ", "monitor", [("แผนที่โอกาสพบซาก", None), ("วิเคราะห์จุดที่คลิก", None), ("ทายชื่อซาก", None)])]
    cw, gap = 1.8, 0.266
    top, bot = 2.2, 6.55
    for c, (name, ic, boxes) in enumerate(cols):
        x = X0 + c * (cw + gap)
        s.icon(ic, "ink", x, Y0 + 0.02, 0.36, 0.36, "l")
        s.text(x + 0.42, Y0 - 0.02, cw - 0.42, 0.44, P(f"{c + 1} {name}", 13, INK, HEADM, space=0, line=1.0), "m")
        n = len(boxes)
        bh = min(1.0, (bot - top - (n - 1) * 0.14) / n)
        y0 = top + ((bot - top) - (n * bh + (n - 1) * 0.14)) / 2
        for k, (t, col) in enumerate(boxes):
            y = y0 + k * (bh + 0.14)
            if col:
                box(s, x, y, cw, bh, t, T1S if col == T1 else T2S, col, 11.5, INK, lw=1.4)
            else:
                box(s, x, y, cw, bh, t, SURF if c else PANEL, LINE, 11.5, INK2)
        if c < len(cols) - 1:
            arrow(s, x + cw + 0.03, (top + bot) / 2, x + cw + gap - 0.03, (top + bot) / 2, DIM, 1.5)
    s.text(X0, 6.7, XW, 0.35, P("Python · scikit-learn · XGBoost · LightGBM · PyTorch (WangchanBERTa) · FastAPI · MapLibre", 11, DIM, align="c", space=0))
    D.append(s)

    # ===== 11 Target label
    s = Slide(4, "Target Label: แต่ละงานทายอะไร", "// TARGET LABEL", (
        "เริ่มจาก Target Label ของแต่ละงาน งาน 1 ทายสกุลซึ่งแยกได้จากชื่อวิทยาศาสตร์ มีป้ายกำกับ 3,944 ชิ้น 131 สกุลรวมกับกลุ่มอื่น ๆ "
        "ชนิดไม่ได้ฝึกแยก แต่ใช้ชนิดที่พบบ่อยที่สุดของสกุลนั้น ส่วนยุคของแหล่งมี 11 ยุค ฝึกจาก 166 แหล่งที่ ทธ. บันทึกยุคไว้ "
        "[คลิก] งาน 2 มี 2 ขั้น ขั้น A คือพบซากหรือไม่ โดยแหล่ง ทธ. 518 แหล่งเป็นคลาส 1 และจุดพื้นหลัง 5,000 จุดเป็นคลาส 0 "
        "ขั้น B คือประเภทซาก 16 ประเภท แหล่งหนึ่งมีได้หลายประเภท และแยก PBDB 290 จุดไว้ทดสอบอิสระ"))
    two(s, "ทายชื่อและยุคของซาก", "คาดการณ์จากพิกัด")

    def labels(x, items):
        y = 2.2
        for t, b, h in items:
            card(s, x, y, HW, h, t, b, tsize=15.5, bsize=13)
            y += h + 0.16

    labels(LX, [("สกุล (Genus) · หลายคลาส", "แยกจากชื่อวิทยาศาสตร์ → มีป้าย **3,944 ชิ้น** · **131 สกุล + “อื่น ๆ”** (132 คลาส) · เสนอ 3 อันดับ", 1.38),
                ("ชนิด (Species)", "ไม่ฝึกแยก: ใช้ชนิดที่พบบ่อยที่สุดของสกุลนั้นในแหล่ง หรือทั้งประเทศ", 1.12),
                ("ยุคของแหล่ง (Period) · 11 คลาส", "แคมเบรียน → ควอเทอร์นารี · ฝึกบน **166 แหล่ง**ที่ ทธ. บันทึกยุค (+ ค่าเติมที่ PBDB รองรับ)", 1.38)])
    with s.reveal():
        right_head(s)
        labels(RX, [("ขั้น A: พบซาก / ไม่พบ · สองคลาส", "1 = แหล่ง ทธ. **518 แหล่ง** · 0 = จุดพื้นหลัง **5,000 จุด** (2 ชุด: สุ่มทั่วประเทศ / ถ่วงการสำรวจ)", 1.38),
                    ("ขั้น B: ประเภทซาก · หลายป้าย", "**16 ประเภท** เช่น ไดโนเสาร์ แบรคิโอพอด ปะการัง · แหล่งหนึ่งมีได้หลายประเภท (490 แหล่ง)", 1.12),
                    ("ชุดทดสอบอิสระ", "PBDB **290 collection** — ไม่ใช้ฝึกเลย · จุดพื้นหลังไม่ใช่จุดที่ยืนยันว่าไม่มีซาก", 1.38)])
    D.append(s)

    # ===== 12 ชุดข้อมูล
    s = Slide(4, "Data Sources: ข้อมูลที่ใช้ 9 ชุด", "// DATASETS", (
        "ข้อมูลที่ใช้มี 9 ชุด ชุดหลักคือทะเบียนตัวอย่างและแหล่งซากของ ทธ. ซึ่งไม่มีคีย์ร่วมกัน "
        "ข้อมูลเชิงพื้นที่ได้แก่แผนที่ธรณีวิทยา แบบจำลองความสูง Copernicus DEM ภาพดาวเทียม Sentinel-2 หน้าแล้ง 188 ภาพ และแบบจำลองแผ่นเปลือกโลกโบราณ 15 ช่วงเวลา "
        "PBDB เป็นฐานข้อมูลนานาชาติที่ใช้ทั้งเติมยุคและเป็นชุดทดสอบอิสระ ส่วน ThaiLex เป็นพจนานุกรมหน่วยหินของไทย"))
    rows = [["ชุดข้อมูล", "ขนาด / ความละเอียด", "ใช้ทำอะไร"],
            ["ทะเบียนซากดึกดำบรรพ์ ทธ.", "8,521 ชิ้น · 11 คอลัมน์ · ไม่มีพิกัด", "ป้ายสกุล / ชนิด · คุณลักษณะเชิงภาษา"],
            ["แหล่งซากดึกดำบรรพ์ ทธ.", "518 แหล่ง · 54 คอลัมน์ · บันทึกยุค 166", "จุดพบซาก (งาน 2) · ป้ายยุค"],
            ["แผนที่ธรณีวิทยา 1:250,000 (ทธ.)", "เวกเตอร์ทั้งประเทศ", "คุณลักษณะธรณี · หมวดหินของพิกัดใหม่"],
            ["Paleobiology Database (PBDB)", "290 collection ในไทย", "เติมยุค · ชุดทดสอบอิสระ"],
            ["ThaiLex", "หมวด / กลุ่มหินของไทย", "ชนิดหินและสภาพแวดล้อม"],
            ["Copernicus DEM GLO-30", "30 เมตร", "คุณลักษณะภูมิประเทศ"],
            ["Sentinel-2 L2A", "10–20 เมตร · ภาพหน้าแล้ง 188 ภาพ", "ดัชนีสเปกตรัม"],
            ["PALEOMAP · Merdith et al.", "15 ช่วงเวลา (480 ล้านปี – ปัจจุบัน)", "ภูมิศาสตร์โบราณ"],
            ["ขอบเขตจังหวัด (ทธ.)", "77 จังหวัด", "หาจังหวัดจากพิกัด · แผนที่รายจังหวัด"]]
    s.table(X0, Y0, [3.75, 4.0, 4.38], rows, size=13, rowh=0.5, head=INK, hl={1: T1S, 2: T1S})
    s.text(X0, 6.72, XW, 0.35, P("แถวไฮไลต์ = ข้อมูลหลักสองชุดของ ทธ. ที่ไม่มีคีย์ร่วมกัน", 11, DIM, space=0))
    D.append(s)

    # ===== 13 ทะเบียนตัวอย่าง
    s = Slide(4, "ทะเบียนตัวอย่าง ทธ.: 8,521 ชิ้น 11 คอลัมน์", "// DATA DICTIONARY · SPECIMENS", (
        "ทะเบียนตัวอย่างมี 11 คอลัมน์ คอลัมน์ที่ใช้หลักคือชื่อวิทยาศาสตร์ ชื่อสามัญ กลุ่มซาก ประเภทซาก ส่วนของซาก และชื่อจังหวัด อำเภอ ตำบล "
        "สังเกตว่าส่วนของซากเป็นข้อความอิสระถึง 991 แบบ และตำบลมีค่าเพียงร้อยละ 72 ส่วนรหัส ผู้ค้นพบ และเลขทะเบียน ไม่ใช้ในแบบจำลอง "
        "กราฟขวาจาก notebook สำรวจข้อมูลคือประเภทซากที่พบมากสุด 20 อันดับ สีน้ำเงินคือสัตว์มีกระดูกสันหลัง สีส้มคือสัตว์ไม่มีกระดูกสันหลัง ไดโนเสาร์มากที่สุด 3,091 ชิ้น สิ่งสำคัญคือทะเบียนไม่มีพิกัด จึงต้องเชื่อมกับแหล่งก่อนใช้ข้อมูลเชิงพื้นที่"))
    rows = [["คอลัมน์", "ความหมาย", "ค่าที่พบ", "มีค่า"],
            ["sci_name", "ชื่อวิทยาศาสตร์", "740 ค่า · 5 รูปแบบ", "94%"],
            ["com_name", "ชื่อสามัญภาษาไทย", "141 ค่า เช่น ไดโนเสาร์", "99%"],
            ["f_group", "กลุ่มซาก", "5 กลุ่ม", "100%"],
            ["f_type", "ประเภทซาก", "62 ค่า", "100%"],
            ["f_part", "ส่วนของซาก", "ข้อความอิสระ 991 แบบ", "93%"],
            ["province", "จังหวัด", "58 จังหวัด (เติมจากอำเภอ → 98%)", "92%"],
            ["district", "อำเภอ", "236 ค่า", "92%"],
            ["tambom", "ตำบล", "279 ค่า", "72%"],
            ["fossil_id · dis_name · regis_code", "รหัส / ผู้ค้นพบ / เลขทะเบียน", "ไม่ใช้ในแบบจำลอง", "—"]]
    s.table(X0, Y0, [2.2, 1.75, 2.35, 0.65], rows, size=12, rowh=0.47, head=INK, align=["l", "l", "l", "r"])
    s.text(7.8, Y0, 4.93, 0.4, P("ประเภทซากที่พบบ่อย (f_type)", 14, INK, HEADM, space=0), "m")
    frame_img(s, C / "nb_ftype.png", 7.8, 2.05, 4.93, 3.28)
    s.rect(7.8, 5.5, 4.93, 1.15, None, ACC, 1.2, 0.12)
    s.text(7.97, 5.53, 4.6, 1.09, P("**ไม่มีพิกัด ไม่มีรหัสแหล่ง** → ต้องเชื่อมกับตารางแหล่งก่อน จึงจะได้ตำแหน่ง หมวดหิน และยุคของตัวอย่าง",
                                   13.5, INK2, bold_color=ACC, space=0), "m")
    D.append(s)

    # ===== 14 ตารางแหล่ง
    s = Slide(4, "แหล่งซาก ทธ.: 518 แหล่ง 54 คอลัมน์", "// DATA DICTIONARY · SITES", (
        "ตารางแหล่งมี 518 แหล่ง 54 คอลัมน์ ทุกแหล่งมีพิกัด ชื่อแหล่ง และคำบรรยายซาก แต่หมวดหินมีเพียงร้อยละ 33 และยุคมีเพียงร้อยละ 32 "
        "ยุคเขียนไว้ 27 แบบ เช่นชื่อยุค ช่วงย่อย หรือช่วงคร่อมยุค จึงจัดให้เป็น 11 ยุคก่อนใช้ และมีอีก 18 คอลัมน์ที่ว่างทุกแหล่ง "
        "กราฟขวาคือยุคของ 166 แหล่งที่ ทธ. บันทึกไว้ ส่วนใหญ่เป็นเพอร์เมียน 88 แหล่ง อีก 352 แหล่งหรือร้อยละ 68 ไม่มียุค "
        "รหัสแหล่งของ ทธ. ซ้ำกัน 8 รหัส จึงใช้ OBJECTID เป็นกุญแจแทน"))
    rows = [["คอลัมน์", "ความหมาย", "ค่าที่พบ", "มีค่า"],
            ["OBJECTID", "รหัสระเบียน", "ไม่ซ้ำ → ใช้เป็นกุญแจ", "100%"],
            ["NAME_TH", "ชื่อแหล่ง", "เช่น ไดโนเสาร์บ้านทุ่งสว่าง", "100%"],
            ["PROVINCE / DISTRICT / TAMBON", "ที่ตั้ง", "39 จังหวัด 151 อำเภอ 270 ตำบล", "100%"],
            ["จุดพิกัด", "ตำแหน่ง (WGS 84)", "ลองจิจูด 98.40–105.29", "100%"],
            ["FOS_DES_TH", "คำบรรยายซาก", "ข้อความอิสระ 390 แบบ", "97%"],
            ["FORMATION", "หมวดหิน", "51 ค่า เช่น อุ้มลูก เขาพระ", "33%"],
            ["PERIODFROM", "ยุค", "27 แบบ → จัดเป็น 11 ยุค", "32%"],
            ["อีก 18 คอลัมน์", "ชื่ออังกฤษ ผู้ค้นพบ ฯลฯ", "ว่างทุกแหล่ง", "0%"]]
    s.table(X0, Y0, [2.85, 1.9, 2.75, 0.75], rows, size=12.5, rowh=0.5, head=INK, align=["l", "l", "l", "r"])
    s.text(9.15, Y0, 3.58, 0.4, P("ยุคที่ ทธ. บันทึกไว้", 14, INK, HEADM, space=0), "m")
    frame_img(s, C / "nb_period_recorded.png", 9.15, 2.05, 3.58, 3.75)
    s.text(9.15, 5.88, 3.58, 0.45, P("**352 แหล่ง (68%)** ไม่มีข้อมูลยุค", 14, INK2, bold_color=ACC, space=0), "m")
    s.text(9.15, 6.36, 3.58, 0.45, P("SITE_CODE ซ้ำ 8 รหัส (16 แหล่ง) จึงไม่ใช้เป็นกุญแจ", 11.5, DIM, space=0), "m")
    D.append(s)

    # ===== 15 ความยุ่งของข้อมูล
    s = Slide(4, "ความยุ่งของข้อมูลจริง", "// DATA QUALITY", (
        "ตัวอย่างความยุ่งของข้อมูลจริงที่ต้องจัดการ เช่น ประเภทเดียวกันสะกดหลายแบบ ชื่อเขตในทะเบียนสะกดต่างจากชื่อเขตจริง "
        "อำเภอภูเวียงถูกแบ่งเป็นอำเภอเวียงเก่า และชื่อสถานที่อย่างถ้ำค้างคาวมีชื่อสัตว์แต่ไม่ได้หมายถึงซากค้างคาว "
        "ถ้าไม่จัดการ แหล่งสัตว์เลี้ยงลูกด้วยนมจะได้ป้ายผิดถึง 14 จาก 25 แหล่ง นอกจากนี้ยุคเขียนได้ 27 แบบ และมีชื่อสกุลปนอยู่ในช่องส่วนของซาก ซึ่งเป็นการรั่วของคำตอบ"))
    ex = [("สะกดหลายแบบ", "“ฟอแรมมินิเฟอร์”\n“ฟอแรมินิเฟอร์”\n“ฟอแรมินิเฟอรา”", "รวมเป็นชื่อเดียวก่อนใช้"),
          ("ชื่อเขตสะกดต่าง", "ทะเบียน “ร่อนพิพูลย์” แต่ชื่อเขต “ร่อนพิบูลย์”", "จับคู่แบบคลุมเครือ (คะแนน 91)"),
          ("เขตการปกครองเปลี่ยน", "อำเภอภูเวียงถูกแบ่งเป็นอำเภอเวียงเก่า", "ค้นชื่ออำเภอ / ตำบลในชื่อแหล่งแทน"),
          ("ชื่อสถานที่มีชื่อสัตว์", "“เขาประทับช้าง” “ถ้ำค้างคาว” ไม่ใช่ซากช้างหรือค้างคาว", "ตัดชื่อสถานที่ก่อนทำป้ายประเภท (เดิมผิด 14 จาก 25 แหล่ง)"),
          ("ยุคเขียน 27 แบบ", "“ไทรแอสสิก / ไทรแอสซิก” “จูแรสสิกตอนปลายถึงครีเทเชียสตอนต้น”", "จัดเป็น 11 ยุค"),
          ("คำตอบปนในข้อมูล", "ชื่อสกุลอยู่ในช่องส่วนของซาก (9 ชิ้น)", "ลบชื่อสกุลก่อนสร้างคุณลักษณะ")]
    for k, (t, f, fix) in enumerate(ex):
        x, y = 0.6 + (k % 3) * 4.11, Y0 + (k // 3) * 2.62
        s.rect(x, y, 3.91, 2.42, SURF, LINE, 1.0, 0.14)
        s.text(x + 0.2, y + 0.14, 3.5, 0.42, P(t, 15.5, INK, HEADM, space=0), "m")
        s.text(x + 0.2, y + 0.62, 3.5, 1.0, P(f, 13, INK2, space=0, line=1.1))
        s.rect(x + 0.2, y + 1.65, 3.51, 0.6, T1S, None, 0, 0.1, ps=P("→ " + fix, 12, T1, HEADM, space=0, line=1.05), valign="m", inset=0.1)
    D.append(s)

    # ===== 16a เชื่อมข้อมูล 1/3: merge แบบตรงตัว (ตัวเลขจาก notebooks/02_merge_fossils ส่วน 1–2)
    s = Slide(5, "Merge 1/3: จับคู่ชื่อเขตแบบตรงตัว (A · B · C)", "// NORMAL MERGE", (
        "ขั้นแรกของการเตรียมข้อมูลคือเชื่อมตัวอย่างแต่ละชิ้นเข้ากับแหล่ง เพราะทะเบียนไม่มีพิกัด ใช้ได้แค่ชื่อจังหวัด อำเภอ ตำบล "
        "เริ่มจาก merge แบบตรงตัว 3 ระดับ ก่อน merge ทำความสะอาดชื่อเขตก่อน ซึ่งทำให้จับคู่ได้เพิ่มราว 500 ถึง 600 ชิ้นในทุกระดับ "
        "ระดับตำบลแคบที่สุด เฉลี่ย 3.4 แหล่งต่อชิ้น แต่จับคู่ได้แค่ร้อยละ 44 ระดับอำเภอได้ร้อยละ 74 "
        "ระดับจังหวัดได้ร้อยละ 92 แต่แถวหลัง merge พุ่งเป็น 160,332 แถว เฉลี่ย 20 แหล่งต่อชิ้น ซึ่งแทบไม่บอกว่าตัวอย่างมาจากแหล่งไหน"))
    LW = 5.6
    merges = [("A · ระดับตำบล", "จังหวัด + อำเภอ + ตำบล", ["แถวหลัง merge: **12,794**", "match: 3,750 ชิ้น (44.0%) · ทำความสะอาดชื่อ +551",
                                                          "แหล่ง / ชิ้น: เฉลี่ย 3.4 · สูงสุด 11",
                                                          "แคบสุด แต่ตำบลหาย / สะกดต่าง → หลุด"]),
              ("B · ระดับอำเภอ", "จังหวัด + อำเภอ", ["แถวหลัง merge: **38,429**", "match: 6,341 ชิ้น (74.4%) · ทำความสะอาดชื่อ +628",
                                                    "แหล่ง / ชิ้น: เฉลี่ย 6.1 · สูงสุด 26", "หลุดตัวอย่างที่ไม่มีอำเภอ"]),
              ("C · ระดับจังหวัด", "จังหวัดอย่างเดียว", ["แถวหลัง merge: **160,332**", "match: 7,853 ชิ้น (92.2%) · ทำความสะอาดชื่อ +522",
                                                       "แหล่ง / ชิ้น: เฉลี่ย 20.4 · สูงสุด 57", "ได้แหล่งทั้งจังหวัด · 9% อยู่นอกอำเภอที่บันทึก"])]
    for k, (t, key, lines) in enumerate(merges):
        y = Y0 + k * 1.76
        s.text(X0, y, LW, 0.42, P(f"• {t}  {{c:{DIM}}}on {key}{{/c}}", 17, INK, HEAD, space=0) , "m")
        keybox(s, X0, y + 0.46, LW, 1.18, lines, 11, head=None)
    code_panel(s, 6.45, Y0, 6.28, 4.72, """
# ทำความสะอาดชื่อเขตก่อน: ตัดคำนำหน้า จังหวัด / อ. / ต.
# และแก้คำผิดที่พบ เช่น เมืองสตูน → เมืองสตูล
KEYCOLS = {
    "A": ["province_key", "district_key", "tambon_key"],
    "B": ["province_key", "district_key"],
    "C": ["province_key"],
}

def exact_candidates(method):
    keys = KEYCOLS[method]
    m = SPEC_KEYS.dropna(subset=keys).merge(
        SITE_KEYS.dropna(subset=keys),
        on=keys, how="inner")   # ชื่อต้องตรงทุกตัวอักษร
    return m[["fossil_id", "OBJECTID"]]

cands = {m: exact_candidates(m) for m in "ABC"}

# A→B→C: ใช้ระดับที่ละเอียดที่สุดที่เจอ (ถอยระดับ)
taken, parts = set(), []
for m in "ABC":
    c = cands[m][~cands[m].fossil_id.isin(taken)]
    parts.append(c)
    taken |= set(c.fossil_id)
cands["A→B→C"] = pd.concat(parts)
""", 10, "Normal Merge", "notebooks/02_merge_fossils.ipynb")
    D.append(s)

    # ===== 16b เชื่อมข้อมูล 2/3: merge คลุมเครือตามลำดับชั้น (วิธีที่เลือก)
    s = Slide(5, "Merge 2/3: จับคู่คลุมเครือตามลำดับชั้น (D · เลือกใช้)", "// FUZZY HIERARCHICAL MERGE", (
        "วิธีที่เลือกคือ merge แบบคลุมเครือตามลำดับชั้น ไล่จากจังหวัด ไปอำเภอที่อยู่ในจังหวัดนั้น แล้วไปตำบลที่อยู่ในอำเภอนั้น "
        "ยอมให้สะกดต่างกันได้เล็กน้อย เช่น ร่อนพิพูลย์กับร่อนพิบูลย์ได้คะแนน 91 ถ้าระดับไหนไม่ตรง จะค้นชื่ออำเภอหรือตำบลในชื่อแหล่งแทน เช่นกรณีภูเวียง "
        "ผลคือจับคู่ได้ร้อยละ 92 เท่ากับระดับจังหวัด แต่เหลือแหล่งที่เป็นไปได้เฉลี่ย 5.7 แหล่งต่อชิ้น "
        "และเมื่อเทียบกับแบบตรงตัวที่ถอยระดับ ย้ายตัวอย่างราว 400 ชิ้นจากระดับจังหวัดไปยังระดับที่ละเอียดขึ้น"))
    s.text(X0, Y0, LW, 0.42, P(f"• D · Fuzzy Hierarchical  {{c:{DIM}}}เลือกใช้{{/c}}", 17, INK, HEAD, space=0), "m")
    keybox(s, X0, Y0 + 0.46, LW, 1.55, ["แถวหลัง merge: **44,906** (C = 160,332)", "match: 7,853 ชิ้น (92.2%) เท่ากับ C",
                                        "แหล่ง / ชิ้น: เฉลี่ย **5.7** (C = 20.4)", "เกณฑ์: จังหวัด ≥ 90 · อำเภอ ≥ 85 · ตำบล ≥ 75",
                                        "ไม่เจอ → ค้นชื่ออำเภอ / ตำบลในชื่อแหล่ง ≥ 90"], 11)
    s.text(X0, Y0 + 2.14, LW, 0.42, P(f"• match_level  {{c:{DIM}}}เทียบแบบตรงตัวถอยระดับ{{/c}}", 17, INK, HEAD, space=0), "m")
    keybox(s, X0, Y0 + 2.6, LW, 1.42, ["tambon     3,750 → 3,811   (+61)", "locality       0 → **414**   (+414)",
                                       "district   2,591 → 2,522   (−69)", "province   1,512 → **1,106**   (−406)"], 11,
           head="level      A→B→C → D")
    keybox(s, X0, Y0 + 4.14, LW, 1.08, ["ร่อนพิพูลย์ → ร่อนพิบูลย์  (คะแนน 91)", "ภูเวียง ~ “แหล่งขุดค้นไดโนเสาร์**ภูเวียง** หลุมที่ 3”"], 11,
           head="ตัวอย่างที่จับคู่ได้:")
    code_panel(s, 6.45, Y0, 6.28, 4.72, """
def fuzzy_match(prov, dist, tamb):
    # 1) จังหวัด: ตรงตัว หรือ fuzzy ≥ 90
    P = prov if prov in SITES_BY_PROV else process.extractOne(
        prov, list(SITES_BY_PROV), scorer=fuzz.ratio,
        score_cutoff=90)[0]
    S = SITES_BY_PROV[P]
    # 2) อำเภอ: ค้นเฉพาะในจังหวัดนั้น ≥ 85
    D = process.extractOne(dist, S.district_key.unique(),
                           scorer=name_sim, score_cutoff=85)
    if D:
        SD = S[S.district_key.eq(D[0])]
        # 3) ตำบล: ค้นเฉพาะในอำเภอนั้น ≥ 75
        T = process.extractOne(tamb, SD.tambon_key.unique(),
                               scorer=name_sim, score_cutoff=75)
        if T:
            return SD[SD.tambon_key.eq(T[0])], "tambon"
        # 4) ไม่เจอ → หาชื่อตำบลในชื่อแหล่ง ≥ 90
        hit = locality_hits(SD, [tamb], 90)
        if hit is not None:
            return hit, "locality"
        return SD, "district"
    hit = locality_hits(S, [dist, tamb], 90)
    if hit is not None:
        return hit, "locality"
    return S, "province"
""", 10, "Fuzzy Hierarchical Merge", "notebooks/02_merge_fossils.ipynb")
    D.append(s)

    # ===== 16c เชื่อมข้อมูล 3/3: จัดอันดับแหล่ง → fossil_list_merged.csv
    s = Slide(5, "Merge 3/3: จัดอันดับแหล่ง → fossil_list_merged.csv", "// RANKING + OUTPUT", (
        "ตัวอย่างหนึ่งชิ้นยังมีแหล่งที่เป็นไปได้หลายแหล่ง จึงให้คะแนนทุกคู่ด้วยหลักฐาน 3 อย่าง คือคำในชื่อหรือคำบรรยายแหล่ง กลุ่มซากตรงกัน และชื่อเขตคล้ายกัน "
        "แล้วเลือกแหล่งอันดับ 1 และติดระดับความเชื่อมั่นจากระดับที่จับคู่ได้และการเสมอกัน ผลลัพธ์คือไฟล์ fossil_list_merged 8,521 แถว "
        "ความเชื่อมั่นสูงร้อยละ 21 กลางร้อยละ 49 ต่ำร้อยละ 23 และเชื่อมไม่ได้ร้อยละ 8 "
        "เมื่อวัดกับป้ายกำกับแบบเงิน 957 ชิ้น แหล่งอันดับ 1 ตรงร้อยละ 48 และห่างแหล่งที่ถูกไม่เกิน 10 กิโลเมตรร้อยละ 91 "
        "และเมื่อเปลี่ยนน้ำหนัก 36 ชุด ผลแทบไม่เปลี่ยน"))
    s.text(X0, Y0, LW, 0.42, P("• fossil_list_merged.csv", 17, INK, HEAD, space=0), "m")
    keybox(s, X0, Y0 + 0.46, LW, 1.32, ["**8,521** แถว × 39 คอลัมน์ · 1 แถวต่อ 1 ชิ้น", "score = 0.5·kw + 0.3·grp + 0.2·adm",
                                        "เลือกแหล่งอันดับ 1 · เสมอกัน → บันทึก n_top", "แหล่งที่มีตัวอย่างเชื่อม ≥ 1 ชิ้น: 438 / 518"], 11)
    s.text(X0, Y0 + 1.9, LW, 0.42, P(f"• confidence  {{c:{DIM}}}ความเชื่อมั่นของการเชื่อม{{/c}}", 17, INK, HEAD, space=0), "m")
    keybox(s, X0, Y0 + 2.36, LW, 1.3, ["high     1,769  (21%)  ตำบล / ชื่อแหล่ง ไม่เสมอ", "medium   4,144  (49%)",
                                       "low      1,940  (23%)  อำเภอที่เสมอ / จังหวัด", "none       668   (8%)  เชื่อมไม่ได้"], 11, head=None)
    s.text(X0, Y0 + 3.78, LW, 0.42, P(f"• วัดผล  {{c:{DIM}}}ป้ายกำกับแบบเงิน 957 ชิ้น{{/c}}", 17, INK, HEAD, space=0), "m")
    keybox(s, X0, Y0 + 4.24, LW, 0.98, ["ตรงแหล่ง **48.0%** · ห่าง ≤ 10 กม. **90.8%** · ตรงอำเภอ 100%",
                                        "เปลี่ยนน้ำหนัก 36 ชุด → ตรงแหล่ง 47.9–48.0%"], 11, head=None)
    code_panel(s, 6.45, Y0, 6.28, 4.72, """
# 1) ให้คะแนนทุกคู่ (ตัวอย่าง, แหล่งที่เป็นไปได้)
P["kw"] = [kw_score(t, o)        # คำในชื่อ / คำบรรยายแหล่ง
           for t, o in zip(P.terms, P.OBJECTID)]
P["kw"] = P.kw.where(P.kw >= 0.8, 0.0)
P["grp"] = [float(SITE.at[o, g])  # กลุ่มซากตรงกัน
            for g, o in zip(P.f_group, P.OBJECTID)]
P["adm"] = [adm_score(d, t, o)    # ชื่ออำเภอ / ตำบลคล้ายกัน
            for d, t, o in zip(P.district_key,
                               P.tambon_key, P.OBJECTID)]
P["score"] = 0.5 * P.kw + 0.3 * P.grp + 0.2 * P.adm

# 2) เลือกแหล่งอันดับ 1 ของแต่ละชิ้น
cD["rank"] = cD.groupby("fossil_id").score.rank(
    method="min", ascending=False)

# 3) ระดับความเชื่อมั่น
def confidence(r):
    fine = r.match_level in ("tambon", "locality")
    if fine and r.n_top == 1:
        return "high"
    if fine or (r.match_level == "district" and r.n_top == 1):
        return "medium"
    return "low"
""", 10, "Ranking + Confidence", "notebooks/02_merge_fossils.ipynb")
    D.append(s)

    # ===== 17 ตัวอย่างการเชื่อม
    s = Slide(5, "ตัวอย่างจริง: เศษกระดูกซอโรพอด อ.หนองบัวแดง", "// WORKED EXAMPLE", (
        "ตัวอย่างจริง ระเบียนนี้เป็นเศษกระดูกซอโรพอด บันทึกว่าพบที่อำเภอหนองบัวแดง ตำบลท่าใหญ่ แต่ไม่มีพิกัด "
        "ในอำเภอนี้มีแหล่งที่เป็นไปได้ 3 แห่ง ระบบให้คะแนนด้วยสมการ 0.5 คูณ kw บวก 0.3 คูณ grp บวก 0.2 คูณ adm "
        "kw คือคะแนนที่ชื่อสามัญหรือประเภทซากปรากฏในชื่อหรือคำบรรยายแหล่ง grp คือแหล่งมีกลุ่มซากเดียวกัน และ adm คือความคล้ายของชื่ออำเภอและตำบล "
        "แหล่งไดโนเสาร์บ้านทุ่งสว่างได้ 0.90 สูงสุด ตัวอย่างนี้จึงได้พิกัด หมวดหินน้ำพอง และยุคไทรแอสสิก"))
    s.rect(X0, Y0, 3.55, 5.2, SURF, LINE, 1.0, 0.14)
    s.text(X0 + 0.2, Y0 + 0.12, 3.2, 0.42, P("ระเบียนในทะเบียน", 15, INK, HEADM, space=0), "m")
    fields = [("รหัส", "20210300767"), ("ชื่อวิทยาศาสตร์", "N/A (Dinosaur)"), ("ชื่อสามัญ", "ซอโรพอด"), ("ประเภท", "ไดโนเสาร์"),
              ("กลุ่ม", "Vertebrate"), ("ส่วนของซาก", "เศษกระดูก"), ("จังหวัด", "ชัยภูมิ"), ("อำเภอ", "หนองบัวแดง"), ("ตำบล", "ท่าใหญ่")]
    for k, (a, b) in enumerate(fields):
        y = Y0 + 0.62 + k * 0.42
        s.text(X0 + 0.2, y, 1.45, 0.4, P(a, 12.5, DIM, space=0), "m")
        s.text(X0 + 1.65, y, 1.8, 0.4, P(b, 12.5, INK, BODYM, space=0), "m")
    s.text(X0 + 0.2, Y0 + 4.42, 3.2, 0.5, P("พิกัด: **ไม่มี**", 13, INK2, bold_color=ACC, space=0), "m")
    arrow(s, 4.2, 4.2, 4.5, 4.2, DIM, 1.5)
    s.rect(4.55, Y0, 4.0, 5.2, PANEL, None, 0, 0.14)
    s.text(4.75, Y0 + 0.12, 3.6, 0.42, P("คะแนนของแหล่งที่เป็นไปได้", 15, INK, HEADM, space=0), "m")
    s.rect(4.75, Y0 + 0.65, 3.6, 0.7, SURF, None, 0, 0.1, ps=P("score = 0.5·kw + 0.3·grp + 0.2·adm", 15, INK, HEAD, "c", space=0), inset=0.04)
    legend = [("kw", "ชื่อสามัญ / ประเภทของตัวอย่างอยู่ในชื่อหรือคำบรรยายแหล่ง"), ("grp", "แหล่งมีกลุ่มซากเดียวกับตัวอย่าง"),
              ("adm", "ชื่ออำเภอและตำบลของแหล่งคล้ายกับที่บันทึกไว้")]
    for k, (a, b) in enumerate(legend):
        y = Y0 + 1.55 + k * 0.85
        s.text(4.75, y, 0.62, 0.4, P(a, 14, ACC, HEAD, space=0), "t")
        s.text(5.4, y, 2.95, 0.8, P(b, 12.5, INK2, space=0, line=1.08), "t")
    s.text(4.75, Y0 + 4.2, 3.6, 0.8, P("แหล่งใน อ.หนองบัวแดงที่เป็นไปได้ **3 แห่ง**", 12.5, MUTED, bold_color=INK, space=0), "m")
    arrow(s, 8.6, 4.2, 8.85, 4.2, DIM, 1.5)
    cands = [("ไดโนเสาร์บ้านทุ่งสว่าง", "1 · 1 · 0.50", 0.90, True), ("รอยตีนไดโนเสาร์บ้านโนนตูม", "1 · 0 · 0.50", 0.60, False),
             ("แบรคิโอพอดห้วยหินลับ", "0 · 0 · 0.56", 0.11, False)]
    for k, (n, comp, sc, win) in enumerate(cands):
        y = Y0 + k * 1.08
        s.rect(8.9, y, 3.83, 0.96, T1S if win else SURF, GREEN if win else LINE, 1.4 if win else 1.0, 0.12)
        s.text(9.05, y + 0.05, 2.75, 0.42, P(n, 13, INK, HEADM, space=0), "m")
        s.text(11.75, y + 0.05, 0.85, 0.42, P(f"{sc:.2f}", 16, GREEN if win else MUTED, HEAD, "r", space=0), "m")
        s.text(9.05, y + 0.5, 2.2, 0.36, P(f"kw · grp · adm = {comp}", 10.5, DIM, space=0), "m")
        s.rect(11.3, y + 0.6, 1.3, 0.16, PANEL, None, 0, 0.04)
        s.rect(11.3, y + 0.6, 1.3 * sc, 0.16, GREEN if win else BASE, None, 0, 0.04)
    s.rect(8.9, Y0 + 3.4, 3.83, 1.8, None, GREEN, 1.2, 0.12)
    s.text(9.05, Y0 + 3.45, 3.55, 1.7, P(["**ได้จากการเชื่อม**", "พิกัด 16.085, 101.632", "หมวดหินน้ำพอง · ยุคไทรแอสสิก",
                                           "ความเชื่อมั่นปานกลาง (เชื่อมระดับอำเภอ)"], 12.5, INK2, bold_color=GREEN, space=2), "m")
    D.append(s)

    # ===== 18 เติมยุค
    s = Slide(5, "เติมยุคให้แหล่ง: ไม่ทราบยุค 352 → 129 แหล่ง", "// PERIOD FILLING", (
        "ขั้นต่อมาคือเติมยุคให้ 352 แหล่งที่ไม่มีข้อมูล ใช้หลักฐาน 2 ทาง ทางแรกจาก PBDB คือชื่อสกุลในคำบรรยายแหล่ง และ collection ที่ใกล้ที่สุดภายใน 10 กิโลเมตร "
        "ทางที่สองจากชื่อซากในแหล่ง โดยหาแหล่งอ้างอิงที่มีชื่อซากตรงกันแล้วโหวตยุค ทดลองจับคู่ชื่อ 3 แบบ แบบผสมดีที่สุด "
        "ถูกจากทั้งหมดร้อยละ 75 กับแหล่ง ทธ. และร้อยละ 89.9 กับ PBDB เทียบกับการเดาเพอร์เมียนทุกแหล่งที่ได้ราวร้อยละ 50 "
        "สุดท้ายแหล่งที่ไม่ทราบยุคเหลือ 129 แหล่ง และทุกค่าที่เติมมีระดับความเชื่อมั่นกำกับ"))
    box(s, X0, Y0, 5.6, 0.62, "แหล่งที่ไม่มียุค **352 แหล่ง**", T2S, None, 14, INK)
    for k, (t, b) in enumerate([("ทางที่ 1 · PBDB", "ชื่อสกุลในคำบรรยาย (taxon)\ncollection ใกล้สุด ≤ 10 กม. (nearby)"),
                                ("ทางที่ 2 · ชื่อซากในแหล่ง", "จับคู่ชื่อกับแหล่งอ้างอิง\nแล้วโหวตยุค")]):
        x = X0 + k * 2.9
        arrow(s, x + 1.35, Y0 + 0.66, x + 1.35, Y0 + 0.98, DIM, 1.4)
        card(s, x, Y0 + 1.02, 2.7, 1.55, t, b, tsize=14, bsize=12.5)
        arrow(s, x + 1.35, Y0 + 2.62, x + 1.35, Y0 + 2.94, DIM, 1.4)
    box(s, X0, Y0 + 2.98, 5.6, 0.82, "รวมเป็นยุคสุดท้าย + ระดับความเชื่อมั่น\n(บันทึก / สูง / ปานกลาง / ต่ำ)", SURF, LINE, 13, INK)
    arrow(s, X0 + 2.8, Y0 + 3.84, X0 + 2.8, Y0 + 4.16, DIM, 1.4)
    box(s, X0, Y0 + 4.2, 5.6, 0.62, "เหลือไม่ทราบยุค **129 แหล่ง**", T1S, None, 14, INK)
    s.text(6.65, Y0, 6.08, 0.4, P("ทดลองจับคู่ชื่อ 3 แบบ (ถูกจากทั้งหมด · แหล่ง ทธ. 48 / PBDB 673)", 13.5, INK, HEADM, space=0), "m")
    rows = [["การจับคู่ชื่อ", "ทายได้", "ถูกจากทั้งหมด"],
            ["baseline: เพอร์เมียนทุกแหล่ง", "100% / 100%", "54.2% / 49.5%"],
            ["ชื่อชนิด", "47.9% / 73.4%", "37.5% / 73.0%"],
            ["ชื่อสกุล", "91.7% / 92.0%", "70.8% / 88.4%"],
            ["**แบบผสม (ใช้จริง)**", "**91.7% / 92.0%**", "**75.0% / 89.9%**"]]
    s.table(6.65, 2.1, [2.6, 1.7, 1.78], rows, size=12.5, rowh=0.5, head=INK, align=["l", "r", "r"], hl={4: PILL})
    s.rect(6.65, 4.85, 6.08, 1.6, None, ACC, 1.2, 0.12)
    s.text(6.85, 4.9, 5.7, 1.5, P(["**แบบผสม** = ชื่อชนิดตรงตัว → ชนิดคลุมเครือ → ถอยไปชื่อสกุล", "ได้ทั้งความแม่นของชื่อชนิดและความครอบคลุมของชื่อสกุล "
                                    "· วัดแบบปิดยุคทีละแหล่ง (Leave-One-Out)"], 13, INK2, bold_color=ACC, space=4), "m")
    D.append(s)

    # ===== 18b ข้อมูลหลังเตรียม: หน้าตา ขนาด และงานที่ใช้ (ตัวอย่างแถวจริงจาก data/processed และ data/features)
    s = Slide(5, "ข้อมูลหลังเตรียม: หน้าตา ขนาด และงานที่ใช้", "// PREPARED DATA", (
        "หลังเตรียมข้อมูลได้ตาราง 3 ชุดที่ส่งต่อให้ 2 งาน ชุดแรก fossil_list_final มี 8,521 แถว 44 คอลัมน์ หนึ่งแถวคือตัวอย่างหนึ่งชิ้น "
        "รวมข้อมูลในทะเบียนกับแหล่งที่เชื่อมได้ ความเชื่อมั่น และยุคของแหล่ง งาน 1 ใช้ 3,944 ชิ้นที่มีสกุลเป็นป้ายฝึก แล้วทายจริงให้ 4,577 ชิ้น "
        "งาน 2 ใช้ประเภทซากของตัวอย่างที่เชื่อมได้มาทำป้ายประเภทของแหล่ง ชุดที่สอง sites_period_final มี 518 แถว 17 คอลัมน์ หนึ่งแถวคือแหล่งหนึ่งแหล่ง "
        "บอกยุคสุดท้าย ที่มา และความเชื่อมั่น งาน 1 ใช้เป็นป้ายยุค ส่วนงาน 2 ใช้ 518 แหล่งเป็นจุดที่พบซาก "
        "ชุดที่สามคือตารางคุณลักษณะของงาน 2 มี 10,808 จุด ได้แก่ แหล่ง ทธ. PBDB และจุดพื้นหลัง 2 ชุด แต่ละจุดมีคุณลักษณะเชิงพื้นที่ที่แบบจำลองใช้ 152 ตัว"))
    PW = 8.9
    datasets = [
        ("fossil_list_final.csv", "8,521 แถว × 44 คอลัมน์", "1 แถว = ตัวอย่าง 1 ชิ้น",
         [0.98, 2.12, 0.92, 0.92, 1.85, 0.8, 1.07],
         [["fossil_id", "sci_name", "com_name", "f_type", "site_NAME_TH", "confidence", "site_final_old"],
          ["20240600002", "Phuwiangosaurus sirindhornae", "ซอโรพอด", "ไดโนเสาร์", "ไดโนเสาร์บ้านนาไคร้", "high", "Cretaceous"],
          ["20210300701", "N/A (Dinosaur)", "ไดโนเสาร์", "ไดโนเสาร์", "ไดโนเสาร์สเตโกซอรัสคำม่วง", "medium", "Jurassic"],
          ["20201200008", "–", "แบรคิโอพอด", "แบรคิโอพอด", "แบรคิโอพอดห้วยหลวง", "low", "Carboniferous"]],
         [("งาน 1", T1, "ป้ายสกุล 3,944 ชิ้น · ทายจริง 4,577 ชิ้น"), ("งาน 2", T2, "ประเภทซากของตัวอย่าง → ป้ายประเภทแหล่ง")]),
        ("sites_period_final.csv", "518 แถว × 17 คอลัมน์", "1 แถว = แหล่ง 1 แหล่ง",
         [0.8, 2.45, 0.95, 1.0, 1.2, 1.35, 1.15],
         [["OBJECTID", "NAME_TH", "PROVINCE", "PERIODFROM", "PERIOD_FINAL", "final_source", "final_confidence"],
          ["11177", "สัตว์ไม่มีกระดูกสันหลังศาลเจ้าพ่อ…", "ราชบุรี", "เพอร์เมียน", "เพอร์เมียน", "DMR record", "recorded"],
          ["11176", "สัตว์ไม่มีกระดูกสันหลังเขาแดง", "สตูล", "–", "คาร์บอนิเฟอรัส", "03 PBDB nearby", "medium"],
          ["11175", "รอยตีนไดโนเสาร์เขาใหญ่", "ปราจีนบุรี", "–", "–", "missing", "–"]],
         [("งาน 1", T1, "ป้ายยุค 166 + ค่าเติม 129 · ทายจริง 352"), ("งาน 2", T2, "518 แหล่ง = จุดที่พบซาก")]),
        ("all_features.parquet", "10,808 แถว × 199 คอลัมน์", "1 แถว = จุด 1 จุด · แบบจำลองใช้ 152 คุณลักษณะ",
         [1.0, 0.95, 0.75, 0.7, 1.15, 1.1, 0.85, 1.15, 1.25],
         [["point_id", "source", "lon", "lat", "geo_rock_sym", "geo_era_old", "dem_elev", "s2_ndvi_mean", "… +191"],
          ["dmr_11177", "dmr_site", "99.67", "13.61", "Pr", "Permian", "61", "0.65", "…"],
          ["pbdb_8103", "pbdb", "99.84", "18.62", "Png1", "Permian", "650", "0.56", "…"],
          ["bgb_0", "bg_bias", "99.61", "18.78", "Qa", "Quaternary", "323", "0.41", "…"]],
         [("งาน 2", T2, "แหล่ง 518 + พื้นหลัง 10,000 → ฝึก / ทดสอบ"), ("งาน 2", T2, "PBDB 290 จุด → ทดสอบอิสระ")]),
    ]
    for k, (fname, shape, unit, colw, rows, uses) in enumerate(datasets):
        y = Y0 + k * 1.76
        s.rect(X0, y, PW, 1.64, SURF, LINE, 1.0, 0.12)
        s.icon("file-text", "ink", X0 + 0.12, y + 0.08, 0.3, 0.3)
        s.text(X0 + 0.48, y + 0.06, 3.2, 0.34, P(fname, 12.5, INK, MONO, space=0), "m")
        w = pill(s, X0 + 3.55, y + 0.08, shape, PILL, INK, 11.5, HEADM, 0.3, 0.13)
        s.text(X0 + 3.55 + w + 0.14, y + 0.06, PW - 3.75 - w, 0.34, P(unit, 11, MUTED, space=0), "m")
        f = (PW - 0.24) / sum(colw)
        s.table(X0 + 0.12, y + 0.46, [c * f for c in colw], rows, size=9.5, rowh=0.27, head=INK2, align=["l"] * len(rows[0]), first_bold=False)
        for j, (lab, col, txt) in enumerate(uses):
            yy = y + 0.12 + j * 0.76
            s.rect(X0 + PW + 0.18, yy, 3.05, 0.66, T1S if col == T1 else T2S, None, 0, 0.1)
            pw = pill(s, X0 + PW + 0.28, yy + 0.17, lab, col, "FFFFFF", 11, HEAD, 0.32, 0.12)
            s.text(X0 + PW + 0.38 + pw, yy + 0.02, 3.05 - pw - 0.26, 0.62, P(txt, 11, INK2, space=0, line=1.05), "m")
        arrow(s, X0 + PW + 0.02, y + 0.82, X0 + PW + 0.16, y + 0.82, DIM, 1.4, False)
    D.append(s)

    # ===== 19 เตรียมและแบ่งข้อมูล (รวมหน้า "เตรียมข้อมูลของแต่ละงาน" + "การฝึกและการแบ่งข้อมูล")
    s = Slide(5, "เตรียมและแบ่งข้อมูล: ฝึก · ทดสอบ · ทายจริง", "// DATA SPLIT", (
        "หน้านี้รวมการเตรียมข้อมูลกับการแบ่งข้อมูลของทั้งสองงาน รูปซ้ายบนคือการแบ่งข้อมูล 3 แบบบนพื้นที่จริงในภาคใต้ "
        "แบบสุ่มทำให้ข้อสอบมีแหล่งฝึกอยู่ห่างแค่ 3.1 กิโลเมตร แบบกลุ่มแหล่งให้แหล่งที่ห่างกันไม่เกิน 2 กิโลเมตรอยู่ฝั่งเดียวกัน "
        "และแบบบล็อกพื้นที่ยกทั้งบล็อก 50 คูณ 50 กิโลเมตรไปเป็นข้อสอบ ระยะถึงแหล่งฝึกจึงเพิ่มเป็น 20.4 กิโลเมตร ใกล้กับพื้นที่ใหม่ "
        "ทุกแบบแบ่งเป็น 5 ส่วน ฝึก 4 ส่วน ทดสอบ 1 ส่วน แล้ววนจนครบ ด้านขวาคือแต่ละงานใช้ข้อมูลอะไรฝึกและทดสอบ "
        "สกุลใช้ 3,944 ชิ้นที่มีสกุล แหล่งเดิมแบ่งแบบสุ่ม แหล่งใหม่แบ่งแบบบล็อก แล้วทายจริงให้ 4,577 ชิ้น "
        "ยุคใช้ 166 แหล่งที่บันทึกยุค ค่าเติมอีก 129 แหล่งใช้ฝึกเท่านั้น แล้วทายจริงให้ 352 แหล่ง "
        "งาน 2 ขั้น A ใช้แหล่ง 518 แหล่งกับจุดพื้นหลัง 5,000 จุด ฝึกแยกตามชุดพื้นหลัง และขั้น B ใช้ 490 แหล่งที่มีป้ายประเภท "
        "ทั้งสองขั้นแบ่งแบบบล็อกพื้นที่ และมี PBDB 290 จุดเป็นข้อสอบอิสระที่ไม่เคยใช้ฝึก แผนที่ซ้ายล่างคือชุดจุดทั้งหมดของงาน 2 "
        "เกณฑ์เลือกแบบจำลองของงาน 1 คือชื่อแรกถูกและชื่อจริงอยู่ใน 3 ชื่อ ส่วนงาน 2 คือ AUC แบบบล็อกพื้นที่และผลกับ PBDB"))
    LWD = 6.5
    frame_img(s, A / "maps" / "map_cv.png", X0, Y0, LWD, 3.22, pad=0.08)
    frame_img(s, A / "maps" / "map_points.png", X0, 4.98, 3.65, 1.86, pad=0.06)
    s.text(X0 + 3.82, 4.98, LWD - 3.82, 1.86, P(["**ชุดจุดงาน 2 · 10,808 จุด**",
                                                 "แหล่ง ทธ. 518 · PBDB 290 (ทดสอบอิสระ) · พื้นหลังสุ่ม 5,000 · พื้นหลังถ่วงการสำรวจ 5,000",
                                                 "**เกณฑ์เลือกแบบจำลอง:** งาน 1 accuracy + top-3 · งาน 2 AUC บล็อก + PBDB"], 11, INK2,
                                                bold_color=INK, space=4, line=1.05), "m")
    RXD, RWD = X0 + LWD + 0.22, XW - LWD - 0.22
    TR_FILL, TE_FILL = INK2, PILL
    lx = RXD
    for lab, fill, fg, ln, dash in [("ฝึก", TR_FILL, "FFFFFF", None, False), ("ทดสอบ", TE_FILL, INK, None, False),
                                    ("ทดสอบอิสระ", TE_FILL, INK, ACC, False), ("ทายจริง", SURF, INK2, DIM, True)]:
        w = tw(lab, 10.5, HEADM) + 0.28
        s.rect(lx, Y0 + 0.02, w, 0.28, fill, ln, 1.0, 0.14, dash, ps=P(lab, 10.5, fg, HEADM, "c", space=0, line=1.0), inset=0.02)
        lx += w + 0.1
    cards = [
        ("งาน 1", T1, "สกุล", "3,944 ชิ้นที่มีสกุล (จากชื่อวิทยาศาสตร์)", "ก. สุ่ม = แหล่งเดิม · ค. บล็อก = แหล่งใหม่", None,
         "4,577 ชิ้นที่ยังไม่รู้สกุล"),
        ("งาน 1", T1, "ยุค", "166 แหล่งที่บันทึกยุค (27 แบบ → 11 ยุค)", "ก · ข · ค · ตัวเลขหลัก = ค", ("+ ค่าเติม 129 · ฝึกเท่านั้น", False),
         "352 แหล่ง"),
        ("งาน 2", T2, "ขั้น A", "แหล่ง 518 (=1) + พื้นหลัง 5,000 (=0)", "ค. บล็อก 50 กม. · ฝึกแยก 2 ชุดพื้นหลัง", ("PBDB 290", True),
         "กริด 20,706 ช่อง"),
        ("งาน 2", T2, "ขั้น B", "490 แหล่ง · ป้าย 16 ประเภท (ตัดชื่อสถานที่)", "ค. บล็อก 50 กม.", ("PBDB", True), "ประเภท 3 อันดับ"),
    ]
    CH = 1.16
    for k, (task, col, sub, data, how, extra, pred) in enumerate(cards):
        y = Y0 + 0.42 + k * (CH + 0.06)
        s.rect(RXD, y, RWD, CH, SURF, LINE, 1.0, 0.1)
        pw = pill(s, RXD + 0.12, y + 0.08, f"{task} · {sub}", col, "FFFFFF", 11, HEAD, 0.3, 0.12)
        s.text(RXD + 0.24 + pw, y + 0.06, RWD - pw - 0.34, 0.34, P(data, 11, INK2, space=0), "m")
        for i in range(5):
            tr = i < 4
            s.rect(RXD + 0.12 + i * 0.4, y + 0.46, 0.37, 0.28, TR_FILL if tr else TE_FILL, None, 0, 0.04,
                   ps=P("ฝึก" if tr else "สอบ", 9, "FFFFFF" if tr else INK, HEADM, "c", space=0, line=1.0), inset=0.0)
        s.text(RXD + 2.2, y + 0.43, RWD - 2.3, 0.34, P(how, 10.5, MUTED, space=0), "m")
        xx = RXD + 0.12
        if extra:
            t, indep = extra
            w = pill(s, xx, y + 0.82, ("ทดสอบอิสระ · " + t) if indep else t, TE_FILL if indep else TR_FILL, INK if indep else "FFFFFF",
                     10, HEADM, 0.26, 0.1, line=ACC if indep else None)
            xx += w + 0.12
        s.rect(xx, y + 0.82, RXD + RWD - 0.12 - xx, 0.26, SURF, DIM, 1.0, 0.13, dash=True,
               ps=P(f"→ ทายจริง: {pred}", 10, INK2, HEADM, "c", space=0, line=1.0), inset=0.02)
    D.append(s)

    # ===== 20 สร้างคุณลักษณะ
    s = Slide(5, "สร้างคุณลักษณะ (Feature Engineering)", "// FEATURE ENGINEERING", (
        "การสร้างคุณลักษณะ งาน 1 ทายสกุลด้วย 2 เส้นทาง เส้นทางแหล่งเดิมรับ 225 ตัว รวมคุณลักษณะเชิงพื้นที่ของแหล่ง 152 ตัวและ TF-IDF ของข้อความ "
        "ส่วนเส้นทางแหล่งใหม่รับเพียง 55 ตัว ตัดข้อมูลแหล่งออกทั้งหมด เหลือข้อมูลตัวอย่าง หมวดหินกับ ThaiLex และเวกเตอร์ WangchanBERTa "
        "[คลิก] งาน 2 ใช้คุณลักษณะเชิงพื้นที่ 152 ตัวต่อจุด แบ่ง 4 กลุ่ม ภูมิศาสตร์โบราณ 85 ตัว ธรณีวิทยา 30 ตัว ดาวเทียม 22 ตัว และภูมิประเทศ 15 ตัว"))
    two(s, "ข้อมูลตัวอย่าง + ข้อความภาษาไทย", "เชิงพื้นที่ 152 ตัวต่อจุด")
    rows = [["กลุ่มคุณลักษณะ", "แหล่งเดิม", "แหล่งใหม่"],
            ["ข้อมูลตัวอย่าง (ชื่อสามัญ ประเภท กลุ่ม ส่วน จังหวัด อำเภอ)", "6", "6"],
            ["คุณลักษณะเชิงพื้นที่ของแหล่ง", "152", "–"],
            ["พิกัด + ประเภทซากในแหล่ง", "2 + 17", "–"],
            ["เวกเตอร์ภาพ CNN", "16", "–"],
            ["ข้อความ", "TF-IDF 32", "WangchanBERTa 32"],
            ["หมวดหิน + ยุค / ชนิดหินจาก ThaiLex", "–", "2 + 15"],
            ["**รวม**", "**225**", "**55**"]]
    s.table(LX, 2.2, [3.15, 1.15, 1.45], rows, size=12, rowh=0.4, rowhs=[0.4, 0.55, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], head=T1, align=["l", "c", "c"], hl={7: T1S})
    s.text(LX, 5.6, HW, 1.25, P(["ชื่อสามัญ + ประเภท + ส่วนของซาก → **TF-IDF หั่นตัวอักษร 2–4 ตัว → SVD 32 มิติ** (แหล่งเดิม)",
                                 "ข้อความสั้นและซ้ำกันมากในแหล่งเดียวกัน → **WangchanBERTa (เฉลี่ยเวกเตอร์) → PCA 32 มิติ** (แหล่งใหม่)"], 12.5, INK2,
                                bold_color=T1,
                                bullet=DIM, space=4))
    with s.reveal():
        right_head(s)
        segs = [("ภูมิศาสตร์โบราณ", 85, GREEN), ("ธรณีวิทยา", 30, ACC), ("ดาวเทียม", 22, ACCHI), ("ภูมิประเทศ", 15, SAGE)]
        x = RX
        for _, v, col in segs:
            w = HW * v / 152
            s.rect(x, 2.22, w, 0.42, col, None, 0, 0, ps=P(str(v), 12.5, "FFFFFF", HEAD, "c", space=0), inset=0.01)
            x += w
        grp = [("globe", "ภูมิศาสตร์โบราณ · 85", "ละติจูดโบราณ ระยะเคลื่อนตามละติจูด ความสูงโบราณ ทะเล / บกโบราณ แผ่นเปลือกโลก (PALEOMAP, Merdith)", GREEN),
               ("layers", "ธรณีวิทยา · 30", "หน่วยหินและยุค ณ จุด ชนิดหิน สัดส่วนหินแต่ละยุคในรัศมี 5 กม. ระยะถึงถ้ำ", ACC),
               ("satellite", "ดาวเทียม · 22", "6 แบนด์ NDVI NDWI ดัชนีดินโล่ง แร่ดินเหนียว / เหล็กออกไซด์ (Sentinel-2)", ACCHI),
               ("mountain", "ภูมิประเทศ · 15", "ความสูง ความชัน TPI ค่าเฉลี่ย / ส่วนเบี่ยงเบนในรัศมี 1 และ 5 กม. (Copernicus DEM)", SAGE)]
        for k, (ic, t, b, col) in enumerate(grp):
            y = 2.85 + k * 1.0
            s.rect(RX, y, HW, 0.88, SURF, LINE, 1.0, 0.1)
            s.rect(RX + 0.16, y + 0.2, 0.48, 0.48, col, None, 0, shape="ellipse")
            s.icon(ic, "white", RX + 0.25, y + 0.29, 0.3, 0.3)
            s.text(RX + 0.78, y + 0.04, HW - 0.9, 0.36, P(t, 13.5, INK, HEADM, space=0), "m")
            s.text(RX + 0.78, y + 0.38, HW - 0.9, 0.5, P(b, 11.5, MUTED, space=0, line=1.05))
    D.append(s)
    return D



def build_deck2():
    D = []

    # ===== 21 ข้อมูลรั่ว
    s = Slide(5, "ข้อมูลรั่ว (Data Leakage): ตรวจเจอและแก้ก่อนวัดผล", "// DATA LEAKAGE", (
        "ข้อมูลรั่วคือการที่แบบจำลองเห็นคำตอบทางอ้อมตอนฝึก ทำให้ผลดูดีเกินจริง โครงงานตรวจเจอหลายจุด จุดใหญ่ที่สุดคือชื่อสามัญที่เป็นชื่อเฉพาะชนิด "
        "ช่องสถานะการศึกษา และข้อมูลการเชื่อม ทำให้ความแม่นสกุลสูงเกินจริงราว 12 จุด จาก 0.785 เหลือ 0.661 หลังแก้ "
        "จุดอื่น เช่น ค่าเติมยุคที่อ้างยุคของแหล่งอื่น หมวดหินที่ ทธ. บันทึกคู่กับยุค และการปรับสเกลหรือ TF-IDF ด้วยข้อมูลทั้งหมด "
        "ตัวเลขหลังแก้ต่ำลง แต่เป็นตัวเลขที่ใช้งานจริงได้"))
    s.rect(X0, Y0, 3.4, 5.2, T2S, None, 0, 0.14)
    s.text(X0 + 0.25, Y0 + 0.3, 2.9, 1.1, P("12 จุด", 44, ACC, HEAD, space=0, line=1.0), "m")
    s.text(X0 + 0.25, Y0 + 1.45, 2.9, 1.1, P("ความแม่นสกุลสูงเกินจริงก่อนแก้", 15, INK, HEADM, space=0))
    s.text(X0 + 0.25, Y0 + 2.2, 2.9, 0.7, P("accuracy แหล่งเดิม", 12.5, MUTED, space=0))
    s.text(X0 + 0.25, Y0 + 2.6, 2.9, 0.8, P("0.785 → 0.661", 24, INK, HEAD, space=0), "m")
    s.text(X0 + 0.25, Y0 + 3.6, 2.9, 1.4, P("ตัวเลขหลังแก้ต่ำลง แต่เป็นตัวเลขที่ใช้งานจริงได้", 13.5, INK2, space=0))
    rows = [["สิ่งที่พบ", "ผลกระทบ", "ก่อน → หลัง"],
            ["ชื่อสามัญเฉพาะชนิด · ช่องสถานะการศึกษา · ข้อมูลการเชื่อม", "ความแม่นสกุลเกินจริง", "0.785 → 0.661"],
            ["ค่าเติมยุคที่อ้างยุคของแหล่งอื่น (28 แหล่ง)", "คำตอบของข้อสอบรั่วเข้าชุดฝึก", "ประโยชน์ +0.025 → +0.003"],
            ["หมวดหินที่ ทธ. บันทึกคู่กับยุค (84% เทียบ 9%)", "ยุคดูดีแต่ใช้จริงแย่", "0.729 → 0.697 (ไม่ใช้)"],
            ["ป้ายประเภทจากชื่อสถานที่", "ป้ายผิด", "ผิด 14 จาก 25 แหล่ง"],
            ["ชื่อสกุลในช่องส่วนของซาก (9 ชิ้น)", "ข้อความเห็นคำตอบ", "ลบชื่อสกุลก่อน"],
            ["ปรับสเกล / TF-IDF / PCA ด้วยข้อมูลทั้งหมด", "เห็นข้อสอบล่วงหน้า", "สร้างจากชุดฝึกแต่ละรอบ"]]
    s.table(4.25, Y0, [4.0, 2.3, 2.18], rows, size=12.5, rowh=0.74, rowhs=[0.42] + [0.79] * 6, head=INK, align=["l", "l", "l"])
    D.append(s)

    # ===== 22 ลองอะไรบ้าง
    s = Slide(5, "ลองอะไรไปบ้าง: อันไหนได้ผล อันไหนไม่", "// EXPERIMENTS", (
        "สรุปสิ่งที่ทดลอง งาน 1 TF-IDF ช่วยแหล่งเดิมได้ทุกรอบ หมวดหินกับ WangchanBERTa ช่วย top-3 ที่แหล่งใหม่จาก 0.546 เป็น 0.587 "
        "และการเพิ่มคุณลักษณะผิวดินช่วยแบบจำลองยุค ส่วนที่ไม่ได้ผลคือเวกเตอร์ภาพ CNN และการใช้คุณลักษณะของแหล่งกับพื้นที่ใหม่ "
        "[คลิก] งาน 2 XGBoost กับคุณลักษณะ 152 ตัวได้ AUC 0.887 สูงกว่าแค่จำพิกัดมาก ภูมิศาสตร์โบราณช่วยผลกับ PBDB จึงเก็บไว้ "
        "แต่ CNN จากภาพดาวเทียมและคุณลักษณะชุดใหม่ 54 ตัวทำให้ผลกับ PBDB ลดลง จึงไม่ใช้"))
    two(s, "ทายชื่อและยุคของซาก", "คาดการณ์จากพิกัด")

    def rows_of(x, items):
        y = 2.2
        for kind, txt in items:
            badge(s, x, y + 0.1, kind)
            s.text(x + 1.05, y, HW - 1.05, 0.72, P(txt, 13, INK2, bold_color=INK, space=0, line=1.08), "m")
            y += 0.78
    rows_of(LX, [("good", "**TF-IDF** กับแหล่งเดิม: accuracy 0.656 → 0.668 (ดีขึ้นทุกรอบ)"),
                 ("good", "**หมวดหิน + WangchanBERTa** แหล่งใหม่: top-3 0.546 → 0.587"),
                 ("good", "**เพิ่มผิวดิน** ให้แบบจำลองยุค: 0.704 → 0.719"),
                 ("same", "**มิติข้อความ** 16 / 64 / 128 แทน 32: ต่างไม่เกิน 0.004"),
                 ("bad", "**เวกเตอร์ภาพ CNN** แหล่งใหม่: top-3 0.526 → 0.431"),
                 ("bad", "**คุณลักษณะแหล่งกับพื้นที่ใหม่**: 0.275 เทียบ 0.389 (ชี้ผิดทาง)")])
    with s.reveal():
        right_head(s)
        rows_of(RX, [("good", "**XGBoost + 152 คุณลักษณะ**: AUC 0.887 (kNN บนพิกัด 0.566)"),
                     ("good", "**ภูมิศาสตร์โบราณ**: ตัดออกแล้ว PBDB ลด 0.796 → 0.783 → เก็บไว้"),
                     ("same", "**XGBoost vs LightGBM**: ต่างไม่เกิน 0.015 (น้อยกว่าความผันผวน ~0.03)"),
                     ("bad", "**CNN ภาพดาวเทียม**: AUC 0.792 · PBDB 0.797 → 0.729 → ไม่ใช้"),
                     ("bad", "**คุณลักษณะชุดใหม่ 54 ตัว**: PBDB 0.796 → 0.784 → กลับไปรุ่นเดิม")])
    D.append(s)

    # ===== 23 แบบจำลองและค่าตั้ง
    s = Slide(6, "แบบจำลองที่เปรียบเทียบและการตั้งค่า", "// MODELS & SETTINGS", (
        "แบบจำลองที่เปรียบเทียบมีตั้งแต่ baseline คือทายตามความถี่ และ kNN ที่ใช้แค่พิกัด ซึ่งวัดว่าแบบจำลองเก่งกว่าการจำตำแหน่งแค่ไหน "
        "ไปจนถึงแบบจำลองกลุ่มต้นไม้ Random Forest LightGBM XGBoost โครงข่ายประสาท MLP และ CNN สำหรับภาพดาวเทียม "
        "โครงงานไม่ได้ค้นหา hyperparameter เพราะข้อมูลน้อยและผลแกว่งระหว่างรอบมาก จึงใช้ค่ากลางคงที่ "
        "ผลที่รายงานจึงอาจยังไม่สูงสุด แต่ไม่สูงเกินจริงจากการจูนให้เข้ากับข้อสอบ"))
    rows = [["แบบจำลอง", "ค่าตั้งหลัก", "บทบาท"],
            ["prior", "ทายตามความถี่ของคลาสในชุดฝึก", "baseline"],
            ["kNN บนพิกัด", "k = 15 ถ่วงด้วยระยะ ใช้พิกัดอย่างเดียว — วัดว่าแค่ “จำตำแหน่ง” ได้แค่ไหน", "baseline"],
            ["Random Forest", "400 ต้น · min_samples_leaf 2 · ถ่วงน้ำหนักคลาส · ทายหลายป้ายพร้อมกัน", "**ใช้จริง:** ประเภทซาก"],
            ["LightGBM", "400 ต้น (หลายคลาส 300) · lr 0.03 · 31 ใบ · subsample 0.8 · colsample 0.7 · ถ่วงน้ำหนักคลาส", "**ใช้จริง:** สกุลแหล่งใหม่ · ยุค"],
            ["XGBoost", "400 ต้น (หลายคลาส 300) · ลึก 5 · lr 0.03 · hist · ถ่วงน้ำหนักเฉพาะงานสองคลาส", "**ใช้จริง:** โอกาสพบซาก · สกุลแหล่งเดิม"],
            ["MLP", "256 → 128 · BatchNorm + ReLU + Dropout 0.3 · AdamW + cosine · หยุดเมื่อไม่ดีขึ้น 8 รอบ", "เปรียบเทียบ"],
            ["CNN", "Sentinel-2 6 แบนด์ 32 × 32 · 3 บล็อก 32 → 64 → 128 · Global Average Pooling", "เปรียบเทียบ (ไม่ใช้)"],
            ["WangchanBERTa", "เฉลี่ยเวกเตอร์ของโทเค็น → PCA 32 มิติ", "ตัวสร้างคุณลักษณะข้อความ"]]
    s.table(X0, Y0, [2.15, 6.6, 3.38], rows, size=12.5, rowh=0.47, head=INK, align=["l", "l", "l"], hl={3: T2S, 4: T1S, 5: PILL})
    s.rect(X0, 6.02, XW, 0.82, None, ACC, 1.2, 0.12)
    s.text(X0 + 0.2, 6.04, XW - 0.4, 0.78, P("**ไม่จูน hyperparameter:** ข้อมูลน้อย ผลแกว่งระหว่างรอบมาก จึงใช้ค่ากลางคงที่ — ผลอาจยังไม่สูงสุด แต่ไม่สูงเกินจริงจากการจูนกับข้อสอบ "
                                             "· MLP หยุดก่อนด้วยชุดแบ่งภายใน 15%", 13, INK2, bold_color=ACC, space=0), "m")
    D.append(s)

    # ===== 24 โครงสร้างแบบจำลอง
    s = Slide(6, "โครงสร้างแบบจำลองของ 2 งาน", "// MODEL ARCHITECTURE", (
        "โครงสร้างของงาน 1 ระบบทายสกุลเลือกแบบจำลองตามสถานการณ์ ถ้าแหล่งของตัวอย่างมีตัวอย่างอื่นที่ระบุสกุลแล้วในชุดฝึก ใช้ XGBoost 225 คุณลักษณะ "
        "ถ้าไม่มีหรือเป็นพิกัดใหม่ ใช้ LightGBM 55 คุณลักษณะ แล้วผ่านตัวกรองความสอดคล้อง ซึ่งตัดสกุลที่ขัดกับกลุ่มซากทิ้ง เช่น ไดโนเสาร์จะไม่ถูกทายเป็นสกุลของเต่า "
        "ส่วนยุคใช้ LightGBM แยกอีกตัว [คลิก] งาน 2 รับพิกัด สร้างคุณลักษณะ 152 ตัว ขั้น A XGBoost ให้คะแนนโอกาสพบซาก ขั้น B Random Forest จัดอันดับประเภทซาก "
        "และ AOA ตรวจว่าจุดนั้นคล้ายข้อมูลฝึกพอหรือไม่"))
    two(s, "ระบบทายสกุล + ยุค", "ขั้น A → ขั้น B → AOA")
    cx = LX + HW / 2
    box(s, LX, 2.15, HW, 0.5, "ข้อมูลตัวอย่าง + แหล่ง / พิกัด (ถ้ามี)", SURF, LINE, 13)
    arrow(s, cx, 2.68, cx, 2.92, T1)
    box(s, LX + 0.4, 2.95, HW - 0.8, 0.58, "แหล่งนี้มีตัวอย่างที่ระบุสกุลในชุดฝึกไหม?", PILL, None, 13, INK, HEADM)
    arrow(s, LX + 1.4, 3.55, LX + 1.4, 3.8, T1)
    arrow(s, LX + HW - 1.4, 3.55, LX + HW - 1.4, 3.8, T1)
    card(s, LX, 3.83, 2.78, 1.3, "มี → แหล่งเดิม", "**XGBoost** 225 ตัว\nข้อมูลแหล่ง + TF-IDF", tsize=13.5, bsize=12.5, bold_color=T1)
    card(s, LX + 2.97, 3.83, 2.78, 1.3, "ไม่มี → แหล่งใหม่", "**LightGBM** 55 ตัว\nหมวดหิน + WangchanBERTa", tsize=13.5, bsize=12.5, bold_color=T1)
    arrow(s, LX + 1.4, 5.16, cx - 0.3, 5.38, T1)
    arrow(s, LX + HW - 1.4, 5.16, cx + 0.3, 5.38, T1)
    box(s, LX, 5.4, HW, 0.55, "ตัวกรองความสอดคล้อง: ตัดสกุลที่ขัดกับกลุ่มซาก", T1S, None, 13, INK)
    arrow(s, cx, 5.98, cx, 6.18, T1)
    box(s, LX, 6.2, 3.6, 0.6, "สกุล 3 อันดับ → ชนิดที่พบบ่อย", SURF, T1, 13, T1, HEADM, lw=1.4)
    box(s, LX + 3.75, 6.2, 2.0, 0.6, "ยุค: LightGBM\n+ ผิวดิน (11 ยุค)", SURF, LINE, 11.5)
    with s.reveal():
        right_head(s)
        cx = RX + HW / 2
        steps = [("พิกัด (lat, long)", SURF, LINE, INK, 0.5),
                 ("คุณลักษณะเชิงพื้นที่ 152 ตัว (+ ภาพ Sentinel-2 สด)", SURF, LINE, INK, 0.55),
                 ("**ขั้น A · XGBoost** — โอกาสพบซาก\nฝึก 2 แบบตามจุดพื้นหลัง", T2S, None, INK, 0.78),
                 ("**ขั้น B · Random Forest** — ประเภทซาก 16 ประเภท\nเสนอ 3 อันดับ", T2S, None, INK, 0.78),
                 ("**AOA** — ระยะความไม่คล้าย (DI) → ในขอบเขต / นอกขอบเขต", PILL, None, INK, 0.58),
                 ("คะแนน + อันดับเทียบกริด · ประเภท 3 อันดับ · แหล่งที่คล้าย 3 แหล่ง", SURF, T2, T2, 0.6)]
        y = 2.15
        for k, (txt, fill, line, col, h) in enumerate(steps):
            s.rect(RX, y, HW, h, fill, line, 1.4 if line == T2 else 1.0, 0.1, ps=P(txt, 13, col, BODYM if k == len(steps) - 1 else BODY, "c",
                                                                                     bold_color=T2, space=0, line=1.08), inset=0.1)
            if k < len(steps) - 1:
                arrow(s, cx, y + h + 0.02, cx, y + h + 0.2, T2)
            y += h + 0.22
    D.append(s)

    # ===== 24b ตัวอย่างจริง: ระบบทายสกุล (ข้อมูลจาก data/processed/task1_genus_predictions.csv)
    s = Slide(6, "ตัวอย่างจริง: ระบบทายสกุลทำงานอย่างไร", "// TASK 1 · WALK-THROUGH", (
        "ตัวอย่างจริง 2 ชิ้นจากไฟล์ผลลัพธ์ ชิ้นแรกเป็นฟันซอโรพอดที่พบในอำเภอคำม่วง กาฬสินธุ์ เชื่อมได้กับแหล่งไดโนเสาร์สเตโกซอรัสคำม่วง "
        "ซึ่งมีตัวอย่างที่ระบุสกุลแล้ว จึงไปเส้นทางแหล่งเดิม XGBoost ใช้ข้อมูลของแหล่งกับข้อความของตัวอย่าง ตัวกรองตัดสกุลที่ไม่ใช่ไดโนเสาร์ "
        "ได้ Phuwiangosaurus ร้อยละ 54 และชนิดที่พบบ่อยคือ Phuwiangosaurus sirindhornae "
        "ชิ้นที่สองเป็นแบรคิโอพอดที่ไม่มีข้อมูลที่พบเลย จึงไปเส้นทางแหล่งใหม่ LightGBM ใช้ข้อมูลของตัวอย่างและข้อความผ่าน WangchanBERTa "
        "ได้ Kasetia ร้อยละ 29 ความมั่นใจต่ำกว่าชิ้นแรกมากเพราะไม่มีข้อมูลแหล่งช่วย ส่วนคำว่าอื่น ๆ คือสกุลที่มีตัวอย่างน้อยกว่า 5 ชิ้นรวมกัน"))
    cols = [("1", "ข้อมูลเข้า", 2.75), ("2", "รู้จักแหล่งนี้ไหม?", 1.95), ("3", "แบบจำลอง + ข้อมูลที่ใช้", 2.85), ("4", "ตัวกรอง", 1.55),
            ("5", "ผลลัพธ์ 3 อันดับ", 2.4)]
    GAP = (XW - sum(c[2] for c in cols)) / 4
    xs = []
    x = X0
    for num, t, w in cols:
        xs.append(x)
        s.rect(x, Y0, 0.34, 0.34, T1, None, 0, 0.17, shape="ellipse", ps=P(num, 12, "FFFFFF", HEAD, "c", space=0, line=1.0), inset=0)
        s.text(x + 0.42, Y0 - 0.02, w - 0.42, 0.38, P(t, 13.5, INK, HEADM, space=0), "m")
        x += w + GAP
    lanes = [
        (2.08, "tooth", "**ฟันซอโรพอด**  {c:" + DIM + "}20210301024{/c}",
         ["ชื่อสามัญ: ซอโรพอด · ประเภท: ไดโนเสาร์", "ส่วนของซาก: กระดูกฟัน (Tooth)", "ที่พบ: กาฬสินธุ์ · อ.คำม่วง",
          "→ แหล่ง “ไดโนเสาร์สเตโกซอรัสคำม่วง”"],
         ("มี ✓", "แหล่งนี้มีตัวอย่างที่ระบุสกุลแล้ว", "แหล่งเดิม", T1S),
         ("XGBoost", "225 คุณลักษณะ", ["ข้อมูลแหล่ง: หิน ยุค ดาวเทียม ซากที่พบในแหล่ง", "ข้อความ “ซอโรพอด | ไดโนเสาร์ | ฟัน” → TF-IDF"]),
         "ตัดสกุลที่ไม่ใช่ไดโนเสาร์ เช่น เต่า หอย",
         [("Phuwiangosaurus", 0.54), ("อื่น ๆ", 0.40), ("Isanosaurus", 0.05)], "ชนิดที่พบบ่อย: P. sirindhornae"),
        (4.38, "brachiopod", "**เปลือกแบรคิโอพอด**  {c:" + DIM + "}20201200026{/c}",
         ["ชื่อสามัญ / ประเภท: แบรคิโอพอด", "ที่พบ: **ไม่ระบุ**", "(ไม่มีจังหวัด อำเภอ หรือแหล่ง)"],
         ("ไม่มี ✗", "ไม่รู้ว่ามาจากแหล่งไหน", "แหล่งใหม่", PILL),
         ("LightGBM", "55 คุณลักษณะ", ["ข้อมูลตัวอย่าง: ประเภท กลุ่ม", "ข้อความ → WangchanBERTa", "หมวดหิน: จากแผนที่ธรณีถ้ามีพิกัด (รายนี้ไม่มี)"]),
         "เหลือเฉพาะสกุลที่เป็นแบรคิโอพอด",
         [("Kasetia", 0.29), ("Dyschrestia", 0.23), ("Brachythyrina", 0.14)], "ชนิดที่พบบ่อย: K. kaseti · มั่นใจต่ำกว่า"),
    ]
    LH = 2.08
    for y, ic, head, inp, dec, model, filt, res, sp in lanes:
        w = [c[2] for c in cols]
        # 1 ข้อมูลเข้า
        s.rect(xs[0], y, w[0], LH, SURF, LINE, 1.0, 0.12)
        s.icon(ic, "ink", xs[0] + 0.15, y + 0.12, 0.5, 0.42, "l")
        s.text(xs[0] + 0.72, y + 0.1, w[0] - 0.82, 0.46, P(head, 12.5, INK, HEADM, bold_color=INK, space=0, line=1.05), "m")
        s.text(xs[0] + 0.15, y + 0.62, w[0] - 0.27, LH - 0.7, P(inp, 11, INK2, bold_color=ACC, space=2, line=1.05))
        # 2 รู้จักแหล่งไหม
        yes, why, route, fill = dec
        s.rect(xs[1], y, w[1], LH, fill, None, 0, 0.12)
        s.text(xs[1] + 0.1, y + 0.12, w[1] - 0.2, 0.46, P(yes, 16, T1 if fill == T1S else ACC, HEAD, "c", space=0), "m")
        s.text(xs[1] + 0.12, y + 0.62, w[1] - 0.24, 0.62, P(why, 11.5, INK2, align="c", space=0, line=1.05), "m")
        s.rect(xs[1] + 0.2, y + 1.36, w[1] - 0.4, 0.46, SURF, None, 0, 0.23, ps=P("→ " + route, 13, INK, HEADM, "c", space=0, line=1.0),
               inset=0.02)
        # 3 แบบจำลอง
        name, nf, used = model
        s.rect(xs[2], y, w[2], LH, T1S, None, 0, 0.12)
        s.text(xs[2] + 0.15, y + 0.1, w[2] - 0.3, 0.42, P(f"**{name}**  {{c:{MUTED}}}{nf}{{/c}}", 14, T1, HEADM, bold_color=T1, space=0), "m")
        s.text(xs[2] + 0.15, y + 0.58, w[2] - 0.27, LH - 0.66, P(used, 11, INK2, bullet=T1, space=3, line=1.05))
        # 4 ตัวกรอง
        s.rect(xs[3], y, w[3], LH, PANEL, None, 0, 0.12)
        s.icon("shield-check", "green", xs[3] + w[3] / 2 - 0.22, y + 0.2, 0.44, 0.44)
        s.text(xs[3] + 0.1, y + 0.75, w[3] - 0.2, LH - 0.85, P(filt, 11.5, INK2, align="c", space=0, line=1.05), "t")
        # 5 ผลลัพธ์
        s.rect(xs[4], y, w[4], LH, SURF, GREEN, 1.4, 0.12)
        for k, (g, pv) in enumerate(res):
            yy = y + 0.14 + k * 0.5
            txt = f"{k + 1}. {{i}}{g}{{/i}}" if g != "อื่น ๆ" else f"{k + 1}. {g}"
            s.text(xs[4] + 0.15, yy, w[4] - 0.85, 0.26, P(txt, 11.5, INK, BODYM, space=0), "m")
            s.text(xs[4] + w[4] - 0.75, yy, 0.6, 0.26, P(f"{pv:.0%}", 12, T1 if k == 0 else MUTED, HEADM, "r", space=0), "m")
            s.rect(xs[4] + 0.15, yy + 0.29, w[4] - 0.3, 0.1, PANEL, None, 0, 0.05)
            s.rect(xs[4] + 0.15, yy + 0.29, max(0.04, (w[4] - 0.3) * pv), 0.1, GREEN if k == 0 else SAGE, None, 0, 0.05)
        s.text(xs[4] + 0.15, y + 1.62, w[4] - 0.3, 0.38, P(sp, 10.5, MUTED, space=0, line=1.0), "m")
        for k in range(4):   # ลูกศรระหว่างขั้น
            arrow(s, xs[k] + w[k] + 0.03, y + LH / 2, xs[k + 1] - 0.03, y + LH / 2, T1, 1.6, False)
    s.text(X0, 6.58, XW, 0.32, P("“อื่น ๆ” = สกุลที่มีตัวอย่างน้อยกว่า 5 ชิ้นรวมกัน · ยุคทายแยกรายแหล่งด้วย LightGBM + ผิวดิน "
                                 "เช่น แหล่งถ้ำเสือแก้ว ชุมพร → เพอร์เมียน 99.8%", 11, MUTED, space=0), "m")
    D.append(s)

    # ===== 24c ตัวอย่างจริง: งาน 2 (ค่าจาก models/task2/predictions/grid_5km_predictions.parquet + predict_lib.location_features)
    s = Slide(6, "ตัวอย่างจริง: ระบบคาดการณ์จากพิกัดทำงานอย่างไร", "// TASK 2 · WALK-THROUGH", (
        "ตัวอย่างจริง 2 จุดจากกริด 5 กิโลเมตรที่ใช้ทำแผนที่บนเว็บ จุดแรกอยู่ที่ภูเวียง ขอนแก่น ห่างแหล่งขุดค้นไดโนเสาร์ไม่ถึง 1 กิโลเมตร "
        "ระบบสร้างคุณลักษณะ 152 ตัว เช่น อยู่บนหมวดหินเสาขัว ยุคครีเทเชียส บนแผ่นอินโดจีน ขั้น A XGBoost ให้คะแนน 0.863 สูงกว่าร้อยละ 99.7 ของประเทศ จึงเป็นระดับสูงมาก "
        "ขั้น B Random Forest จัดอันดับประเภทซาก ไดโนเสาร์มาเป็นอันดับหนึ่ง และ AOA บอกว่าจุดนี้คล้ายข้อมูลฝึก ผลจึงเชื่อถือได้ "
        "จุดที่สองอยู่บนเทือกเขาหินแกรนิตยุคไทรแอสซิกในเชียงใหม่ ห่างแหล่งที่ใกล้สุด 88 กิโลเมตร คะแนนต่ำมาก และอยู่นอกขอบเขต AOA "
        "เว็บจึงแสดงว่าข้อมูลไม่พอ แทนที่จะให้ประเภทซากที่ดูมั่นใจ"))
    cols = [("1", "พิกัดที่ถาม", 2.15), ("2", "สร้างคุณลักษณะ 152 ตัว", 2.9), ("3", "ขั้น A · โอกาสพบซาก", 2.2), ("4", "ขั้น B · ประเภทซาก", 2.05),
            ("5", "AOA → ผลบนเว็บ", 1.85)]
    GAP = (XW - sum(c[2] for c in cols)) / 4
    xs, x = [], X0
    for num, t, w in cols:
        xs.append(x)
        s.rect(x, Y0, 0.34, 0.34, T2, None, 0, 0.17, shape="ellipse", ps=P(num, 12, "FFFFFF", HEAD, "c", space=0, line=1.0), inset=0)
        s.text(x + 0.42, Y0 - 0.02, w - 0.42, 0.38, P(t, 13.5, INK, HEADM, space=0), "m")
        x += w + GAP
    LH = 2.08
    lanes = [
        (2.08, "map-pin", "**ภูเวียง · ขอนแก่น**", ["16.662, 102.234", "ห่างแหล่งไดโนเสาร์ 0.9 กม."],
         ["**ธรณี:** หมวดหินเสาขัว · ครีเทเชียส", "หินตะกอน · ครีเทเชียส 55% ใน 5 กม.",
          "**ภูมิประเทศ:** สูง 327 ม. · ชัน 9°", "**โบราณ:** แผ่นอินโดจีน", "**ดาวเทียม:** Sentinel-2 22 ค่า"],
         (0.863, "99.7", "สูงมาก", "10% แรกของประเทศ"),
         [("ไดโนเสาร์", 0.77), ("หอยสองฝา", 0.18), ("ร่องรอย", 0.16)], True,
         ("ใน AOA ✓", "เชื่อถือได้", "แหล่งที่คล้าย: ภูเวียงหลุม 3 · ภูนกเขียน")),
        (4.38, "mountain", "**ภูเขา · เชียงใหม่**", ["19.038, 98.762", "ห่างแหล่งที่ใกล้สุด 88 กม."],
         ["**ธรณี:** หินแกรนิต · ไทรแอสซิก", "หินอัคนี · ไม่มีครีเทเชียสรอบ 5 กม.",
          "**ภูมิประเทศ:** สูง 1,165 ม. · ชัน 18°", "**โบราณ:** แผ่นฉาน-ไทย", "**ดาวเทียม:** Sentinel-2 22 ค่า"],
         (0.001, "2.5", "ต่ำ", "ต่ำกว่าครึ่งประเทศ"),
         [("พืช", 0.37), ("เซฟาโลพอด", 0.29), ("หอยสองฝา", 0.29)], False,
         ("นอก AOA ✗", "ข้อมูลไม่พอ", "ไม่เหมือนแหล่งใดในข้อมูลฝึก")),
    ]
    for y, ic, head, inp, feats, (pot, pct, level, _), types, inside, (aoa, verdict, why) in lanes:
        w = [c[2] for c in cols]
        s.rect(xs[0], y, w[0], LH, SURF, LINE, 1.0, 0.12)
        s.icon(ic, "rust", xs[0] + 0.15, y + 0.14, 0.42, 0.42, "l")
        s.text(xs[0] + 0.65, y + 0.12, w[0] - 0.75, 0.46, P(head, 13, INK, HEADM, bold_color=INK, space=0, line=1.05), "m")
        s.text(xs[0] + 0.15, y + 0.7, w[0] - 0.27, LH - 0.8, P(inp, 11.5, INK2, space=3, line=1.05))
        s.rect(xs[1], y, w[1], LH, T2S, None, 0, 0.12)
        s.text(xs[1] + 0.12, y + 0.1, w[1] - 0.2, LH - 0.16, P(feats, 11, INK2, bold_color=T2, bullet=T2, space=3, line=1.02), "m")
        s.rect(xs[2], y, w[2], LH, SURF, LINE, 1.0, 0.12)
        s.text(xs[2] + 0.15, y + 0.1, w[2] - 0.3, 0.34, P("XGBoost · คะแนน", 11, MUTED, space=0), "m")
        s.text(xs[2] + 0.15, y + 0.42, w[2] - 0.3, 0.5, P(f"{pot:.3f}", 24, T2 if inside else MUTED, HEAD, space=0, line=1.0), "m")
        s.text(xs[2] + 0.15, y + 0.95, w[2] - 0.3, 0.36, P(f"สูงกว่า {pct}% ของประเทศ", 11.5, INK2, space=0), "m")
        s.rect(xs[2] + 0.15, y + 1.38, w[2] - 0.3, 0.5, T2 if inside else PANEL, None, 0, 0.25,
               ps=P(f"ระดับ: {level}", 13.5, "FFFFFF" if inside else INK2, HEADM, "c", space=0, line=1.0), inset=0.02)
        s.rect(xs[3], y, w[3], LH, SURF, LINE, 1.0, 0.12)
        for k, (t, pv) in enumerate(types):
            yy = y + 0.14 + k * 0.5
            s.text(xs[3] + 0.15, yy, w[3] - 0.85, 0.26, P(f"{k + 1}. {t}", 11.5, INK if inside else MUTED, BODYM, space=0), "m")
            s.text(xs[3] + w[3] - 0.7, yy, 0.55, 0.26, P(f"{pv:.2f}", 12, (T2 if k == 0 else MUTED) if inside else DIM, HEADM, "r", space=0), "m")
            s.rect(xs[3] + 0.15, yy + 0.29, w[3] - 0.3, 0.1, PANEL, None, 0, 0.05)
            s.rect(xs[3] + 0.15, yy + 0.29, max(0.04, (w[3] - 0.3) * pv), 0.1, (T2 if k == 0 else ACCHI) if inside else BASE, None, 0, 0.05)
        s.text(xs[3] + 0.15, y + 1.62, w[3] - 0.3, 0.38, P("Random Forest · 16 ประเภท", 10, DIM, space=0), "m")
        s.rect(xs[4], y, w[4], LH, T1S if inside else PILL, None, 0, 0.12)
        s.text(xs[4] + 0.1, y + 0.12, w[4] - 0.2, 0.42, P(aoa, 15, T1 if inside else ACC, HEAD, "c", space=0), "m")
        s.rect(xs[4] + 0.15, y + 0.62, w[4] - 0.3, 0.46, SURF, None, 0, 0.23, ps=P(verdict, 13, INK, HEADM, "c", space=0, line=1.0), inset=0.02)
        s.text(xs[4] + 0.12, y + 1.15, w[4] - 0.24, 0.85, P(why, 10.5, INK2, align="c", space=0, line=1.05), "m")
        for k in range(4):
            arrow(s, xs[k] + w[k] + 0.03, y + LH / 2, xs[k + 1] - 0.03, y + LH / 2, T2, 1.6, False)
    s.text(X0, 6.58, XW, 0.32, P("ระดับ: สูงมาก = 10% แรกของประเทศ · สูง = 25% แรก · ปานกลาง = สูงกว่าครึ่งประเทศ · ต่ำ · "
                                 "ค่าจากกริด 5 กม. 20,706 ช่องที่ใช้ทำแผนที่บนเว็บ", 11, MUTED, space=0), "m")
    D.append(s)

    # ===== 26 เลือกแบบจำลอง
    s = Slide(6, "เลือกแบบจำลองจากผลจริง", "// MODEL SELECTION", (
        "การเลือกแบบจำลองมาจากผลวัดจริง งาน 1 ที่แหล่งเดิม XGBoost ได้ accuracy สูงสุด 0.661 แต่ที่แหล่งใหม่เหลือแค่ 0.275 "
        "เพราะคุณลักษณะของแหล่งชี้ไปหาสกุลของแหล่งอื่นที่คล้ายกัน ส่วน LightGBM ที่ใช้ข้อมูลตัวอย่างอย่างเดียวได้ 0.389 ที่แหล่งใหม่ จึงสลับแบบจำลองตามสถานการณ์ "
        "ยุคใช้ LightGBM เพราะห่างไม่เกิน 1 ยุคสูงกว่า [คลิก] งาน 2 ขั้น A XGBoost ได้ AUC สูงสุด 0.887 และรับค่าว่างจากภาพที่มีเมฆได้ "
        "กราฟจากรายงานแสดง AUC แบบบล็อกพื้นที่ของทุกแบบจำลอง แท่งเข้มคือพื้นหลังแบบถ่วงการสำรวจ แท่งอ่อนคือพื้นหลังสุ่มทั่วประเทศ "
        "ขั้น B Random Forest ได้ macro AUC สูงสุด 0.758"))
    two(s, "accuracy ของแต่ละงานย่อย", "AUC ขั้นโอกาสพบซาก (บล็อกพื้นที่)")
    rows = [["งาน", "ใช้จริง", "เทียบกับ"],
            ["สกุล · แหล่งเดิม", "**XGBoost 0.661**", "LightGBM 0.600 · MLP 0.556 · RF 0.539"],
            ["สกุล · แหล่งใหม่", "**LightGBM 0.389**", "XGBoost 0.275 · LightGBM + แหล่ง 0.266 · MLP 0.256"],
            ["ยุค (acc / ±1 ยุค)", "**LightGBM 0.704 / 0.906**", "RF 0.703 / 0.867 · XGBoost 0.682 / 0.906"]]
    s.table(LX, 2.2, [1.55, 1.75, 2.45], rows, size=12.5, rowh=0.8, rowhs=[0.42, 0.85, 0.85, 0.85], head=T1, align=["l", "l", "l"])
    s.rect(LX, 5.4, HW, 1.4, T1S, None, 0, 0.12)
    s.text(LX + 0.18, 5.45, HW - 0.36, 1.3, P("XGBoost เก่งเมื่อแหล่งมีตัวอย่างแล้ว แต่คุณลักษณะแหล่ง **ชี้ผิดทาง** ในพื้นที่ใหม่ → ระบบ**สลับแบบจำลองตามสถานการณ์**",
                                              13.5, INK2, bold_color=T1, space=0), "m")
    with s.reveal():
        right_head(s)
        frame_img(s, C / "chart_stageA.png", RX, 2.15, HW, 3.62)
        s.rect(RX, 5.87, HW, 0.98, T2S, None, 0, 0.12)
        s.text(RX + 0.18, 5.88, HW - 0.36, 0.96, P(["**ขั้น B (macro AUC):** RF 0.758 · XGBoost 0.745 · LightGBM 0.742 · MLP 0.737 → กราฟหน้าถัดไป",
                                                    "XGBoost รับค่าว่างจากภาพที่มีเมฆได้ในตัว · RF ใช้ต้นไม้ชุดเดียวกับทั้ง 16 ประเภท เหมาะกับ 490 แหล่ง"],
                                                   12, INK2, bold_color=T2, space=2, line=1.05), "m")
    D.append(s)

    # ===== 26b งาน 2 ขั้น B: เลือกแบบจำลองประเภทซาก (กราฟจาก notebooks/10_task2_models)
    s = Slide(6, "งาน 2 ขั้น B: เลือกแบบจำลองทายประเภทซาก", "// TASK 2 · STAGE B", (
        "ขั้น B ทายว่าบริเวณนั้นน่าจะพบซากประเภทไหนจาก 16 ประเภท กราฟซ้ายคือ macro AUC กราฟขวาคือ top-3 คือประเภทที่พบจริงอยู่ใน 3 อันดับที่เสนอ "
        "แท่งสีเทาคือแบ่งแบบสุ่ม สีฟ้าอ่อนคือแบ่งตามกลุ่มแหล่ง และสีน้ำเงินเข้มคือบล็อกพื้นที่ซึ่งเป็นตัวเลขหลัก "
        "Random Forest ได้สูงสุดทั้งสองค่า macro AUC 0.758 และ top-3 0.776 เหนือการทายตามความถี่ที่ได้ 0.5 และ 0.528 ชัดเจน "
        "XGBoost LightGBM และ MLP ได้ใกล้กัน ส่วน CNN ที่ใช้แค่ภาพดาวเทียมได้เพียง 0.587 "
        "เลือก Random Forest เพราะใช้ต้นไม้ชุดเดียวทายทั้ง 16 ประเภท ทุกประเภทจึงใช้ข้อมูลร่วมกัน เหมาะกับข้อมูลเพียง 490 แหล่ง "
        "และเมื่อตรวจกับ PBDB ที่ไม่เคยใช้ฝึกได้ macro AUC 0.776 และ top-3 0.651"))
    frame_img(s, C / "nb_stageB.png", X0, Y0, XW, 3.72, pad=0.1)
    s.text(X0, Y0 + 3.78, XW, 0.34, P("เทา = สุ่ม · ฟ้าอ่อน = กลุ่มแหล่ง · **น้ำเงินเข้ม = บล็อกพื้นที่ (ตัวเลขหลัก)** · prior = ทายตามความถี่ · "
                                      "knn_coords = kNN บนพิกัด · rf = Random Forest · lgbm = LightGBM · xgb = XGBoost",
                                      10, MUTED, bold_color=INK2, space=0), "m")
    cards = [("Random Forest (ใช้จริง)", "macro AUC **0.758** · top-3 **0.776** สูงสุดทั้งสองค่า · กับ PBDB 0.776 / 0.651"),
             ("ทำไมเลือก RF", "ต้นไม้ชุดเดียวทายทั้ง 16 ประเภท ทุกประเภทใช้ข้อมูลร่วมกัน เหมาะกับข้อมูลเพียง 490 แหล่ง"),
             ("เทียบตัวอื่น", "XGBoost 0.745 / 0.759 · LightGBM 0.742 / 0.771 · MLP 0.737 / 0.734 · CNN 0.587 / 0.458 · ทายตามความถี่ 0.5 / 0.528")]
    cw = (XW - 0.4) / 3
    for k, (t, b) in enumerate(cards):
        x = X0 + k * (cw + 0.2)
        s.rect(x, 5.88, cw, 0.98, T2S if k == 0 else SURF, None if k == 0 else LINE, 1.0, 0.12)
        s.text(x + 0.16, 5.9, cw - 0.32, 0.94, P([f"**{t}**", b], 11, INK2, bold_color=T2, space=2, line=1.05), "m")
    D.append(s)

    # ===== 27 Final combined format
    s = Slide(6, "Final Combined Format: ไฟล์ผลลัพธ์สุดท้าย", "// FINAL COMBINED FORMAT", (
        "ผลทั้งหมดรวมเป็นไฟล์ผลลัพธ์สุดท้าย 5 ไฟล์ ไฟล์แรก fossil_list_final รวมทะเบียน 8,521 ชิ้นกับผลการเชื่อมโยง ข้อมูลแหล่ง และยุคสุดท้าย "
        "sites_period_final คือยุคของ 518 แหล่งพร้อมที่มาและความเชื่อมั่น ไฟล์ของงาน 1 คือสกุล 3 อันดับของ 4,577 ชิ้นที่ยังไม่รู้สกุล และยุคที่ทายของ 352 แหล่ง "
        "ไฟล์ของงาน 2 คือกริด 5 กิโลเมตร 20,706 ช่อง เว็บต้นแบบอ่านไฟล์เหล่านี้ผ่าน API และทุกผลมีรหัสตรวจสอบข้อมูลไว้ตรวจย้อนได้"))
    files = [("fossil_list_final.csv", "8,521 × 44", "ทะเบียน + ผลเชื่อม (match_level, score, confidence) + ข้อมูลแหล่ง site_* + ยุคสุดท้าย site_final_*", INK, "ข้อมูลรวม"),
             ("sites_period_final.csv", "518 × 17", "PERIOD_FINAL · final_source · final_confidence · alternative · rejected_by_fossils", INK, "ข้อมูลรวม"),
             ("task1_genus_predictions.csv", "4,577 × 13", "route (แหล่งเดิม / ใหม่) · genus_top1–3 · p_top1–3 · species_guess", T1, "TASK 1"),
             ("task1_period_predictions.csv", "352 × 10", "pred_period · p · pred_period_th · agree_with_05", T1, "TASK 1"),
             ("grid_5km_predictions.parquet", "20,706 × 29", "potential / rank × 2 พื้นหลัง · inside (AOA) · top_type · type_* 16 ประเภท", T2, "TASK 2")]
    for k, (fn, size, cols, col, tag) in enumerate(files):
        y = Y0 + k * 1.0
        s.rect(X0, y, 8.55, 0.88, SURF, LINE, 1.0, 0.1)
        s.icon("file-text", "green" if col == T1 else ("rust" if col == T2 else "ink"), X0 + 0.16, y + 0.2, 0.46, 0.46)
        s.text(X0 + 0.75, y + 0.04, 4.0, 0.4, P(fn, 14, INK, HEADM, space=0), "m")
        pill(s, X0 + 0.8 + tw(fn, 14, HEADM) + 0.15, y + 0.08, size, PANEL, INK2, 11, BODYM, 0.3)
        w = tw(tag, 11, HEAD) + 0.3
        s.rect(X0 + 8.55 - w - 0.14, y + 0.08, w, 0.3, col, None, 0, 0.15, ps=P(tag, 11, "FFFFFF", HEAD, "c", space=0), inset=0.02)
        s.text(X0 + 0.75, y + 0.44, 7.65, 0.42, P(cols, 11.5, MUTED, space=0), "m")
    s.rect(9.4, Y0, 3.33, 4.88, PANEL, None, 0, 0.14)
    s.icon("database", "ink", 9.6, Y0 + 0.2, 0.46, 0.46, "l")
    s.text(10.2, Y0 + 0.18, 2.4, 0.5, P("ไฟล์ทั้งหมด", 14, INK, HEADM, space=0), "m")
    arrow(s, 11.07, Y0 + 0.8, 11.07, Y0 + 1.2, DIM, 1.5)
    box(s, 9.6, Y0 + 1.25, 2.93, 0.75, "API ของเว็บต้นแบบ\n(FastAPI)", SURF, LINE, 13)
    arrow(s, 11.07, Y0 + 2.05, 11.07, Y0 + 2.45, DIM, 1.5)
    box(s, 9.6, Y0 + 2.5, 2.93, 0.75, "แผนที่ · วิเคราะห์จุด\nทายชื่อซาก", SURF, LINE, 13)
    s.text(9.6, Y0 + 3.4, 2.93, 1.4, P("ทุกผลบันทึก **รหัสตรวจสอบข้อมูล (Fingerprint)** ของข้อมูลและโค้ด → ตรวจย้อนได้ว่าผลมาจากข้อมูลชุดไหน",
                                       12, INK2, bold_color=INK, space=0))
    D.append(s)

    # ===== 28 วัดผลอย่างไร
    s = Slide(7, "วัดผลอย่างไรให้ซื่อตรง", "// VALIDATION DESIGN", (
        "การวัดผลออกแบบให้สะท้อนการใช้งานจริง ภาพซ้ายคือการแบ่งข้อมูล 3 แบบบนพื้นที่เดียวกันทางภาคใต้ แบบสุ่มทำให้ข้อสอบมีจุดใกล้ ๆ อยู่ในชุดฝึก "
        "ระยะถึงแหล่งฝึกที่ใกล้ที่สุดมัธยฐานแค่ 3.1 กิโลเมตร จึงใช้วัดงานเติมชื่อในแหล่งเดิม ส่วนแบบบล็อกพื้นที่ 50 กิโลเมตร ระยะมัธยฐานเพิ่มเป็น 20.4 กิโลเมตร "
        "เป็นพื้นที่ที่แบบจำลองไม่เคยเห็น จึงเป็นตัวเลขหลัก นอกจากนี้ยังตรวจกับ PBDB ที่ไม่เคยใช้ฝึก และใช้ AOA บอกขอบเขตที่เชื่อถือได้"))
    frame_img(s, A / "maps" / "map_cv.png", X0, Y0, 7.6, 4.15, pad=0.08)
    s.text(X0, 5.82, 7.6, 0.4, P("ก. แบบสุ่ม (ระยะถึงแหล่งฝึกที่ใกล้สุด มัธยฐาน 3.1 กม.) · ข. แบบกลุ่มแหล่ง (5.9 กม.) · ค. แบบบล็อกพื้นที่ (20.4 กม.)",
                                 11, DIM, space=0))
    ev = [("shuffle", "แบบสุ่ม → แหล่งเดิม", "ข้อสอบมีจุดใกล้ ๆ ในชุดฝึก · วัดงานเติมชื่อในแหล่งที่รู้จัก"),
          ("map", "บล็อก 50 กม. → แหล่งใหม่", "**ตัวเลขหลัก** — ทั้งบล็อกเป็นพื้นที่ที่ไม่เคยเห็น"),
          ("database", "PBDB 290 จุด → ทดสอบอิสระ", "ข้อมูลคนละแหล่ง ไม่เคยใช้ฝึก"),
          ("shield-check", "AOA → บอกขอบเขต", "เตือนเมื่อจุดไม่คล้ายข้อมูลฝึก")]
    for k, (ic, t, b) in enumerate(ev):
        y = Y0 + k * 1.08
        card(s, 8.45, y, 4.28, 0.96, t, b, ic if ic != "shuffle" else "git-branch", tsize=14, bsize=12, isize=0.36, pad=0.14, bold_color=ACC)
    s.text(X0, 6.28, 1.7, 0.42, P("ตัวชี้วัดที่ใช้", 13, INK, HEADM, space=0), "m")
    x = X0 + 1.55
    for m in ["AUC", "PR-AUC", "top-3", "accuracy", "ห่างไม่เกิน 1 ยุค", "F1 macro / weighted", "precision@3"]:
        x += pill(s, x, 6.33, m, PILL, INK, 12, BODYM, 0.34) + 0.12
    D.append(s)

    # ===== 29 ผลหลัก
    s = Slide(7, "ผลลัพธ์หลัก", "// RESULTS", (
        "ผลหลักของงาน 1 ชื่อสกุลจริงอยู่ใน 3 ชื่อแรกร้อยละ 86.3 ที่แหล่งเดิม และร้อยละ 58.7 ที่แหล่งใหม่ แบบจำลองยุคทายถูกร้อยละ 71.9 "
        "และทายห่างไม่เกิน 1 ยุคร้อยละ 90 ถ้าผู้ใช้บอกแค่หน้าตาของซาก ชื่อจริงยังอยู่ใน 3 ชื่อแรกร้อยละ 72.3 "
        "[คลิก] งาน 2 ขั้นโอกาสพบซากได้ AUC 0.887 ในพื้นที่ที่ไม่เคยเห็น และ 0.796 กับ PBDB ที่ไม่เคยใช้ฝึก "
        "ถ้าสำรวจแค่ร้อยละ 10 ของประเทศที่คะแนนสูงสุด จะครอบคลุมจุด PBDB ได้ร้อยละ 48 ราว 5 เท่าของการสุ่ม ส่วนประเภทซากได้ macro AUC 0.758"))
    two(s, "ทายชื่อและยุคของซาก", "คาดการณ์จากพิกัด")

    def grid4(x, items, col, tint):
        for k, (v, lab) in enumerate(items):
            cx, cy = x + (k % 2) * 2.95, 2.2 + (k // 2) * 2.3
            s.rect(cx, cy, 2.8, 2.12, tint, None, 0, 0.14)
            stat(s, cx + 0.2, cy + 0.15, 2.45, v, lab, col, 36, 12.5, h=1.85)
    grid4(LX, [("86.3%", "สกุลจริงอยู่ใน 3 ชื่อแรก · แหล่งเดิม (accuracy 66.7%)"), ("58.7%", "สกุลจริงอยู่ใน 3 ชื่อแรก · แหล่งใหม่ (accuracy 39.6%)"),
               ("71.9%", "ทายยุคของแหล่งถูก · ห่างไม่เกิน 1 ยุค 90.0%"), ("72.3%", "top-3 เมื่อบอกแค่หน้าตาของซาก (แหล่งใหม่ 47.2%)")], T1, T1S)
    with s.reveal():
        right_head(s)
        grid4(RX, [("0.887", "AUC โอกาสพบซาก พื้นที่ที่ไม่เคยเห็น (0.930 พื้นหลังสุ่ม)"), ("0.796", "AUC กับ PBDB ที่ไม่เคยใช้ฝึก"),
                   ("48%", "ของจุด PBDB อยู่ใน 10% ของประเทศที่คะแนนสูงสุด (สุ่ม ≈ 10%)"), ("0.758", "macro AUC ประเภทซาก 16 ประเภท · top-3 0.776")], T2, T2S)
    D.append(s)

    # ===== 30 เส้นทางพัฒนาระบบทายสกุล
    s = Slide(7, "งาน 1: เส้นทางพัฒนาระบบทายสกุล (top-3)", "// TASK 1 · GENUS", (
        "กราฟนี้จากรายงานแสดงเส้นทางพัฒนาระบบทายสกุล วัดด้วย top-3 แท่งสีเข้มคือแหล่งเดิม แท่งสีอ่อนคือแหล่งใหม่ "
        "รอบแรก XGBoost ดูดีมาก 0.892 แต่มีคุณลักษณะรั่ว พอตัดออกเหลือ 0.852 ซึ่งเป็นตัวเลขจริง "
        "การสลับแบบจำลองตามสถานการณ์และตัวกรองช่วยแหล่งใหม่ และการเพิ่มข้อความกับหมวดหินช่วยแหล่งใหม่มากที่สุด จาก 0.546 เป็น 0.587"))
    frame_img(s, C / "chart_genus.png", X0, Y0, 7.9, 3.86)
    s.text(X0, Y0 + 4.02, 7.9, 1.0, P(["แท่งเข้ม = แหล่งเดิม (วัดแบบสุ่ม) · แท่งอ่อน = แหล่งใหม่ (วัดแบบบล็อกพื้นที่ 50 กม.)",
                                       "แถวล่างสุด (ตัวหนา) = ระบบที่ใช้จริงบนเว็บ"], 12.5, MUTED, bullet=DIM, space=4))
    ins = [("ตัวเลขลดหลังแก้ข้อมูลรั่ว", "0.892 → 0.852 (แหล่งเดิม) — เป็นตัวเลขที่ใช้งานจริงได้"),
           ("สลับแบบจำลองช่วยแหล่งใหม่", "0.515 → 0.546 เพราะไม่ใช้คุณลักษณะแหล่งที่ชี้ผิดทาง"),
           ("ข้อความ + หมวดหินช่วยมากที่สุด", "แหล่งใหม่ 0.546 → 0.587 (WangchanBERTa + ThaiLex)")]
    for k, (t, b) in enumerate(ins):
        card(s, 8.8, Y0 + k * 1.72, 3.93, 1.56, t, b, tsize=14.5, bsize=12.5)
    D.append(s)

    # ===== 31 แผนที่งาน 2
    s = Slide(7, "งาน 2: แผนที่โอกาสพบซากทั่วประเทศ", "// TASK 2 · MAPS", (
        "ผลของงาน 2 นำมาทำแผนที่ทั้งประเทศในกริด 5 กิโลเมตร ภาพซ้ายคือโอกาสพบซากรวม พื้นที่สีเข้มคือคะแนนสูง "
        "ภาพขวาคือแผนที่รายประเภท ไดโนเสาร์เด่นบนที่ราบสูงโคราช แบรคิโอพอดเด่นในหินปูนทางเหนือและตะวันตก "
        "ถ้าสำรวจแค่ร้อยละ 10 ของประเทศที่คะแนนสูงสุดจะครอบคลุมจุด PBDB ร้อยละ 48 และถ้าร้อยละ 25 จะครอบคลุมร้อยละ 69"))
    frame_img(s, A / "maps" / "potential_bg_bias.png", X0, Y0, 3.3, 4.95, pad=0.1)
    s.text(X0, 6.6, 3.3, 0.3, P("โอกาสพบซาก (XGBoost ขั้น A)", 10.5, DIM, align="c", space=0))
    for k, (f, lab) in enumerate([("type_dino", "ไดโนเสาร์"), ("type_brachiopod", "แบรคิโอพอด"), ("type_mammal", "สัตว์เลี้ยงลูกด้วยนม")]):
        x = 4.15 + k * 2.03
        frame_img(s, A / "maps" / f"{f}.png", x, Y0, 1.9, 3.3, pad=0.06)
        s.text(x, Y0 + 3.35, 1.9, 0.36, P(lab, 12.5, INK, HEADM, "c", space=0), "m")
    for k, (v, lab) in enumerate([("48%", "ของจุด PBDB อยู่ใน 10% ของประเทศที่คะแนนสูงสุด"), ("69%", "อยู่ใน 25% ของประเทศที่คะแนนสูงสุด"),
                                  ("93.3%", "ของกริดอยู่ในขอบเขตที่เชื่อถือได้ (AOA)")]):
        y = Y0 + k * 1.75
        s.rect(10.4, y, 2.33, 1.6, T2S, None, 0, 0.12)
        stat(s, 10.55, y + 0.1, 2.05, v, lab, T2, 28, 11.5, h=1.45)
    s.text(4.15, Y0 + 3.85, 6.0, 1.3, P(["สีเข้ม = โอกาสสูง · แผนที่รายประเภท = คะแนนประเภท × โอกาสพบซาก",
                                         "ไดโนเสาร์เด่นบนที่ราบสูงโคราช · แบรคิโอพอดเด่นในหินปูนทางเหนือและตะวันตก"], 12.5, MUTED, bullet=DIM, space=4))
    D.append(s)

    # ===== 32 precision recall F1
    s = Slide(7, "Precision / Recall / F1: ดูว่าพลาดแบบไหน", "// PRECISION · RECALL · F1", (
        "นอกจาก accuracy และ AUC ยังวัด precision recall และ F1 เพื่อดูว่าระบบพลาดแบบไหน งาน 1 F1 แบบ macro ต่ำกว่าแบบ weighted มากทุกงาน "
        "แปลว่าสกุลหรือยุคที่พบบ่อยทายได้ดี แต่ที่พบน้อยยังทายได้ไม่ดี จึงเสนอ 3 ชื่อให้ผู้เชี่ยวชาญพิจารณา และ XGBoost ที่ไม่ได้ถ่วงน้ำหนักคลาสหาสกุลที่พบน้อยเจอน้อยกว่า LightGBM "
        "ภาพซ้ายคือตารางความสับสนของการทายยุคแบบบล็อกพื้นที่จากรายงาน ยุคที่มีมากอย่างเพอร์เมียนถูกเกือบทั้งหมด แต่ยุคที่มีแหล่งน้อยอย่างแคมเบรียน ดีโวเนียน และควอเทอร์นารี ถูกทายเป็นยุคข้างเคียงหรือยุคที่พบบ่อย "
        "[คลิก] งาน 2 PR-AUC 0.626 ดีกว่าการเดาสุ่มราว 6.7 เท่า ขั้นประเภทซาก ใน 3 ประเภทที่เสนอ ถูกจริงเฉลี่ยราว 1.3 ประเภท "
        "และถ้าใช้เกณฑ์ 0.5 ตายตัว Random Forest จะหาเจอแค่ร้อยละ 21 เว็บจึงเสนอ 3 อันดับแรกแทน"))
    two(s, "F1 แบบ macro เทียบ weighted", "ขั้น A และขั้น B")
    frame_img(s, C / "cm_period_spatial_block.png", LX, 2.2, 3.25, 3.25)
    s.table(3.98, 2.2, [0.85, 0.68, 0.85], [["F1", "macro", "weighted"], ["สกุล เดิม", "0.401", "0.634"], ["สกุล ใหม่", "0.118", "0.348"],
                                           ["ยุค", "0.491", "0.764"]], size=12, rowh=0.42, head=T1, align=["l", "r", "r"])
    s.text(3.98, 4.0, 2.38, 1.45, P("ภาพซ้าย: ทายยุคแบบบล็อกพื้นที่ — แถว = ยุคจริง · คอลัมน์ = ยุคที่ทาย · ตัวเลข = % ของแถว",
                                   11.5, MUTED, space=0, line=1.08), "b")
    bullets(s, LX, 5.6, HW, 1.25, ["macro ต่ำกว่า weighted มาก → **สกุล / ยุคที่พบน้อยยังทายได้ไม่ดี** จึงเสนอ 3 ชื่อให้ผู้เชี่ยวชาญ",
                                   "XGBoost ไม่ถ่วงน้ำหนักคลาส: recall macro 0.367 (LightGBM 0.502)"], 12.5, bold_color=T1, space=4)
    with s.reveal():
        right_head(s)
        items = [("6.7×", "PR-AUC 0.626 เทียบการเดาสุ่ม 0.094 (ขั้น A)"), ("0.594", "F1 สูงสุด · precision 0.616 / recall 0.573"),
                 ("≈ 1.3 / 3", "ประเภทที่เสนอ 3 อันดับ ถูกจริงเฉลี่ย (precision@3 0.419 · recall@3 0.494)"),
                 ("21%", "หาเจอที่เกณฑ์ 0.5 ตายตัว → เว็บใช้ 3 อันดับแรกแทน")]
        for k, (v, lab) in enumerate(items):
            cx, cy = RX + (k % 2) * 2.95, 2.2 + (k // 2) * 2.3
            s.rect(cx, cy, 2.8, 2.12, T2S, None, 0, 0.14)
            stat(s, cx + 0.2, cy + 0.15, 2.45, v, lab, T2, 32, 12, h=1.85)
    D.append(s)

    # ===== 33 เกณฑ์ความสำเร็จ
    s = Slide(7, "เกณฑ์ความสำเร็จ 15 ข้อ: ผ่านขั้นต่ำครบ ถึงเป้าหมาย 7 ข้อ", "// SUCCESS CRITERIA", (
        "โครงงานตั้งเกณฑ์ความสำเร็จไว้ 15 ข้อ แต่ละข้อมีค่าขั้นต่ำที่ถือว่าใช้งานได้ และค่าเป้าหมายที่ถือว่าดี ผลคือผ่านเกณฑ์ขั้นต่ำครบทุกข้อ "
        "และถึงเป้าหมาย 7 ข้อ ขั้นโอกาสพบซากแข็งแรงที่สุด ส่วนประเภทซาก สกุลที่แหล่งใหม่ และยุค อยู่ระหว่างขั้นต่ำกับเป้าหมาย "
        "ซึ่งข้อจำกัดหลักมาจากข้อมูล"))
    crit = [(1, "โอกาสพบซาก AUC บล็อก", "0.887", "≥ 0.80", True, T2), (2, "โอกาสพบซาก AUC PBDB", "0.796", "≥ 0.80", False, T2),
            (3, "PBDB ใน 10% บนสุด", "0.479", "≥ 0.40", True, T2), (4, "AUC สูงกว่า kNN พิกัด", "+0.320", "≥ 0.20", True, T2),
            (5, "ประเภท macro AUC", "0.758", "≥ 0.80", False, T2), (6, "ประเภท PBDB top-3", "0.651", "≥ 0.70", False, T2),
            (7, "ประเภท สัดส่วนใน AOA", "0.752", "≥ 0.90", False, T2), (8, "สกุล top-3 แหล่งเดิม", "0.863", "≥ 0.90", False, T1),
            (9, "สกุล top-3 แหล่งใหม่", "0.587", "≥ 0.70", False, T1), (10, "สกุล ทายข้ามกลุ่มซาก", "0 ครั้ง", "0", True, T1),
            (11, "ยุค accuracy บล็อก", "0.719", "≥ 0.80", False, T1), (12, "ยุค ห่างไม่เกิน 1 ยุค", "0.900", "≥ 0.95", False, T1),
            (13, "ไม่มีข้อมูลรั่ว", "ผ่าน", "ผ่าน", True, INK), (14, "ทุกผลมีรหัสตรวจสอบ", "ผ่าน", "ผ่าน", True, INK),
            (15, "ทำนายตรงกับตอนฝึก", "100%", "ผ่าน", True, INK)]
    for k, (n, lab, v, goal, hit, col) in enumerate(crit):
        x, y = X0 + (k % 5) * 2.45, Y0 + (k // 5) * 1.62
        s.rect(x, y, 2.3, 1.48, T1S if hit else PILL, None, 0, 0.12)
        s.rect(x + 0.14, y + 0.14, 0.42, 0.3, col, None, 0, 0.15, ps=P(str(n), 11, "FFFFFF", HEAD, "c", space=0), inset=0.0)
        s.text(x + 0.64, y + 0.08, 1.6, 0.42, P(lab, 11, INK2, space=0, line=1.0), "m")
        s.text(x + 0.14, y + 0.52, 2.05, 0.55, P(v, 22, INK, HEAD, space=0, line=1.0), "m")
        s.text(x + 0.14, y + 1.08, 2.05, 0.32, P(f"เป้าหมาย {goal}", 10.5, MUTED, space=0), "m")
    for k, (lab, fill) in enumerate([("ถึงเป้าหมาย (7)", T1S), ("ผ่านขั้นต่ำ ยังไม่ถึงเป้าหมาย (8)", PILL)]):
        s.rect(X0 + k * 2.6, 6.55, 0.3, 0.3, fill, LINE, 1.0, 0.06)
        s.text(X0 + 0.4 + k * 2.6, 6.5, 2.2, 0.4, P(lab, 12, INK2, space=0), "m")
    for k, (lab, col) in enumerate([("งาน 2", T2), ("งาน 1", T1), ("ระบบ", INK)]):
        s.rect(7.2 + k * 1.4, 6.6, 0.42, 0.24, col, None, 0, 0.12)
        s.text(7.68 + k * 1.4, 6.5, 0.9, 0.4, P(lab, 12, INK2, space=0), "m")
    D.append(s)

    # ===== 34 ข้อจำกัด: ข้อมูลน้อย โครงข่ายประสาทจึงยังไม่แม่น (ตัวเลขจากรายงาน 3.1.5 และตารางที่ 4.14)
    s = Slide(7, "ข้อจำกัด: ข้อมูลน้อยเกินไป โครงข่ายประสาทจึงยังไม่แม่น", "// LIMITATIONS", (
        "ข้อจำกัดหลักของโครงงานคือข้อมูลที่มีคำตอบมีน้อย แหล่งซากมี 518 แหล่ง แหล่งที่มีป้ายประเภทมี 490 แหล่ง แหล่งที่รู้ยุคมีเพียง 166 แหล่ง "
        "และตัวอย่างที่มีสกุล 3,944 ชิ้นกระจายอยู่ในหลายสกุล หลายสกุลมีไม่ถึง 5 ชิ้น "
        "โครงข่ายประสาทเทียมมีพารามิเตอร์มาก MLP ราวหนึ่งแสน CNN ราวสองแสนเก้าหมื่น จึงต้องการข้อมูลมากกว่านี้ "
        "ผลคือแบบจำลองกลุ่มต้นไม้ชนะโครงข่ายประสาทในทุกงาน เช่น ขั้นโอกาสพบซาก XGBoost ได้ 0.887 แต่ MLP 0.846 และ CNN 0.792 "
        "สกุลที่แหล่งใหม่ LightGBM ได้ 0.389 แต่ MLP เพียง 0.256 ซึ่งตรงกับงานของ Grinsztajn และคณะที่พบว่าต้นไม้เหมาะกับข้อมูลตารางขนาดกลาง "
        "ส่วน CNN ยังมีอีกเหตุผลคือพื้นที่ส่วนใหญ่ของไทยมีพืชคลุม ภาพจึงเห็นหินน้อย และเมื่อตรวจกับ PBDB ผลลดลง "
        "ข้อมูลน้อยยังทำให้สกุลและยุคที่พบน้อยทายได้ไม่ดี จูนค่าตั้งและปรับแต่งแบบจำลองภาษาไม่ได้ แนวทางแก้จึงต้องเริ่มจากการเพิ่มข้อมูล"))
    LWL = 4.55
    s.text(X0, Y0, LWL, 0.4, P("ข้อมูลที่มีคำตอบมีน้อย", 15, INK, HEADM, space=0), "m")
    tiles = [("518", "แหล่งซาก (ขั้นโอกาสพบซาก)", T2), ("490", "แหล่งที่มีป้ายประเภท", T2), ("166", "แหล่งที่รู้ยุค", T1),
             ("3,944", "ชิ้นที่มีสกุล · หลายสกุลไม่ถึง 5 ชิ้น", T1)]
    tw2 = (LWL - 0.15) / 2
    for k, (v, lab, col) in enumerate(tiles):
        x, y = X0 + (k % 2) * (tw2 + 0.15), 2.12 + (k // 2) * 1.42
        s.rect(x, y, tw2, 1.3, T1S if col == T1 else T2S, None, 0, 0.12)
        stat(s, x + 0.16, y + 0.08, tw2 - 0.3, v, lab, col, 28, 11.5, h=1.15)
    s.rect(X0, 5.05, LWL, 1.12, None, ACC, 1.2, 0.12)
    s.text(X0 + 0.18, 5.07, LWL - 0.36, 1.08, P(["**โครงข่ายประสาทมีพารามิเตอร์มาก**",
                                                 "MLP ≈ 100,000 · CNN ≈ 290,000 พารามิเตอร์ แต่มีจุดฝึกราว 4,400 จุดต่อรอบ และภาพเพียงหลักพันภาพ"],
                                                11.5, INK2, bold_color=ACC, space=2, line=1.05), "m")
    RXL = X0 + LWL + 0.3
    RWL = XW - LWL - 0.3
    s.text(RXL, Y0, RWL, 0.4, P("ต้นไม้ชนะโครงข่ายประสาททุกงาน (วัดชุดเดียวกัน)", 15, INK, HEADM, space=0), "m")
    rows = [["งาน · ตัวชี้วัด", "ต้นไม้ (ใช้จริง)", "MLP", "CNN"],
            ["โอกาสพบซาก · AUC", "**XGBoost 0.887**", "0.846", "0.792"],
            ["ประเภทซาก · macro AUC", "**RF 0.758**", "0.737", "0.587"],
            ["สกุลแหล่งเดิม · accuracy", "**XGBoost 0.661**", "0.556", "–"],
            ["สกุลแหล่งใหม่ · accuracy", "**LightGBM 0.389**", "0.256", "–"],
            ["ยุค · accuracy", "**LightGBM 0.704**", "0.655", "–"]]
    s.table(RXL, 2.12, [2.55, 2.0, 1.21, 1.21], rows, size=12.5, rowh=0.46, head=INK, align=["l", "l", "r", "r"], hl={1: T2S, 2: T2S})
    s.text(RXL, 4.98, RWL, 1.2, P(["สอดคล้องกับ Grinsztajn และคณะ (2022): ต้นไม้เหมาะกับข้อมูลตารางขนาดกลาง",
                                   "CNN: ไทยส่วนใหญ่มีพืชคลุม ภาพเห็นหินน้อย · รวม CNN แล้วผลกับ PBDB ลดลง (0.797 → 0.729)"],
                                  12, INK2, bullet=DIM, space=5, line=1.05))
    s.rect(X0, 6.3, XW, 0.56, PANEL, None, 0, 0.12)
    s.text(X0 + 0.2, 6.3, XW - 0.4, 0.56, P("**ข้อมูลน้อยยังทำให้:** สกุล / ยุคที่พบน้อยทายได้ไม่ดี (F1 macro สกุลแหล่งใหม่ 0.118) · จูนค่าตั้งไม่ได้ · "
                                             "ปรับแต่ง WangchanBERTa ไม่ได้ → เพิ่มข้อมูลก่อน (หน้า Future Plans)", 12, INK2, bold_color=ACC, space=0), "m")
    D.append(s)

    # ===== 35 สถาปัตยกรรมเว็บ
    s = Slide(8, "เว็บต้นแบบ: สถาปัตยกรรม", "// DEPLOYMENT", (
        "เว็บต้นแบบแบ่งเป็นหน้าเว็บกับเซิร์ฟเวอร์ หน้าเว็บมีแผนที่ MapLibre ภาพสามมิติ และรองรับภาษาไทยกับอังกฤษ เซิร์ฟเวอร์เป็น FastAPI "
        "ซึ่งเรียกโค้ดทำนายชุดเดียวกับตอนฝึกแบบจำลอง จึงได้ผลตรงกับตอนฝึกร้อยละ 100 ตามชุดทดสอบอัตโนมัติ 18 ข้อ "
        "แผนที่และข้อมูลที่ไม่เปลี่ยนคำนวณไว้ล่วงหน้า ส่วนตอนคลิกวิเคราะห์จุด เซิร์ฟเวอร์ดึงภาพ Sentinel-2 ของจุดนั้นมาคำนวณสด"))
    lay = [(X0, "monitor", "หน้าเว็บ (เบราว์เซอร์)", "HTML / JS · แผนที่ MapLibre\nภาพสามมิติ · ไทย / อังกฤษ\nโหมดสว่าง / มืด"),
           (4.75, "cog", "เซิร์ฟเวอร์ FastAPI", "webapp/server.py\nเรียก predict_lib = โค้ดทำนาย\nชุดเดียวกับตอนฝึก"),
           (8.9, "database", "โมเดลและข้อมูล", "โมเดล .joblib ราว 140 MB\nกริด 5 กม. (คำนวณล่วงหน้า)\nแผนที่ธรณี · DEM · แหล่ง")]
    for x, ic, t, b in lay:
        card(s, x, Y0, 3.83, 2.05, t, b, ic, tsize=15.5, bsize=12.5, isize=0.46)
    arrow(s, 4.47, Y0 + 1.0, 4.72, Y0 + 1.0, DIM, 1.6)
    arrow(s, 8.62, Y0 + 1.0, 8.87, Y0 + 1.0, DIM, 1.6)
    box(s, 4.75, Y0 + 2.3, 3.83, 0.62, "ภาพ Sentinel-2 (Planetary Computer) ดึงสดตอนวิเคราะห์จุด", PILL, None, 12, INK)
    arrow(s, 6.66, Y0 + 2.27, 6.66, Y0 + 2.08, DIM, 1.4)
    rows = [["API", "หน้าที่"],
            ["GET /api/grid · /api/meta · /api/options", "กริด 5 กม. · ข้อมูลแบบจำลอง · ตัวเลือกของฟอร์ม"],
            ["GET /api/geology · /api/provinces · /api/pbdb", "ชั้นยุคของหิน · ขอบเขตจังหวัด · จุด PBDB"],
            ["GET /api/showcase · /api/period_profile", "ซากตัวอย่างหน้าแรก · ยุคที่พบของแต่ละสกุล"],
            ["POST /api/predict_location", "วิเคราะห์พิกัด: โอกาสพบซาก ประเภท AOA แหล่งที่คล้าย"],
            ["POST /api/predict_specimen", "ทายชื่อซากจากประเภทหรือหน้าตา + สถานที่พบ"]]
    s.table(X0, 4.72, [5.4, 6.73], rows, size=12, rowh=0.36, head=INK, align=["l", "l"], hl={4: T2S, 5: T1S})
    D.append(s)

    # ===== 36 หน้าจอ: หน้าแรก + แผนที่
    s = Slide(8, "หน้าจอเว็บ: หน้าแรก และแผนที่โอกาสพบซาก", "// DEMO · 1", (
        "หน้าแรกของเว็บแสดงซากตัวอย่างข้ามยุคบนชั้นหินแบบสามมิติ พร้อมตัวเลขสรุปของข้อมูล เพื่อดึงความสนใจของคนทั่วไป "
        "หน้าแผนที่แสดงโอกาสพบซากรายจังหวัดและอันดับจังหวัดที่น่าสำรวจ ผู้ใช้เลือกดูได้ทั้งโอกาสพบซากรวมและรายประเภท"))
    for k, (f, t, b) in enumerate([("web_index", "หน้าแรก", "ซากตัวอย่างข้ามยุคบนชั้นหิน + ตัวเลขสรุป"),
                                   ("web_map", "แผนที่", "โอกาสพบซากรายจังหวัด + อันดับจังหวัดที่น่าสำรวจ")]):
        x = X0 + k * 6.17
        frame_img(s, A / f"{f}.png", x, Y0, 5.96, 3.83, pad=0.06)
        s.text(x, Y0 + 3.95, 5.96, 0.42, P(t, 15, INK, HEADM, space=0), "m")
        s.text(x, Y0 + 4.38, 5.96, 0.75, P(b, 13, MUTED, space=0))
    D.append(s)

    # ===== 37 หน้าจอ: วิเคราะห์จุด + ชั้นหิน
    s = Slide(8, "หน้าจอเว็บ: วิเคราะห์จุดที่คลิก และชั้นยุคของหิน", "// DEMO · 2", (
        "เมื่อคลิกจุดบนแผนที่ เช่น แหล่งภูเวียง เว็บจะดึงภาพดาวเทียมของจุดนั้น คำนวณคุณลักษณะ แล้วบอกโอกาสพบซาก อันดับเทียบทั้งประเทศ "
        "ประเภทซาก 3 อันดับ และแหล่งที่คล้ายกัน ส่วนชั้นยุคของหินแสดงหน่วยหินจากแผนที่ 1:250,000 และบอกรายละเอียดหมวดหินเมื่อชี้เมาส์"))
    for k, (f, t, b) in enumerate([("web_point", "วิเคราะห์จุดที่คลิก (ภูเวียง)", "โอกาสพบซาก · อันดับเทียบทั้งประเทศ · ประเภท 3 อันดับ · แหล่งที่คล้าย"),
                                   ("web_geology", "ชั้นยุคของหิน", "หน่วยหินจากแผนที่ 1:250,000 · รายละเอียดหมวดหินเมื่อชี้เมาส์")]):
        x = X0 + k * 6.17
        frame_img(s, A / f"{f}.png", x, Y0, 5.96, 3.83, pad=0.06)
        s.text(x, Y0 + 3.95, 5.96, 0.42, P(t, 15, INK, HEADM, space=0), "m")
        s.text(x, Y0 + 4.38, 5.96, 0.75, P(b, 13, MUTED, space=0))
    D.append(s)

    # ===== 38 หน้าจอ: ทายชื่อซาก
    s = Slide(8, "หน้าจอเว็บ: ทายชื่อซากจากหน้าตา", "// DEMO · 3", (
        "หน้าทายชื่อซากออกแบบให้คนทั่วไปใช้ได้ แม้ไม่รู้ประเภทของซาก ผู้ใช้เลือกการ์ดหน้าตา เช่น ฟัน แล้วตอบคำถามต่อ เช่น รูปกรวย "
        "บอกสถานที่พบ แล้วเว็บจะเสนอสกุล 3 อันดับ ยุคที่มักพบ และภาพสามมิติ ในการทดลอง ถ้าบอกแค่หน้าตา ชื่อจริงอยู่ใน 3 ชื่อแรกร้อยละ 72.3 ที่แหล่งเดิม"))
    frame_img(s, A / "web_identify.png", X0, Y0, 7.9, 5.08, pad=0.06)
    steps = [("1", "เลือกการ์ดหน้าตา", "เช่น ฟัน → รูปกรวย (ไม่ต้องรู้ประเภทซาก)"), ("2", "บอกที่พบ", "แหล่ง / จังหวัด / พิกัด"),
             ("3", "ได้ผล", "สกุล 3 อันดับ + ยุคที่มักพบ + ภาพสามมิติ")]
    for k, (n, t, b) in enumerate(steps):
        y = Y0 + k * 1.12
        s.rect(8.8, y, 0.5, 0.5, T1, None, 0, shape="ellipse", ps=P(n, 14, "FFFFFF", HEAD, "c", space=0), inset=0)
        s.text(9.45, y - 0.04, 3.3, 0.4, P(t, 15, INK, HEADM, space=0), "m")
        s.text(9.45, y + 0.36, 3.3, 0.66, P(b, 12.5, MUTED, space=0))
    s.rect(8.8, 5.05, 3.93, 1.65, T1S, None, 0, 0.14)
    stat(s, 9.0, 5.12, 3.6, "72.3%", "top-3 เมื่อบอกแค่หน้าตา (แหล่งเดิม) · แหล่งใหม่ 47.2%", T1, 30, 12, h=1.5)
    D.append(s)

    # ===== 39 สรุป: สิ่งที่ทำได้ (ตัวเลขจากรายงานบทที่ 4–5)
    s = Slide(None, "สรุป: สิ่งที่โครงงานทำได้", "// CONCLUSION", (
        "สรุปสิ่งที่โครงงานทำได้ 5 เรื่อง เรื่องแรกจัดการข้อมูลซากของ ทธ. ให้เป็นระเบียบและใช้ร่วมกันได้ เชื่อมทะเบียน 8,521 ชิ้นกับแหล่ง 518 แหล่งได้ร้อยละ 92 "
        "เติมยุคจนแหล่งที่ไม่ทราบยุคลดจาก 352 เหลือ 129 แหล่ง และได้ไฟล์ข้อมูลรวมพร้อมใช้ "
        "เรื่องที่สองสร้างคุณลักษณะเชิงพื้นที่ 152 ตัวต่อจุด และคุณลักษณะจากข้อความภาษาไทย "
        "เรื่องที่สามแบบจำลองงาน 1 สกุลจริงอยู่ใน 3 ชื่อแรกร้อยละ 86.3 ที่แหล่งเดิม และ 58.7 ที่แหล่งใหม่ ทายยุคถูกร้อยละ 71.9 "
        "และเติมชื่อให้ 4,577 ชิ้น ยุคให้ 352 แหล่ง เรื่องที่สี่แบบจำลองงาน 2 ได้ AUC 0.887 ในพื้นที่ที่ไม่เคยเห็น และ 0.796 กับ PBDB "
        "ถ้าสำรวจเพียงร้อยละ 10 ของประเทศที่คะแนนสูงสุดจะครอบคลุมจุด PBDB ร้อยละ 48 เรื่องที่ห้าทำเว็บต้นแบบภาษาไทยและอังกฤษ "
        "ให้ดูแผนที่โอกาสพบซาก วิเคราะห์จุดที่คลิก และทายชื่อซากจากหน้าตา ทั้งหมดผ่านเกณฑ์ความสำเร็จขั้นต่ำครบ 15 ข้อ"))
    steps = [("database", "ink", INK, "จัดการข้อมูล", "92%", "ทะเบียน 8,521 ชิ้น เชื่อมกับแหล่งได้",
              ["ยุคของแหล่ง: ไม่ทราบ 352 → 129", "ทำความสะอาดชื่อเขต · ยุค 27 แบบ → 11 ยุค", "ได้ไฟล์ข้อมูลรวมพร้อมใช้"]),
             ("layers", "ink", INK, "สร้างคุณลักษณะ", "152", "คุณลักษณะเชิงพื้นที่ต่อจุด",
              ["ภูมิศาสตร์โบราณ · ธรณี · ภูมิประเทศ · ดาวเทียม", "ข้อความไทย: TF-IDF · WangchanBERTa", "หมวดหิน → ThaiLex"]),
             ("search", "green", T1, "แบบจำลองงาน 1", "86.3%", "สกุลจริงอยู่ใน 3 ชื่อ (แหล่งเดิม)",
              ["แหล่งใหม่ 58.7%", "ยุคถูก 71.9% · ห่างไม่เกิน 1 ยุค 90%", "เติมชื่อ 4,577 ชิ้น · ยุค 352 แหล่ง"]),
             ("map", "rust", T2, "แบบจำลองงาน 2", "0.887", "AUC โอกาสพบซาก (พื้นที่ใหม่)",
              ["ตรวจกับ PBDB: AUC 0.796", "สำรวจ 10% ของประเทศ ครอบคลุม PBDB 48%", "ประเภทซาก 16 ประเภท top-3 77.6%"]),
             ("monitor", "ink", INK, "เว็บต้นแบบ", "3", "เครื่องมือ · ไทย / อังกฤษ",
              ["แผนที่โอกาสพบซากทั่วประเทศ", "วิเคราะห์จุดที่คลิก + ชั้นยุคของหิน", "ทายชื่อซากจากหน้าตา"])]
    GAPC = 0.2
    cw = (XW - GAPC * 4) / 5
    for k, (ic, icol, col, title, big, lab, items) in enumerate(steps):
        x = X0 + k * (cw + GAPC)
        tint = T1S if col == T1 else T2S if col == T2 else SURF
        s.rect(x, Y0, cw, 4.55, tint, None if col in (T1, T2) else LINE, 1.0, 0.14)
        s.rect(x + 0.16, Y0 + 0.16, 0.34, 0.34, col, None, 0, 0.17, shape="ellipse", ps=P(str(k + 1), 12, "FFFFFF", HEAD, "c", space=0, line=1.0),
               inset=0)
        s.icon(ic, icol, x + cw - 0.6, Y0 + 0.12, 0.42, 0.42)
        s.text(x + 0.16, Y0 + 0.6, cw - 0.3, 0.42, P(title, 15.5, col, HEAD, space=0), "m")
        s.text(x + 0.16, Y0 + 1.08, cw - 0.3, 0.62, P(big, 32, col, HEAD, space=0, line=1.0), "m")
        s.text(x + 0.16, Y0 + 1.72, cw - 0.3, 0.56, P(lab, 11.5, INK, HEADM, space=0, line=1.05), "t")
        s.line(x + 0.16, Y0 + 2.36, x + cw - 0.16, Y0 + 2.36, LINE, 1.0)
        s.text(x + 0.16, Y0 + 2.46, cw - 0.3, 2.0, P(items, 11, INK2, bullet=col, space=5, line=1.05))
        if k < 4:
            arrow(s, x + cw + 0.02, Y0 + 0.33, x + cw + GAPC - 0.02, Y0 + 0.33, DIM, 1.3, False)
    s.rect(X0, 6.3, XW, 0.56, PANEL, None, 0, 0.12)
    s.text(X0 + 0.2, 6.3, XW - 0.4, 0.56, P("**ผ่านเกณฑ์ความสำเร็จขั้นต่ำครบ 15 ข้อ** (ถึงเป้าหมาย 7 ข้อ) · ข้อจำกัดหลักอยู่ที่ข้อมูล → แนวทางพัฒนาต่อหน้าถัดไป",
                                             13, INK2, bold_color=ACC, space=0), "m")
    D.append(s)

    # ===== 40 แนวทางพัฒนาต่อ (รายงาน 5.3)
    s = Slide(None, "Future Plans: แนวทางพัฒนาต่อ", "// FUTURE PLANS", (
        "แนวทางพัฒนาต่อแบ่งเป็น 3 ด้าน ด้านที่สำคัญที่สุดคือข้อมูล เพราะเป็นต้นเหตุหลักของข้อจำกัด ข้อเสนอแรกคือให้ทะเบียนบันทึกรหัสแหล่งหรือพิกัดของตัวอย่าง "
        "ซึ่งแก้ทั้งการเชื่อมโยงและป้ายประเภทซากได้ตรงจุดที่สุด รวมทั้งบันทึกยุคโดยแยกอายุหินกับอายุซาก และเพิ่มตัวอย่างของกลุ่มที่มีน้อย "
        "ด้านแบบจำลอง จะเตรียมข้อความให้ WangchanBERTa ตามรูปแบบตอนฝึก ซึ่งในการทดลองได้ top-3 ที่แหล่งใหม่ 0.597 สูงกว่าที่ใช้จริง 0.587 "
        "ถ่วงน้ำหนักคลาสให้ XGBoost หรือรวมกับ LightGBM และจูนค่าตั้งด้วยการตรวจสอบไขว้ซ้อนสองชั้นแบบบล็อกพื้นที่ ส่วนงาน 2 จะลองภาพดาวเทียมที่กว้างขึ้นและแบบจำลองที่ฝึกมาแล้ว "
        "ด้านระบบ จะเชื่อมกับระบบข้อมูลของ ทธ. และเปิดช่องทางให้ผู้พบซากแจ้งพร้อมพิกัดและภาพถ่าย "
        "เป้าหมายระยะยาวคือวงจรข้อมูล ผู้พบซากแจ้งผ่านเว็บ ทธ. ขึ้นทะเบียนพร้อมตำแหน่ง แบบจำลองได้ข้อมูลใหม่ไปฝึก และเว็บแนะนำได้แม่นขึ้น"))
    CW = (XW - 0.44) / 3
    plans = [("database", "ข้อมูล", True, [
                 "**บันทึกรหัสแหล่งหรือพิกัด**ของตัวอย่างในทะเบียน → แก้การเชื่อมโยงและป้ายประเภทซากได้ตรงจุดที่สุด",
                 "บันทึกยุคของแหล่ง **แยกอายุหินกับอายุซาก** พร้อมระบุที่มาของยุค",
                 "แยกชื่อหมวดหินที่บันทึกรวมกัน (11 ชื่อ 13 แหล่ง) ก่อนจับคู่ ThaiLex · แจ้งชื่อที่พิมพ์ผิด",
                 "เพิ่มตัวอย่างที่ระบุสกุลของกลุ่มที่มีน้อย เช่น พืช หอยฝาเดียว เอไคโนเดิร์ม"]),
             ("cog", "แบบจำลอง", False, [
                 "เตรียมข้อความ WangchanBERTa ตามรูปแบบตอนฝึก (<_>) → top-3 แหล่งใหม่ **0.597** ในการทดลอง (ใช้จริง 0.587)",
                 "ถ่วงน้ำหนักคลาสให้ XGBoost หรือรวมกับ LightGBM · เสนอผลระดับวงศ์เมื่อสกุลมีตัวอย่างน้อย",
                 "จูนค่าตั้งด้วย **Nested Spatial CV** เมื่อมีข้อมูลมากขึ้น",
                 "งาน 2: ภาพดาวเทียมกว้างกว่า 640 ม. · แบบจำลองหลายแบนด์ที่ฝึกมาแล้ว · หน่วยหินย่อย"]),
             ("monitor", "ระบบ", False, [
                 "เชื่อมระบบข้อมูลของ ทธ. → **ปรับข้อมูลอัตโนมัติ**เมื่อขึ้นทะเบียนใหม่ · รองรับผู้ใช้จำนวนมาก",
                 "ให้ผู้พบซาก**แจ้ง ทธ. พร้อมพิกัดและภาพถ่าย**จากเว็บ → ได้ข้อมูลใหม่กลับมาฝึกแบบจำลอง",
                 "รับภาพถ่ายซากเพื่อพัฒนาแบบจำลองจำแนกภาพ · ทดสอบการทายจากหน้าตากับผู้ใช้จริง",
                 "ผู้เชี่ยวชาญตรวจคำแปลอังกฤษ และแปลชื่อแหล่งซากอย่างเป็นทางการ"])]
    for k, (ic, t, top, items) in enumerate(plans):
        x = X0 + k * (CW + 0.22)
        s.rect(x, Y0, CW, 3.98, SURF, ACC if top else LINE, 1.6 if top else 1.0, 0.14)
        s.icon(ic, "rust" if top else "ink", x + 0.2, Y0 + 0.17, 0.42, 0.42)
        s.text(x + 0.74, Y0 + 0.14, 1.8, 0.48, P(t, 17, ACC if top else INK, HEAD, space=0), "m")
        if top:
            s.rect(x + CW - 1.42, Y0 + 0.22, 1.22, 0.34, T2S, None, 0, 0.17, ps=P("สำคัญที่สุด", 11.5, ACC, HEADM, "c", space=0, line=1.0),
                   inset=0.02)
        s.line(x + 0.2, Y0 + 0.76, x + CW - 0.2, Y0 + 0.76, LINE, 1.0)
        bullets(s, x + 0.2, Y0 + 0.88, CW - 0.4, 3.0, items, 12.5, bcolor=ACC if top else DIM, space=7, bold_color=INK)
    s.text(X0, 5.78, 1.75, 0.95, P("เป้าหมายระยะยาว", 12, DIM, HEADM, space=0) + P("วงจรข้อมูล", 16, ACC, HEAD, space=0), "m")
    loop = ["ผู้พบซากแจ้งผ่านเว็บ\nพิกัด + ภาพถ่าย", "ทธ. ขึ้นทะเบียน\nพร้อมรหัสแหล่ง", "ฝึกแบบจำลองใหม่\nด้วยข้อมูลที่มีตำแหน่ง",
            "เว็บแนะนำแม่นขึ้น\nคนใช้และแจ้งมากขึ้น"]
    bw, gap = 2.24, 0.44
    for k, t in enumerate(loop):
        bx = 2.45 + k * (bw + gap)
        box(s, bx, 5.83, bw, 0.86, t, T2S if k == 3 else SURF, ACC if k == 3 else LINE, 12, INK2)
        if k < 3:
            arrow(s, bx + bw + 0.05, 6.26, bx + bw + gap - 0.05, 6.26, ACC, 1.5, False)
    D.append(s)

    # ===== อ้างอิง 1/2: ข้อมูล
    s = Slide(None, "References 1/2: ข้อมูลที่นำมาใช้", "// REFERENCES · DATA", (
        "หน้านี้คือแหล่งข้อมูลที่อ้างอิง ข้อมูลหลักมาจากกรมทรัพยากรธรณี ทั้งทะเบียนตัวอย่าง แหล่งซาก และแผนที่ธรณีวิทยา "
        "PBDB ใช้ทั้งเติมยุคและเป็นชุดทดสอบอิสระ ThaiLex ให้ข้อมูลยุคและชนิดหินของหมวดหิน ข้อมูลเชิงพื้นที่มาจาก Copernicus DEM, Sentinel-2 "
        "และแบบจำลองแผ่นเปลือกโลก PALEOMAP กับ Merdith และคณะ ส่วนชื่อและอายุของยุคใช้มาตราธรณีกาลสากลของ ICS"))
    rows = [["อ้างอิง", "นำอะไรมาใช้", "เพราะอะไร"],
            ["**กรมทรัพยากรธรณี (2567)** บัญชีซากดึกดำบรรพ์ · data.go.th", "ทะเบียนตัวอย่าง 8,521 ชิ้น", "ข้อมูลหลัก: ป้ายสกุล / ชนิด และข้อความภาษาไทย"],
            ["**กรมทรัพยากรธรณี** ระบบบริการข้อมูลภูมิสารสนเทศ (ArcGIS REST)", "แหล่งซาก 518 แหล่ง · แผนที่ธรณีวิทยา 1:250,000", "จุดพบซากของงาน 2 · ป้ายยุค · คุณลักษณะธรณีวิทยา"],
            ["**The Paleobiology Database** · Peters & McClennen (2016) PBDB API", "290 collection ในไทย", "เติมยุคของแหล่ง + ชุดทดสอบอิสระที่ไม่ใช้ฝึก"],
            ["**ThaiLex** Thailand Lexicon of Stratigraphic Units", "พจนานุกรมหมวด / กลุ่มหิน", "ยุค ชนิดหิน และสภาพแวดล้อมของหมวดหิน (งาน 1 แหล่งใหม่)"],
            ["**ESA** Copernicus DEM GLO-30", "ความสูงความละเอียด 30 ม.", "คุณลักษณะภูมิประเทศ 15 ตัว"],
            ["**Drusch et al. (2012)** Sentinel-2 · Microsoft Planetary Computer", "ภาพ L2A หน้าแล้ง", "ดัชนีสเปกตรัม 22 ตัว · ดึงภาพสดตอนวิเคราะห์จุด"],
            ["**Scotese & Wright (2018)** PALEOMAP · **Merdith et al. (2021)** · Müller et al. (2018) GPlates",
             "แบบจำลองแผ่นเปลือกโลก + ความสูงโบราณ", "ย้อนตำแหน่งจุดไปในอดีต → คุณลักษณะภูมิศาสตร์โบราณ 85 ตัว"],
            ["**Cohen et al. (2013)** ICS Chronostratigraphic Chart", "มาตราธรณีกาลสากล", "ชื่อ อายุ และสีมาตรฐานของ 11 ยุค"],
            ["**พ.ร.บ.คุ้มครองซากดึกดำบรรพ์ พ.ศ. 2551**", "บทบาทของ ทธ.", "หน่วยงานหลักที่ขึ้นทะเบียนและดูแลข้อมูลซาก"]]
    s.table(X0, Y0, [4.6, 3.05, 4.48], rows, size=12, rowh=0.5, rowhs=[0.4] + [0.5] * 9, head=INK, align=["l", "l", "l"], first_bold=False)
    s.text(X0, 6.64, XW, 0.3, P("รายการอ้างอิงฉบับเต็ม 55 รายการอยู่ในบรรณานุกรมของรายงาน", 11, DIM, space=0))
    D.append(s)

    # ===== อ้างอิง 2/2: วิธีการ
    s = Slide(None, "References 2/2: วิธีการที่นำมาใช้", "// REFERENCES · METHODS", (
        "หน้านี้คืองานวิจัยด้านวิธีการที่นำมาใช้ การเชื่อมโยงระเบียนอิงทฤษฎีของ Fellegi และ Sunter และขั้นตอนของ Christen "
        "ข้อความใช้ TF-IDF และ WangchanBERTa แบบจำลองหลักคือ Random Forest, XGBoost และ LightGBM โดยมีงานของ Grinsztajn และคณะเป็นเหตุผลที่เลือกแบบจำลองต้นไม้ "
        "การเลือกจุดพื้นหลังแบบถ่วงการสำรวจมาจาก Phillips และคณะ การตรวจสอบไขว้แบบบล็อกมาจาก Roberts และคณะ AOA มาจาก Meyer และ Pebesma "
        "และการตรวจข้อมูลรั่วอิงงานของ Kaufman และคณะ กับ Kapoor และ Narayanan"))
    rows = [["อ้างอิง", "นำอะไรมาใช้", "เพราะอะไร"],
            ["**Fellegi & Sunter (1969)** · **Christen (2012)**", "ทฤษฎีและขั้นตอนการเชื่อมโยงระเบียน", "ทะเบียนกับแหล่งไม่มีคีย์ร่วม → จับคู่คลุมเครือ + ให้คะแนนแหล่ง"],
            ["**Salton & Buckley (1988)**", "TF-IDF (ใช้กับกลุ่มตัวอักษร 2–4 ตัว)", "เร็ว เหมาะกับข้อความสั้นที่ซ้ำกันในแหล่งเดิม"],
            ["**Lowphansirikul et al. (2021)** WangchanBERTa · Phatthiyaphaibun et al. (2023) PyThaiNLP",
             "แบบจำลองภาษาไทย · ถอดเสียงอักษรไทย", "จับความหมายข้อความที่แหล่งใหม่ · ถอดเสียงชื่อหมวดหินเพื่อจับคู่ ThaiLex"],
            ["**Breiman (2001)** · **Chen & Guestrin (2016)** · **Ke et al. (2017)**", "Random Forest · XGBoost · LightGBM",
             "RF ทายหลายป้ายพร้อมกัน · XGBoost รับค่าว่างจากภาพที่มีเมฆ · LightGBM เร็ว"],
            ["**Grinsztajn et al. (2022)**", "เหตุผลที่ใช้แบบจำลองต้นไม้", "ต้นไม้ดีกว่าการเรียนรู้เชิงลึกบนข้อมูลตาราง → ทดสอบซ้ำกับ MLP"],
            ["**Phillips et al. (2009)** · Elith & Leathwick (2009) · **Block et al. (2016)**", "จุดพื้นหลังแบบถ่วงการสำรวจ · สถิติ KS",
             "แหล่งซากเป็นข้อมูลแบบพบเท่านั้นและมีอคติการสำรวจ"],
            ["**Roberts et al. (2017)** · Valavi et al. (2019) · Ploton et al. (2020)", "ตรวจสอบไขว้แบบบล็อกพื้นที่", "แบบสุ่มให้ผลสูงเกินจริงกับพื้นที่ใหม่"],
            ["**Meyer & Pebesma (2021)**", "AOA และดัชนีความไม่คล้าย (DI)", "บอกพื้นที่ที่แผนที่ทั้งประเทศเชื่อถือได้"],
            ["**Kaufman et al. (2012)** · **Kapoor & Narayanan (2023)**", "กรอบตรวจหาข้อมูลรั่ว", "พบความแม่นเกินจริง 12 จุด และแก้ก่อนวัดผล"],
            ["**Anemone et al. (2011)** · **Peters et al. (2014)**", "งานต้นแบบในต่างประเทศ", "ML + ภาพดาวเทียมหาแหล่งซาก (ที่มาของการทดลอง CNN) · สกัดข้อมูลจากข้อความ"]]
    s.table(X0, Y0, [4.6, 3.05, 4.48], rows, size=11.5, rowh=0.46, rowhs=[0.38] + [0.46] * 10, head=INK, align=["l", "l", "l"], first_bold=False)
    D.append(s)

    # ===== 40 ขอบคุณ
    s = Slide(kind="title", notes="ขอบคุณค่ะ ยินดีตอบคำถาม")
    s.rect(0.63, 2.05, 0.17, 0.17, None, ACC, 1.6, shape="diamond")
    s.text(0.9, 1.95, 8.0, 0.36, P("PROJECT JURASSIC · SENIOR PROJECT 2569", 12, ACC, HEADM, space=0, spacing=2), "m")
    s.text(0.6, 2.4, 8.0, 1.2, P("ขอบคุณ", 60, INK, HEAD, space=0, line=1.0), "m")
    s.text(0.6, 3.6, 8.0, 0.7, P("Q & A · ถาม–ตอบ", 26, ACC, HEADM, space=0), "m")
    s.text(0.6, 4.7, 8.0, 1.2, P(["**สิดาพร จิตระวีวรรณ** · 66070203", "อาจารย์ที่ปรึกษา **อาจารย์เฉลิมพล ศิริกายน**",
                                  "สาขาวิชาวิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ · คณะเทคโนโลยีสารสนเทศ สจล."], 13, MUTED, bold_color=INK, space=2))
    sil = ["trilobite", "brachiopod", "ammonoid", "fish", "phuwiangosaurus", "siamraptor", "elephant"]
    for k, ic in enumerate(sil):
        s.icon(ic, "dim", 0.6 + k * 1.75, 6.2, 1.4, 0.75)
    s.rect(9.15, 1.2, 3.58, 4.6, "FFFFFF", LINE, 1.0, 0.16)
    s.img(A / "maps" / "potential_bg_bias.png", 9.3, 1.35, 3.28, 4.3)
    D.append(s)
    return D


# ---------------------------------------------------------------- หัว / ท้ายสไลด์
def chrome(s, k, n):
    pre = []

    def add(**it):
        it["step"] = 0
        pre.append(it)
    if s.kind == "content":
        add(k="rect", x=0.62, y=0.5, w=0.16, h=0.16, fill=None, line=ACC, lw=1.6, radius=0, dash=False, ps=[], valign="m", inset=0, shape="diamond")
        kick = f"{s.sec:02d} · {SECTIONS[s.sec]}" if s.sec else "PROJECT JURASSIC"
        add(k="text", x=0.88, y=0.39, w=7.0, h=0.38, ps=P(kick, 12, ACC, HEADM, space=0, spacing=2), valign="m", inset=0.03)
        add(k="text", x=7.9, y=0.39, w=4.83, h=0.38, ps=P(s.tag, 11, DIM, HEADM, "r", space=0, spacing=1.5), valign="m", inset=0.03)
        add(k="text", x=X0, y=0.74, w=XW, h=0.74, ps=P(s.title, 28, INK, HEAD, space=0, line=1.0), valign="m", inset=0.03)
    if s.kind != "title":
        add(k="text", x=X0, y=7.04, w=5.0, h=0.3, ps=P("PROJECT JURASSIC · SENIOR PROJECT 2569", 9, DIM, HEADM, space=0, spacing=1.5), valign="m", inset=0.03)
        add(k="text", x=11.73, y=7.04, w=1.0, h=0.3, ps=P(f"{k:02d} / {n:02d}", 10, DIM, HEADM, "r", space=0), valign="m", inset=0.03)
    s.items = pre + s.items


# ---------------------------------------------------------------- PowerPoint
def write_pptx(slides, out, base=None, finish=None):
    # base = ไฟล์ .pptx ที่มีอยู่ (เพิ่มสไลด์ต่อท้ายไฟล์นั้น) · finish(prs) = จัดลำดับ / ลบ / แก้เลขหน้า ก่อนบันทึก
    from lxml import etree
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Inches, Pt

    prs = Presentation(str(base)) if base else Presentation()
    if not base:
        prs.slide_width, prs.slide_height = Inches(W_IN), Inches(H_IN)
    blank = next(lay for lay in prs.slide_layouts if lay.name == "Blank")
    def rgb(h):
        return RGBColor.from_string(h)
    AL = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
    VA = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}

    def set_faces(rPr, font):
        for tag in ("a:latin", "a:ea", "a:cs"):
            el = rPr.find(qn(tag))
            if el is None:
                el = etree.SubElement(rPr, qn(tag))
            el.set("typeface", BODY if font == MONO and tag != "a:latin" else font)   # ฟอนต์โค้ดไม่มีอักษรไทย

    def fill_text(tf, ps, valign, inset):
        tf.word_wrap = True
        tf.auto_size = None
        tf.vertical_anchor = VA[valign]
        tf.margin_left = tf.margin_right = Inches(inset)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        for i, p in enumerate(ps):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = AL[p["align"]]
            para.line_spacing = p["line"]
            para.space_after = Pt(p["space"])
            if p["bullet"]:
                pPr = para._p.get_or_add_pPr()
                pPr.set("marL", str(int(Inches(0.24))))
                pPr.set("indent", str(int(-Inches(0.24))))
                bc = etree.SubElement(pPr, qn("a:buClr"))
                etree.SubElement(bc, qn("a:srgbClr")).set("val", p["bullet"])
                etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
                etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
            for text, bold, color, italic in p["runs"]:
                r = para.add_run()
                r.text = text
                f = r.font
                f.size = Pt(p["size"])
                f.bold = bold
                f.italic = italic
                f.color.rgb = rgb(color)
                rPr = r._r.get_or_add_rPr()
                if p["spacing"]:
                    rPr.set("spc", str(int(p["spacing"] * 100)))
                set_faces(rPr, p["font"])

    def dash(line_fmt):
        line_fmt.dash_style = MSO_LINE_DASH_STYLE.DASH

    def add_chart(sl, it):
        kind = it["kind"]
        cd = CategoryChartData(number_format=it["fmt"])
        cd.categories = it["cats"]
        for name, vals, _ in it["series"]:
            cd.add_series(name, vals)
        ct = {"bar": XL_CHART_TYPE.BAR_CLUSTERED, "col": XL_CHART_TYPE.COLUMN_CLUSTERED, "donut": XL_CHART_TYPE.DOUGHNUT}[kind]
        gf = sl.shapes.add_chart(ct, Inches(it["x"]), Inches(it["y"]), Inches(it["w"]), Inches(it["h"]), cd)
        ch = gf.chart
        ch.font.size = Pt(it["size"])
        ch.font.color.rgb = rgb(INK2)
        ch.has_legend = bool(it["legend"])
        if it["legend"]:
            ch.legend.position = XL_LEGEND_POSITION.TOP
            ch.legend.include_in_layout = False
            ch.legend.font.size = Pt(it["size"])
        plot = ch.plots[0]
        if kind == "donut":
            plot.has_data_labels = False
            ser = plot.series[0]
            for i, c in enumerate(it["point_colors"]):
                pt = ser.points[i]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = rgb(c)
                pt.format.line.color.rgb = rgb(BG)
            hole = ch._chartSpace.find(".//" + qn("c:doughnutChart"))
            hs = hole.find(qn("c:holeSize"))
            if hs is None:
                hs = etree.SubElement(hole, qn("c:holeSize"))
            hs.set("val", "62")
        else:
            plot.gap_width = it["gap"]
            plot.overlap = -8 if len(it["series"]) > 1 else 0
            plot.has_data_labels = True
            dl = plot.data_labels
            dl.number_format = it["fmt"]
            dl.number_format_is_linked = False
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
            dl.font.size = Pt(it["size"])
            dl.font.color.rgb = rgb(INK)
            for si, ser in enumerate(plot.series):
                ser.format.fill.solid()
                ser.format.fill.fore_color.rgb = rgb(it["series"][si][2])
                ser.invert_if_negative = False
                if it["point_colors"] and si == 0:
                    for i, c in enumerate(it["point_colors"]):
                        pt = ser.points[i]
                        pt.format.fill.solid()
                        pt.format.fill.fore_color.rgb = rgb(c)
            va = ch.value_axis
            va.visible = False
            va.has_major_gridlines = False
            va.minimum_scale = 0
            if it["vmax"]:
                va.maximum_scale = it["vmax"]
            ca = ch.category_axis
            ca.has_major_gridlines = False
            ca.tick_labels.font.size = Pt(it["size"])
            ca.tick_labels.font.color.rgb = rgb(INK2)
            ca.format.line.color.rgb = rgb(LINE)
            if kind == "bar":
                ca.reverse_order = True
        for d in ch._chartSpace.iter(qn("a:defRPr")):          # ฟอนต์ไทยของทุกข้อความในกราฟ
            set_faces(d, BODY)
        return gf

    def add_reveal(sl, shapes):
        # แอนิเมชัน: คลิกครั้งเดียว ทุกชิ้นของงาน 2 ค่อย ๆ ปรากฏพร้อมกัน (fade 0.4 วินาที)
        P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
        nid = [3]

        def next_id():
            nid[0] += 1
            return nid[0]
        effects, blds = [], []
        for i, (spid, kind) in enumerate(shapes):
            a, b, c = next_id(), next_id(), next_id()
            node = "clickEffect" if i == 0 else "withEffect"
            effects.append(
                f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" grpId="0" nodeType="{node}">'
                f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
                f'<p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
                f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>'
                f'</p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
                f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{c}" dur="400"/>'
                f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect>'
                f'</p:childTnLst></p:cTn></p:par>')
            if kind == "sp":
                blds.append(f'<p:bldP spid="{spid}" grpId="0" animBg="1"/>')
            elif kind == "graphicFrame":
                blds.append(f'<p:bldGraphic spid="{spid}" grpId="0"><p:bldAsOne/></p:bldGraphic>')
        xml = (f'<p:timing xmlns:p="{P_NS}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:tnLst><p:par>'
               f'<p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
               f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
               f'<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst>'
               f'<p:par><p:cTn id="{next_id()}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
               + "".join(effects) +
               '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>'
               '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
               '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
               '</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
               + (f'<p:bldLst>{"".join(blds)}</p:bldLst>' if blds else "") + '</p:timing>')
        timing = etree.fromstring(xml)
        sld = sl._element
        ext = sld.find(qn("p:extLst"))
        (ext.addprevious(timing) if ext is not None else sld.append(timing))

    for s in slides:
        sl = prs.slides.add_slide(blank)
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = rgb(BG)
        step1 = []
        for it in s.items:
            kind = it["k"]
            shp = None
            if kind == "rect":
                shape = {"rect": MSO_SHAPE.ROUNDED_RECTANGLE if it["radius"] else MSO_SHAPE.RECTANGLE, "ellipse": MSO_SHAPE.OVAL,
                         "diamond": MSO_SHAPE.DIAMOND}[it["shape"]]
                shp = sl.shapes.add_shape(shape, Inches(it["x"]), Inches(it["y"]), Inches(it["w"]), Inches(it["h"]))
                if it["radius"] and it["shape"] == "rect":
                    shp.adjustments[0] = min(0.5, it["radius"] / min(it["w"], it["h"]))
                if it["fill"]:
                    shp.fill.solid()
                    shp.fill.fore_color.rgb = rgb(it["fill"])
                else:
                    shp.fill.background()
                if it["line"]:
                    shp.line.color.rgb = rgb(it["line"])
                    shp.line.width = Pt(it["lw"])
                    if it["dash"]:
                        dash(shp.line)
                else:
                    shp.line.fill.background()
                shp.shadow.inherit = False
                fill_text(shp.text_frame, it["ps"], it["valign"], it["inset"])
            elif kind == "text":
                shp = sl.shapes.add_textbox(Inches(it["x"]), Inches(it["y"]), Inches(it["w"]), Inches(it["h"]))
                fill_text(shp.text_frame, it["ps"], it["valign"], it["inset"])
            elif kind == "line":
                shp = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(it["x1"]), Inches(it["y1"]), Inches(it["x2"]), Inches(it["y2"]))
                shp.line.color.rgb = rgb(it["color"])
                shp.line.width = Pt(it["lw"])
                if it["dash"]:
                    dash(shp.line)
                if it["head"]:
                    ln = shp.line._get_or_add_ln()
                    te = etree.SubElement(ln, qn("a:tailEnd"))
                    te.set("type", "triangle")
                    te.set("w", "med")
                    te.set("len", "med")
            elif kind == "img":
                shp = sl.shapes.add_picture(str(it["path"]), Inches(it["x"]), Inches(it["y"]), Inches(it["w"]), Inches(it["h"]))
            elif kind == "table":
                rows, colw = it["rows"], it["colw"]
                rh = it["rowhs"] or [it["rowh"]] * len(rows)
                gf = sl.shapes.add_table(len(rows), len(colw), Inches(it["x"]), Inches(it["y"]), Inches(sum(colw)), Inches(sum(rh)))
                tbl = gf.table
                tbl._tbl.tblPr.set("firstRow", "0")
                tbl._tbl.tblPr.set("bandRow", "0")
                sty = tbl._tbl.tblPr.find(qn("a:tableStyleId"))
                if sty is not None:
                    sty.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"     # No Style, Table Grid → ใช้สีของเราเอง
                for j, cw in enumerate(colw):
                    tbl.columns[j].width = Inches(cw)
                for i, row in enumerate(rows):
                    tbl.rows[i].height = Inches(rh[i])
                    for j, val in enumerate(row):
                        c = tbl.cell(i, j)
                        c.fill.solid()
                        c.fill.fore_color.rgb = rgb(it["head"] if i == 0 else it["hl"].get(i, SURF if i % 2 else "F0F0EA"))
                        c.margin_left = c.margin_right = Inches(0.09)
                        c.margin_top = c.margin_bottom = Inches(0.03)
                        c.vertical_anchor = MSO_ANCHOR.MIDDLE
                        hdr = i == 0
                        col = "FFFFFF" if hdr else INK2
                        font = HEADM if hdr else (BODYM if (j == 0 and it["first_bold"]) else BODY)
                        ps = P(val, it["size"], col, font, it["align"][j] if not hdr else ("l" if it["align"][j] == "l" else "c"),
                               bold_color=INK if not hdr else "FFFFFF", space=0, line=1.0)
                        c.text_frame.text = ""
                        fill_text(c.text_frame, ps, "m", 0.0)
                        tcPr = c._tc.get_or_add_tcPr()
                        for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                            ln = etree.SubElement(tcPr, qn(side))
                            ln.set("w", str(int(Pt(0.75))))
                            sf = etree.SubElement(ln, qn("a:solidFill"))
                            etree.SubElement(sf, qn("a:srgbClr")).set("val", LINE if side in ("a:lnT", "a:lnB") else (it["head"] if hdr else LINE))
                shp = gf
            elif kind == "chart":
                shp = add_chart(sl, it)
            if shp is not None and it["step"]:
                tag = etree.QName(shp._element).localname
                step1.append((shp.shape_id, tag))
        if step1:
            add_reveal(sl, step1)
        ns = sl.notes_slide
        if ns.notes_text_frame is None:
            # ไฟล์ที่บันทึกจาก PowerPoint อาจมี notes master ไม่มีช่องบทพูด → คัดลอกช่องจากหน้า notes ที่มีอยู่แล้ว
            import copy
            donor = next(x.notes_slide for x in prs.slides if x.has_notes_slide and x.notes_slide.notes_text_frame is not None)
            for ph in donor.placeholders:
                ns.shapes._spTree.append(copy.deepcopy(ph._element))
        ns.notes_text_frame.text = s.notes
    if finish:
        finish(prs)
    if not base:
        prs.core_properties.title = "Project Jurassic — Senior Project"
        prs.core_properties.author = "สิดาพร จิตระวีวรรณ"
    prs.save(out)


# ---------------------------------------------------------------- HTML สำหรับตรวจหน้าตา
def write_html(slides, out):
    PX = 96
    faces = []
    for (fam, bold), f in FONT_FILES.items():
        faces.append(f"@font-face{{font-family:'{fam}';font-weight:{700 if bold else 400};src:url('{(FONTS / (f + '.ttf')).as_uri()}')}}")
    def q(v):
        return f"{v * PX:.2f}px"

    def html_ps(ps):
        o = []
        for p in ps:
            r = ""
            for t, b, c, i in p["runs"]:
                r += (f"<span style='color:#{c};font-weight:{700 if b else 400};font-style:{'italic' if i else 'normal'};"
                      f"letter-spacing:{p['spacing'] * 1.333:.1f}px'>{H.escape(t)}</span>")
            bul = (f"padding-left:{0.24 * PX}px;text-indent:-{0.24 * PX}px;" if p["bullet"] else "")
            mark = f"<span style='display:inline-block;width:{0.24 * PX}px;text-indent:0;color:#{p['bullet']}'>•</span>" if p["bullet"] else ""
            fam = f'"{p["font"]}"' + (f',"{BODY}"' if p["font"] == MONO else "")
            pre = "white-space:pre;" if p["font"] == MONO else ""
            o.append(f"<p style='margin:0 0 {p['space'] * 1.333:.1f}px;font-family:{fam};font-size:{p['size'] * 1.3333:.2f}px;{pre}"
                     f"line-height:{p['line'] * 1.22:.3f};{bul}text-align:{ {'l': 'left', 'c': 'center', 'r': 'right'}[p['align']] }'>{mark}{r}</p>")
        return "".join(o)

    def chart_svg(it):
        W_, H_ = it["w"] * PX, it["h"] * PX
        fs = it["size"] * 1.333
        cats, series = it["cats"], it["series"]
        fmt = it["fmt"]
        f = (lambda v: f"{v:,.0f}") if fmt in ("0", "#,##0") else (lambda v: f"{v:.{len(fmt.split('.')[1])}f}" if "." in fmt else str(v))
        el = []
        if it["kind"] == "donut":
            vals = series[0][1]
            tot, a0 = sum(vals), -90
            cx, cy, r = W_ / 2, H_ / 2, min(W_, H_) / 2
            for v, c in zip(vals, it["point_colors"]):
                a1 = a0 + 360 * v / tot
                import math
                x0, y0 = cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0))
                x1, y1 = cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1))
                el.append(f"<path d='M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 {1 if a1 - a0 > 180 else 0} 1 {x1:.1f},{y1:.1f} Z' fill='#{c}' stroke='#{BG}' stroke-width='2'/>")
                a0 = a1
            el.append(f"<circle cx='{cx}' cy='{cy}' r='{r * 0.62}' fill='#{BG}'/>")
            return f"<svg width='{W_:.0f}' height='{H_:.0f}'>{''.join(el)}</svg>"
        vmax = it["vmax"] or max(max(v) for _, v, _ in series) * 1.0
        ns, nc = len(series), len(cats)
        top = fs * 1.6 if it["legend"] else 0
        if it["legend"]:
            lx = 0
            for name, _, c in series:
                el.append(f"<rect x='{lx}' y='2' width='{fs * 0.8}' height='{fs * 0.8}' fill='#{c}'/>"
                          f"<text x='{lx + fs}' y='{fs * 0.8}' font-size='{fs}' fill='#{INK2}' font-family='{BODY}'>{H.escape(name)}</text>")
                lx += fs * 1.6 + len(name) * fs * 0.55
        if it["kind"] == "bar":
            lw = (it["label_w"] or 1.4) * PX
            band = (H_ - top) / nc
            bh = band / (1 + it["gap"] / 100) / ns
            for i, cat in enumerate(cats):
                y = top + i * band + (band - bh * ns) / 2
                el.append(f"<text x='{lw - 6}' y='{y + bh * ns / 2 + fs * 0.35}' text-anchor='end' font-size='{fs}' fill='#{INK2}' font-family='{BODY}'>{H.escape(cat)}</text>")
                for si, (_, vals, c) in enumerate(series):
                    col = it["point_colors"][i] if (it["point_colors"] and si == 0) else c
                    bw = (W_ - lw - fs * 3.2) * vals[i] / vmax
                    el.append(f"<rect x='{lw}' y='{y + si * bh}' width='{max(bw, 0):.1f}' height='{bh * 0.96:.1f}' fill='#{col}'/>"
                              f"<text x='{lw + bw + 4}' y='{y + si * bh + bh / 2 + fs * 0.35}' font-size='{fs}' fill='#{INK}' font-family='{BODY}'>{f(vals[i])}</text>")
            el.append(f"<line x1='{lw}' y1='{top}' x2='{lw}' y2='{H_}' stroke='#{LINE}'/>")
        else:
            lab_h = fs * 2.6
            band = W_ / nc
            bw = band / (1 + it["gap"] / 100) / ns
            ph = H_ - top - lab_h - fs * 1.4
            for i, cat in enumerate(cats):
                x = i * band + (band - bw * ns) / 2
                el.append(f"<foreignObject x='{i * band}' y='{H_ - lab_h}' width='{band}' height='{lab_h}'><div xmlns='http://www.w3.org/1999/xhtml' "
                          f"style='font:{fs}px \"{BODY}\";color:#{INK2};text-align:center;line-height:1.15'>{H.escape(cat)}</div></foreignObject>")
                for si, (_, vals, c) in enumerate(series):
                    col = it["point_colors"][i] if (it["point_colors"] and si == 0) else c
                    bh = ph * vals[i] / vmax
                    y = H_ - lab_h - bh
                    el.append(f"<rect x='{x + si * bw}' y='{y:.1f}' width='{bw * 0.96:.1f}' height='{bh:.1f}' fill='#{col}'/>"
                              f"<text x='{x + si * bw + bw / 2}' y='{y - 4:.1f}' text-anchor='middle' font-size='{fs}' fill='#{INK}' font-family='{BODY}'>{f(vals[i])}</text>")
            el.append(f"<line x1='0' y1='{H_ - lab_h}' x2='{W_}' y2='{H_ - lab_h}' stroke='#{LINE}'/>")
        return f"<svg width='{W_:.0f}' height='{H_:.0f}' style='overflow:visible'>{''.join(el)}</svg>"

    body = []
    for k, s in enumerate(slides, 1):
        el = []
        for it in s.items:
            kind = it["k"]
            st = " step1" if it["step"] else ""
            if kind == "rect":
                x, y, w, h = it["x"], it["y"], it["w"], it["h"]
                bd = f"border:{it['lw'] * 1.333:.1f}px {'dashed' if it['dash'] else 'solid'} #{it['line']};" if it["line"] else ""
                bg = f"background:#{it['fill']};" if it["fill"] else ""
                jc = {"t": "flex-start", "m": "center", "b": "flex-end"}[it["valign"]]
                if it["shape"] == "diamond":
                    el.append(f"<svg class='e{st}' style='left:{q(x)};top:{q(y)}' width='{w * PX:.1f}' height='{h * PX:.1f}'>"
                              f"<polygon points='{w * PX / 2},1 {w * PX - 1},{h * PX / 2} {w * PX / 2},{h * PX - 1} 1,{h * PX / 2}' "
                              f"fill='{('#' + it['fill']) if it['fill'] else 'none'}' stroke='#{it['line'] or 'none'}' stroke-width='{it['lw'] * 1.333}'/></svg>")
                    continue
                rad = "50%" if it["shape"] == "ellipse" else f"{min(it['radius'], min(w, h) / 2) * PX:.1f}px"
                el.append(f"<div class='e{st}' style='left:{q(x)};top:{q(y)};width:{q(w)};height:{q(h)};{bg}{bd}border-radius:{rad};"
                          f"display:flex;flex-direction:column;justify-content:{jc};padding:{0.02 * PX}px {it['inset'] * PX:.1f}px'>{html_ps(it['ps'])}</div>")
            elif kind == "text":
                jc = {"t": "flex-start", "m": "center", "b": "flex-end"}[it["valign"]]
                el.append(f"<div class='e t{st}' style='left:{q(it['x'])};top:{q(it['y'])};width:{q(it['w'])};height:{q(it['h'])};justify-content:{jc};"
                          f"padding:{0.02 * PX}px {it['inset'] * PX:.1f}px'>{html_ps(it['ps'])}</div>")
            elif kind == "line":
                x1, y1, x2, y2 = (v * PX for v in (it["x1"], it["y1"], it["x2"], it["y2"]))
                mx, my = min(x1, x2) - 8, min(y1, y2) - 8
                w, h = abs(x2 - x1) + 16, abs(y2 - y1) + 16
                mk = (f"<defs><marker id='m{k}{len(el)}' markerWidth='8' markerHeight='8' refX='6' refY='4' orient='auto' markerUnits='userSpaceOnUse'>"
                      f"<path d='M0,0 L8,4 L0,8 Z' fill='#{it['color']}'/></marker></defs>") if it["head"] else ""
                el.append(f"<svg class='e{st}' style='left:{mx:.1f}px;top:{my:.1f}px;overflow:visible' width='{w:.1f}' height='{h:.1f}'>{mk}"
                          f"<line x1='{x1 - mx:.1f}' y1='{y1 - my:.1f}' x2='{x2 - mx:.1f}' y2='{y2 - my:.1f}' stroke='#{it['color']}' "
                          f"stroke-width='{it['lw'] * 1.333:.1f}' {'stroke-dasharray=\"6 4\"' if it['dash'] else ''} "
                          f"{('marker-end=\"url(#m' + str(k) + str(len(el)) + ')\"') if it['head'] else ''}/></svg>")
            elif kind == "img":
                el.append(f"<img class='e{st}' src='{it['path'].as_uri()}' style='left:{q(it['x'])};top:{q(it['y'])};width:{q(it['w'])};height:{q(it['h'])}'>")
            elif kind == "table":
                rh = it["rowhs"] or [it["rowh"]] * len(it["rows"])
                trs = []
                for i, row in enumerate(it["rows"]):
                    tds = []
                    for j, v in enumerate(row):
                        hdr = i == 0
                        bg = it["head"] if hdr else it["hl"].get(i, SURF if i % 2 else "F0F0EA")
                        font = HEADM if hdr else (BODYM if (j == 0 and it["first_bold"]) else BODY)
                        al = it["align"][j] if not hdr else ("l" if it["align"][j] == "l" else "c")
                        ps = P(v, it["size"], "FFFFFF" if hdr else INK2, font, al, bold_color="FFFFFF" if hdr else INK, space=0, line=1.0)
                        tds.append(f"<td style='width:{q(it['colw'][j])};height:{q(rh[i])};background:#{bg};border-top:1px solid #{LINE};"
                                   f"border-bottom:1px solid #{LINE}'>{html_ps(ps)}</td>")
                    trs.append("<tr>" + "".join(tds) + "</tr>")
                el.append(f"<table class='e{st}' style='left:{q(it['x'])};top:{q(it['y'])};width:{q(sum(it['colw']))}'>{''.join(trs)}</table>")
            elif kind == "chart":
                el.append(f"<div class='e{st}' style='left:{q(it['x'])};top:{q(it['y'])};width:{q(it['w'])};height:{q(it['h'])}'>{chart_svg(it)}</div>")
        body.append(f"<section class='s' id='s{k}'>{''.join(el)}</section>")
    css = f"""{''.join(faces)}
body{{margin:0;background:#777}} .s{{position:relative;width:{W_IN * PX:.0f}px;height:{H_IN * PX:.0f}px;overflow:hidden;background:#{BG};margin:0 0 20px}}
.e{{position:absolute;box-sizing:border-box}} .t{{display:flex;flex-direction:column}} p{{margin:0}}
table{{border-collapse:collapse;table-layout:fixed}} td{{padding:{0.03 * PX}px {0.09 * PX}px;box-sizing:border-box;vertical-align:middle}}
body.before .step1{{visibility:hidden}}"""
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(body)}</body></html>")


def screenshots(html, outdir, n, only=None):
    from playwright.sync_api import sync_playwright
    outdir.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", args=["--allow-file-access-from-files"])
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        pg.goto(html.as_uri())
        pg.wait_for_timeout(1500)
        for k in range(1, n + 1):
            if only and k not in only:
                continue
            pg.locator(f"#s{k}").screenshot(path=str(outdir / f"slide_{k:02d}.png"))
        b.close()


def write_script(slides, out):
    lines = ["# บทพูดประกอบสไลด์ — Project Jurassic (Senior Project)", "",
             "> ไฟล์สไลด์: `Project_Jurassic_Senior_Project.pptx` · บทพูดเดียวกันนี้อยู่ใน Speaker notes ของทุกสไลด์",
             "> [คลิก] = คลิกให้ส่วนของงาน 2 (ครึ่งขวา) ปรากฏ", ""]
    for k, s in enumerate(slides, 1):
        title = s.title or ("ปก" if k == 1 else "")
        lines += [f"## สไลด์ {k} — {title}", f"*ประมาณ {max(20, round(len(s.notes) / 10))} วินาที*", "", s.notes, ""]
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    import sys
    deck = build_deck()
    if "build_deck2" in globals():
        deck += build_deck2()
    for k, s in enumerate(deck, 1):
        chrome(s, k, len(deck))
    # ผู้จัดทำแก้ไฟล์ .pptx ใน PowerPoint ได้ — ถ้าไฟล์ไม่ตรงกับที่สคริปต์สร้างครั้งล่าสุด จะไม่เขียนทับ แต่เขียนเป็นไฟล์ใหม่แทน
    import hashlib
    stamp = SL / ".senior_pptx.sha256"
    def digest(f):
        return hashlib.sha256(f.read_bytes()).hexdigest()
    out = OUT
    if OUT.exists() and (not stamp.exists() or digest(OUT) != stamp.read_text().strip()):
        out = OUT.with_name(OUT.stem + "_สร้างใหม่.pptx")
        print(f"!! {OUT.name} ถูกแก้นอกสคริปต์ (เช่นใน PowerPoint) จึงไม่เขียนทับ -> เขียนที่ {out.name}")
    write_pptx(deck, out)
    if out == OUT:
        stamp.write_text(digest(OUT))
    html = SL / "_preview.html"
    write_html(deck, html)
    only = {int(a) for a in sys.argv[1:] if a.isdigit()} or None
    screenshots(html, SL / "preview", len(deck), only)
    html.unlink()
    write_script(deck, SL / "บทพูด_สไลด์_senior.md")
    print("slides:", len(deck), "->", out)
