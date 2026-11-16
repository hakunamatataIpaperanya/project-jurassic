# สร้างรายงาน (.docx) จากไฟล์เนื้อหา docs/report_src/parts/*.md + อ้างอิง docs/report_src/refs.py + รูปใน docs/report_src/figures/
#   รูปแบบตามคู่มือการจัดเล่มรายงานฉบับสมบูรณ์ คณะเทคโนโลยีสารสนเทศ สจล. (docs/คู่มือการจัดเล่ม...pdf):
#   กระดาษ A4 ขอบบน/ขวา/ล่าง 1" ซ้าย 1.5" ทั้งเล่ม (ข้อ 4.4), TH SarabunPSK 16 pt, "บทที่" 20 pt หนา, ชื่อบทและหัวเรื่องส่วนหน้า 24 pt หนา,
#   หัวข้อ 18 / 16 pt หนา, สมการ Times New Roman (ข้อ 4.2.10), เลขหน้าโรมันกลางล่างห่างขอบล่าง 1" (ข้อ 4.5.1),
#   เลขหน้าเนื้อหามุมขวาบนห่างขอบบน 0.5" หน้าแรกของบทไม่แสดง (ข้อ 4.5.2), คำบรรยายรูป/ตารางหลายบรรทัดเยื้องตรงอักษรแรกของคำบรรยาย,
#   เว้น 1 บรรทัดก่อนรูปและหลังรูป/ตาราง, ตารางที่ไม่จบในหน้าเดียวขึ้นหน้าใหม่พร้อม "ตารางที่ x.y (ต่อ)" (ข้อ 4.6–4.8),
#   อ้างอิงแบบตัวเลข [n] เรียงตามลำดับที่อ้างครั้งแรก ชื่อหนังสือตัวหนา (ข้อ 4.10)
#
# ขั้นตอน (python-docx + Playwright/Chrome + PyMuPDF):
#   1. แปลงไฟล์เนื้อหาเป็นรายการบล็อก
#   2. สร้างหน้าพรีวิว HTML ที่จัดหน้าเหมือน Word (ฟอนต์ ขนาด ระยะบรรทัด 1.31 เท่า ขอบกระดาษ) → พิมพ์เป็น PDF ด้วย Chrome
#   3. หาเลขหน้าของทุกหัวข้อ / รูป / ตาราง จากเครื่องหมายที่ซ่อนไว้ใน PDF → ใส่ในสารบัญ / สารบัญรูป / สารบัญตาราง
#   4. สร้าง .docx — สารบัญเป็นฟิลด์ TOC ของ Word ที่ใส่รายการและเลขหน้าไว้แล้ว และตั้งให้ Word อัปเดตเลขหน้าเมื่อเปิดไฟล์
# รัน:  <python ที่มี python-docx, playwright, pymupdf> scripts/build_report.py [ไฟล์.docx] [โฟลเดอร์พรีวิว] [--force]
#   ไม่เขียนทับไฟล์ที่เปิดอยู่ใน Word หรือไฟล์ที่ถูกแก้หลังสร้างครั้งล่าสุด (สำรองไว้ใน docs/report_src/_backup/ ก่อนเสมอ;
#   นำการแก้ไขเข้า parts/*.md แล้วจึงรันด้วย --force)
#   หน้าปกทั้งชุด (ปก ปกใน ปกภาษาอังกฤษ ลิขสิทธิ์ ใบรับรอง) จัดใน Word แล้วเก็บไว้ที่ docs/report_src/cover.docx — คัดลอกเข้าเล่มตามต้นฉบับ
#   เก็บปกจากไฟล์ที่แก้ใน Word:  scripts/build_report.py --take-cover ไฟล์.docx   (ไม่มี cover.docx = สร้างปกหน้าเดียวจาก refs.py)
#
# รูปแบบไฟล์เนื้อหา (บรรทัดต่อบรรทัด):
#   @cover / @front / @toc / @lof / @lot / @bibliography / @pagebreak     โครงเล่ม
#   @space              บรรทัดว่าง 1 บรรทัด (ใช้จัดหน้าเฉพาะจุด ตามที่ผู้จัดทำแทรกไว้ใน Word)
#   @right ข้อความ      ย่อหน้าชิดขวา (ชื่อผู้จัดทำท้ายกิตติกรรมประกาศ)
#   @bio figures/รูป    หน้าประวัติผู้เขียน (ขึ้นหน้าใหม่ + รูปผู้เขียนชิดขวา) — ทุกบรรทัดหลังจากนี้เป็นบรรทัดข้อมูลชิดซ้าย
#   @info th|en                                                          ข้อมูลโครงงานบนหน้าบทคัดย่อ
#   # ชื่อ {front}      หัวข้อส่วนหน้า      # บทที่ 1 | บทนำ   หัวบท (ขึ้นหน้าใหม่ 2 บรรทัด)
#   ## / ###            หัวข้อระดับ 2 / 3 (อยู่ในสารบัญ),  #### หัวข้อย่อย (ไม่อยู่ในสารบัญ)
#   - / "  - "          รายการ / รายการย่อย (รายการสัญลักษณ์ของ Word; เยื้องตามบริบท: ต่อจากย่อหน้า 0.5", ต่อจากหัวข้อระดับ 3 0.75",
#                       ต่อจากหัวข้อย่อยตัวหนาบรรทัดเดียว เช่น **ขอบเขตด้านข้อมูล** 1.0"; รายการย่อย +0.5")
#   ![รูปที่ x.y คำบรรยาย](figures/ไฟล์.png){w=6}                         รูป + คำบรรยายใต้รูป
#   Table: ตารางที่ x.y คำบรรยาย {w=1.5,2,...}  ตามด้วย | a | b |        ตาราง (แถวแรก = หัวตาราง)
#                       แถวคั่น |---| = จัดแนวอัตโนมัติ (คอลัมน์แรกชิดซ้าย ข้อความสั้นกึ่งกลาง); |:---:| กึ่งกลาง, |:---| ชิดซ้าย ทั้งคอลัมน์
#   $$ สูตร $$          สูตรกึ่งกลาง
#   ข้อความ: **หนา**, *เอียง*, [@key] หรือ [@a; @b] = อ้างอิง, <br> = ขึ้นบรรทัดในย่อหน้าเดียวกัน (Shift+Enter)
#   ชื่อเรื่องใน refs.py: "\n" = ขึ้นบรรทัดบนปกและหน้าบทคัดย่อ
import copy
import hashlib
import html as H
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report_src"
COVER = SRC / "cover.docx"         # หน้าปกที่ผู้จัดทำจัดใน Word (ทุกอย่างก่อนตัวแบ่งส่วนแรก)
STAMP = SRC / "_last_build.sha256"  # ค่าแฮชของ .docx ที่สร้างครั้งล่าสุด ใช้ตรวจว่าไฟล์ถูกแก้ใน Word หรือไม่
FONT = "TH SarabunPSK"
FONT_DIR = "file:///Applications/Microsoft%20Word.app/Contents/Resources/DFonts/"
FONT_FILE = Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts")
BODY_PT, CHAP_PT, TITLE_PT, H2_PT, H3_PT, TABLE_PT, CAPTION_PT = 16, 20, 24, 18, 16, 16, 16
EQ_FONT, EQ_PT = "Times New Roman", 12   # สมการ (ข้อ 4.2.10); 12 pt สูงพอ ๆ กับ TH SarabunPSK 16 pt
LINE = 1.31                        # ระยะบรรทัดเดี่ยวของ TH SarabunPSK (ascent + descent ของฟอนต์ / unitsPerEm) — ตรงกับ Word (20.9 pt ที่ 16 pt)
BLANK_PT = round(BODY_PT * LINE, 1)  # 1 บรรทัดว่าง
PAGE_W, PAGE_H = 210 / 25.4, 297 / 25.4   # A4 (นิ้ว)
TOP, BOTTOM, LEFT, RIGHT = 1.0, 1.0, 1.5, 1.0
HEADER_IN, FOOTER_IN = 0.5, 1.0    # เลขหน้าเนื้อหาห่างขอบบน 0.5", เลขหน้าโรมันห่างขอบล่าง 1"
TEXT_W = round(PAGE_W - LEFT - RIGHT, 2)   # ความกว้างเนื้อหา 5.77"
TEXT_H = PAGE_H - TOP - BOTTOM             # ความสูงเนื้อหา 9.69"
FIT = 0.85                         # ส่วนของหน้าที่ให้ตารางแต่ละช่วงใช้ (เผื่อ Word ตัดบรรทัดในเซลล์ต่างจาก Chrome ได้ราว 10–15%)


def text_pt(text, bold=False, size=CAPTION_PT):
    # ความกว้างข้อความ (pt) จากฟอนต์ของ Word — สระบน/ล่างและวรรณยุกต์กว้าง 0
    from PIL import ImageFont
    f = ImageFont.truetype(str(FONT_FILE / ("thsarabun-bold.ttf" if bold else "thsarabun.ttf")), size * 10)
    return f.getlength(text) / 10


def sentence_spaces(text):
    # ข้อ 4.9: หลังมหัพภาค (.) จบประโยคเว้น 2 ช่วงอักษร — ประโยคภาษาอังกฤษ (ภาษาไทยในเล่มใช้จุดเฉพาะคำย่อ เช่น พ.ศ. ทธ.)
    return re.sub(r"(?<=[a-z0-9)%])\. (?=[A-Z])", ".  ", text)


# ---------------------------------------------------------------- อ่านไฟล์เนื้อหา → บล็อก
def parse(text):
    blocks, lines, i, bio = [], text.split("\n"), 0, False
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        i += 1
        if not s or s.startswith("<!--"):
            continue
        if s.startswith("@bio "):
            blocks.append(("bio", s[5:].strip()))
            bio = True
            continue
        if bio:
            blocks.append(("line", s))
            continue
        if s in ("@cover", "@front", "@pagebreak", "@bibliography", "@space"):
            blocks.append((s[1:],))
            continue
        if s in ("@toc", "@lof", "@lot"):
            blocks.append(("toc", s[1:]))
            continue
        if s.startswith("@right "):
            blocks.append(("right", s[7:].strip()))
            continue
        if s.startswith("@info"):
            blocks.append(("info", s.split()[1]))
            continue
        if s.startswith("# "):
            t = s[2:].strip()
            blocks.append(("front_title", t[:-7].strip()) if t.endswith("{front}") else ("chapter", t))
            continue
        m = re.match(r"(#{2,4}) (.+)", s)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip()))
            continue
        m = re.match(r"!\[(.+)\]\((.+?)\)(\{w=([\d.]+)\})?", s)
        if m:
            blocks.append(("fig", m.group(1), m.group(2), min(float(m.group(4) or TEXT_W), TEXT_W)))
            continue
        if s.startswith("Table:"):
            cap, widths = s[6:].strip(), None
            mw = re.search(r"\{w=([\d.,]+)\}$", cap)
            if mw:
                widths = [float(x) for x in mw.group(1).split(",")]
                if sum(widths) > TEXT_W:                  # กำหนดไว้ตามหน้า Letter (6.0") — ย่อตามสัดส่วนให้พอดีหน้า A4
                    widths = [round(w * TEXT_W / sum(widths), 3) for w in widths]
                cap = cap[:mw.start()].strip()
            rows, aligns = [], None
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    aligns = [("c" if c.endswith(":") else "l") if c.startswith(":") else None for c in cells]
                else:
                    rows.append(cells)
                i += 1
            blocks.append(("table", cap, rows, widths, aligns))
            continue
        if s.startswith("$$") and s.endswith("$$"):
            blocks.append(("formula", s[2:-2].strip()))
            continue
        if line.startswith(("  - ", "    - ")):
            blocks.append(("bullet", 1, s[2:]))
            continue
        if s.startswith("- "):
            blocks.append(("bullet", 0, s[2:]))
            continue
        blocks.append(("para", s))
    return blocks


class Citations:
    def __init__(self, refs):
        self.refs, self.order = refs, []

    def number(self, key):
        if key not in self.refs:
            raise KeyError(f"ไม่มีอ้างอิง {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def sub(self, text):
        return re.sub(r"\[(@[^\]]+)\]", lambda m: " ".join(f"[{self.number(k.strip().lstrip('@'))}]" for k in m.group(1).split(";")), text)


def split_caption(caption, word):
    m = re.match(rf"({word}\s+[\d.]+)\s*(.*)", caption)
    return (m.group(1), m.group(2)) if m else ("", caption)


def caption_parts(caption, word, cont=False):
    # (ป้าย, คำบรรยาย, ระยะเยื้อง pt, ยาวเกิน 1 บรรทัดไหม) — บรรทัดต่อไปเริ่มตรงอักษรตัวแรกของคำบรรยาย (คู่มือข้อ 4.6, 4.8)
    num, rest = split_caption(caption, word)
    if cont:
        num += " (ต่อ)"
    hang = round(text_pt(num + " ", bold=True), 1)
    plain = re.sub(r"\[@[^\]]+\]", "[00]", re.sub(r"\*+", "", rest))
    return num, rest, hang, hang + text_pt(plain) > TEXT_W * 72 - 2


def toc_entries(blocks):
    # รายการของสารบัญ: (ชนิด, ระดับ, ข้อความ) ตามลำดับในเล่ม — ชนิด toc / lof / lot
    out = []
    for b in blocks:
        if b[0] == "front_title":
            out.append(("toc", 1, b[1]))
        elif b[0] == "toc":
            out.append(("toc", 1, {"toc": "สารบัญ", "lof": "สารบัญรูป", "lot": "สารบัญตาราง"}[b[1]]))
        elif b[0] == "chapter":
            out.append(("toc", 1, b[1]))                 # คีย์ = ข้อความดิบ (มี |) ตรงกับเครื่องหมายในพรีวิว; แสดงผลแบบบรรทัดเดียว
        elif b[0] == "h" and b[1] <= 3:
            out.append(("toc", b[1], b[2]))
        elif b[0] == "fig":
            out.append(("lof", 1, b[1]))
        elif b[0] == "table":
            out.append(("lot", 1, b[1]))
        elif b[0] == "bibliography":
            out.append(("toc", 1, "บรรณานุกรม"))
    return out


def bullet_indents(blocks):
    # ระยะเยื้องซ้าย (นิ้ว) ของรายการระดับบนสุด ตามบล็อกก่อนหน้ารายการ — แบบที่ผู้จัดทำจัดในบทที่ 1
    out, base = [], 0.5
    for i, b in enumerate(blocks):
        if b[0] == "bullet":
            if i == 0 or blocks[i - 1][0] != "bullet":
                prev = blocks[i - 1] if i else ("", "")
                if prev[0] == "h" and prev[1] == 3:
                    base = 0.75
                elif prev[0] == "para" and re.fullmatch(r"\*\*[^*]+\*\*", prev[1].strip()):
                    base = 1.0
                else:
                    base = 0.5
            out.append(base + 0.5 * b[1])
        else:
            out.append(None)
    return out


def cell_center(i, j, txt, aligns):
    # หัวตารางกึ่งกลางเสมอ; คอลัมน์ที่กำหนดในแถวคั่น (:---: / :---) ใช้ตามนั้น; นอกนั้นข้อความสั้น (≤ 16 ตัว) นอกคอลัมน์แรกกึ่งกลาง
    if i == 0:
        return True
    if aligns and j < len(aligns) and aligns[j]:
        return aligns[j] == "c"
    return j > 0 and len(re.sub(r"\*", "", txt)) <= 16


def roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    s = ""
    for v, r in vals:
        while n >= v:
            s, n = s + r, n - v
    return s


# ---------------------------------------------------------------- ตัวช่วย XML ของ Word
def set_run_font(run, size=None, bold=None, italic=None, font=FONT):
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), font)
    def flag(tag, on):
        el = rpr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            rpr.append(el)
        el.set(qn("w:val"), on)
    if size is not None:
        run.font.size = Pt(size)
        flag("w:szCs", str(int(size * 2)))
    if bold is not None:
        run.font.bold = bold
        flag("w:bCs", "1" if bold else "0")
    if italic is not None:
        run.font.italic = italic
        flag("w:iCs", "1" if italic else "0")


def style_font(style, size, bold=False):
    style.font.name, style.font.size, style.font.bold, style.font.italic = FONT, Pt(size), bold, False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), FONT)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(k)) is not None:
            del fonts.attrib[qn(k)]
    for tag, val in (("w:szCs", str(size * 2)), ("w:bCs", "1" if bold else "0")):
        el = rpr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            rpr.append(el)
        el.set(qn("w:val"), val)
    lang = rpr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rpr.append(lang)
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:bidi"), "th-TH")


def para_format(p, align=None, first=None, left=None, before=0, after=0, keep_next=False):
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    if first is not None:
        pf.first_line_indent = Inches(first)
    if left is not None:
        pf.left_indent = Inches(left)
    pf.space_before, pf.space_after, pf.keep_with_next = Pt(before), Pt(after), keep_next


def thai_justify(p):
    ppr = p._element.get_or_add_pPr()
    jc = ppr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        ppr.append(jc)
    jc.set(qn("w:val"), "thaiDistribute")


def fld_char(paragraph, kind):
    r = paragraph.add_run()
    el = OxmlElement("w:fldChar")
    el.set(qn("w:fldCharType"), kind)
    if kind == "begin":
        el.set(qn("w:dirty"), "true")
    r._element.append(el)


def instr(paragraph, text):
    r = paragraph.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {text} "
    r._element.append(it)


def add_field(paragraph, text, placeholder=""):
    fld_char(paragraph, "begin")
    instr(paragraph, text)
    fld_char(paragraph, "separate")
    if placeholder:
        set_run_font(paragraph.add_run(placeholder), size=BODY_PT)
    fld_char(paragraph, "end")


def cell_shade(cell, fill):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", fill)):
        shd.set(qn(k), v)
    tcpr.append(shd)


def row_prop(row, tag):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement(tag)
    el.set(qn("w:val"), "true")
    trpr.append(el)


BIO_PHOTO_W, BIO_PHOTO_TOP = 1.53, 0.21    # รูปผู้เขียน (นิ้ว) ขนาดและระยะจากบนย่อหน้าหัวเรื่อง ตามที่ผู้จัดทำวางใน Word


def float_right(run, top_in):
    # แปลงรูปในบรรทัด (wp:inline) เป็นรูปลอยชิดขอบขวา ห่างจากบนย่อหน้า top_in นิ้ว ข้อความไหลรอบรูป
    inline = run._r.find(f".//{qn('wp:inline')}")
    anchor = OxmlElement("wp:anchor")
    for k, v in (("distT", "0"), ("distB", "0"), ("distL", "114300"), ("distR", "114300"), ("simplePos", "0"),
                 ("relativeHeight", "251658240"), ("behindDoc", "0"), ("locked", "0"), ("layoutInCell", "1"), ("allowOverlap", "1")):
        anchor.set(k, v)
    sp = OxmlElement("wp:simplePos")
    sp.set("x", "0")
    sp.set("y", "0")
    ph, al = OxmlElement("wp:positionH"), OxmlElement("wp:align")
    ph.set("relativeFrom", "margin")
    al.text = "right"
    ph.append(al)
    pv, off = OxmlElement("wp:positionV"), OxmlElement("wp:posOffset")
    pv.set("relativeFrom", "paragraph")
    off.text = str(int(Emu(Inches(top_in))))
    pv.append(off)
    wrap = OxmlElement("wp:wrapSquare")
    wrap.set("wrapText", "bothSides")
    anchor.extend([sp, ph, pv, copy.deepcopy(inline.find(qn("wp:extent")))])
    eff = inline.find(qn("wp:effectExtent"))
    if eff is not None:
        anchor.append(copy.deepcopy(eff))
    anchor.append(wrap)
    for tag in ("wp:docPr", "wp:cNvGraphicFramePr", "a:graphic"):
        el = inline.find(qn(tag))
        if el is not None:
            anchor.append(copy.deepcopy(el))
    inline.getparent().replace(inline, anchor)


def print_image(path, width_in, dpi=300):
    # ย่อรูปให้พอดีความละเอียดพิมพ์ (300 dpi ที่ขนาดจริงในเล่ม) — ภาพหน้าจอเว็บเก็บเป็น JPEG, แผนภาพ / กราฟ / แผนที่คง PNG
    from PIL import Image
    cache = SRC / "_print"
    cache.mkdir(exist_ok=True)
    im = Image.open(path)
    target = int(width_in * dpi)
    photo = path.name.startswith("web_")
    out = cache / (path.stem + (".jpg" if photo else ".png"))
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    if im.width > target:
        im = im.resize((target, round(im.height * target / im.width)), Image.LANCZOS)
    if photo:
        im.convert("RGB").save(out, quality=88, optimize=True)
    else:
        im.save(out, optimize=True)
    return out


# ---------------------------------------------------------------- ตัวสร้าง .docx
class DocxReport:
    def __init__(self, meta, refs, pages):
        self.meta, self.cite, self.pages = meta, Citations(refs), pages      # pages: {(ชนิด, ข้อความ): ป้ายเลขหน้า}
        self.doc = Document()
        self.kind = None
        self.first_in_section = True
        self.cover_docx = False
        self._styles()
        self._page(self.doc.sections[0])

    def _styles(self):
        st = self.doc.styles
        style_font(st["Normal"], BODY_PT)
        st["Normal"].paragraph_format.space_after = Pt(0)
        st["Normal"].paragraph_format.line_spacing = 1.0
        for name, size in (("Heading 1", TITLE_PT), ("Heading 2", H2_PT), ("Heading 3", H3_PT), ("Heading 4", BODY_PT)):
            style_font(st[name], size, bold=True)
            pf = st[name].paragraph_format
            pf.space_before, pf.space_after, pf.keep_with_next = Pt(0), Pt(0), True
        for lvl, ind in ((1, 0.0), (2, 0.25), (3, 0.55)):
            s = st[f"TOC {lvl}"] if f"TOC {lvl}" in [x.name for x in st] else st.add_style(f"TOC {lvl}", WD_STYLE_TYPE.PARAGRAPH)
            style_font(s, BODY_PT)
            s.base_style = st["Normal"]
            s.paragraph_format.left_indent = Inches(ind)
            s.paragraph_format.space_after = Pt(0)
            s.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        # TableCaptionCont = คำบรรยาย "ตารางที่ x.y (ต่อ)" — คนละสไตล์ จึงไม่ซ้ำในสารบัญตาราง (ฟิลด์ TOC \t "TableCaption,1")
        for name in ("FigureCaption", "TableCaption", "TableCaptionCont"):
            s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            s.base_style = st["Normal"]
            style_font(s, CAPTION_PT)
        # หัวบท 2 บรรทัด ("บทที่ 1" / "บทนำ"): ไม่ใช่ Heading 1 เพราะสารบัญจะแตกเป็น 2 บรรทัดตามการขึ้นบรรทัด —
        # ใส่ฟิลด์ TC "บทที่ 1 บทนำ" แทน (สารบัญแสดงบรรทัดเดียวแบบตัวอย่าง; ไม่ตั้งระดับเค้าร่าง เพื่อไม่ให้ซ้ำในสารบัญ)
        s = st.add_style("ChapterTitle", WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st["Normal"]
        style_font(s, TITLE_PT, bold=True)

    def _page(self, s):
        s.page_width, s.page_height = Mm(210), Mm(297)
        s.top_margin, s.bottom_margin = Inches(TOP), Inches(BOTTOM)
        s.left_margin, s.right_margin = Inches(LEFT), Inches(RIGHT)
        s.header_distance, s.footer_distance = Inches(HEADER_IN), Inches(FOOTER_IN)

    def _pgnum(self, s, fmt, start=None):
        pg = s._sectPr.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType")
            s._sectPr.append(pg)
        pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))
        elif pg.get(qn("w:start")) is not None:             # add_section คัดลอกค่าของส่วนก่อนหน้ามา: ลบออกเพื่อให้นับหน้าต่อ
            del pg.attrib[qn("w:start")]

    def _header_number(self, s, show_first):
        s.header.is_linked_to_previous = False
        hp = s.header.paragraphs[0]
        para_format(hp, align=WD_ALIGN_PARAGRAPH.RIGHT)
        add_field(hp, "PAGE", "1")
        for r in hp.runs:
            set_run_font(r, size=BODY_PT)
        s.different_first_page_header_footer = not show_first
        if not show_first:
            s.first_page_header.is_linked_to_previous = False
        for f in (s.footer, s.first_page_footer):           # ท้ายกระดาษว่าง (ไม่รับเลขหน้าโรมันจากส่วนหน้า)
            f.is_linked_to_previous = False

    def _footer_number(self, s, footers):
        # เลขหน้าโรมันกลางท้ายกระดาษ ห่างขอบล่าง 1" (คู่มือข้อ 4.5.1)
        for f in footers:
            f.is_linked_to_previous = False
            fp = f.paragraphs[0]
            para_format(fp, align=WD_ALIGN_PARAGRAPH.CENTER)
            add_field(fp, "PAGE", "I")
            for r in fp.runs:
                set_run_font(r, size=BODY_PT)

    def section(self, kind):
        if self.kind is None:
            s = self.doc.sections[0]
        else:
            s = self.doc.add_section(WD_SECTION.NEW_PAGE)
            self._page(s)
            if self.kind == "cover" and self.cover_docx:
                # ตัวแบ่งส่วนอยู่ท้ายย่อหน้าสุดท้ายของปก (แบบไฟล์ Word ของผู้จัดทำ) ไม่เพิ่มย่อหน้าว่างอีกหนึ่งย่อหน้า
                brk = self.doc.element.body.sectPr.getprevious()
                ppr = brk.getprevious().find(qn("w:pPr"))
                sect = brk.find(qn("w:pPr")).find(qn("w:sectPr"))
                change = ppr.find(qn("w:pPrChange"))
                ppr.insert(list(ppr).index(change), sect) if change is not None else ppr.append(sect)
                brk.getparent().remove(brk)
        if kind == "cover":
            s.header.is_linked_to_previous = False
            s.footer.is_linked_to_previous = False
        elif kind == "front":
            self._pgnum(s, "upperRoman", 1)
            s.header.is_linked_to_previous = False
            self._footer_number(s, [s.footer])
        else:
            self._pgnum(s, "decimal", 1 if self.kind != "chapter" else None)
            self._header_number(s, False)
        self.kind, self.first_in_section = kind, True
        return s

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def front_page(self):
        if not self.first_in_section:
            self.page_break()
        self.first_in_section = False

    def runs(self, p, text, size=BODY_PT, bold=False, italic=False):
        for n, seg in enumerate(text.split("<br>")):
            if n:
                p.add_run().add_break(WD_BREAK.LINE)
            for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", self.cite.sub(seg)):
                if not tok:
                    continue
                b, i, t = bold, italic, tok
                if tok.startswith("**") and tok.endswith("**"):
                    b, t = True, tok[2:-2]
                elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                    i, t = True, tok[1:-1]
                set_run_font(p.add_run(t), size=size, bold=b, italic=i)

    def para(self, text):
        p = self.doc.add_paragraph()
        thai_justify(p)
        para_format(p, first=0.5)
        self.runs(p, sentence_spaces(text))

    def right(self, text):
        p = self.doc.add_paragraph()
        para_format(p, align=WD_ALIGN_PARAGRAPH.RIGHT)
        self.runs(p, text)

    def _bullet_num(self):
        # รายการสัญลักษณ์ของ Word: ระดับ 1 = จุด (Symbol), ระดับ 2 = o (Courier New) — สร้างครั้งเดียวในส่วน numbering
        if getattr(self, "num_id", None):
            return self.num_id
        numbering = self.doc.part.numbering_part.element
        ids = [int(n.get(qn("w:abstractNumId"))) for n in numbering.findall(qn("w:abstractNum"))]
        aid = max(ids + [0]) + 1
        an = OxmlElement("w:abstractNum")
        an.set(qn("w:abstractNumId"), str(aid))
        mlt = OxmlElement("w:multiLevelType")
        mlt.set(qn("w:val"), "hybridMultilevel")
        an.append(mlt)
        for lvl, (sym, font) in enumerate([("\uf0b7", "Symbol"), ("o", "Courier New")]):
            lv = OxmlElement("w:lvl")
            lv.set(qn("w:ilvl"), str(lvl))
            for tag, val in (("w:start", "1"), ("w:numFmt", "bullet"), ("w:lvlText", sym), ("w:lvlJc", "left")):
                e = OxmlElement(tag)
                e.set(qn("w:val"), val)
                lv.append(e)
            ppr, ind = OxmlElement("w:pPr"), OxmlElement("w:ind")
            ind.set(qn("w:left"), str(720 * (lvl + 1)))
            ind.set(qn("w:hanging"), "360")
            ppr.append(ind)
            lv.append(ppr)
            rpr, rf = OxmlElement("w:rPr"), OxmlElement("w:rFonts")
            for a in ("w:ascii", "w:hAnsi"):
                rf.set(qn(a), font)
            rf.set(qn("w:hint"), "default")
            rpr.append(rf)
            lv.append(rpr)
            an.append(lv)
        first_num = numbering.find(qn("w:num"))
        if first_num is not None:
            first_num.addprevious(an)
        else:
            numbering.append(an)
        nid = max([int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))] + [0]) + 1
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(nid))
        ref = OxmlElement("w:abstractNumId")
        ref.set(qn("w:val"), str(aid))
        num.append(ref)
        numbering.append(num)
        self.num_id = nid
        return nid

    def bullet(self, level, text, left=0.5):
        p = self.doc.add_paragraph(style="List Paragraph")
        thai_justify(p)
        para_format(p, left=left, first=-0.25)
        ppr = p._p.get_or_add_pPr()
        npr = OxmlElement("w:numPr")
        for tag, val in (("w:ilvl", str(level)), ("w:numId", str(self._bullet_num()))):
            e = OxmlElement(tag)
            e.set(qn("w:val"), val)
            npr.append(e)
        pstyle = ppr.find(qn("w:pStyle"))            # ลำดับใน pPr: pStyle ก่อน numPr
        if pstyle is None:
            ppr.insert(0, npr)
        else:
            pstyle.addnext(npr)
        self.runs(p, sentence_spaces(text))

    def title(self, text, style="Heading 1"):
        parts = [t.strip() for t in text.split("|")]
        p = self.doc.add_paragraph(style="ChapterTitle" if len(parts) > 1 else style)
        para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, after=12, keep_next=True)
        if len(parts) > 1:
            fld_char(p, "begin")
            instr(p, f'TC "{" ".join(parts)}" \\l 1')
            fld_char(p, "end")
        for k, t in enumerate(parts):
            r = p.add_run(t)
            set_run_font(r, size=CHAP_PT if k < len(parts) - 1 else TITLE_PT, bold=True)   # "บทที่ x" 20, ชื่อบท 24 (ข้อ 4.2.3–4.2.4)
            if k < len(parts) - 1:
                r.add_break()

    def heading(self, level, text):
        p = self.doc.add_paragraph(style=f"Heading {level}")
        size = {2: H2_PT, 3: H3_PT, 4: BODY_PT}[level]
        para_format(p, left={2: 0, 3: 0.25, 4: 0.5}[level], before=12 if level == 2 else 6, after=3, keep_next=True)
        self.runs(p, text, size=size, bold=True)

    def formula(self, text):
        p = self.doc.add_paragraph()
        para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=4)
        set_run_font(p.add_run(text), size=EQ_PT, italic=True, font=EQ_FONT)

    def caption(self, style, caption, word, before, after, keep_next, center=False, cont=False):
        # ป้ายตัวหนา + คำบรรยาย; หลายบรรทัด = ชิดซ้าย เยื้องแบบแขวนให้บรรทัดต่อไปตรงอักษรแรกของคำบรรยาย
        c = self.doc.add_paragraph(style=style)
        num, rest, hang, multi = caption_parts(caption, word, cont)
        if center and not multi:
            para_format(c, align=WD_ALIGN_PARAGRAPH.CENTER, before=before, after=after, keep_next=keep_next)
        else:
            para_format(c, align=WD_ALIGN_PARAGRAPH.LEFT, before=before, after=after, keep_next=keep_next)
            c.paragraph_format.left_indent, c.paragraph_format.first_line_indent = Pt(hang), Pt(-hang)
        if cont:
            c.paragraph_format.page_break_before = True
        if num:
            set_run_font(c.add_run(num + " "), size=CAPTION_PT, bold=True)
        self.runs(c, rest, size=CAPTION_PT)

    def figure(self, caption, path, width):
        # เว้น 1 บรรทัดก่อนรูป และหลังคำบรรยาย (ข้อ 4.8)
        p = self.doc.add_paragraph()
        para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=BLANK_PT, keep_next=True)
        p.add_run().add_picture(str(print_image(SRC / path, width)), width=Inches(width))
        self.caption("FigureCaption", caption, "รูปที่", 0, BLANK_PT, False, center=True)

    def table(self, caption, rows, widths, aligns=None, chunks=None):
        # แต่ละช่วงของตารางอยู่หน้าเดียวกันทั้งช่วง (keep with next ทุกแถว) — ช่วงถัดไปขึ้นหน้าใหม่พร้อม "ตารางที่ x.y (ต่อ)" + หัวตาราง (ข้อ 4.6)
        ncol = max(len(r) for r in rows)
        for n, (a, b) in enumerate(chunks or [(1, len(rows))]):
            self.caption("TableCaptionCont" if n else "TableCaption", caption, "ตารางที่", 0 if n else 8, 0, True, cont=n > 0)
            part = [rows[0]] + rows[a:b]
            t = self.doc.add_table(rows=len(part), cols=ncol)
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = widths is None
            for i, row in enumerate(part):
                for j in range(ncol):
                    cell = t.cell(i, j)
                    txt = row[j] if j < len(row) else ""
                    cp = cell.paragraphs[0]
                    center = cell_center(i, j, txt, aligns)
                    para_format(cp, align=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT, keep_next=i < len(part) - 1)
                    self.runs(cp, txt, size=TABLE_PT, bold=(i == 0))
                    if i == 0:
                        cell_shade(cell, "E7EDF0")
                    if widths:
                        cell.width = Inches(widths[j])
                if i == 0:
                    row_prop(t.rows[i], "w:tblHeader")
                row_prop(t.rows[i], "w:cantSplit")
        self.doc.add_paragraph()                       # เว้น 1 บรรทัดหลังตาราง (ข้อ 4.6)

    def cover(self):
        if COVER.exists():
            # ปกที่จัดใน Word: คัดลอกย่อหน้าทั้งหมดและขนาด/ขอบกระดาษของส่วนปกตามต้นฉบับ
            els, sect = cover_parts()
            body = self.doc.element.body
            for e in els:
                body.sectPr.addprevious(copy.deepcopy(e))
            mine = self.doc.sections[0]._sectPr
            for tag in ("w:pgSz", "w:pgMar"):
                if sect.find(qn(tag)) is not None:
                    mine.replace(mine.find(qn(tag)), copy.deepcopy(sect.find(qn(tag))))
            self.cover_docx = True
            return
        m = self.meta
        def line(text, size, bold=False, before=0):
            p = self.doc.add_paragraph()
            para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=before)
            for n, t in enumerate(text.split("\n")):
                if n:
                    p.add_run().add_break(WD_BREAK.LINE)
                set_run_font(p.add_run(t), size=size, bold=bold)
        line(m["title_th"], 20, True, 24)
        line(m["title_en"].upper(), 20, True, 8)
        line("โดย", 18, True, 110)
        line(m["author_th"], 18, True, 10)
        line(m["author_en"].upper(), 18, True)
        for k, t in enumerate(m["cover_lines"]):
            line(t, 18, True, 140 if k == 0 else 0)

    def info(self, lang):
        for label, value in self.meta["info_th" if lang == "th" else "info_en"]:
            for n, v in enumerate(value.split("\n")):         # ชื่อเรื่องหลายบรรทัด: บรรทัดต่อไปเยื้องตรงกับค่า
                p = self.doc.add_paragraph()
                para_format(p, left=1.75, first=-1.75)
                p.paragraph_format.tab_stops.add_tab_stop(Inches(1.75))
                if not n:
                    set_run_font(p.add_run(label), size=BODY_PT, bold=True)
                set_run_font(p.add_run("\t" + v), size=BODY_PT)
        self.doc.add_paragraph()

    def toc_section(self, kind):
        # สารบัญ / สารบัญรูป / สารบัญตาราง ขึ้นหน้าใหม่เป็นส่วนของตัวเอง เลขหน้าโรมันนับต่อ
        # เลขหน้าโรมันกลางท้ายกระดาษทุกหน้า; หน้าต่อ ๆ ไปมี "สารบัญ (ต่อ)" ตรงกลางหัวกระดาษ (หน้าแรกหัวกระดาษว่าง)
        # อยู่ในหัวกระดาษจึงตามการแบ่งหน้าจริงของ Word เสมอ และไม่หายเมื่อ Word อัปเดตสารบัญ
        title = {"toc": "สารบัญ", "lof": "สารบัญรูป", "lot": "สารบัญตาราง"}[kind]
        s = self.doc.add_section(WD_SECTION.NEW_PAGE)
        self._page(s)
        self._pgnum(s, "upperRoman")
        s.different_first_page_header_footer = True
        s.first_page_header.is_linked_to_previous = False
        s.header.is_linked_to_previous = False
        tp = s.header.paragraphs[0]
        para_format(tp, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_run_font(tp.add_run(f"{title} (ต่อ)"), size=TITLE_PT, bold=True)
        self._footer_number(s, [s.first_page_footer, s.footer])
        self.first_in_section = False

    def toc(self, kind, entries):
        title = {"toc": "สารบัญ", "lof": "สารบัญรูป", "lot": "สารบัญตาราง"}[kind]
        self.title(title)
        rows = [e for e in entries if e[0] == kind]
        code = {"toc": 'TOC \\o "1-3" \\f \\h \\z', "lof": 'TOC \\h \\z \\t "FigureCaption,1"', "lot": 'TOC \\h \\z \\t "TableCaption,1"'}[kind]
        # ฟิลด์ TOC ที่มีผลลัพธ์ล่วงหน้า (รายการ + เลขหน้าจากพรีวิว) — Word แทนด้วยค่าจริงเมื่ออัปเดตฟิลด์
        for k, (_, lvl, text) in enumerate(rows):
            p = self.doc.add_paragraph(style=f"TOC {lvl}")
            if k == 0:
                fld_char(p, "begin")
                instr(p, code)
                fld_char(p, "separate")
            label = " ".join(t.strip() for t in text.split("|"))
            set_run_font(p.add_run(self.cite.sub(label) if "[@" in label else label), size=BODY_PT)
            set_run_font(p.add_run("\t" + self.pages.get((kind, text), "")), size=BODY_PT)
            if k == len(rows) - 1:
                fld_char(p, "end")

    def bibliography(self):
        self.title("บรรณานุกรม")
        for k, key in enumerate(self.cite.order, 1):
            p = self.doc.add_paragraph()
            para_format(p, left=0.5, first=-0.5, after=2)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.5))
            set_run_font(p.add_run(f"[{k}]\t"), size=BODY_PT)
            self.runs(p, self.cite.refs[key])           # **ชื่อหนังสือ** = ตัวหนา (ข้อ 4.10.1)

    def bio(self, path):
        # หน้าประวัติผู้เขียน (ตามที่ผู้จัดทำจัดใน Word): หัวเรื่องชิดซ้าย 24 pt หนา + รูปผู้เขียนลอยชิดขวา
        self.page_break()
        p = self.doc.add_paragraph()
        para_format(p, after=2)
        run = p.add_run()
        run.add_picture(str(SRC / path), width=Inches(BIO_PHOTO_W))
        float_right(run, BIO_PHOTO_TOP)
        set_run_font(p.add_run("ประวัติผู้เขียน"), size=TITLE_PT, bold=True)
        para_format(self.doc.add_paragraph(), after=2)

    def line(self, text):
        p = self.doc.add_paragraph()
        para_format(p, left=0.5, first=-0.5, after=2)
        self.runs(p, text)

    def settings(self):
        st = self.doc.settings.element
        el = OxmlElement("w:updateFields")         # Word ถามเพื่ออัปเดตเลขหน้าในสารบัญเมื่อเปิดไฟล์
        el.set(qn("w:val"), "true")
        st.append(el)


def build_docx(blocks, meta, refs, pages, out, plan):
    R = DocxReport(meta, refs, pages)
    entries = toc_entries(blocks)
    indents = bullet_indents(blocks)
    for i, b in enumerate(blocks):
        k = b[0]
        if k == "cover":
            R.section("cover")
            R.cover()
        elif k == "front":
            R.section("front")
        elif k == "front_title":
            if blocks[i - 1][0] != "info":      # หน้าบทคัดย่อ: ข้อมูลโครงงานอยู่เหนือหัวเรื่อง และขึ้นหน้าใหม่ไปแล้ว
                R.front_page()
            R.title(b[1])
        elif k == "info":
            if blocks[i - 1][0] != "front_title":
                R.front_page()
            R.info(b[1])
        elif k == "toc":
            R.toc_section(b[1])
            R.toc(b[1], entries)
        elif k == "chapter":
            R.section("chapter")
            R.title(b[1])
        elif k == "bibliography":
            R.section("chapter")
            R.bibliography()
        elif k == "h":
            R.heading(b[1], b[2])
        elif k == "fig":
            R.figure(*b[1:])
        elif k == "table":
            R.table(*b[1:], chunks=plan.get(i))
        elif k == "bio":
            R.bio(b[1])
        elif k == "line":
            R.line(b[1])
        elif k == "space":
            R.doc.add_paragraph()
        elif k == "formula":
            R.formula(b[1])
        elif k == "bullet":
            R.bullet(b[1], b[2], indents[i])
        elif k == "pagebreak":
            R.page_break()
        elif k == "para":
            R.para(b[1])
        elif k == "right":
            R.right(b[1])
    R.settings()
    R.doc.save(out)
    return R.cite


# ---------------------------------------------------------------- หน้าปกจาก cover.docx
def cover_parts():
    body = Document(COVER).element.body
    return [e for e in body.iterchildren() if e.tag != qn("w:sectPr")], body.sectPr


def take_cover(src):
    # เก็บหน้าปกจากไฟล์ที่แก้ใน Word: ทุกอย่างก่อนตัวแบ่งส่วนแรก + ขนาด/ขอบกระดาษของส่วนนั้น
    d = Document(src)
    body = d.element.body
    els = list(body.iterchildren())
    end = next(k for k, e in enumerate(els) if e.find(f"{qn('w:pPr')}/{qn('w:sectPr')}") is not None)
    sect = els[end].find(f"{qn('w:pPr')}/{qn('w:sectPr')}")
    sect.getparent().remove(sect)
    for e in els[end + 1:]:
        body.remove(e)
    for ref in sect.findall(qn("w:headerReference")) + sect.findall(qn("w:footerReference")):
        sect.remove(ref)
    body.append(sect)
    xml = d.element.xml
    for rid, rel in list(d.part.rels.items()):          # รูป / หัวกระดาษของเนื้อหาที่ตัดออกแล้ว
        if rel.reltype.rsplit("/", 1)[-1] in ("image", "header", "footer") and f'"{rid}"' not in xml:
            d.part.rels.pop(rid)
    d.save(COVER)
    print(f"saved {COVER} | ย่อหน้า {end + 1}")


def cover_html():
    # ปกในพรีวิวแบบคร่าว ๆ (การจัดแนว ระยะห่าง ขนาด ตัวหนา แท็บ ขึ้นหน้าใหม่) — ใช้ดูหน้าตาและนับหน้าเท่านั้น
    els, sect = cover_parts()
    def W(e, a):
        return e.get(qn(f"w:{a}"))
    def tw(v):
        return int(v or 0) / 20                     # twip → pt
    m = sect.find(qn("w:pgMar"))
    o = [f"<style>@page cover {{ margin: {tw(W(m, 'top'))}pt {tw(W(m, 'right'))}pt {tw(W(m, 'bottom'))}pt {tw(W(m, 'left'))}pt; }}"
         ".wcover { page: cover; } .wcover .tab { display: inline-block; width: 0.5in; }</style><div class='wcover'>"]
    def para(e):
        ppr = e.find(qn("w:pPr"))
        def get(path):
            return ppr.find(path) if ppr is not None else None
        jc, sp, ind, msz = get(qn("w:jc")), get(qn("w:spacing")), get(qn("w:ind")), get(f"{qn('w:rPr')}/{qn('w:sz')}")
        css = ["text-align:" + {"center": "center", "right": "right", "both": "justify", "thaiDistribute": "justify"}.get(W(jc, "val") if jc is not None else "", "left")]
        if sp is not None:
            css.append(f"margin:{tw(W(sp, 'before'))}pt 0 {tw(W(sp, 'after'))}pt")
            if W(sp, "line") and W(sp, "lineRule") in (None, "auto"):
                css.append(f"line-height:{LINE * int(W(sp, 'line')) / 240:.3f}")
        if ind is not None:
            css.append(f"margin-left:{tw(W(ind, 'left'))}pt;text-indent:{tw(W(ind, 'firstLine')) - tw(W(ind, 'hanging'))}pt")
        css.append(f"font-size:{int(W(msz, 'val')) / 2 if msz is not None else BODY_PT}pt")
        html, breaks = [], 0
        for r in e.iter(qn("w:r")):
            sz, b = r.find(f"{qn('w:rPr')}/{qn('w:sz')}"), r.find(f"{qn('w:rPr')}/{qn('w:b')}")
            st = (f"font-size:{int(W(sz, 'val')) / 2}pt;" if sz is not None else "") + ("font-weight:bold" if b is not None and W(b, "val") not in ("0", "false") else "")
            for c in r:
                if c.tag == qn("w:t"):
                    html.append(f"<span style='{st}'>{H.escape(c.text or '')}</span>")
                elif c.tag == qn("w:tab"):
                    html.append("<span class='tab'></span>")
                elif c.tag == qn("w:br") and W(c, "type") == "page":
                    breaks += 1
                elif c.tag == qn("w:br"):
                    html.append("<br>")
        pre = "<div class='pb'></div>" if get(qn("w:pageBreakBefore")) is not None else ""
        return pre + f"<p style='{';'.join(css)}'>{''.join(html) or '&nbsp;'}</p>" + "<div class='pb'></div>" * breaks

    for e in els:
        if e.tag == qn("w:p"):
            o.append(para(e))
        elif e.tag == qn("w:tbl"):                       # ตารางไม่มีเส้น (เช่น ช่องลงนามคู่กัน)
            o.append("<table style='width:100%;border-collapse:collapse;font-size:inherit'>" + "".join(
                "<tr>" + "".join("<td style='border:0;padding:0;width:50%'>" + "".join(para(p) for p in tc.iter(qn("w:p"))) + "</td>"
                                 for tc in tr.iter(qn("w:tc"))) + "</tr>" for tr in e.iter(qn("w:tr"))) + "</table>")
    return "".join(o) + "</div>"


# ---------------------------------------------------------------- พรีวิว HTML (จัดหน้าแบบเดียวกับ Word เพื่อหาเลขหน้าและตรวจหน้าตา)
CSS = f"""
@font-face {{ font-family: "THS"; src: url("{FONT_DIR}thsarabun.ttf"); }}
@font-face {{ font-family: "THS"; font-weight: bold; src: url("{FONT_DIR}thsarabun-bold.ttf"); }}
@font-face {{ font-family: "THS"; font-style: italic; src: url("{FONT_DIR}thsarabun-italic.ttf"); }}
@font-face {{ font-family: "THS"; font-weight: bold; font-style: italic; src: url("{FONT_DIR}thsarabun-bold-italic.ttf"); }}
@page {{ size: 210mm 297mm; margin: {TOP}in {RIGHT}in {BOTTOM}in {LEFT}in; }}
@page front {{ margin-bottom: calc({FOOTER_IN}in + {BLANK_PT}pt); }}   /* เลขหน้าโรมันท้ายกระดาษห่างขอบล่าง 1": Word ดันเนื้อหาขึ้นเหนือเลขหน้า */
.front {{ page: front; }}
@media screen {{ body {{ width: {TEXT_W}in; }} }}   /* วัดความสูงแถวตารางที่ความกว้างเนื้อหาจริง */
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "THS"; font-size: {BODY_PT}pt; line-height: {LINE}; color: #000; }}
p {{ margin: 0; }}
.para {{ text-indent: 0.5in; text-align: justify; }}
.pb {{ break-before: page; }}
.title {{ text-align: center; font-weight: bold; font-size: {TITLE_PT}pt; margin: 0 0 12pt; break-after: avoid; }}
.title.chap {{ font-size: {CHAP_PT}pt; }} .title.chap span {{ font-size: {TITLE_PT}pt; }}
.h2 {{ font-weight: bold; font-size: {H2_PT}pt; margin: 12pt 0 3pt; break-after: avoid; }}
.h3 {{ font-weight: bold; font-size: {H3_PT}pt; margin: 6pt 0 3pt 0.25in; break-after: avoid; }}
.h4 {{ font-weight: bold; margin: 6pt 0 3pt 0.5in; break-after: avoid; }}
.bul {{ text-align: justify; text-indent: -0.25in; }}
.bul .sym {{ display: inline-block; width: 0.25in; text-indent: 0; }}
.formula {{ text-align: center; font-family: "Times New Roman"; font-style: italic; font-size: {EQ_PT}pt; line-height: 1.15; margin: 4pt 0; }}
.fig {{ text-align: center; margin-top: {BLANK_PT}pt; break-after: avoid; break-inside: avoid; }}
.fig img {{ display: block; margin: 0 auto; }}
.cap {{ text-align: center; font-size: {CAPTION_PT}pt; margin: 0 0 {BLANK_PT}pt; }}
.tcap {{ font-size: {CAPTION_PT}pt; margin: 8pt 0 0; break-after: avoid; }}
.tcap.cont {{ margin-top: 0; }}
.hang {{ text-align: left; }}
.tblk {{ break-inside: avoid; }}
table {{ border-collapse: collapse; margin: 0 auto; font-size: {TABLE_PT}pt; line-height: {LINE}; }}
td, th {{ border: 0.5pt solid #000; padding: 0 0.08in; vertical-align: top; }}
th {{ background: #e7edf0; font-weight: bold; text-align: center; }}
tr {{ break-inside: avoid; }}
thead {{ display: table-header-group; }}
.c {{ text-align: center; }}
.after-table {{ height: calc({BODY_PT}pt * {LINE}); }}
.info p {{ margin-left: 1.75in; text-indent: -1.75in; }}
.info b {{ display: inline-block; width: 1.75in; text-indent: 0; }}
.toc p {{ display: flex; align-items: baseline; }}
.toc p .t {{ flex: none; max-width: 85%; }}
.toc p .d {{ flex: 1; border-bottom: 1.5pt dotted #000; margin: 0 4pt; transform: translateY(-4pt); }}
.toc .l2 {{ margin-left: 0.25in; }} .toc .l3 {{ margin-left: 0.55in; }}
.bib p {{ margin: 0 0 2pt 0.5in; text-indent: -0.5in; }}
.bib .n {{ display: inline-block; width: 0.5in; text-indent: 0; }}
.biot {{ font-weight: bold; font-size: {TITLE_PT}pt; margin: 0 0 2pt; }}
.biot img {{ float: right; width: {BIO_PHOTO_W}in; margin-top: {BIO_PHOTO_TOP}in; margin-left: 0.12in; }}
.line {{ margin: 0 0 2pt 0.5in; text-indent: -0.5in; }}
.cover p {{ text-align: center; font-weight: bold; font-size: 18pt; }}
.mk {{ font-size: 1pt; color: #fff; line-height: 0; }}
"""


class HtmlReport:
    def __init__(self, meta, refs, pages, plan=None):
        self.meta, self.cite, self.pages, self.plan = meta, Citations(refs), pages, plan or {}
        self.out, self.marks, self.first = [], [], True

    def mark(self, kind, text):
        # เครื่องหมายซ่อน (ตัวอักษรสีขาวขนาด 1 pt) ไว้หาเลขหน้าจาก PDF
        self.marks.append((kind, text))
        return f"<span class='mk'>QZMK{len(self.marks) - 1:04d}</span>"

    def esc_runs(self, text):
        t = H.escape(self.cite.sub(text), quote=False)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t).replace("&lt;br&gt;", "<br>")
        return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", t)

    def pb(self):
        if self.first:
            self.first = False
            return ""
        return " pb"

    def caption(self, cls, caption, word, center, cont=False, mark=""):
        num, rest, hang, multi = caption_parts(caption, word, cont)
        style = "" if center and not multi else f" style='padding-left:{hang}pt;text-indent:-{hang}pt'"
        extra = "" if center and not multi else " hang"
        return f"<p class='{cls}{extra}'{style}>{mark}<b>{num}</b> {self.esc_runs(rest)}</p>"

    def build(self, blocks):
        entries = toc_entries(blocks)
        indents = bullet_indents(blocks)
        o = self.out
        o.append(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>")
        in_front = False
        for i, b in enumerate(blocks):
            k = b[0]
            if k == "chapter" and in_front:
                o.append("</div>")
                in_front = False
            if k == "cover" and COVER.exists():
                o.append(cover_html())
                self.first = False
            elif k == "cover":
                m = self.meta
                o.append(f"<div class='cover'><p style='font-size:20pt;margin-top:24pt'>{H.escape(m['title_th']).replace(chr(10), '<br>')}</p>"
                         f"<p style='font-size:20pt;margin-top:8pt'>{H.escape(m['title_en'].upper())}</p>"
                         f"<p style='margin-top:110pt'>โดย</p><p style='margin-top:10pt'>{H.escape(m['author_th'])}</p>"
                         f"<p>{H.escape(m['author_en'].upper())}</p>"
                         + "".join(f"<p style='margin-top:{140 if j == 0 else 0}pt'>{H.escape(t)}</p>" for j, t in enumerate(m["cover_lines"])) + "</div>")
                self.first = False
            elif k == "front":
                self.first = True
                in_front = True
                o.append("<div class='pb'></div><div class='front'>")
            elif k == "front_title":
                pb = self.pb() if blocks[i - 1][0] != "info" else ""
                o.append(f"<p class='title{pb}'>{self.mark('toc', b[1])}{H.escape(b[1])}</p>")
            elif k == "info":
                pb = self.pb() if blocks[i - 1][0] != "front_title" else ""
                o.append(f"<div class='info{pb}'>" + "".join(f"<p><b>{H.escape(a)}</b>{H.escape(v).replace(chr(10), '<br>')}</p>"
                         for a, v in self.meta["info_th" if b[1] == "th" else "info_en"]) + "</div><p>&nbsp;</p>")
            elif k == "toc":
                title = {"toc": "สารบัญ", "lof": "สารบัญรูป", "lot": "สารบัญตาราง"}[b[1]]
                o.append(f"<p class='title{self.pb()}'>{self.mark('toc', title)}{title}</p><div class='toc'>")
                for kind, lvl, text in entries:
                    if kind == b[1]:
                        label = " ".join(t.strip() for t in text.split("|"))
                        o.append(f"<p class='l{lvl}'><span class='t'>{H.escape(label)}</span><span class='d'></span>"
                                 f"<span>{self.pages.get((kind, text), '00')}</span></p>")
                o.append("</div>")
            elif k in ("chapter", "bibliography"):
                text = b[1] if k == "chapter" else "บรรณานุกรม"
                parts = [H.escape(t.strip()) for t in text.split("|")]
                inner = "<br>".join(parts[:-1] + [f"<span>{parts[-1]}</span>"]) if len(parts) > 1 else parts[0]
                o.append(f"<p class='title pb{' chap' if len(parts) > 1 else ''}'>{self.mark('toc', text)}{inner}</p>")
                if k == "bibliography":
                    o.append("<div class='bib'>" + "".join(f"<p><span class='n'>[{n}]</span>{self.esc_runs(self.cite.refs[key])}</p>"
                                                          for n, key in enumerate(self.cite.order, 1)) + "</div>")
            elif k == "h":
                mk = self.mark("toc", b[2]) if b[1] <= 3 else ""
                o.append(f"<p class='h{b[1]}'>{mk}{self.esc_runs(b[2])}</p>")
            elif k == "fig":
                cap, path, w = b[1], b[2], b[3]
                o.append(f"<div class='fig'><img src='{(SRC / path).as_uri()}' style='width:{w}in'></div>"
                         + self.caption("cap", cap, "รูปที่", True, mark=self.mark("lof", cap)))
            elif k == "table":
                cap, rows, widths, aligns = b[1], b[2], b[3], b[4]
                tw = f" style='table-layout:fixed;width:{sum(widths)}in'" if widths else ""
                cols = ("<colgroup>" + "".join(f"<col style='width:{w}in'>" for w in widths) + "</colgroup>") if widths else ""
                html = [f"<div class='mk'>QZTB{i:05d}</div>"]      # จุดที่ตารางจะเริ่ม (ไม่เลื่อนตามเมื่อทั้งช่วงถูกดันไปหน้าถัดไป)
                for n, (a, z) in enumerate(self.plan.get(i) or [(1, len(rows))]):
                    head = self.caption("tcap cont" if n else "tcap", cap, "ตารางที่", False, cont=n > 0, mark="" if n else self.mark("lot", cap))
                    html.append(f"<div class='tblk{' pb' if n else ''}'>{head}<table data-t='{i}'{tw}>{cols}")
                    for r, row in enumerate([rows[0]] + rows[a:z]):
                        tag = "th" if r == 0 else "td"
                        cells = "".join(f"<{tag} class='{'c' if cell_center(r, j, c, aligns) else ''}'>{self.esc_runs(c)}</{tag}>"
                                        for j, c in enumerate(row))
                        html.append(("<thead>" if r == 0 else "") + f"<tr>{cells}</tr>" + ("</thead>" if r == 0 else ""))
                    html.append("</table></div>")
                html.append("<div class='after-table'></div>")
                o.append("".join(html))
            elif k == "formula":
                o.append(f"<p class='formula'>{H.escape(b[1])}</p>")
            elif k == "bullet":
                sym = "•" if b[1] == 0 else "o"
                text = self.esc_runs(sentence_spaces(b[2])).replace(".  ", ".&nbsp; ")
                o.append(f"<p class='bul' style='margin-left:{indents[i]}in'><span class='sym'>{sym}</span>{text}</p>")
            elif k == "pagebreak":
                o.append("<div class='pb'></div>")
            elif k == "space":
                o.append("<p>&nbsp;</p>")
            elif k == "para":
                o.append(f"<p class='para'>{self.esc_runs(sentence_spaces(b[1])).replace('.  ', '.&nbsp; ')}</p>")
            elif k == "right":
                o.append(f"<p style='text-align:right'>{self.esc_runs(b[1])}</p>")
            elif k == "bio":
                o.append(f"<p class='biot pb'><img src='{(SRC / b[1]).as_uri()}'>ประวัติผู้เขียน</p><p>&nbsp;</p>")
            elif k == "line":
                o.append(f"<p class='line'>{self.esc_runs(b[1])}</p>")
        o.append("</body></html>")
        return "".join(o)


MEASURE_JS = """() => [...document.querySelectorAll('table[data-t]')].map(t => ({
    t: +t.dataset.t,
    rows: [...t.rows].map(r => r.getBoundingClientRect().height * 0.75),
    cap: t.previousElementSibling.getBoundingClientRect().height * 0.75 + 8 }))"""


def print_pdf(html, html_path, pdf_path, measure=False):
    # measure=True: คืนความสูง (pt) ของหัวตาราง / แถว / คำบรรยาย ของทุกตาราง วัดบนหน้าจอที่ความกว้างเนื้อหาจริง
    from playwright.sync_api import sync_playwright
    html_path.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--allow-file-access-from-files"])
        pg = b.new_page()
        pg.goto(html_path.as_uri())
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(500)
        sizes = pg.evaluate(MEASURE_JS) if measure else None
        pg.pdf(path=str(pdf_path), prefer_css_page_size=True, print_background=True)
        b.close()
    return sizes


def table_tops(pdf_path):
    # ตำแหน่งที่แต่ละตารางจะเริ่ม (pt จากขอบบนกระดาษ) = ท้ายเนื้อหาก่อนตาราง จากเครื่องหมาย QZTB ใน PDF: {ลำดับบล็อก: y}
    import pymupdf
    out = {}
    for page in pymupdf.open(str(pdf_path)):
        for w in page.get_text("words"):
            for m in re.finditer(r"QZTB(\d{5})", w[4]):
                out[int(m.group(1))] = w[1]
    return out


def plan_tables(blocks, sizes, tops):
    # แบ่งตารางที่ยาวเกินหน้าเป็นช่วง ๆ (แถวเนื้อหา [a, b)) ให้แต่ละช่วงสูงไม่เกิน FIT ของพื้นที่ — ช่วงแรกเท่าที่เหลือในหน้าที่ตารางเริ่ม
    # ตารางที่สั้นกว่าหน้าไม่แบ่ง (อยู่หน้าเดียวกันทั้งตาราง)
    page_h = TEXT_H * 72
    bottom = (PAGE_H - BOTTOM) * 72
    plan = {}
    for s in sizes:
        i = s["t"]
        hdr, body, cap = s["rows"][0], s["rows"][1:], s["cap"]
        if cap + hdr + sum(body) <= page_h * FIT:
            continue
        top = tops.get(i)
        room = (bottom - top) if top is not None else 0         # cap รวมระยะ 8 pt เหนือคำบรรยายแล้ว
        full, cur = (page_h - cap - hdr) * FIT, (room - cap - hdr) * FIT
        chunks, start, h = [], 1, 0
        for r, rh in enumerate(body, 1):
            if h + rh > cur and r > start:
                chunks.append((start, r))
                start, h, cur = r, 0, full
            elif h + rh > cur and r == start and not chunks:
                cur = full                       # ช่วงแรกไม่พอแม้แต่แถวเดียว: ทั้งช่วงขึ้นหน้าใหม่เอง (keep together)
            h += rh
        chunks.append((start, len(body) + 1))
        plan[i] = chunks
    return plan


def page_labels(pdf_path, marks):
    # เลขหน้าตามแบบของเล่ม: ปก = ไม่มี, ส่วนหน้า = I, II, ...; เนื้อหาเริ่ม 1 ที่หน้าแรกของบทที่ 1
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    where = {}
    for i, page in enumerate(doc):
        for m in re.findall(r"QZMK(\d{4})", page.get_text()):
            where[int(m)] = i
    first_body = min(where[k] for k, (kind, text) in enumerate(marks) if kind == "toc" and text.startswith("บทที่"))
    first_front = min(where.values())          # ปกมีได้หลายหน้า: ส่วนหน้าเริ่ม I ที่หน้าแรกที่มีหัวข้อ
    labels = {}
    for k, (kind, text) in enumerate(marks):
        i = where.get(k)
        if i is None:
            continue
        labels[(kind, text)] = str(i - first_body + 1) if i >= first_body else roman(i - first_front + 1)
    return labels, len(doc), first_body


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_output(out, force):
    # กันเขียนทับงานที่ผู้ใช้แก้เองใน Word: มีไฟล์ล็อก ~$ = เปิดอยู่; ค่าแฮชไม่ตรงกับครั้งล่าสุด = ถูกแก้หลังสร้าง
    lock = [p for p in out.parent.glob("~$*") if out.name.endswith(p.name[2:])]
    if lock:
        sys.exit(f"หยุด: {out.name} เปิดอยู่ใน Word ({lock[0].name}) — ปิดไฟล์ก่อนแล้วรันใหม่")
    if not out.exists() or (STAMP.exists() and STAMP.read_text().strip() == sha256(out)):
        return
    bak = SRC / "_backup"
    bak.mkdir(exist_ok=True)
    dst = bak / f"{out.stem}_{datetime.fromtimestamp(out.stat().st_mtime):%Y%m%d-%H%M%S}.docx"
    shutil.copy2(out, dst)
    if not force:
        sys.exit(f"หยุด: {out.name} ถูกแก้หลังสร้างครั้งล่าสุด (เช่น แก้ใน Word) — สำรองไว้ที่ {dst}\n"
                 "นำการแก้ไขเข้า docs/report_src/parts/*.md ก่อน แล้วรันใหม่ด้วย --force")
    print(f"สำรองไฟล์เดิมที่ถูกแก้ไว้ที่ {dst}")


def main():
    sys.path.insert(0, str(SRC))
    import refs as REF  # noqa: E402
    if "--take-cover" in sys.argv:
        take_cover(Path(sys.argv[sys.argv.index("--take-cover") + 1]))
        return
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    out = Path(args[0]) if args else ROOT / "docs" / "report.docx"
    prev = Path(args[1]) if len(args) > 1 else SRC / "_preview"
    guard_output(out, force)
    prev.mkdir(parents=True, exist_ok=True)
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((SRC / "parts").glob("*.md")))
    blocks = parse(text)
    # รอบที่ 1: หาเลขหน้า (สารบัญใส่ 00 ไว้ก่อน — ความยาวเท่ากัน ไม่ทำให้หน้าเลื่อน) + วัดความสูงแถวของทุกตาราง
    h1 = HtmlReport(REF.META, REF.REFS, {})
    sizes = print_pdf(h1.build(blocks), prev / "report_pass1.html", prev / "report_pass1.pdf", measure=True)
    # แบ่งตารางยาวตามตำแหน่งที่ตารางเริ่ม แล้วจัดหน้าใหม่จนการแบ่งคงที่ (ตำแหน่งขึ้นกับการแบ่งของตารางก่อนหน้า)
    plan, tops = {}, table_tops(prev / "report_pass1.pdf")
    for _ in range(5):
        new = plan_tables(blocks, sizes, tops)
        if new == plan:
            break
        plan = new
        hp = HtmlReport(REF.META, REF.REFS, {}, plan)
        print_pdf(hp.build(blocks), prev / "report_pass1.html", prev / "report_pass1.pdf")
        tops = table_tops(prev / "report_pass1.pdf")
    h1 = HtmlReport(REF.META, REF.REFS, {}, plan)
    h1.build(blocks)
    labels, n, fb = page_labels(prev / "report_pass1.pdf", h1.marks)
    # รอบที่ 2: พรีวิวที่มีเลขหน้าในสารบัญ + .docx
    h2 = HtmlReport(REF.META, REF.REFS, labels, plan)
    print_pdf(h2.build(blocks), prev / "report_preview.html", prev / "report_preview.pdf")
    labels2, n2, _ = page_labels(prev / "report_preview.pdf", h2.marks)
    changed = [k for k in labels if labels2.get(k) != labels[k]]
    cite = build_docx(blocks, REF.META, REF.REFS, labels2, out, plan)
    split = ", ".join(f"{blocks[i][1].split()[1]}×{len(c)}" for i, c in plan.items())
    print(f"ตารางที่แบ่งหน้า (ต่อ): {split or '-'}")
    STAMP.write_text(sha256(out) + "\n")
    missing = sorted(set(REF.REFS) - set(cite.order))
    print(f"saved {out} | หน้าพรีวิว {n2} (เนื้อหาเริ่มหน้า PDF ที่ {fb + 1}) | หัวข้อ {sum(1 for k in labels2 if k[0] == 'toc')} "
          f"รูป {sum(1 for k in labels2 if k[0] == 'lof')} ตาราง {sum(1 for k in labels2 if k[0] == 'lot')} | อ้างอิง {len(cite.order)} "
          f"| ไม่ได้อ้าง {missing} | เลขหน้าเปลี่ยนระหว่างสองรอบ {len(changed)}")


if __name__ == "__main__":
    main()
