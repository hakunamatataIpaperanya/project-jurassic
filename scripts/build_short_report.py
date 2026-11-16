# สร้างรายงานฉบับย่อ (docs/รายงานฉบับย่อ.docx) จาก docs/report_src/short/short.md + อ้างอิง docs/report_src/short/refs_ieee.py
#   ใช้ไฟล์ docs/รูปแบบการเขียนรายงานฉบับย่อ 8 หน้า Update ปีการศึกษา 2568.docx เป็นต้นแบบ (กระดาษ ขอบ สองคอลัมน์ สไตล์)
#   รูปแบบตามไฟล์ตัวอย่าง: ชื่อเรื่อง 24 pt ตัวหนากึ่งกลาง เว้น 12 pt ก่อนชื่อผู้เขียน, ผู้เขียน 16 pt ตัวหนาชิดขวา,
#   หน่วยงาน / อีเมล 13 pt ตัวเอียงชิดขวา, บทคัดย่อภาษาไทย 1 คอลัมน์ (หัวข้อ 16 pt ตัวหนา เนื้อหา 16 pt), คำสำคัญ 13 pt,
#   เนื้อหา 2 คอลัมน์ 13 pt กระจายแบบไทย ย่อหน้าแรกใต้หัวข้อไม่เยื้อง ย่อหน้าต่อไปเยื้อง 1 แท็บ,
#   หัวข้อหลัก 16 pt ตัวหนา / หัวข้อย่อย 14 pt ตัวหนา (เว้นก่อน 12 pt หลัง 3 pt), คำบรรยายรูป (ใต้รูป) และตาราง (เหนือตาราง) 12 pt,
#   สมการกึ่งกลางคอลัมน์ เลขสมการชิดขวา เว้น 1 บรรทัดก่อนและหลัง, ไม่ใส่เลขหน้า,
#   เอกสารอ้างอิง 18 pt ตัวหนากึ่งกลาง ไม่มีเลขหัวข้อ เรียงตามลำดับที่อ้างครั้งแรก แบบ IEEE
#   รูปหรือตารางที่กว้างเกินคอลัมน์ (@span) วางกึ่งกลางเต็มความกว้างสองคอลัมน์ ด้วยตัวแบ่งส่วนแบบต่อเนื่อง
# ขั้นตอน: พรีวิว HTML ที่จัดหน้าแบบเดียวกับ Word (สองคอลัมน์ ฟอนต์ ระยะบรรทัด) → PDF ด้วย Chrome เพื่อนับหน้า → สร้าง .docx
# รัน:  <python ที่มี python-docx, playwright, pymupdf> scripts/build_short_report.py [--force]
#   ไม่เขียนทับไฟล์ที่เปิดอยู่ใน Word หรือไฟล์ที่ถูกแก้หลังสร้างครั้งล่าสุด (สำรองไว้ใน docs/report_src/_backup/ ก่อนเสมอ;
#   นำการแก้ไขเข้า short.md แล้วจึงรันด้วย --force)
#
# รูปแบบไฟล์เนื้อหา: แบบเดียวกับรายงานฉบับเต็ม (scripts/build_report.py) และเพิ่ม
#   @title / @authors / @affil / @email / @abstract / @keywords    ส่วนหัวบทความ (1 คอลัมน์)
#   # 1. หัวข้อหลัก   ## 1.1. หัวข้อย่อย   ### หัวข้อย่อยลำดับที่สอง   # เอกสารอ้างอิง + @bibliography
#   @span               รูปหรือตารางถัดไปกว้างเต็มสองคอลัมน์
#   @pagebreak          ขึ้นหน้าใหม่ (เช่น ให้เนื้อหาเริ่มหน้า 2 ตามที่ผู้จัดทำจัดใน Word)
#   ![คำบรรยาย](a.png|b.png){w=2.8}                                     รูปสองภาพเรียงกัน (กว้างภาพละ w นิ้ว)
#   $$ สมการ     (1) $$                                                 สมการพร้อมเลขสมการ
import html as H
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as BR  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt  # noqa: E402

ROOT, SRC = BR.ROOT, BR.SRC
SHORT = SRC / "short"
TEMPLATE = ROOT / "docs" / "รูปแบบการเขียนรายงานฉบับย่อ 8 หน้า Update ปีการศึกษา 2568.docx"
OUT = ROOT / "docs" / "รายงานฉบับย่อ.docx"
STAMP = SHORT / "_last_build.sha256"
PREVIEW = SHORT / "_preview"
MAX_PAGES = 8
TITLE_PT, AUTHOR_PT, AFFIL_PT, ABS_PT, BODY_PT, H1_PT, H2_PT, CAP_PT, TABLE_PT, REFS_PT = 24, 16, 13, 16, 13, 16, 14, 12, 12, 18
LINE = BR.LINE                     # ระยะบรรทัดเดี่ยวของ TH SarabunPSK
PAGE_W, PAGE_H = 11909 / 1440, 16834 / 1440                 # ขนาดกระดาษของไฟล์รูปแบบ (A4)
MARGIN = (1.0, 1.0, 1.0, 1.5)                                # บน ขวา ล่าง ซ้าย (นิ้ว)
GAP = 708 / 1440                                             # ระยะห่างระหว่างคอลัมน์
TEXT_W = PAGE_W - MARGIN[1] - MARGIN[3]
COL_W = (TEXT_W - GAP) / 2
INDENT, REF_HANG = 0.5, 0.4                                  # เยื้อง 1 แท็บ / ระยะแขวนของรายการอ้างอิง
META_KEYS = ("title", "authors", "affil", "email", "abstract", "keywords")


# ---------------------------------------------------------------- อ่านไฟล์เนื้อหา
def read_source():
    meta, rest = {}, []
    for line in (SHORT / "short.md").read_text(encoding="utf-8").split("\n"):
        m = re.match(r"@(\w+) (.+)", line.strip())
        if m and m.group(1) in META_KEYS:
            meta[m.group(1)] = m.group(2).strip()
        else:
            rest.append(line)
    blocks, span = [], False
    for b in BR.parse("\n".join(rest)):
        if b == ("para", "@span"):
            span = True
            continue
        blocks.append(b + (span,) if b[0] in ("fig", "table") else b)
        span = False
    return meta, blocks


def split_formula(text):
    m = re.match(r"(.*?)\s*\((\d+)\)$", text)
    return (m.group(1).strip(), m.group(2)) if m else (text, "")


def tokens(text):
    # **หนา** / *เอียง* / ข้อความปกติ
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if tok.startswith("**") and tok.endswith("**"):
            yield tok[2:-2], True, False
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            yield tok[1:-1], False, True
        elif tok:
            yield tok, False, False


# ---------------------------------------------------------------- ตัวสร้าง .docx
def set_cols(sect, n):
    cols = OxmlElement("w:cols")
    if n > 1:
        cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), "708")
    old = sect.find(qn("w:cols"))
    if old is not None:
        sect.replace(old, cols)
    else:
        sect.append(cols)


def tiny_mark(p):
    # ย่อหน้าที่มีแต่ตัวแบ่งส่วน: ทำเครื่องหมายย่อหน้าให้เล็กที่สุด ไม่ให้เกิดบรรทัดว่างเพิ่ม
    ppr = p.find(qn("w:pPr"))
    rpr = OxmlElement("w:rPr")
    for tag in ("w:sz", "w:szCs"):
        el = OxmlElement(tag)
        el.set(qn("w:val"), "2")
        rpr.append(el)
    ppr.find(qn("w:sectPr")).addprevious(rpr)
    sp = OxmlElement("w:spacing")
    for k in ("w:before", "w:after", "w:line"):
        sp.set(qn(k), "0" if k != "w:line" else "40")
    sp.set(qn("w:lineRule"), "exact")
    ppr.insert(0, sp)


def borders(table):
    tblpr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", "000000")):
            el.set(qn(k), v)
        b.append(el)
    tblpr.append(b)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr.append(layout)


class ShortDocx:
    def __init__(self, meta, refs):
        self.meta, self.cite = meta, BR.Citations(refs)
        self.doc = Document(str(TEMPLATE))
        body = self.doc.element.body
        for el in list(body):                                # เก็บเฉพาะการตั้งค่าส่วนท้ายเอกสาร (A4, 2 คอลัมน์) ของไฟล์รูปแบบ
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        sect = body.sectPr
        for tag in ("w:headerReference", "w:footerReference", "w:pgNumType"):   # ไม่ใส่เลขหน้า
            for el in sect.findall(qn(tag)):
                sect.remove(el)
        for rid, rel in list(self.doc.part.rels.items()):
            if rel.reltype.rsplit("/", 1)[-1] in ("header", "footer"):
                self.doc.part.rels.pop(rid)
        set_cols(sect, 2)
        st = self.doc.styles
        BR.style_font(st["Normal"], BODY_PT)
        st["Normal"].paragraph_format.space_after = Pt(0)
        st["Normal"].paragraph_format.space_before = Pt(0)
        st["Normal"].paragraph_format.line_spacing = 1.0
        cp = self.doc.core_properties
        cp.author = cp.last_modified_by = "สิดาพร จิตระวีวรรณ"
        cp.title = re.sub(r"\s*<br>\s*", "", meta["title"])
        cp.subject, cp.keywords, cp.comments = "รายงานฉบับย่อ", meta["keywords"], ""
        self.indent_next = False

    def runs(self, p, text, size, bold=False, italic=False):
        for n, seg in enumerate(self.cite.sub(text).split("<br>")):
            if n:
                p.add_run().add_break(WD_BREAK.LINE)
            for t, b, i in tokens(seg):
                BR.set_run_font(p.add_run(t), size=size, bold=bold or b, italic=italic or i)

    def par(self, text="", size=BODY_PT, align=None, bold=False, italic=False, before=0, after=0, first=None, left=None, keep=False):
        p = self.doc.add_paragraph()
        BR.para_format(p, align=align, first=first, left=left, before=before, after=after, keep_next=keep)
        if text:
            self.runs(p, text, size, bold, italic)
        return p

    def section_break(self, ending_cols, next_cols):
        self.doc.add_section(WD_SECTION.CONTINUOUS)
        body = self.doc.element.body
        brk = body.sectPr.getprevious()
        set_cols(brk.find(qn("w:pPr")).find(qn("w:sectPr")), ending_cols)
        set_cols(body.sectPr, next_cols)
        tiny_mark(brk)

    def front(self):
        m = self.meta
        self.par(m["title"], TITLE_PT, WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=12)
        self.par(m["authors"], AUTHOR_PT, WD_ALIGN_PARAGRAPH.RIGHT, bold=True)
        self.par(m["affil"], AFFIL_PT, WD_ALIGN_PARAGRAPH.RIGHT, italic=True)
        self.par(m["email"], AFFIL_PT, WD_ALIGN_PARAGRAPH.RIGHT, italic=True)
        self.par()
        self.par("บทคัดย่อ", ABS_PT, bold=True)
        BR.thai_justify(self.par(m["abstract"], ABS_PT, first=INDENT))
        p = self.par(before=12, after=10, left=0.2)
        BR.set_run_font(p.add_run("คำสำคัญ – "), size=AFFIL_PT, bold=True)
        for k, word in enumerate(m["keywords"].split(";")):
            if k:
                BR.set_run_font(p.add_run("; "), size=AFFIL_PT, italic=True)
            BR.set_run_font(p.add_run(word.strip()), size=AFFIL_PT)
        self.section_break(1, 2)

    def heading(self, level, text):
        if text == "เอกสารอ้างอิง":
            self.par(text, REFS_PT, WD_ALIGN_PARAGRAPH.CENTER, bold=True, before=12, after=3, keep=True)
        elif level == 1:
            self.par(text, H1_PT, bold=True, before=12, after=3, keep=True)
        elif level == 2:
            self.par(text, H2_PT, bold=True, before=12, after=3, keep=True)
        else:
            self.par(text, BODY_PT, italic=True, before=6, after=3, keep=True)
        self.indent_next = False

    def para(self, text):
        BR.thai_justify(self.par(text, first=INDENT if self.indent_next else 0))
        self.indent_next = True

    def formula(self, text):
        expr, num = split_formula(text)
        line = BODY_PT * LINE
        p = self.par(before=line, after=line)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(COL_W / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(COL_W), WD_TAB_ALIGNMENT.RIGHT)
        BR.set_run_font(p.add_run("\t" + expr), size=BODY_PT, italic=True)
        BR.set_run_font(p.add_run(f"\t({num})" if num else ""), size=BODY_PT)
        self.indent_next = False                          # ย่อหน้าที่ต่อจากสมการ (เมื่อ ...) ไม่เยื้อง

    def caption(self, cap, word, align, before, after, keep):
        num, rest = BR.split_caption(cap, word)
        p = self.par(align=align, before=before, after=after, keep=keep)
        if num:
            BR.set_run_font(p.add_run(num + " "), size=CAP_PT, bold=True)
        self.runs(p, rest, CAP_PT)

    def figure(self, cap, path, width, span):
        if span:
            self.section_break(2, 1)
        p = self.par(align=WD_ALIGN_PARAGRAPH.CENTER, before=6, keep=True)
        for k, one in enumerate(path.split("|")):
            if k:
                BR.set_run_font(p.add_run("   "), size=BODY_PT)
            p.add_run().add_picture(str(BR.print_image(SRC / one, width)), width=Inches(width))
        self.caption(cap, "รูปที่", WD_ALIGN_PARAGRAPH.CENTER, 2, 6, False)
        if span:
            self.section_break(1, 2)

    def table(self, cap, rows, widths, aligns, span):
        if span:
            self.section_break(2, 1)
        self.caption(cap, "ตารางที่", WD_ALIGN_PARAGRAPH.CENTER if span else WD_ALIGN_PARAGRAPH.LEFT, 6, 3, True)
        ncol = max(len(r) for r in rows)
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        borders(t)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, row in enumerate(rows):
            for j in range(ncol):
                cell, txt = t.cell(i, j), row[j] if j < len(row) else ""
                cp = cell.paragraphs[0]
                center = BR.cell_center(i, j, txt, aligns)
                BR.para_format(cp, align=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT)
                self.runs(cp, txt, TABLE_PT, bold=(i == 0))
                cell.width = Inches(widths[j])
            if i == 0:
                BR.row_prop(t.rows[i], "w:tblHeader")
            BR.row_prop(t.rows[i], "w:cantSplit")
        self.par(size=6, after=6).paragraph_format.line_spacing = Pt(6)
        if span:
            self.section_break(1, 2)

    def page_break(self):
        # แบบเดียวกับที่ผู้จัดทำแทรกใน Word: ย่อหน้าที่มีตัวแบ่งหน้า เครื่องหมายย่อหน้าตกไปเป็นบรรทัดแรกของหน้าถัดไป
        p = self.par(after=8)
        r = p.add_run()
        BR.set_run_font(r, size=H1_PT, bold=True)
        r.add_break(WD_BREAK.PAGE)
        self.indent_next = False

    def bibliography(self):
        for k, key in enumerate(self.cite.order, 1):
            p = self.par(left=REF_HANG, first=-REF_HANG)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(REF_HANG))
            BR.set_run_font(p.add_run(f"[{k}]\t"), size=BODY_PT)
            self.runs(p, self.cite.refs[key], BODY_PT)

    def build(self, blocks, out):
        self.front()
        for b in blocks:
            k = b[0]
            if k == "chapter":
                self.heading(1, b[1])
            elif k == "h":
                self.heading(b[1], b[2])
            elif k == "para":
                self.para(b[1])
            elif k == "formula":
                self.formula(b[1])
            elif k == "fig":
                self.figure(b[1], b[2], b[3], b[4])
            elif k == "table":
                self.table(b[1], b[2], b[3], b[4], b[5])
            elif k == "bibliography":
                self.bibliography()
            elif k == "pagebreak":
                self.page_break()
            else:
                raise ValueError(f"ไม่รองรับบล็อก {k} ในรายงานฉบับย่อ")
        self.doc.save(out)
        return self.cite


# ---------------------------------------------------------------- พรีวิว HTML (หน้า A4 สองคอลัมน์แบบเดียวกับ Word เพื่อนับหน้า)
CSS = BR.CSS.split("@page")[0] + f"""
@page {{ size: {PAGE_W}in {PAGE_H}in; margin: {MARGIN[0]}in {MARGIN[1]}in {MARGIN[2]}in {MARGIN[3]}in; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "THS"; font-size: {BODY_PT}pt; line-height: {LINE}; color: #000; }}
p {{ margin: 0; }}
.title {{ text-align: center; font-weight: bold; font-size: {TITLE_PT}pt; margin-bottom: 12pt; }}
.author {{ text-align: right; font-weight: bold; font-size: {AUTHOR_PT}pt; }}
.affil {{ text-align: right; font-style: italic; font-size: {AFFIL_PT}pt; }}
.abs-h {{ font-weight: bold; font-size: {ABS_PT}pt; }}
.abs {{ font-size: {ABS_PT}pt; text-indent: {INDENT}in; text-align: justify; }}
.kw {{ font-size: {AFFIL_PT}pt; margin: 12pt 0 10pt 0.2in; }}
.cols {{ column-count: 2; column-gap: {GAP}in; }}
.span {{ column-span: all; }}
.h1 {{ font-weight: bold; font-size: {H1_PT}pt; margin: 12pt 0 3pt; break-after: avoid; }}
.h2 {{ font-weight: bold; font-size: {H2_PT}pt; margin: 12pt 0 3pt; break-after: avoid; }}
.h3 {{ font-style: italic; margin: 6pt 0 3pt; break-after: avoid; }}
.refs-h {{ font-weight: bold; font-size: {REFS_PT}pt; text-align: center; margin: 12pt 0 3pt; break-after: avoid; }}
.p {{ text-align: justify; }}
.ind {{ text-indent: {INDENT}in; }}
.formula {{ display: flex; margin: {BODY_PT * LINE}pt 0; }}
.formula .e {{ flex: 1; text-align: center; font-style: italic; }}
.fig {{ text-align: center; margin-top: 6pt; break-inside: avoid; }}
.fig img {{ margin: 0 0.05in; }}
.cap {{ text-align: center; font-size: {CAP_PT}pt; margin: 2pt 0 6pt; }}
.tcap {{ font-size: {CAP_PT}pt; margin: 6pt 0 3pt; break-after: avoid; }}
.tcap.c {{ text-align: center; }}
table {{ border-collapse: collapse; margin: 0 auto 6pt; font-size: {TABLE_PT}pt; line-height: {LINE}; table-layout: fixed; }}
td, th {{ border: 0.5pt solid #000; padding: 0 0.08in; vertical-align: top; }}
th {{ font-weight: bold; text-align: center; }}
tr {{ break-inside: avoid; }}
.c {{ text-align: center; }}
.pbreak {{ break-before: page; font-size: {H1_PT}pt; margin-bottom: 8pt; }}
.cols.pb {{ break-before: page; }}
.ref {{ margin-left: {REF_HANG}in; text-indent: -{REF_HANG}in; }}
.ref .n {{ display: inline-block; width: {REF_HANG}in; text-indent: 0; }}
"""


class ShortHtml:
    def __init__(self, meta, refs):
        self.meta, self.cite = meta, BR.Citations(refs)

    def esc(self, text):
        out = []
        for n, seg in enumerate(self.cite.sub(text).split("<br>")):
            out.append("<br>" if n else "")
            for t, b, i in tokens(seg):
                t = H.escape(t, quote=False)
                out.append(f"<b>{t}</b>" if b else f"<i>{t}</i>" if i else t)
        return "".join(out)

    def build(self, blocks):
        m = self.meta
        kw = "<i>; </i>".join(H.escape(w.strip()) for w in m["keywords"].split(";"))
        o = [f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
             f"<p class='title'>{self.esc(m['title'])}</p><p class='author'>{self.esc(m['authors'])}</p>",
             f"<p class='affil'>{self.esc(m['affil'])}</p><p class='affil'>{self.esc(m['email'])}</p><p>&nbsp;</p>",
             f"<p class='abs-h'>บทคัดย่อ</p><p class='abs'>{self.esc(m['abstract'])}</p>",
             f"<p class='kw'><b>คำสำคัญ – </b>{kw}</p>"]
        # Chrome ไม่ขึ้นหน้าใหม่เมื่อตัวแบ่งหน้าเป็นสิ่งแรกในกล่องสองคอลัมน์ จึงย้ายไปขึ้นหน้าที่ตัวกล่องแทน
        o.append("<div class='cols pb'>" if blocks and blocks[0][0] == "pagebreak" else "<div class='cols'>")
        ind = False
        for b in blocks:
            k = b[0]
            if k in ("chapter", "h"):
                level, text = (1, b[1]) if k == "chapter" else (b[1], b[2])
                cls = "refs-h" if text == "เอกสารอ้างอิง" else f"h{min(level, 3)}"
                o.append(f"<p class='{cls}'>{self.esc(text)}</p>")
                ind = False
            elif k == "para":
                o.append(f"<p class='p{' ind' if ind else ''}'>{self.esc(b[1])}</p>")
                ind = True
            elif k == "formula":
                expr, num = split_formula(b[1])
                o.append(f"<p class='formula'><span class='e'>{H.escape(expr)}</span><span>({num})</span></p>")
                ind = False
            elif k == "fig":
                cap, path, w, span = b[1], b[2], b[3], b[4]
                num, rest = BR.split_caption(cap, "รูปที่")
                imgs = "".join(f"<img src='{BR.print_image(SRC / one, w).as_uri()}' style='width:{w}in'>" for one in path.split("|"))
                o.append(f"<div class='fig{' span' if span else ''}'>{imgs}<p class='cap'><b>{num}</b> {self.esc(rest)}</p></div>")
            elif k == "table":
                cap, rows, widths, aligns, span = b[1:]
                num, rest = BR.split_caption(cap, "ตารางที่")
                html = [f"<div class='{'span' if span else ''}'><p class='tcap{' c' if span else ''}'><b>{num}</b> {self.esc(rest)}</p>"
                        f"<table style='width:{sum(widths)}in'><colgroup>" + "".join(f"<col style='width:{w}in'>" for w in widths) + "</colgroup>"]
                for i, row in enumerate(rows):
                    tag = "th" if i == 0 else "td"
                    html.append("<tr>" + "".join(f"<{tag}{' class=c' if BR.cell_center(i, j, c, aligns) else ''}>{self.esc(c)}</{tag}>"
                                                 for j, c in enumerate(row)) + "</tr>")
                o.append("".join(html) + "</table></div>")
            elif k == "pagebreak":
                o.append("<p class='pbreak' style='break-before:auto'>&nbsp;</p>" if b is blocks[0] else "<p class='pbreak'>&nbsp;</p>")
                ind = False
            elif k == "bibliography":
                o.append("".join(f"<p class='ref'><span class='n'>[{n}]</span>{self.esc(self.cite.refs[key])}</p>"
                                 for n, key in enumerate(self.cite.order, 1)))
        return "".join(o) + "</div></body></html>"


# ---------------------------------------------------------------- main
def guard_output(out, force):
    lock = [p for p in out.parent.glob("~$*") if out.name.endswith(p.name[2:])]
    if lock:
        sys.exit(f"หยุด: {out.name} เปิดอยู่ใน Word ({lock[0].name}) — ปิดไฟล์ก่อนแล้วรันใหม่")
    if not out.exists() or (STAMP.exists() and STAMP.read_text().strip() == BR.sha256(out)):
        return
    bak = SRC / "_backup"
    bak.mkdir(exist_ok=True)
    dst = bak / f"{out.stem}_{datetime.fromtimestamp(out.stat().st_mtime):%Y%m%d-%H%M%S}.docx"
    shutil.copy2(out, dst)
    if not force:
        sys.exit(f"หยุด: {out.name} ถูกแก้หลังสร้างครั้งล่าสุด (เช่น แก้ใน Word) — สำรองไว้ที่ {dst}\n"
                 "นำการแก้ไขเข้า docs/report_src/short/short.md ก่อน แล้วรันใหม่ด้วย --force")
    print(f"สำรองไฟล์เดิมที่ถูกแก้ไว้ที่ {dst}")


def main():
    sys.path.insert(0, str(SHORT))
    import refs_ieee as REF  # noqa: E402
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    out = Path(args[0]) if args else OUT
    prev = Path(args[1]) if len(args) > 1 else PREVIEW
    if out == OUT:                 # ไฟล์ทดลองที่ส่งพาธอื่นมาไม่ต้องตรวจ
        guard_output(out, force)
    prev.mkdir(parents=True, exist_ok=True)
    meta, blocks = read_source()
    BR.print_pdf(ShortHtml(meta, REF.REFS).build(blocks), prev / "short_preview.html", prev / "short_preview.pdf")
    import pymupdf
    pages = len(pymupdf.open(str(prev / "short_preview.pdf")))
    cite = ShortDocx(meta, REF.REFS).build(blocks, out)
    if out == OUT:
        STAMP.write_text(BR.sha256(out) + "\n")
    unused = sorted(set(REF.REFS) - set(cite.order))
    words = len(re.sub(r"<br>", " ", meta["abstract"]))
    print(f"saved {out} | หน้าพรีวิว {pages} (ไม่เกิน {MAX_PAGES}) | อ้างอิง {len(cite.order)} | ไม่ได้อ้าง {unused} | บทคัดย่อ {words} ตัวอักษร")
    if pages > MAX_PAGES:
        print(f"เตือน: ยาวเกิน {MAX_PAGES} หน้า")


if __name__ == "__main__":
    main()
