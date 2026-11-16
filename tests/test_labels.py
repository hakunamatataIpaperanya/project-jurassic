# Label construction: fossil-type keywords, Thai period text, genus parsing.
import re

import pandas as pd
import pytest

from conftest import fl, ml

PLACE_WORDS = r"(เขา|ถ้ำ|บ้าน|คอก|ศาล|ห้วย|ดอย|เกาะ|วัด|บ่อ|หนอง|ภู|พุง|ทุ่ง|ลำ|แม่น้ำ|อ่าง)"


@pytest.mark.parametrize("text,expect,absent", [
    ("สัตว์ไม่มีกระดูกสันหลังศาลเจ้าพ่อเขาประทับช้าง", set(), {"สัตว์เลี้ยงลูกด้วยนม"}),
    ("สัตว์ไม่มีกระดูกสันหลังถ้ำค้างคาว ไคร์นอยด์ ปะการัง", {"ไครนอยด์", "ปะการัง"}, {"สัตว์เลี้ยงลูกด้วยนม"}),
    ("หอยกาบคู่ สกุล ไพลคาโตยูนิโอ (กวางสีคอนชา)", {"หอยสองฝา"}, {"สัตว์เลี้ยงลูกด้วยนม"}),
    ("ช้างมาสโตดอน กระจงหมู หนู", {"สัตว์เลี้ยงลูกด้วยนม"}, set()),
    ("กระดูกและฟันของวัว - ควาย ช้าง กวางป่า", {"สัตว์เลี้ยงลูกด้วยนม"}, set()),
    ("ปลากินพืชพุทธบุตร", {"ปลา", "พืช"}, set()),          # classes_of ดิบจับทั้งสอง (specimen_classes ใช้ f_type ก่อนจึงไม่พลาด)
])
def test_classes_of(text, expect, absent):
    got = fl.classes_of(text)
    assert expect <= got and not (absent & got), got


def test_no_class_from_place_names(raw_sites):
    # คำของประเภทซากที่อยู่หลังคำบอกสถานที่ในชื่อแหล่ง (เช่น "เขาช้าง") ต้องไม่ทำให้ได้ประเภทนั้น ถ้าคำบรรยายซากไม่ได้กล่าวถึง
    bad = []
    for name, desc in zip(raw_sites.NAME_TH.fillna(""), raw_sites.FOS_DES_TH.fillna("")):
        from_name = fl.classes_of(name) - fl.classes_of(desc)
        for c in from_name:
            for k in fl.FOSSIL_CLASSES[c]:
                for m in re.finditer(re.escape(k), name):
                    before = name[max(0, m.start() - 6):m.start()]
                    if re.search(PLACE_WORDS + r"\S{0,3}$", before) and not before.endswith("หิน"):   # "หินปะการัง" = หินปูนแนวปะการัง (ถูก)
                        bad.append((name, c, k))
    assert not bad, bad[:5]


def test_specimen_classes_prefers_f_type():
    assert fl.specimen_classes("ปลากระดูกแข็ง", "ปลากินพืชพุทธบุตร") == {"ปลา"}


def test_every_recorded_period_parses(raw_sites):
    v = raw_sites.PERIODFROM.dropna().astype(str).str.strip()
    v = v[v.ne("")]
    failed = sorted({t for t in v if fl.parse_period_th(t)[0] is None})
    assert not failed, failed


@pytest.mark.parametrize("name,genus", [
    ("Phuwiangosaurus sirindhornae", "Phuwiangosaurus"), ("N/A (Dinosaur)", None), ("Sauropod", None),
    ("Ruminantia", None), ("Primates indet.", None), ("Crocodilia", None), ("Tetraconodontinae indet.", None),
    ("Spinomatinia sp.", "Spinomartinia"), ("cf. Stenocisma sp.", "Stenoscisma"),
    ("Stegosaurid Dinosaur", None), ("Pectinid sp.", None), ("Sinraptorid", None),        # ชื่อเรียกระดับวงศ์แบบไม่เป็นทางการ (-id)
    ("Minimocursor phunoiensis", "Minimocursor"),
])
def test_parse_taxon(name, genus):
    t = fl.parse_taxon(name)
    assert (t[0] if t else None) == genus


def test_parsed_genera_are_genera(spec):
    # ชื่อที่ parse เป็นสกุลต้องไม่ใช่ชื่อระดับสูงกว่า (ตรวจกับลำดับชั้นใน PBDB)
    tx = pd.read_csv(ml.DATA / "pbdb_thailand" / "taxa.csv", low_memory=False)
    occ = pd.read_csv(ml.DATA / "pbdb_thailand" / "occurrences.csv", low_memory=False)
    rank = {**occ.drop_duplicates("accepted_name").set_index("accepted_name").accepted_rank.to_dict(),
            **tx.drop_duplicates("taxon_name").set_index("taxon_name").taxon_rank.to_dict()}
    genera = {t[0] for t in spec.sci_name.map(fl.parse_taxon) if t}
    bad = {g: rank[g] for g in genera if rank.get(g) not in (None, "genus", "subgenus")}
    assert not bad, bad


def test_pbdb_class_mapping_coverage():
    # ทุก occurrence ที่อยู่ในกลุ่มที่โมเดลมีประเภทรองรับต้องแปลงได้ (กลุ่มที่ไม่มีประเภท เช่น ostracod, conodont, ฟองน้ำ ไม่นับ)
    occ = pd.read_csv(ml.DATA / "pbdb_thailand" / "occurrences.csv", low_memory=False)
    occ["cls"] = [fl.pbdb_class(a, b, c) for a, b, c in zip(occ.phylum, occ["class"], occ.order)]
    out_of_scope = occ["class"].isin(["Ostracoda", "Conodonta", "Insecta", "Amphibia", "Aves", "Tentaculita", "Stromatoporoidea",
                                      "Demospongiae", "Mammaliaformes", "Malacostraca", "Branchiopoda", "Hydrozoa", "Scaphopoda"]) \
        | occ.phylum.isin(["Porifera", "NO_PHYLUM_SPECIFIED", "Annelida", "Problematica"]) | occ.phylum.isna()
    miss = occ[~out_of_scope & occ.cls.isna()]
    assert len(miss) == 0, miss.groupby(["phylum", "class"], dropna=False).size().sort_values().tail(5).to_dict()


def test_notebook04_parse_taxon_matches_feature_lib(spec):
    # notebook 04 มีสำเนา parse_taxon ของตัวเอง (notebook แบบ self-contained) ต้องให้ผลเหมือน feature_lib ทุกชื่อ
    import re as _re
    src = (ml.ROOT / "scripts" / "build_taxon_fill_nb.py").read_text(encoding="utf-8")
    start = src.index("# ชื่อไม่เป็นทางการ/กลุ่มกว้างที่ไม่ใช่ชื่อสกุล")
    end = src.index("def latin_in_text")
    ns = {"re": _re}
    exec(src[start:end], ns)
    occ = pd.read_csv(ml.DATA / "pbdb_thailand" / "occurrences.csv", low_memory=False)
    names = set(spec.sci_name.dropna()) | set(occ.identified_name.dropna()) | set(occ.accepted_name.dropna())
    diff = [n for n in names if (ns["parse_taxon"](n) or (None,))[0] != (fl.parse_taxon(n) or (None,))[0]]
    assert not diff, diff[:10]


def test_no_informal_family_names_as_genera(spec):
    # ชื่อลงท้าย -id (Stegosaurid, Pectinid, Productid ...) เป็นชื่อเรียกระดับวงศ์ ห้ามเป็นป้ายสกุล
    genera = {t[0] for t in spec.sci_name.map(fl.parse_taxon) if t}
    assert not {g for g in genera if g.endswith("id")}
