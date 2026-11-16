# Task 1 — "ซากที่พบหน้าตาเป็นอย่างไร" (หน้า IDENTIFY ของเว็บ): คนทั่วไปมักไม่รู้ประเภทซาก (f_type) ซึ่งเป็น input หลักของโมเดลสกุล
#   APPEARANCE  การ์ดรูป -> คำถามต่อ (เฉพาะที่กำกวม) -> "ใบ" = ประเภทที่เข้ากับคำตอบ (ความรู้ทางบรรพชีวินวิทยา) + คำค้นใน f_part
#   appearance_type_weights  น้ำหนักแต่ละประเภท = จำนวนตัวอย่างในทะเบียนที่เป็นประเภทนั้นและ f_part ตรงคำค้น (+0.5)
#                            แล้วตัดหาง (สะสม 97% ไม่เกิน 5 ประเภท) -> โมเดลสกุลรันทีละประเภทแล้วรวมผลตามน้ำหนัก
#   appearance_part          f_part ที่ทะเบียนเขียนบ่อยสุดของประเภทนั้น (ตรงคำค้น) — ข้อความเข้าโมเดลหน้าตาเหมือนตอน train
#   appearance_truth         คำตอบที่ "ถูก" ของตัวอย่างในทะเบียน (ใช้จำลองผู้ใช้ตอนวัดผลใน notebook 15)
# คำตอบคำถามต่อต้องไปถึงโมเดล: ตัวเลือกที่ได้ชุดประเภทเดียวกัน (กระดูกแต่ละส่วน; ขดแบน = แอมโมไนต์ / แท่งมีห้อง = นอติลอยด์ เป็นเซฟาโลพอด
# ทั้งคู่) มีคำบรรยายส่วนของซากของตัวเอง (part_text ตามคำที่ทะเบียนใช้); ตัวเลือกที่ประเภทต่างกันอยู่แล้วใช้คำกลางของการ์ด
# (notebook 15 หัวข้อ 7: คำเจาะจงในกรณีนั้นทำให้แหล่งใหม่แย่ลง เช่น เปลือกสองฝา)
# ทุกฟังก์ชันรับตารางตัวอย่าง (f_type, f_part, f_group) เป็นพารามิเตอร์: เว็บใช้ทะเบียนทั้งหมด, notebook 15 ใช้เฉพาะแถว train ของแต่ละ fold
VERTEBRATE = "@Vertebrate"          # ทุกประเภทที่กลุ่มซาก (f_group ที่พบบ่อยสุด) เป็น Vertebrate
ALL_TYPES = "*"

APPEARANCE = [
    {"id": "tooth", "group": "ส่วนของสัตว์มีกระดูกสันหลัง", "label": "ฟัน", "hint": "มีเคลือบฟันเป็นมัน", "icon": "tooth",
     "ask": "ฟันมีรูปร่างแบบไหน", "part": "ฟัน|เขี้ยว", "part_text": "ฟัน",
     "options": [
         {"id": "blade", "label": "แบนคม รูปสามเหลี่ยมหรือใบมีด", "hint": "ขอบอาจหยักเป็นฟันเลื่อย",
          "types": ["ปลากระดูกอ่อน", "ฉลาม", "ไดโนเสาร์", "สัตว์เลื้อยคลาน"]},
         {"id": "cone", "label": "เป็นกรวยแหลม หน้าตัดกลม", "hint": "คล้ายฟันจระเข้หรือเขี้ยว",
          "types": ["จระเข้", "ไดโนเสาร์", "ซอโรพอด", "ปลากระดูกแข็ง", "ปลา", "สัตว์เลื้อยคลาน", "ไฟโตซอร์", "กิ้งก่าและงู", "สัตว์กินเนื้อ"]},
         {"id": "molar", "label": "ฟันกราม ผิวบนมีปุ่มหรือสันไว้บด", "hint": "แบบฟันวัว หมู ลิง",
          "types": ["สัตว์กีบคู่", "ไพรเมต", "สัตว์กินเนื้อ", "สัตว์ฟันแทะ", "สัตว์กีบคี่", "สัตว์งวง", "โปรบอสซิเดีย", "สัตว์เลี้ยงลูกด้วยนม"]},
         {"id": "crusher", "label": "แบนเตี้ย ผิวเรียบหรือย่น", "hint": "มักเล็ก ใช้บดเปลือกหอย",
          "types": ["ปลากระดูกอ่อน", "ปลากระดูกแข็ง", "ปลา"]},
         {"id": "unsure", "label": "ไม่แน่ใจ", "types": ALL_TYPES},
     ]},
    {"id": "bone", "group": "ส่วนของสัตว์มีกระดูกสันหลัง", "label": "กระดูก", "hint": "เนื้อเป็นรูพรุนเมื่อแตก", "icon": "bone",
     "ask": "เป็นกระดูกส่วนไหน", "types": VERTEBRATE, "part_text": "เศษกระดูก",
     "options": [
         {"id": "vertebra", "label": "กระดูกสันหลังหรือกระดูกหาง", "hint": "เป็นปล้อง มีปุ่มหรือปีกยื่น", "part": "สันหลัง|หาง|เชฟรอน",
          "part_text": "กระดูกสันหลัง"},
         {"id": "limb", "label": "กระดูกขา แขน นิ้ว หรือข้อต่อ", "hint": "ยาว ปลายบาน", "part": "ขา|แขน|นิ้ว|ฝ่า|ข้อ|เท้า|เชิงกราน|สะบัก",
          "part_text": "กระดูกขา"},
         {"id": "rib", "label": "ซี่โครง", "hint": "ยาวโค้ง แบน", "part": "ซี่โครง", "part_text": "กระดูกซี่โครง"},
         {"id": "skull", "label": "ขากรรไกรหรือกะโหลก", "hint": "อาจมีเบ้าฟัน", "part": "ขากรรไกร|กะโหลก|กระโหลก|หัว|ท้ายทอย|กราม",
          "part_text": "กระดูกขากรรไกร"},
         {"id": "frag", "label": "เศษกระดูกชิ้นเล็ก", "hint": "บอกไม่ได้ว่าส่วนไหน", "part": "เศษ"},
         {"id": "unsure", "label": "ไม่แน่ใจ", "part": "กระดูก", "part_text": "nan"},
     ]},
    {"id": "plate", "group": "ส่วนของสัตว์มีกระดูกสันหลัง", "label": "แผ่นกระดอง", "hint": "แผ่นแข็งมีลายหรือหลุม", "icon": "turtle",
     "types": ["เต่า", "เทสทูดีเนต", "จระเข้"], "part": "กระดอง|แผ่น|ผิวหนัง", "part_text": "แผ่นกระดอง"},
    {"id": "fish", "group": "ส่วนของสัตว์มีกระดูกสันหลัง", "label": "ปลา", "hint": "เกล็ด รอยตัวปลา หรือกระดูกปลา", "icon": "fish",
     "ask": "เห็นส่วนไหนของปลา", "types": ["ปลากระดูกแข็ง", "ปลา", "ปลากระดูกอ่อน"], "part": "เกล็ด|ปลา|ครีบ|กะโหลก|กระโหลก", "part_text": "nan",
     "options": [
         {"id": "scales", "label": "เกล็ดเป็นมัน", "hint": "รูปข้าวหลามตัดหรือกลม", "part": "เกล็ด", "part_text": "เกล็ดปลา"},
         {"id": "body", "label": "รอยตัวปลาหรือโครงกระดูกปลา", "hint": "เห็นรูปร่างปลา", "part": "ตัว|โครงกระดูก|สมบูรณ์|ลำตัว|ครีบ", "part_text": "ตัวปลา"},
         {"id": "head", "label": "กระดูกหัวหรือขากรรไกรปลา", "hint": "", "part": "กะโหลก|กระโหลก|หัว|ขากรรไกร|กราม", "part_text": "กระดูกกะโหลก"},
         {"id": "unsure", "label": "ไม่แน่ใจ"},
     ]},

    {"id": "twoshell", "group": "เปลือกและโครงแข็ง", "label": "เปลือกสองฝา", "hint": "หรือฝาเดียวที่โค้งคล้ายหอย", "icon": "bivalve",
     "ask": "ดูฝาใดฝาหนึ่งจากด้านบน", "part": "เปลือก|ฝา|รอยพิมพ์", "part_text": "เปลือก",
     "options": [
         {"id": "sym", "label": "ซ้าย-ขวาเหมือนกัน จะงอยอยู่ตรงกลาง", "hint": "มักมีซี่แผ่จากจะงอย", "icon": "brachiopod", "types": ["แบรคิโอพอด"]},
         {"id": "asym", "label": "จะงอยเอียงไปข้างหนึ่ง", "hint": "ซ้าย-ขวาไม่เท่ากัน", "icon": "bivalve", "types": ["หอยกาบคู่/หอยสองฝา", "หอยกาบคู่"]},
         {"id": "unsure", "label": "ไม่แน่ใจ", "types": ["แบรคิโอพอด", "หอยกาบคู่/หอยสองฝา", "หอยกาบคู่", "ครัสตาเชียน"]},
     ]},
    {"id": "coil", "group": "เปลือกและโครงแข็ง", "label": "เปลือกขดเป็นวง", "hint": "", "icon": "ammonoid",
     "ask": "ขดแบบไหน", "part": "เปลือก|หอย|แอมโม", "part_text": "เปลือก",
     "options": [
         {"id": "flat", "label": "ขดแบนในระนาบเดียว", "hint": "คล้ายเหรียญหรือเขาแกะ", "icon": "ammonoid", "types": ["เซฟาโลพอด"], "part": "แอมโม",
          "part_text": "เปลือกหอยแอมโมนอยด์"},
         {"id": "spire", "label": "ขดเป็นเกลียว ยอดแหลม", "hint": "คล้ายหอยขม", "icon": "gastropod", "types": ["หอยกาบเดี่ยว/หอยฝาเดียว"], "part": "เปลือก|หอย"},
         {"id": "unsure", "label": "ไม่แน่ใจ", "types": ["เซฟาโลพอด", "หอยกาบเดี่ยว/หอยฝาเดียว"]},
     ]},
    {"id": "cone", "group": "เปลือกและโครงแข็ง", "label": "แท่งหรือกรวยยาว", "hint": "", "icon": "orthocone",
     "ask": "ลักษณะแบบไหน", "part": "เปลือก|หอย|นอติลอยด์|Nautiloid|โครงร่าง", "part_text": "เปลือก",
     "options": [
         {"id": "chambers", "label": "มีผนังกั้นเป็นห้อง ๆ ตามยาว", "hint": "เห็นเมื่อแตกหรือขัด", "icon": "orthocone", "types": ["เซฟาโลพอด"], "part": "นอติลอยด์|Nautiloid|โครงร่าง",
          "part_text": "โครงร่างแข็งของนอติลอยด์"},
         {"id": "rings", "label": "กรวยเล็กแหลม มีวงแหวนรอบ", "hint": "ยาวไม่กี่เซนติเมตร", "types": ["เทนทาคิวไลต์"]},
         {"id": "spiral", "label": "เป็นเกลียวยาวคล้ายเจดีย์", "hint": "", "icon": "gastropod", "types": ["หอยกาบเดี่ยว/หอยฝาเดียว"], "part": "เปลือก|หอย"},
         {"id": "unsure", "label": "ไม่แน่ใจ", "types": ["เซฟาโลพอด", "เทนทาคิวไลต์", "หอยกาบเดี่ยว/หอยฝาเดียว"]},
     ]},
    {"id": "trilobite", "group": "เปลือกและโครงแข็ง", "label": "คล้ายแมลง มีสามพู", "hint": "แบ่งสามส่วนตามยาว มีปล้อง", "icon": "trilobite",
     "types": ["ไทรโลไบต์"], "part_text": "ไทรโลไบต์"},
    {"id": "crab", "group": "เปลือกและโครงแข็ง", "label": "กระดองปู / ขาเป็นข้อ", "hint": "", "icon": "crab",
     "types": ["ครัสตาเชียน", "เฮกซาพอด"], "part_text": "กระดองปู"},
    {"id": "grain", "group": "เปลือกและโครงแข็ง", "label": "เม็ดรูปกระสวย", "hint": "คล้ายเมล็ดข้าว ยาวราว 0.5–1 ซม.", "icon": "fusulinid",
     "types": ["ฟอแรมมินิเฟอร์", "ฟอแรมินิเฟอร์", "ฟอแรมินิเฟอรา", "ฟอแรมมินิเฟอ", "เรดิโอลาเรีย"], "part_text": "ฟิวซูลินิด"},
    {"id": "coral", "group": "เปลือกและโครงแข็ง", "label": "ก้อนมีช่องเรียงกัน", "hint": "รังผึ้ง รูพรุน หรือตาข่าย", "icon": "coral",
     "ask": "ช่องเป็นแบบไหน", "part_text": "โครงสร้างแข็ง",
     "options": [
         {"id": "cells", "label": "ช่องเห็นด้วยตา คล้ายรังผึ้ง หรือแท่งเรียงกัน", "hint": "", "icon": "coral", "types": ["แอนโธซัว", "เทบูเลต้า", "ไนดาเรีย"]},
         {"id": "mesh", "label": "แผ่นตาข่ายละเอียดคล้ายผ้าลูกไม้", "hint": "", "icon": "bryozoan", "types": ["ไบรโอซัว"]},
         {"id": "sponge", "label": "ก้อนพรุนคล้ายฟองน้ำ ไม่เป็นระเบียบ", "hint": "", "icon": "sponge", "types": ["พอริเฟอรา", "ฟองน้ำ"]},
         {"id": "unsure", "label": "ไม่แน่ใจ", "types": ["แอนโธซัว", "เทบูเลต้า", "ไนดาเรีย", "ไบรโอซัว", "พอริเฟอรา", "ฟองน้ำ"]},
     ]},
    {"id": "stem", "group": "เปลือกและโครงแข็ง", "label": "แท่งกลมเป็นข้อ ๆ", "hint": "คล้ายเหรียญซ้อนกัน", "icon": "crinoid",
     "types": ["เอไคโนเดิร์ม", "ไครนอยด์"], "part_text": "ก้านไครนอยด์"},
    {"id": "graptolite", "group": "เปลือกและโครงแข็ง", "label": "ลายเส้นคล้ายใบเลื่อย", "hint": "เส้นเล็กบนหินดินดาน", "icon": "graptolite",
     "types": ["แกรปโตไลต์"], "part_text": "แกรปโตไลต์"},

    {"id": "leaf", "group": "พืช", "label": "ใบไม้", "hint": "หรือรอยพิมพ์ใบ", "icon": "leaf",
     "types": ["เศษพืช", "ใบไม้", "พืชกลุ่มเฟิร์น", "เฟิร์น", "สกุล เฟรเนลอปซิส เอ", "สกุล เฟรเนลอปซิส บี", "สาหร่าย"],
     "part": "ใบ|เฟิร์น|พืช", "part_text": "ใบไม้"},
    {"id": "wood", "group": "พืช", "label": "เนื้อไม้", "hint": "เห็นลายเสี้ยนหรือวงปี", "icon": "wood",
     "types": ["ไม้กลายเป็นหิน", "เศษพืช"], "part": "ไม้", "part_text": "ไม้กลายเป็นหิน"},

    {"id": "track", "group": "รอยที่สัตว์ทิ้งไว้", "label": "รอยเท้า", "hint": "เป็นหลุมรูปนิ้ว", "icon": "footprint",
     "types": ["รอยตีน"], "part_text": "รอยตีน"},
    {"id": "burrow", "group": "รอยที่สัตว์ทิ้งไว้", "label": "รอยชอนไช / ก้อนคล้ายมูล", "hint": "ท่อ รู หรือรอยคดเคี้ยวในหิน", "icon": "slab",
     "types": ["รอยชอนไช", "รอยเจาะและชอนไช", "มูลสัตว์", "มวลสารพอก"], "part_text": "รอยชอนไช"},

    {"id": "unknown", "group": "ไม่แน่ใจ", "label": "ไม่ตรงกับแบบไหนเลย", "hint": "ระบบจะเดาจากสถิติทั้งหมด (แม่นน้อยลง)", "icon": "slab",
     "types": ALL_TYPES, "part_text": "nan"},
]


_LEAVES = {}


def appearance_leaves():
    # {leaf_id: {"card", "option", "types", "part", "part_text", "label"}}: ใบ = การ์ดที่ไม่มีคำถามต่อ หรือ (การ์ด, ตัวเลือก)
    if _LEAVES:
        return _LEAVES
    out = _LEAVES
    for c in APPEARANCE:
        base = {"card": c["id"], "types": c.get("types", ALL_TYPES), "part": c.get("part"), "part_text": c.get("part_text", "nan")}
        if not c.get("options"):
            out[c["id"]] = {**base, "option": None, "label": c["label"]}
            continue
        for o in c["options"]:
            out[f"{c['id']}/{o['id']}"] = {**base, "option": o["id"], "label": f"{c['label']}: {o['label']}",
                                          **{k: o[k] for k in ("types", "part", "part_text") if k in o}}
    return out


def appearance_leaf_id(card, option=None):
    c = next((c for c in APPEARANCE if c["id"] == card), None)
    if c is None:
        raise KeyError(card)
    if not c.get("options"):
        return card
    opt = option or "unsure"
    if opt not in {o["id"] for o in c["options"]}:
        raise KeyError(f"{card}/{opt}")
    return f"{card}/{opt}"


def _type_set(types, spec):
    if types == ALL_TYPES:
        return sorted(spec.f_type.dropna().unique())
    if types == VERTEBRATE:
        g = spec.dropna(subset=["f_type", "f_group"]).groupby("f_type").f_group.agg(lambda s: s.value_counts().idxmax())
        return sorted(g[g.eq("Vertebrate")].index)
    return list(types)


def appearance_leaf_types(spec):
    # ประเภทของทุกใบ (คำนวณครั้งเดียวต่อ spec แล้วส่งต่อให้ฟังก์ชันด้านล่าง)
    return {lid: _type_set(leaf["types"], spec) for lid, leaf in appearance_leaves().items()}


def appearance_type_weights(leaf_id, spec, top=5, mass=0.97, prior=0.5, leaf_types=None):
    # spec: ตาราง f_type / f_part / f_group -> Series น้ำหนักประเภท (รวม = 1) เรียงจากมากไปน้อย
    leaf = appearance_leaves()[leaf_id]
    types = (leaf_types or {}).get(leaf_id) or _type_set(leaf["types"], spec)
    rows = spec[spec.f_type.isin(types)]
    if leaf["part"]:
        rows = rows[rows.f_part.fillna("").astype(str).str.contains(leaf["part"], regex=True)]
    w = pd.Series(prior, index=types, dtype=float).add(rows.f_type.value_counts(), fill_value=0)
    w = w.sort_index().sort_values(ascending=False, kind="stable")          # เท่ากัน -> เรียงตามชื่อ (ผลเหมือนกันทุกเครื่อง)
    w = w / w.sum()
    keep = max(1, min(top, int((w.cumsum() < mass).sum()) + 1))
    w = w.iloc[:keep]
    return w / w.sum()


APPEARANCE_PART_POLICY = "generic"   # วิธีเติม "ส่วนของซาก" ที่ใช้จริง — เลือกด้วย CV ใน notebook 15 หัวข้อ 7 (เทสต์ตรวจว่าตรงกับผลที่ดีสุด)


def appearance_part(f_type, leaf_id, spec, policy=None):
    # ส่วนของซากที่ป้อนโมเดลเมื่อผู้ใช้ไม่ได้พิมพ์เอง:
    #   mode    = คำที่ทะเบียนเขียนบ่อยสุดของประเภทนี้ (ตรงคำค้นของคำตอบก่อน)   generic = คำกลางของการ์ด (เช่น "ฟัน")   blank = ไม่ใส่
    policy = policy or APPEARANCE_PART_POLICY
    leaf = appearance_leaves()[leaf_id]
    if policy == "blank":
        return None
    if policy == "generic":
        return None if leaf["part_text"] == "nan" else leaf["part_text"]
    rows = spec[spec.f_type.eq(f_type)]
    part = rows.f_part.dropna().astype(str)
    if leaf["part"]:
        hit = part[part.str.contains(leaf["part"], regex=True)]
        part = hit if len(hit) else part
    return part.value_counts().idxmax() if len(part) else leaf["part_text"]


_PART_N = {}


def _part_count(spec, f_type, pattern):
    # จำนวนตัวอย่างของประเภทนี้ที่ส่วนของซากตรงคำค้น — cache ผูกกับตัว spec จริง (เก็บ reference ไว้ด้วย
    # จึงไม่มีตารางอื่นได้ id ซ้ำมาใช้ค่าผิด เช่นตาราง fold ก่อนหน้าใน notebook 15)
    entry = _PART_N.get(id(spec))
    if entry is None or entry[0] is not spec:
        entry = _PART_N[id(spec)] = (spec, {})
    counts = entry[1]
    if (f_type, pattern) not in counts:
        counts[(f_type, pattern)] = int(spec[spec.f_type.eq(f_type)].f_part.fillna("").astype(str).str.contains(pattern, regex=True).sum())
    return counts[(f_type, pattern)]


def appearance_truth(f_type, f_part, spec, leaf_types=None):
    # คำตอบของผู้ใช้ที่ "มองซากถูก" (จำลองตอนวัดผล) -> (ใบที่ตอบครบ, ใบที่ตอบแค่การ์ดแรกแล้ว "ไม่แน่ใจ")
    #   1) ใบที่มีประเภทจริงและคำค้นตรง f_part  2) ใบที่ไม่ต้องใช้คำค้น  3) f_part ว่าง / ไม่ตรง: การ์ดที่ประเภทนี้มีตัวอย่างตรงคำค้นมากสุด
    #      ในทะเบียน แล้วตอบคำถามต่อว่า "ไม่แน่ใจ" (ทะเบียนไม่ได้บอกว่าเห็นอะไร จึงไม่สมมติว่าผู้ใช้ตอบเจาะจงได้)
    #   เสมอกัน: ตัวเลือกที่ไม่ใช่ "ไม่แน่ใจ" ก่อน แล้วใบที่เจาะจงสุด (ประเภทน้อยสุด)
    part = "" if f_part is None or (isinstance(f_part, float) and np.isnan(f_part)) else str(f_part)
    lt = leaf_types or appearance_leaf_types(spec)
    best, best_key = None, None
    for lid, leaf in appearance_leaves().items():
        if leaf["card"] == "unknown" or f_type not in lt[lid]:
            continue
        if leaf["part"] is None:
            rank, score = 1, 0
        elif re.search(leaf["part"], part):
            rank, score = 0, 0
        else:
            rank, score = 2, _part_count(spec, f_type, leaf["part"])
            if score == 0:
                continue
        key = (rank, -score, leaf["option"] == "unsure", len(lt[lid]))
        if best_key is None or key < best_key:
            best, best_key = lid, key
    if best is None:
        return "unknown", "unknown"
    card_only = appearance_leaf_id(appearance_leaves()[best]["card"])
    return (card_only if best_key[0] == 2 else best), card_only


# ชื่อกลุ่มที่แสดงผู้ใช้ (ชื่อคลาสของตัวกรองบางชื่อแคบเกิน: เอไคโนเดิร์มทั้งหมดอยู่คลาส "ไครนอยด์")
_GROUP_NAME = {"สัตว์เลื้อยคลานอื่น": "สัตว์เลื้อยคลาน", "ฟอแรม/ฟิวซูลินิด": "ฟอแรม / ฟิวซูลินิด", "ไครนอยด์": "เอไคโนเดิร์ม"}


def class_group(cls):
    # ชื่อกลุ่มที่แสดงผู้ใช้ของคลาสประเภทซาก (คลาสของตัวกรอง / fl.pbdb_class) — ชุดชื่อเดียวกับ other_group
    return _GROUP_NAME.get(cls, cls)


def other_group(f_type):
    # กลุ่มของ "อื่น ๆ" (สกุลพบน้อย) = กลุ่มซากเดียวกับตัวกรองความสอดคล้อง (classes_of) — ประเภทที่สะกดต่างแต่เป็นกลุ่มเดียวกัน
    # (ไดโนเสาร์ / ซอโรพอด, ฟอแรมมินิเฟอร์ / ฟอแรมินิเฟอรา ...) ได้กลุ่มเดียวกัน; ประเภทที่ไม่เข้ากลุ่มใดใช้ชื่อประเภทเอง
    if not f_type or (isinstance(f_type, float) and np.isnan(f_type)):
        return None
    c = sorted(specimen_classes(f_type, f_type))
    return " / ".join(class_group(x) for x in c) if c else ("ประเภทอื่น ๆ" if f_type == "อื่น ๆ" else str(f_type))
