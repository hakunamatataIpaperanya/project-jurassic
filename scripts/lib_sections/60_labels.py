# Label vocabularies shared by notebooks 10-12: fossil-type groups (Thai keywords + PBDB taxonomy),
# genus parsing of scientific names, and Thai period text -> standard period.
FOSSIL_CLASSES = {
    "ไดโนเสาร์": ["ไดโนเสาร์", "ซอโรพอด", "เทอโรพอด", "ออร์นิโทพอด", "สเตโกซอรัส", "ซิตตะโกซอรัส"],
    "สัตว์เลื้อยคลานอื่น": ["จระเข้", "เต่า", "ไฟโตซอร์", "กิ้งก่า", "ตะกวด", "เหี้ย", "สัตว์เลื้อยคลาน"],
    "ปลา": ["ปลา", "ฉลาม"],
    "สัตว์เลี้ยงลูกด้วยนม": ["สัตว์กีบคู่", "สัตว์กีบคี่", "สัตว์งวง", "สัตว์เลี้ยงลูกด้วยนม", "ไพรเมต", "สัตว์กินเนื้อ", "ช้าง", "แรด",
                             "วาฬ", "ลิง", "กวาง", "ไฮยีนา", "สัตว์ฟันแทะ", "ค้างคาว", "หมู", "วัว", "สมเสร็จ"],
    "แบรคิโอพอด": ["แบรคิโอพอด", "บราคิโอพอด"],
    "หอยสองฝา": ["หอยกาบคู่", "หอยสองฝา"],
    "หอยฝาเดียว": ["หอยกาบเดี่ยว", "หอยฝาเดียว", "หอยขม", "หอยโข่ง"],
    "เซฟาโลพอด": ["เซฟาโลพอด", "แอมโมไนต์", "แอมโมนอยด์", "นอติลอยด์"],
    "ปะการัง": ["ปะการัง", "ประการัง", "แอนโธซัว", "เทบูเลต"],
    "ไครนอยด์": ["ไครนอยด์", "ไคร์นอยด์", "พลับพลึงทะเล", "เอไคโนเดิร์ม"],
    "ไบรโอซัว": ["ไบรโอซัว"],
    "ไทรโลไบต์": ["ไทรโลไบต์"],
    "แกรปโตไลต์": ["แกรปโตไลต์"],
    "ฟอแรม/ฟิวซูลินิด": ["ฟิวซูลินิด", "ฟอแรม", "คตข้าวสาร"],
    "เรดิโอลาเรีย": ["เรดิโอลาเรีย"],
    "พืช": ["พืช", "ไม้กลายเป็นหิน", "ใบไม้", "สาหร่าย", "สโตรมาโตไลต์", "เฟิร์น"],
    "ร่องรอย": ["รอยตีน", "ร่องรอย", "รอยชอนไช", "มูล", "โคโพรไลต์"],
}


# ชื่อสถานที่ / ชื่อสกุลทับศัพท์ที่มีคำของประเภทซากปนอยู่ — ตัดออกก่อนจับคำ (พบตอนตรวจทาน: 14 แหล่งถูกติด "สัตว์เลี้ยงลูกด้วยนม" ผิด)
# เช่น เขาประทับช้าง, ถ้ำพุงช้าง, ถ้ำค้างคาว, เขาลิงต่าง, "กวางสีคอนชา" (Guangxiconcha = หอยสองฝา)
NOT_FOSSIL_RE = re.compile(r"เขาประทับช้าง|คอกช้าง|พุงช้าง|ช้างเล่น|เขาช้าง|ถ้ำกวางทอง|กวางสี|เขาลิง|ถ้ำค้างคาว|คอกวัว|ภูเต่า|เขาปลาร้า")


def classes_of(text):
    if not isinstance(text, str):
        return set()
    text = NOT_FOSSIL_RE.sub(" ", text)
    return {c for c, kws in FOSSIL_CLASSES.items() if any(k in text for k in kws)}


def specimen_classes(f_type, com_name):
    # f_type first; com_name only when f_type matches nothing (avoids 'ปลากินพืช' -> plant, 'ลิงใบไม้' -> plant)
    return classes_of(f_type) or classes_of(com_name)


def pbdb_class(phylum, cls, order):
    ph, cl, od = str(phylum), str(cls), str(order)
    if cl in ("Saurischia", "Ornithischia") or od in ("Saurischia", "Ornithischia"):
        return "ไดโนเสาร์"
    if cl == "Reptilia" or od in ("Testudines", "Crocodylia", "Phytosauria", "Squamata"):
        return "สัตว์เลื้อยคลานอื่น"
    if cl in ("Osteichthyes", "Actinopterygii", "Chondrichthyes", "Sarcopterygii", "Elasmobranchii"):
        return "ปลา"
    if cl == "Mammalia":
        return "สัตว์เลี้ยงลูกด้วยนม"
    if ph == "Brachiopoda":
        return "แบรคิโอพอด"
    if cl == "Bivalvia":
        return "หอยสองฝา"
    if cl == "Gastropoda":
        return "หอยฝาเดียว"
    if cl == "Cephalopoda":
        return "เซฟาโลพอด"
    if cl == "Anthozoa":
        return "ปะการัง"
    if ph == "Echinodermata" or cl == "Crinoidea":
        return "ไครนอยด์"
    if ph == "Bryozoa":
        return "ไบรโอซัว"
    if cl == "Trilobita":
        return "ไทรโลไบต์"
    if cl == "Graptolithina" or ph == "Hemichordata":
        return "แกรปโตไลต์"
    if ph in ("Foraminifera", "Retaria") or cl == "Fusulinata":
        return "ฟอแรม/ฟิวซูลินิด"
    if ph in ("Radiolaria", "Radiozoa"):
        return "เรดิโอลาเรีย"
    if ph in ("Tracheophyta", "Spermatophyta", "Pteridophyta", "Coniferophyta", "Ginkgophyta", "Cycadophyta", "Chlorophyta", "Rhodophyta",
              "Equisetophyta", "Embryophyta", "Lycopodiophyta", "Pinophyta", "Bryophyta", "Marchantiophyta", "Cyanobacteria",
              "Angiospermae", "Gymnospermae"):
        return "พืช"                       # Cyanobacteria = ผู้สร้างสโตรมาโตไลต์ (อยู่ในกลุ่มพืชตามคำสำคัญ)
    return None


INFORMAL_NAMES = {"Sauropod", "Theropod", "Dinosaur", "Shark", "Turtle", "Coral", "Coprolite", "Phytosaur", "Ooids", "Trilobites",
                  "Prosauropod", "Sinraptorid", "Rhychonellacean", "Fusulinid", "Crocodile", "Brachiopod", "Ammonite", "Bivalve",
                  "Gastropod", "Productid", "Rugosa", "Tabulata", "Plant"}


# ลงท้าย -id (เช่น Stegosaurid, Pectinid) = ชื่อเรียกระดับวงศ์แบบไม่เป็นทางการ ไม่ใช่สกุล — ทะเบียน / PBDB ไม่มีสกุลจริงที่ลงท้ายแบบนี้
# ชื่อระดับสูงกว่าสกุลที่รูปคำเหมือนสกุล (ตรวจกับลำดับชั้นใน PBDB ตอนตรวจทาน) และชื่อสกุลที่สะกดผิดในทะเบียน
HIGHER_TAXA = {"Ruminantia", "Primates", "Rodentia", "Crocodilia", "Cnidaria", "Sauropoda", "Proboscidea", "Actinopterygii",
               "Bryozoa", "Trionychia", "Testudines", "Theropoda", "Ornithopoda", "Mammalia", "Reptilia", "Brachiopoda", "Mollusca"}
GENUS_FIX = {"Spinomatinia": "Spinomartinia", "Cetopangasis": "Cetopangasius", "Stenocisma": "Stenoscisma", "Orthoticia": "Orthotichia",
             "Phricodothynis": "Phricodothyris", "Neoschwaqerina": "Neoschwagerina", "Squemularia": "Squamularia",
             "Steinmanias": "Steinmannia", "Pseudofusulinid": "Pseudofusulina"}


def parse_taxon(name):
    # (genus, "Genus species" or None) for a real genus/species name, else None (N/A, informal or higher-rank names)
    if not isinstance(name, str) or name.lstrip("์").startswith("N/A"):
        return None
    toks = [t for t in re.sub(r"[\"'()\[\]?,]", " ", name).split() if t.lower() not in {"cf.", "cf", "aff.", "aff", "n.", "nov.", "gen.", "?"}]
    if not toks:
        return None
    g = GENUS_FIX.get(toks[0], toks[0])
    if not re.fullmatch(r"[A-Z][a-z]{2,}", g) or g in INFORMAL_NAMES or g in HIGHER_TAXA or re.search(r"(idae|inae|oidea|formes|acea|ida|id)$", g):
        return None
    ep = toks[1] if len(toks) > 1 else None
    if ep and re.fullmatch(r"[a-z][a-z\-]{2,}", ep) and ep not in {"sp", "sp.", "spp.", "isp.", "indet", "indet.", "or", "and", "gen", "et"}:
        return (g, f"{g} {ep}")
    return (g, None)


PERIOD_ORDER = ["Precambrian", "Cambrian", "Ordovician", "Silurian", "Devonian", "Carboniferous", "Permian",
                "Triassic", "Jurassic", "Cretaceous", "Tertiary", "Quaternary"]
PERIOD_TH_NAME = {"Precambrian": "พรีแคมเบรียน", "Cambrian": "แคมเบรียน", "Ordovician": "ออร์โดวิเชียน", "Silurian": "ไซลูเรียน",
                  "Devonian": "ดีโวเนียน", "Carboniferous": "คาร์บอนิเฟอรัส", "Permian": "เพอร์เมียน", "Triassic": "ไทรแอสซิก",
                  "Jurassic": "จูแรสซิก", "Cretaceous": "ครีเทเชียส", "Tertiary": "เทอร์เชียรี", "Quaternary": "ควอเทอร์นารี"}
_PERIOD_TH_RE = {"Cambrian": r"แคมเบรียน", "Ordovician": r"ออร์โด", "Silurian": r"ไซลูเรียน", "Devonian": r"ดีโวเนียน",
                 "Carboniferous": r"คาร์บอนิเฟอรัส", "Permian": r"เพอร์เมียน", "Triassic": r"ไทรแอส", "Jurassic": r"จู(?:แร|รา)ส",
                 "Cretaceous": r"ครีเท", "Tertiary": r"เทอร์เชียรี|นีโอจีน|พาลีโอจีน|ไมโอซีน|ไพลโอซีน|โอลิโกซีน|อีโอซีน",
                 "Quaternary": r"ควอเทอร์นารี|ไพลสโตซีน|โฮโลซีน"}


def parse_period_th(text):
    # (oldest, youngest) standard period named in Thai free text
    if not isinstance(text, str):
        return (None, None)
    found = sorted((p for p, pat in _PERIOD_TH_RE.items() if re.search(pat, text)), key=PERIOD_ORDER.index)
    return (found[0], found[-1]) if found else (None, None)
