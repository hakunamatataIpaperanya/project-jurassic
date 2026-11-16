# Build notebooks/12_inference.ipynb — load the saved models and predict for new coordinates / specimen records.
import sys

import nbformat as nbf

out = sys.argv[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# ขั้นที่ 5: ทำนายด้วยโมเดลที่บันทึกไว้

| ฟังก์ชัน | input | output | โมเดล |
|---|---|---|---|
| `predict_location(lat, lon)` | พิกัด | โอกาสพบซาก (0–1) ทั้ง `bg_bias` / `bg_uniform` + อันดับเปอร์เซ็นไทล์เทียบทั้งประเทศ + ประเภทซาก 3 อันดับ + อยู่ในขอบเขตที่โมเดลใช้ได้ไหม (AOA) + แหล่ง ทธ. ที่คล้ายที่สุด | งานที่ 2 (`models/task2/final/`) |
| `predict_specimen(record)` | ข้อมูลตัวอย่างที่ไม่มีชื่อวิทยาศาสตร์ (+ แหล่งหรือพิกัด) | สกุล 3 อันดับ + ชนิดที่น่าจะเป็น | งานที่ 1 (`models/task1/final/`) |

feature ของพิกัดใหม่คำนวณด้วย `feature_lib` (โค้ดชุดเดียวกับตอน train): พันเจีย, DEM, ธรณีวิทยา และ Sentinel-2
(ดึงสดจาก Planetary Computer ด้วยกติกาเลือกภาพเดียวกัน — ต้องต่ออินเทอร์เน็ต)

ใช้**โมเดลตารางอย่างเดียว** ตามที่ตกลงหลัง notebook 10 (`final_meta.json` → `deploy`); ภาพ Sentinel-2 ยังใช้คำนวณดัชนีและ CNN embedding ของงานที่ 1
""")

code(r"""
import sys

sys.path.insert(0, ".")
import model_lib as ml
import feature_lib as fl
import predict_lib as pl                     # โค้ดทำนายชุดเดียวกับเว็บแอป (webapp/) และเทสต์: scripts/predict_sections/

import pandas as pd

PRED = pl.Predictor()                        # โหลดโมเดลตัวจริง + meta + ตาราง/กริดที่ใช้ประกอบ ครั้งเดียว
predict_location = PRED.predict_location     # (lat, lon) -> (ผล, feature, patch)
predict_specimen = PRED.predict_specimen     # record -> สกุล 3 อันดับ + ชนิดที่น่าจะเป็น
print("งานที่ 2:", {bg: PRED.stage_a[bg]["tabular"] for bg in PRED.deploy["backgrounds"]}, "| B:", PRED.meta2["stage_B"]["tabular"],
      "| CNN:", PRED.deploy["use_cnn"])
print("งานที่ 1:", PRED.genus["models"], "|", PRED.meta1["period"]["model"])
""")

md(r"""
## ทดลองทำนายพิกัด

- 3 จุดแรกเป็น**พิกัดของแหล่ง ทธ. ที่ใช้ train** — คะแนนสูงเป็นธรรมดา ใช้ดูว่าระบบทำงานครบ ไม่ใช่การทดสอบ
  (การทดสอบจริงดู `spatial_block` ใน notebook 10 และ PBDB ใน 10b)
- `inside_*` = อยู่ในขอบเขตที่โมเดลใช้ได้ (AOA) ไหม; `similar_sites` = แหล่ง ทธ. ที่ feature คล้ายที่สุด (อธิบายว่าทำไมได้คะแนนนี้);
  `nearest_site_km` = ระยะถึงแหล่ง ทธ. ที่ใกล้สุดทางภูมิศาสตร์
""")

code(r"""
# พิกัดจริงของแหล่ง ทธ. (จาก fossil_sites.geojson) + จุดทั่วไปที่ไม่มีแหล่งใกล้
examples = {
    "ภูเวียง หลุม 3 ขอนแก่น (แหล่ง ทธ.)": (16.6803, 102.2357),
    "ภูกุ้มข้าว กาฬสินธุ์ (แหล่ง ทธ.)": (16.6947, 103.5247),
    "ฟิวซูลินิดเขาขาว สระบุรี (แหล่ง ทธ.)": (14.6993, 100.8573),
    "เกาะตะรุเตา สตูล (หินพาลีโอโซอิก)": (6.62, 99.66),
    "กรุงเทพฯ (ที่ราบตะกอน)": (13.75, 100.50),
    "ดอยอินทนนท์ เชียงใหม่ (หินแกรนิต)": (18.59, 98.49),
}
rows, cache = [], {}
for name, (la, lo) in ml.tqdm_auto()(examples.items(), desc="ทำนาย"):
    r, df, P = predict_location(la, lo)
    cache[name] = (df, P)
    rows.append({"ตำแหน่ง": name, **{k: v for k, v in r.items() if k not in ("lat", "lon")}})
pd.DataFrame(rows).set_index("ตำแหน่ง")
""")

md(r"""
## ทดลองเติมชื่อวิทยาศาสตร์ของตัวอย่าง

`record` ต้องมีข้อมูลแบบเดียวกับทะเบียน (`com_name`, `f_type`, `f_group`, `f_part`, `province`, `district`)
และ **แหล่ง** (`site_OBJECTID`) หรือ **พิกัด** (`lat`, `lon`) อย่างใดอย่างหนึ่ง
(`dis_name`, `match_level`, `confidence` ไม่ใช้แล้ว — ตัดออกกัน data leak ใน notebook 11)
""")

md(r"""
`predict_specimen` สลับตามกรณีแบบเดียวกับ notebook 11 (แหล่งที่มีตัวอย่างระบุสกุลแล้ว → โมเดล known + TF-IDF; แหล่งใหม่ / พิกัดใหม่ →
โมเดล new + หมวดหิน + ThaiLex + WangchanBERTa) แล้วตัดสกุลที่ขัดกับกลุ่มซาก (ตัวกรองความสอดคล้อง) และเดาชนิดจากชนิดที่พบบ่อยสุดในแหล่ง → ทั้งประเทศ
""")

md(r"""
### ตรวจความสอดคล้องกับ notebook 11

ทำนายตัวอย่างจริงจากทะเบียน (ที่ไม่มีสกุล) ด้วย `predict_specimen` แล้วเทียบกับผลที่ notebook 11 บันทึกไว้ — ต้องได้สกุลอันดับ 1 ตรงกัน
""")

code(r"""
REG = pd.read_csv(fl.DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str, "site_OBJECTID": "Int64"},
                  encoding="utf-8-sig", low_memory=False).set_index("fossil_id")
PRED11 = pd.read_csv(fl.DATA / "processed" / "task1_genus_predictions.csv", dtype={"fossil_id": str}, encoding="utf-8-sig")
check = PRED11.groupby("route", group_keys=False).apply(lambda g: g.sample(min(len(g), 20), random_state=0))
rows = []
for r in ml.tqdm_auto()(check.itertuples(), total=len(check), desc="ตรวจ"):
    rec = REG.loc[r.fossil_id, ["com_name", "f_type", "f_group", "f_part", "province", "district", "site_OBJECTID"]].to_dict()
    rec["site_OBJECTID"] = None if pd.isna(rec["site_OBJECTID"]) else int(rec["site_OBJECTID"])
    out = predict_specimen(rec)
    rows.append({"fossil_id": r.fossil_id, "route": out["route"], "notebook 11": r.genus_top1, "notebook 12": out["top"][0][0],
                 "ชนิด (11)": r.species_guess, "ชนิด (12)": out["species"][0]})
chk = pd.DataFrame(rows)
same_sp = chk["ชนิด (11)"].fillna("-").eq(chk["ชนิด (12)"].fillna("-"))
print(f"ตรงกัน {chk['notebook 11'].eq(chk['notebook 12']).mean():.0%} จาก {len(chk)} ตัวอย่าง | ชนิดที่เดาตรงกัน {same_sp.mean():.0%}")
chk.head(8)
""")

md("### ตัวอย่างใหม่ (มีแต่พิกัด)")

code(r"""
tests = [
    {"com_name": "ไดโนเสาร์", "f_type": "ไดโนเสาร์", "f_group": "Vertebrate", "f_part": "ฟัน", "lat": 16.70, "lon": 103.62},
    {"com_name": "แบรคิโอพอด", "f_type": "แบรคิโอพอด", "f_group": "Invertebrate", "f_part": "เปลือกแบรคิโอพอด", "lat": 17.49, "lon": 101.72},
    {"com_name": "ฟันฉลาม", "f_type": "ฉลาม", "f_group": "Vertebrate", "f_part": "ฟัน", "lat": 16.45, "lon": 103.10},
]
pd.DataFrame([{"ตัวอย่าง": t["com_name"], **predict_specimen(t)} for t in tests])
""")

md(r"""
## สรุป

**`predict_location(lat, lon)`** (งานที่ 2) คำนวณ feature สดด้วย `feature_lib` ชุดเดียวกับตอน train (พันเจีย, DEM, ธรณีวิทยา, ภาพ Sentinel-2 จาก Planetary Computer)
แล้วคืนโอกาสพบซากทั้งสอง background + อันดับเทียบทั้งประเทศ, ประเภทซาก 3 อันดับ, ธง AOA และแหล่ง ทธ. ที่คล้ายที่สุด

| ตำแหน่ง | ผล | อ่านอย่างไร |
|---|---|---|
| ภูเวียง / ภูกุ้มข้าว / เขาขาว (แหล่ง ทธ.) | อันดับ ~100%, ประเภทถูก (ไดโนเสาร์ / ปะการัง-ฟิวซูลินิด) | เป็นจุด train — ตรวจว่าระบบทำงานครบ |
| เกาะตะรุเตา | โอกาสต่ำ (อันดับ 24–58%) แต่ประเภท = ไทรโลไบต์ 0.752 | หินแคมเบรียน; ประเภทถูกตามธรณีวิทยา แม้จุดนี้ไม่ใช่แหล่งที่รู้จัก |
| กรุงเทพฯ | `bg_bias` อันดับ 92% แต่**นอก AOA**; `bg_uniform` 71% | คล้ายแหล่งหอยนางรมยักษ์ ปทุมธานี (ห่าง 38 กม.) — ใช้อย่างระวัง |
| ดอยอินทนนท์ | อันดับ ~30%, **นอก AOA ทุกโมเดล** | หินแกรนิตพรีแคมเบรียน ไม่มีข้อมูล train ที่คล้าย — ไม่ควรเชื่อคะแนน |

**`predict_specimen(record)`** (งานที่ 1) ทำนายสกุลแบบเดียวกับ notebook 11: สลับโมเดลตามว่าแหล่งมีตัวอย่างระบุสกุลแล้วหรือไม่
+ ตัวกรองความสอดคล้องทางอนุกรมวิธาน — ทดสอบกับตัวอย่างจริง 40 ชิ้น**ตรงกับ notebook 11 ทุกชิ้น**;
ตัวอย่างที่มีแต่พิกัดจะหาจังหวัดจากขอบเขตจังหวัดของ ทธ. ให้อัตโนมัติ

**ข้อจำกัด**: ภาพ Sentinel-2 ต้องดึงจากอินเทอร์เน็ต (ถ้าไม่มีภาพ feature ดาวเทียมเป็นค่าว่าง ซึ่งโมเดลต้นไม้รับได้แต่แม่นน้อยลง);
ผลทำนายเป็น "ข้อเสนอ" ให้ผู้เชี่ยวชาญตรวจ โดยเฉพาะจุดที่ `inside_* = False`
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (Project_Jurassic)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
