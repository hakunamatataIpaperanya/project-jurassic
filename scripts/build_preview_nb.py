# Build notebooks/17_data_preview.ipynb — หน้าตาข้อมูลหลังเตรียม 3 ไฟล์ (head, ขนาด, คอลัมน์) ที่ส่งต่อให้งาน 1 และงาน 2
# อ่านไฟล์อย่างเดียว ไม่เขียนอะไรกลับ
import sys

import nbformat as nbf

out = sys.argv[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# หน้าตาข้อมูลหลังเตรียม

ไฟล์ 3 ชุดที่ได้จากขั้นเตรียมข้อมูล และส่งต่อให้แบบจำลอง

| ไฟล์ | 1 แถว = | ใช้ที่ |
|---|---|---|
| `data/processed/fossil_list_final.csv` | ตัวอย่าง 1 ชิ้น | งาน 1 (ป้ายสกุล ข้อความ) · งาน 2 (ป้ายประเภทของแหล่ง) |
| `data/processed/sites_period_final.csv` | แหล่ง 1 แหล่ง | งาน 1 (ป้ายยุค) · งาน 2 (จุดที่พบซาก) |
| `data/features/all_features.parquet` | จุด 1 จุด | งาน 2 (แหล่ง + พื้นหลัง ฝึก / ทดสอบ · PBDB ทดสอบอิสระ) |
""")

code(r"""
from pathlib import Path

import pandas as pd

DATA = Path("../data")
pd.set_option("display.max_columns", 250, "display.max_colwidth", 40, "display.width", 200)

fl = pd.read_csv(DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str}, low_memory=False)
sites = pd.read_csv(DATA / "processed" / "sites_period_final.csv")
feat = pd.read_parquet(DATA / "features" / "all_features.parquet")
pd.DataFrame({"แถว": [len(fl), len(sites), len(feat)], "คอลัมน์": [fl.shape[1], sites.shape[1], feat.shape[1]]},
             index=["fossil_list_final.csv", "sites_period_final.csv", "all_features.parquet"])
""")

md("## 1. fossil_list_final.csv — ทะเบียนตัวอย่าง + ผลเชื่อมกับแหล่ง + ยุคของแหล่ง")

code(r"""
fl.head()
""")

md("""
กลุ่มคอลัมน์: ข้อมูลในทะเบียน (`sci_name` … `regis_code`) · คีย์ที่ทำความสะอาดแล้ว (`*_key`, `province_inferred`) ·
ผลการเชื่อม (`match_level`, `score`, `kw`, `grp`, `adm`, `confidence`) · ข้อมูลแหล่งที่เชื่อมได้ (`site_*`) · ยุคสุดท้ายของแหล่ง (`site_final_*`)
""")

code(r"""
pd.DataFrame({"มีค่า": fl.notna().sum(), "ค่าไม่ซ้ำ": fl.nunique(), "ตัวอย่างค่า": fl.iloc[0]})
""")

md("## 2. sites_period_final.csv — ยุคสุดท้ายของแหล่ง พร้อมที่มาและความเชื่อมั่น")

code(r"""
sites.head()
""")

code(r"""
sites.final_source.value_counts(dropna=False).rename("แหล่ง")
""")

md("## 3. all_features.parquet — คุณลักษณะเชิงพื้นที่ของทุกจุดในงาน 2")

code(r"""
feat.source.value_counts().rename("จุด")
""")

code(r"""
groups = feat.columns.str.extract(r"^(pg|geo|s2|dem|oc)_")[0].fillna("ตัวระบุ")
groups.value_counts().rename("คอลัมน์")
""")

code(r"""
# 1 แถวต่อชุดจุด
feat.groupby("source").head(1)
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (Project_Jurassic)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
