# Build notebooks/06_eda_final.ipynb (EDA of the processed datasets: missing values, hidden nulls, other quality checks).
import sys

import nbformat as nbf

out = sys.argv[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# EDA ข้อมูลชุดใหม่: ค่าที่ขาดหาย (NULL) และคุณภาพข้อมูล

สำรวจข้อมูลหลังผ่าน notebook 02–05 แล้ว

| ชุดข้อมูล | ไฟล์ | หน่วย |
|---|---|---|
| **ตัวอย่าง (final)** | `data/processed/fossil_list_final.csv` | ตัวอย่าง 1 ชิ้น (ทะเบียน + แหล่งที่เชื่อม + ยุคสุดท้าย) |
| **แหล่ง (final)** | `data/processed/sites_period_final.geojson` | แหล่ง 1 แห่ง (ยุคสุดท้าย) |
| ต้นฉบับ (ไว้เทียบ) | `data/0301010101_fossil_list-31072567.csv`, `data/dmr_fossil_sites/fossil_sites.geojson` | |

**สารบัญ**
1. ภาพรวมและกลุ่มคอลัมน์
2. ค่าที่ขาดหายระดับตัวอย่าง: แยก "ขาดเพราะโครงสร้าง" กับ "ขาดจริง"
3. รูปแบบการขาดหาย: matrix และความสัมพันธ์ของการขาด
4. ค่าที่ทำหน้าที่เหมือนค่าว่าง (hidden nulls)
5. ก่อน vs หลังทำความสะอาด/เติมข้อมูล
6. ค่าที่ขาดหายระดับแหล่ง
7. คุณภาพข้อมูลด้านอื่น: ค่าซ้ำ ชนิดข้อมูล ความสอดคล้อง ค่าผิดปกติ การสะกดหลายแบบ
8. สรุปปัญหาและข้อแนะนำ
""")

code(r"""
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from rapidfuzz import fuzz, process

DATA = Path("../data")
OUT = DATA / "processed"

_installed = {f.name for f in font_manager.fontManager.ttflist}
THAI_FONT = next((f for f in ["Tahoma", "Thonburi", "Noto Sans Thai", "Leelawadee UI", "Loma", "Garuda"] if f in _installed), None)
plt.rcParams.update({
    "font.family": [THAI_FONT, "DejaVu Sans"] if THAI_FONT else ["DejaVu Sans"], "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
})
pd.set_option("display.max_columns", 50, "display.max_colwidth", 60, "display.width", 160)
INK, MUTED, BAR = "#52514e", "#8a8984", "#2a78d6"

ID_DTYPES = {"fossil_id": str, "SITE_CODE": str, "site_SITE_CODE": str, "site_OBJECTID": "Int64"}
spec = pd.read_csv(OUT / "fossil_list_final.csv", dtype=ID_DTYPES, encoding="utf-8-sig", low_memory=False)
sites = gpd.read_file(OUT / "sites_period_final.geojson")
raw_spec = pd.read_csv(DATA / "0301010101_fossil_list-31072567.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
raw_sites = gpd.read_file(DATA / "dmr_fossil_sites" / "fossil_sites.geojson")

N = len(spec)


def is_text(series):
    # pandas 3 อ่านข้อความเป็น dtype 'str' ไม่ใช่ object จึงต้องเช็กทั้งสองแบบ
    return pd.api.types.is_string_dtype(series) or series.dtype == object


print(f"ตัวอย่าง (final): {spec.shape} | แหล่ง (final): {sites.shape}")
print(f"ต้นฉบับ: ทะเบียน {raw_spec.shape} (pd.read_csv ตรง ๆ) | แหล่ง {raw_sites.shape}")
""")

# ---------------------------------------------------------------- 1
md(r"""
## 1. ภาพรวมและกลุ่มคอลัมน์

ไฟล์ตัวอย่าง (final) มี 44 คอลัมน์ มาจาก 5 ขั้นตอน การขาดหายของแต่ละกลุ่มมีความหมายต่างกัน จึงแบ่งกลุ่มไว้ก่อน
""")

code(r"""
GROUPS = {
    "ทะเบียนเดิม": ["fossil_id", "sci_name", "com_name", "f_group", "f_type", "f_part", "dis_name", "province", "district", "tambon", "regis_code"],
    "สร้างตอนทำความสะอาด": ["province_inferred", "district_key", "tambon_key", "genus"],
    "ผลการ merge (02)": ["match_level", "fuzzy_note", "n_candidates", "n_top", "tied_site_ids", "score", "kw", "grp", "adm", "confidence"],
    "ข้อมูลแหล่งที่เชื่อม": [c for c in spec.columns if c.startswith("site_") and not re.search(r"PERIOD_FINAL|final_", c)],
    "ยุคสุดท้าย (05)": [c for c in spec.columns if re.search(r"site_(PERIOD_FINAL|final_)", c)],
}
GROUP_COLOR = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
COL_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}
assert set(COL_GROUP) == set(spec.columns), set(spec.columns) ^ set(COL_GROUP)

overview = pd.DataFrame({
    "group": [COL_GROUP[c] for c in spec.columns],
    "dtype": spec.dtypes.astype(str).values,
    "non_null": spec.notna().sum().values,
    "missing": spec.isna().sum().values,
    "missing_%": (spec.isna().mean() * 100).round(1).values,
    "unique": spec.nunique().values,
    "example": [spec[c].dropna().astype(str).iloc[0][:40] if spec[c].notna().any() else "" for c in spec.columns],
}, index=spec.columns)
print(f"หน่วยความจำ {spec.memory_usage(deep=True).sum() / 1e6:.1f} MB | คอลัมน์ที่ไม่มีค่าว่างเลย {(overview.missing == 0).sum()} จาก {len(overview)}")
overview
""")

# ---------------------------------------------------------------- 2
md(r"""
## 2. ค่าที่ขาดหายระดับตัวอย่าง

ค่าว่างบางส่วน**เป็นไปตามโครงสร้าง ไม่ใช่ข้อมูลหาย** เช่น ตัวอย่าง 668 ชิ้นที่ไม่ได้เชื่อมกับแหล่งใด (`match_level = none`)
ย่อมไม่มีข้อมูลแหล่ง จึงแยกนับ % ขาดหาย 2 แบบ
- **ทุกแถว**: นับตรง ๆ
- **เฉพาะที่เชื่อมแหล่งได้**: สำหรับคอลัมน์ของ merge / แหล่ง / ยุค นับเฉพาะ 7,853 ชิ้นที่มีแหล่ง เพื่อดูว่า**ขาดจริง**เท่าไร
""")

code(r"""
linked = spec.site_OBJECTID.notna()
structural = set(GROUPS["ผลการ merge (02)"] + GROUPS["ข้อมูลแหล่งที่เชื่อม"] + GROUPS["ยุคสุดท้าย (05)"]) - {"match_level", "confidence"}
miss = pd.DataFrame({
    "group": [COL_GROUP[c] for c in spec.columns],
    "all_rows_%": spec.isna().mean() * 100,
    "linked_rows_%": [spec.loc[linked, c].isna().mean() * 100 if c in structural else np.nan for c in spec.columns],
}, index=spec.columns)
miss["real_missing_%"] = miss["linked_rows_%"].fillna(miss["all_rows_%"])
show = miss[miss["all_rows_%"] > 0].sort_values("real_missing_%", ascending=False)
display(show.style.format({"all_rows_%": "{:.1f}", "linked_rows_%": "{:.1f}", "real_missing_%": "{:.1f}"}, na_rep="-"))

fig, ax = plt.subplots(figsize=(10, 9))
y = np.arange(len(show))[::-1]
ax.barh(y, show["all_rows_%"], color="#e6e5e0", height=0.75, label="ทุกแถว")
ax.barh(y, show["real_missing_%"], color=[GROUP_COLOR[g] for g in show.group], height=0.45, label="ขาดจริง")
for yi, v in zip(y, show["real_missing_%"]):
    ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=7.5, color=INK)
ax.set_yticks(y, show.index, fontsize=8)
ax.set_xlim(0, 105)
ax.set_xlabel("% ขาดหาย")
ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUPS] + [plt.Rectangle((0, 0), 1, 1, color="#e6e5e0")]
ax.legend(handles, list(GROUPS) + ["ทุกแถว (รวมขาดเพราะโครงสร้าง)"], loc="lower right", frameon=False, fontsize=8)
ax.set_title("ค่าที่ขาดหายต่อคอลัมน์ (แท่งสี = ขาดจริง, แท่งเทา = ทุกแถว)")
plt.tight_layout()
""")

md(r"""
อ่านกราฟ: คอลัมน์ข้อมูลแหล่งหลายคอลัมน์ขาดราว 8% ทุกแถว ซึ่ง**ทั้งหมด**มาจากตัวอย่างที่ไม่ได้เชื่อมแหล่ง (ขาดจริง 0%)
ส่วนที่ขาดจริงสูงคือ `regis_code` (เลขทะเบียน THF), คุณสมบัติของแหล่งอย่าง `site_GEO_GROUP`, `site_FORMATION`, `site_PERIODFROM`, `site_LOCALITY`,
และ `genus` (ตัวอย่างที่ยังไม่จำแนกถึงสกุล) ส่วน `site_PERIOD_FINAL` ขาดน้อยกว่า `site_PERIODFROM` มากเพราะเติมใน notebook 03–05

### การขาดหายตามกลุ่มย่อย

ค่าที่ขาดกระจายไม่เท่ากัน ดูว่าขาดมากในจังหวัด/กลุ่มซากไหน
""")

code(r"""
KEY = ["dis_name", "tambon", "genus", "regis_code", "site_OBJECTID", "site_PERIOD_FINAL"]
top_prov = spec.province.value_counts().head(15).index
by_prov = spec[KEY].isna().astype(float).groupby(spec.province).mean().reindex(top_prov)
by_group = spec[KEY].isna().astype(float).groupby(spec.f_group).mean()

blues = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.6, 1]})
for ax, tbl, title in [(axes[0], by_prov, "15 จังหวัดที่มีตัวอย่างมากสุด"), (axes[1], by_group, "กลุ่มซาก")]:
    im = ax.imshow(tbl.values, cmap=blues, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(KEY)), KEY, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(tbl)), tbl.index, fontsize=8)
    ax.grid(False)
    for i in range(tbl.shape[0]):
        for j in range(tbl.shape[1]):
            v = tbl.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
    ax.set_title(f"% ขาดหาย: {title}")
fig.colorbar(im, ax=axes, shrink=0.6, format=plt.matplotlib.ticker.PercentFormatter(1))
""")

# ---------------------------------------------------------------- 3
md(r"""
## 3. รูปแบบการขาดหาย

### Missing matrix
แต่ละแถวคือตัวอย่าง 1 ชิ้น (เรียงตามระดับการเชื่อมแหล่ง) สีเข้ม = ขาด
ถ้าเห็นเป็น**แถบตามแนวนอน** แปลว่าหลายคอลัมน์ขาดพร้อมกันในแถวเดียวกัน (มีสาเหตุร่วม)
""")

code(r"""
cols = [c for c in spec.columns if 0 < spec[c].isna().mean() < 1]
order = spec.assign(_lv=spec.match_level.map({"tambon": 0, "locality": 1, "district": 2, "province": 3, "none": 4}),
                    _p=spec.province.isna()).sort_values(["_lv", "_p", "site_OBJECTID"]).index
M = spec.loc[order, cols].isna().values

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#eef3fa", "#256abf"]))
ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
ax.set_yticks([])
ax.grid(False)
lv = spec.loc[order, "match_level"].values
for name in ["tambon", "locality", "district", "province", "none"]:
    idx = np.where(lv == name)[0]
    if len(idx):
        ax.text(-0.6, idx.mean(), f"{name} ({len(idx):,})", ha="right", va="center", fontsize=8, color=INK)
        ax.axhline(idx.max() + 0.5, color="white", lw=1.2)
ax.set_title("Missing matrix (แถว = ตัวอย่าง เรียงตามระดับการเชื่อมแหล่ง, สีเข้ม = ขาด)")
plt.tight_layout()
""")

md(r"""
### ความสัมพันธ์ของการขาด (nullity correlation)

ค่าสหสัมพันธ์ระหว่าง "คอลัมน์ A ขาด" กับ "คอลัมน์ B ขาด" ใกล้ +1 = ขาดพร้อมกันเสมอ, ใกล้ −1 = ขาดสลับกัน
ตัดคอลัมน์ที่ขาดพร้อมกันเป๊ะ (ข้อมูลแหล่งทั้งชุด) ออกเหลือตัวแทนเดียว จะได้อ่านง่าย
""")

code(r"""
nul = spec[cols].isna().astype(int)
# ยุบคอลัมน์ที่รูปแบบการขาดเหมือนกันทุกแถวให้เหลือตัวแทนเดียว
sig = nul.T.apply(lambda r: hash(r.values.tobytes()), axis=1)
PREFER = ["site_OBJECTID", "site_PERIOD_FINAL", "district", "tambon"]  # ชื่อตัวแทนที่อ่านแล้วเข้าใจง่าย
rep = []
for h_ in sig.drop_duplicates():
    members = sig[sig == h_].index.tolist()
    rep.append(next((c for c in PREFER if c in members), members[0]))
rep = [c for c in cols if c in rep]
merged_note = {c: [d for d in cols if sig[d] == sig[c] and d != c] for c in rep}
corr = nul[rep].corr()

div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr.values, cmap=div, vmin=-1, vmax=1)
labels = [f"{c} (+{len(merged_note[c])})" if merged_note[c] else c for c in rep]
ax.set_xticks(range(len(rep)), labels, rotation=60, ha="right", fontsize=7.5)
ax.set_yticks(range(len(rep)), labels, fontsize=7.5)
ax.grid(False)
for i in range(len(rep)):
    for j in range(len(rep)):
        if i != j and abs(corr.values[i, j]) >= 0.3:
            ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
fig.colorbar(im, ax=ax, shrink=0.7)
ax.set_title("Nullity correlation (+N = มีอีก N คอลัมน์ที่ขาดพร้อมกันเป๊ะ)")
plt.tight_layout()
for c, d in merged_note.items():
    if d:
        print(f"{c} ขาดพร้อมกันเป๊ะกับ: {', '.join(d)}")
""")

md(r"""
- `district` ↔ `district_key` และ `tambon` ↔ `tambon_key` ขาดพร้อมกันเสมอ (คีย์สร้างจากคอลัมน์ต้นทาง)
- ข้อมูลแหล่งทั้งชุดขาดพร้อมกันเป๊ะ (มาจากตัวอย่างที่ไม่ได้เชื่อมแหล่ง)
- `site_FORMATION`, `site_GEO_GROUP`, `site_PERIODFROM` ขาดพร้อมกันสูง (0.7–0.9) แปลว่าแหล่งที่ไม่บันทึกธรณีวิทยา มักไม่บันทึกเลยทั้งชุด
  การเติมยุคจึงต้องพึ่งข้อมูลภายนอก (PBDB, ชื่อซาก) เพราะเดาจากหมวดหินของแหล่งเดียวกันไม่ได้
- ชื่อกลุ่มซากใน `f_type` สะกดหลายแบบด้วย เช่น `ฟอแรมมินิเฟอร์` / `ฟอแรมินิเฟอร์` / `ฟอแรมินิเฟอรา` (ดูส่วน 7.3)
- `sci_name` กับ `genus`: ทุกแถวที่ไม่มี `sci_name` ก็ไม่มี `genus` แต่ `genus` ขาดมากกว่า เพราะชื่อที่ไม่ใช่ระดับสกุล (`N/A (Dinosaur)`, `Sauropod`) ก็นับเป็นขาด

## 4. ค่าที่ทำหน้าที่เหมือนค่าว่าง (hidden nulls)

ค่าที่ไม่ใช่ `NaN` แต่ไม่มีข้อมูลจริง เช่น `ไม่ระบุ`, `N/A`, `No data`, ช่องว่างล้วน, ข้อความ `nan`
นับในไฟล์ต้นฉบับเทียบกับไฟล์ final
""")

code(r"""
HIDDEN = {"ไม่ระบุ", "n/a", "na", "no data", "-", "--", "รอการตรวจสอบ", "nan", "none", "null", "<null>", "?", ""}


def hidden_counts(df, columns):
    out = {}
    for c in columns:
        s = df[c].dropna().astype(str)
        st = s.str.strip()
        out[c] = {
            "ค่าแทนที่ว่าง": int(st.str.lower().isin(HIDDEN - {""}).sum()),
            "ว่าง/ช่องว่างล้วน": int((st == "").sum()),
            "มีช่องว่างหัว/ท้าย": int((s != st).sum()),
            "N/A (กลุ่ม)": int(st.str.match(r"^N/A\s*\(").sum()),
        }
    return pd.DataFrame(out).T


orig_cols = ["sci_name", "com_name", "f_group", "f_type", "f_part", "dis_name", "province", "district", "tambom", "regis_code"]
before = hidden_counts(raw_spec, orig_cols)
after = hidden_counts(spec.rename(columns={"tambon": "tambom"}), orig_cols)
cmp_hidden = pd.concat({"ต้นฉบับ": before, "final": after}, axis=1)
display(cmp_hidden[(cmp_hidden != 0).any(axis=1)])

site_text = [c for c in raw_sites.columns if c != "geometry" and is_text(raw_sites[c])]
hs = hidden_counts(raw_sites, site_text)
print("แหล่ง (ต้นฉบับ) คอลัมน์ที่มี hidden null หรือช่องว่างเกิน:")
hs[(hs != 0).any(axis=1)]
""")

md(r"""
- ไฟล์ final ไม่มีค่าแทนที่ว่างและช่องว่างเกินเหลือแล้ว (ทำความสะอาดใน notebook 02)
- **`N/A (กลุ่ม)` ใน `sci_name` ตั้งใจเก็บไว้** เพราะบอกกลุ่มกว้างของซาก (เช่น `N/A (Dinosaur)`) แต่ต้องระวังเมื่อนับจำนวนชนิด
- ต้นฉบับของแหล่งมีช่องว่างต่อท้ายใน `FOS_GR_*` ซึ่งทำให้นับกลุ่มผิดถ้าไม่ตัดก่อน
  และค่าว่างของแหล่งเก็บเป็นข้อความว่าง `""` ไม่ใช่ NULL จริง จึงต้องแปลงก่อนนับค่าที่ขาด

## 5. ก่อน vs หลัง
""")

code(r"""
raw_junk = (~raw_spec.fossil_id.str.strip().str.fullmatch(r"\d+")).sum()
raw_na = lambda c: raw_spec[c].str.strip().str.lower().isin(HIDDEN).mean() * 100
rows = [
    ("ระเบียนที่อ่านผิด (fossil_id ไม่ใช่ตัวเลข)", raw_junk, 0, "แถว"),
    ("ไม่มีจังหวัด", raw_na("province"), spec.province.isna().mean() * 100, "%"),
    ("ไม่มีอำเภอ", raw_na("district"), spec.district.isna().mean() * 100, "%"),
    ("ไม่มีตำบล", raw_na("tambom"), spec.tambon.isna().mean() * 100, "%"),
    ("ไม่มีพิกัด", 100.0, spec.site_longitude.isna().mean() * 100, "%"),
    ("ไม่มียุค (ของแหล่งที่เชื่อม)", 100.0, spec.site_PERIOD_FINAL.isna().mean() * 100, "%"),
]
ba = pd.DataFrame(rows, columns=["รายการ", "ก่อน", "หลัง", "หน่วย"]).set_index("รายการ")
ba["เปลี่ยน"] = ba["หลัง"] - ba["ก่อน"]
display(ba.style.format({"ก่อน": "{:.1f}", "หลัง": "{:.1f}", "เปลี่ยน": "{:+.1f}"}))

s_rows = [("PERIODFROM / PERIOD_FINAL", sites.PERIODFROM.isna().mean() * 100, sites.PERIOD_FINAL.isna().mean() * 100)]
print("ระดับแหล่ง: ยุคว่าง %.1f%% → %.1f%%" % s_rows[0][1:])

fig, ax = plt.subplots(figsize=(9, 3.6))
pct = ba[ba["หน่วย"].eq("%")]
y = np.arange(len(pct))[::-1]
ax.barh(y + 0.18, pct["ก่อน"], height=0.36, color="#bdbcb6", label="ก่อน (ต้นฉบับ)")
ax.barh(y - 0.18, pct["หลัง"], height=0.36, color=BAR, label="หลัง (final)")
for yi, (b, a) in zip(y, zip(pct["ก่อน"], pct["หลัง"])):
    ax.text(b + 1, yi + 0.18, f"{b:.0f}%", va="center", fontsize=8, color=INK)
    ax.text(a + 1, yi - 0.18, f"{a:.0f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(y, pct.index, fontsize=9)
ax.set_xlim(0, 112)
ax.set_xlabel("% ตัวอย่างที่ขาด")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
ax.set_title("ค่าที่ขาดหายระดับตัวอย่าง: ก่อน vs หลัง")
plt.tight_layout()
""")

# ---------------------------------------------------------------- 6
md(r"""
## 6. ค่าที่ขาดหายระดับแหล่ง

ใช้ไฟล์แหล่งต้นฉบับ (55 คอลัมน์) ดูว่าคอลัมน์ไหนว่างทั้งหมด/ว่างมาก และไฟล์ final ดูยุคที่เติมแล้ว
""")

code(r"""
rs = raw_sites.drop(columns="geometry").copy()
for c in rs.columns:
    if is_text(rs[c]):
        rs[c] = rs[c].astype("string").str.strip().replace({"": pd.NA, "<Null>": pd.NA})
site_miss = (rs.isna().mean() * 100).sort_values(ascending=False)
all_empty = site_miss[site_miss == 100].index.tolist()
print(f"คอลัมน์ว่างทั้งหมด {len(all_empty)} จาก {rs.shape[1]}: {', '.join(all_empty)}")

partial = site_miss[(site_miss > 0) & (site_miss < 100)]
final_cols = sites.drop(columns="geometry").isna().mean() * 100
final_partial = final_cols[final_cols > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})
for ax, ser, title in [(axes[0], partial, "แหล่ง (ต้นฉบับ): คอลัมน์ที่ขาดบางส่วน"), (axes[1], final_partial.sort_values(ascending=False), "แหล่ง (final)")]:
    s_ = ser[::-1]
    ax.barh(s_.index, s_.values, color=BAR, height=0.65)
    for yi, v in enumerate(s_.values):
        ax.text(v + 1, yi, f"{v:.1f}%" if 99 < v < 100 else f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 112)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
plt.tight_layout()
""")

md(r"""
ในไฟล์แหล่ง final คอลัมน์ `alternative`, `disagrees_with_record` ว่างเป็นส่วนใหญ่**โดยตั้งใจ** (มีค่าเฉพาะแหล่งที่คำทายขัดกัน)
และ `est_accuracy` ว่างในแหล่งที่ใช้ค่าเดิมของ ทธ. (`recorded`) ส่วนที่ขาดจริงคือ `PERIOD_FINAL` (25%) และ `PERIODFROM` ต้นฉบับ (68%)
""")

code(r"""
tbl = pd.crosstab(sites.PROVINCE, sites.PERIOD_FINAL.isna()).rename(columns={False: "มียุค", True: "ยังว่าง"})
tbl["%ว่าง"] = tbl["ยังว่าง"] / tbl.sum(axis=1)
print("จังหวัดที่ยังมีแหล่งไม่มียุคมากสุด:")
tbl.sort_values("ยังว่าง", ascending=False).head(10).style.format({"%ว่าง": "{:.0%}"})
""")

# ---------------------------------------------------------------- 7
md(r"""
## 7. คุณภาพข้อมูลด้านอื่น

### 7.1 ค่าซ้ำ
""")

code(r"""
content_cols = [c for c in GROUPS["ทะเบียนเดิม"] if c != "fossil_id"]
dup_content = spec.duplicated(subset=content_cols, keep=False)
print(f"fossil_id ซ้ำ: {spec.fossil_id.duplicated().sum()}")
print(f"ตัวอย่างที่เนื้อหาเหมือนกันทุกคอลัมน์ (ต่างแค่ fossil_id): {dup_content.sum():,} แถว ({dup_content.mean():.0%}) ใน {spec[dup_content].groupby(content_cols, dropna=False).ngroups:,} กลุ่ม")
print("  -> ปกติสำหรับทะเบียนที่ลงชิ้นส่วนหลายชิ้นจากแหล่งเดียวกัน (เช่น ฟันไดโนเสาร์หลายซี่) ไม่ใช่ข้อมูลซ้ำผิด")

print(f"\nแหล่ง: SITE_CODE ซ้ำ {sites.SITE_CODE.duplicated(keep=False).sum()} แถว | ชื่อซ้ำ {sites.NAME_TH.duplicated(keep=False).sum()} แถว")
xy = sites.geometry.apply(lambda g: (round(g.x, 5), round(g.y, 5)))
same_xy = xy.duplicated(keep=False)
print(f"แหล่งที่พิกัดซ้ำกันเป๊ะ: {same_xy.sum()} แถว")
sites.loc[sites.SITE_CODE.duplicated(keep=False) | same_xy, ["OBJECTID", "SITE_CODE", "NAME_TH", "PROVINCE"]].assign(
    xy=xy[sites.SITE_CODE.duplicated(keep=False) | same_xy]).sort_values(["SITE_CODE"])
""")

md("### 7.2 ชนิดข้อมูลและรูปแบบรหัส")

code(r"""
checks = {
    "fossil_id เป็นตัวเลข 11 หลัก": spec.fossil_id.str.fullmatch(r"\d{11}").all(),
    "site_SITE_CODE เป็นตัวเลข 7 หลัก (มี 0 นำหน้า)": spec.site_SITE_CODE.dropna().str.fullmatch(r"\d{7}").all(),
    "SITE_CODE แหล่งเป็นตัวเลข 7 หลัก": sites.SITE_CODE.str.fullmatch(r"\d{7}").all(),
    "site_OBJECTID เป็นจำนวนเต็ม (ไม่ใช่ทศนิยม)": str(spec.site_OBJECTID.dtype) == "Int64",
    "regis_code รูปแบบ THF <ปี พ.ศ.> <ประเภท> <ลำดับ>": spec.regis_code.dropna().str.fullmatch(r"THF \d{4} \d \d{5}").mean(),
    "province_inferred เป็น True/False": set(spec.province_inferred.dropna().astype(str)) <= {"True", "False"},
}
pd.Series(checks, name="ผ่าน").to_frame()
""")

md(r"""
**หมายเหตุ**: ก่อนรอบนี้ `site_SITE_CODE` ในไฟล์ของ notebook 03/05 เสีย 0 นำหน้า (`0192003` → `192003.0`) เพราะอ่าน CSV โดยไม่กำหนดชนิดข้อมูล
แก้แล้วทั้งต้นทาง (02 เขียน `site_OBJECTID` เป็น `Int64`) และปลายทาง (03–05 อ่านคอลัมน์รหัสเป็นข้อความ) **ถ้าอ่านไฟล์เหล่านี้เองให้ใช้**
`pd.read_csv(..., dtype={"fossil_id": str, "SITE_CODE": str, "site_SITE_CODE": str})`

### 7.3 ความสอดคล้องระหว่างคอลัมน์
""")

code(r"""
lk = spec[linked]
prov_mismatch = lk.province.notna() & lk.province.ne(lk.site_PROVINCE)
dist_match = lk.district_key.notna() & lk.match_level.isin(["tambon", "district"])
period_ok = sites.dropna(subset=["final_old"])
ORDER = ["Precambrian", "Cambrian", "Ordovician", "Silurian", "Devonian", "Carboniferous", "Permian",
         "Triassic", "Jurassic", "Cretaceous", "Tertiary", "Quaternary"]
R = {p: i for i, p in enumerate(ORDER)}
ct_type = spec.groupby("com_name").f_type.nunique()
ct_group = spec.groupby("f_type").f_group.nunique()

consistency = pd.DataFrame([
    ("จังหวัดของตัวอย่าง ≠ จังหวัดของแหล่งที่เชื่อม", prov_mismatch.sum(), len(lk)),
    ("ยุค final_old อ่อนกว่า final_young (ช่วงกลับด้าน)", (period_ok.final_old.map(R) > period_ok.final_young.map(R)).sum(), len(period_ok)),
    ("ยุคที่เติม ขัดกับ PERIODFROM เดิม (disagrees_with_record)", sites.disagrees_with_record.notna().sum(), sites.PERIODFROM.notna().sum()),
    ("com_name เดียวแต่มีหลาย f_type", (ct_type > 1).sum(), len(ct_type)),
    ("f_type เดียวแต่มีหลาย f_group", (ct_group > 1).sum(), len(ct_group)),
    ("แหล่งไดโนเสาร์ (ตามชื่อ) ที่ยุคสุดท้ายเก่ากว่าไทรแอสซิก",
     (sites.NAME_TH.str.contains("ไดโนเสาร์", na=False) & sites.final_old.map(R).lt(R["Triassic"])).sum(),
     sites.NAME_TH.str.contains("ไดโนเสาร์", na=False).sum()),
], columns=["การตรวจ", "พบ", "จาก"])
display(consistency)

print("ตัวอย่าง f_type ที่มีหลาย f_group:")
display(spec[spec.f_type.isin(ct_group[ct_group > 1].index)].groupby(["f_type", "f_group"]).size().rename("ตัวอย่าง").to_frame().head(12))
print("แหล่งไดโนเสาร์ที่ยุคเป็นไปไม่ได้:")
sites[sites.NAME_TH.str.contains("ไดโนเสาร์", na=False) & sites.final_old.map(R).lt(R["Triassic"])][
    ["NAME_TH", "PROVINCE", "PERIOD_FINAL", "final_source", "final_confidence", "alternative"]]
""")

md(r"""
ไดโนเสาร์ปรากฏครั้งแรกในยุคไทรแอสซิก แหล่งไดโนเสาร์ที่ยุคเก่ากว่านั้นจึง**ผิดแน่นอน** — notebook 05 ส่วน 2b ใช้กฎนี้
(และกฎของไทรโลไบต์ / แกรปโตไลต์ / สัตว์เลี้ยงลูกด้วยนม) ตัดคำทายที่เป็นไปไม่ได้ก่อนรวมแล้ว ตารางนี้จึงควรว่าง

### 7.4 ค่าผิดปกติ (outliers)
""")

code(r"""
in_th = sites.geometry.x.between(97.3, 105.7) & sites.geometry.y.between(5.6, 20.5)
# นับแบบแบ่งน้ำหนัก: ตัวอย่างที่เสมอกัน k แหล่ง ให้แต่ละแหล่ง 1/k (ไม่นับจาก site_OBJECTID ที่ถูกเลือกตอนตัดสินเสมอ)
tied = spec.dropna(subset=["tied_site_ids"]).assign(sid=lambda d: d.tied_site_ids.str.split(";")).explode("sid")
tied["w"] = 1 / tied.groupby("fossil_id").sid.transform("size")
per_site = tied.groupby("sid").w.sum()
naive_max = spec.groupby("site_OBJECTID").size().max()
print(f"แหล่งที่พิกัดอยู่นอกกรอบประเทศไทย: {(~in_th).sum()}")
print(f"ตัวอย่างต่อแหล่ง (แบ่งน้ำหนักกรณีเสมอ): มัธยฐาน {per_site.median():.1f}, 90th {per_site.quantile(.9):.0f}, สูงสุด {per_site.max():,.0f}"
      f" | ถ้านับจาก site_OBJECTID ตรง ๆ สูงสุดจะเป็น {naive_max:,} (เพี้ยนจากการตัดสินเสมอ)")

fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
axes[0].hist(np.log10(per_site.values.astype(float)), bins=30, color=BAR, edgecolor="white")
axes[0].set_xticks([-1, 0, 1, 2, 3], ["0.1", "1", "10", "100", "1,000"])
axes[0].set_title("ตัวอย่างต่อแหล่ง (แบ่งน้ำหนัก, log)")
axes[0].set_xlabel("ตัวอย่าง")
axes[1].hist(spec.n_candidates.dropna(), bins=range(1, int(spec.n_candidates.max()) + 2), color=BAR, edgecolor="white")
axes[1].set_title("แหล่งผู้สมัครต่อตัวอย่าง (n_candidates)")
axes[2].hist(spec.score.dropna(), bins=20, color=BAR, edgecolor="white")
axes[2].set_title("คะแนนจัดอันดับของแหล่งที่เลือก (score)")
for ax in axes:
    ax.grid(axis="x", visible=False)
plt.tight_layout()
""")

md(r"""
ตัวอย่างต่อแหล่งเบ้ขวามาก (แหล่งขุดค้นไดโนเสาร์ไม่กี่แห่งมีหลายร้อยชิ้น) ค่าสูงสุดจึงไม่ใช่ข้อผิดพลาด
แต่ควรใช้มัธยฐานหรือ log เวลาสรุป

### 7.5 ข้อความอิสระที่สะกดหลายแบบ

`dis_name` (ผู้ค้นพบ) เป็นข้อความอิสระ ชื่อคณะเดียวกันเขียนได้หลายแบบ จับกลุ่มชื่อที่คล้ายกัน (`fuzz.token_set_ratio` ≥ 85)
""")

code(r"""
names = spec.dis_name.value_counts()
uniq = names.index.tolist()
clusters, assigned = [], set()
for nm in uniq:  # เรียงจากพบบ่อยไปน้อย ชื่อแรกของกลุ่มเป็นตัวแทน
    if nm in assigned:
        continue
    sim = process.extract(nm, [u for u in uniq if u not in assigned], scorer=fuzz.token_set_ratio, score_cutoff=85, limit=None)
    members = [m for m, _, _ in sim]
    assigned |= set(members)
    if len(members) > 1:
        clusters.append((nm, len(members), int(names[members].sum()), members[1:4]))
cl = pd.DataFrame(clusters, columns=["ตัวแทน", "ชื่อในกลุ่ม", "ตัวอย่าง", "ตัวอย่างชื่อที่รวม"]).sort_values("ตัวอย่าง", ascending=False)
print(f"dis_name {len(uniq)} ค่า → กลุ่มที่มีหลายชื่อ {len(cl)} กลุ่ม (ครอบคลุม {cl['ชื่อในกลุ่ม'].sum()} ชื่อ)")
cl.head(12)
""")

md(r"""
การจับกลุ่มอัตโนมัติช่วยหาตัวเลือก แต่ยังต้องให้คนตรวจ เพราะบางกลุ่มรวมคนละคณะที่ชื่อคล้ายกัน
`f_part` (ส่วนของซาก ~990 ค่า ผสมไทย/อังกฤษ) ก็มีปัญหาแบบเดียวกัน

## 8. สรุปปัญหาและข้อแนะนำ
""")

code(r"""
summary = pd.DataFrame([
    ("ทะเบียน", "regis_code (เลขทะเบียน THF) ขาด", f"{spec.regis_code.isna().mean():.0%}", "สูง", "ขอข้อมูลเลขทะเบียนจากต้นทาง ใช้ fossil_id เป็นคีย์ไปก่อน"),
    ("ทะเบียน", "dis_name (ผู้ค้นพบ) ขาด + สะกดหลายแบบ", f"{spec.dis_name.isna().mean():.0%} / {len(cl)} กลุ่ม", "กลาง", "ทำตาราง mapping ชื่อคณะ (เริ่มจากกลุ่มในส่วน 7.5)"),
    ("ทะเบียน", "tambon ขาด", f"{spec.tambon.isna().mean():.0%}", "กลาง", "ทำให้เชื่อมแหล่งได้แค่ระดับอำเภอ/จังหวัด"),
    ("ทะเบียน", "ยังไม่จำแนกถึงสกุล (genus ว่าง)", f"{spec.genus.isna().mean():.0%}", "กลาง", "ธรรมชาติของข้อมูล ใช้ broad taxon ใน sci_name แทน"),
    ("merge", "ตัวอย่างที่ไม่ได้เชื่อมแหล่ง (ไม่มีพิกัด)", f"{(~linked).mean():.0%}", "กลาง", "ส่วนใหญ่มาจากจังหวัดที่ไม่มีแหล่งใน DB (เช่น สมุทรสาคร)"),
    ("merge", "เชื่อมแหล่งด้วยความเชื่อมั่น low", f"{spec.confidence.eq('low').mean():.0%}", "กลาง", "กรอง confidence != 'low' เมื่อวิเคราะห์ระดับแหล่ง"),
    ("แหล่ง", "คุณสมบัติธรณีวิทยา (FORMATION, GEO_GROUP) ขาด", f"{rs.FORMATION.isna().mean():.0%}", "กลาง", "เติมจากแผนที่หน่วยหิน (notebook 01 ส่วน 5)"),
    ("แหล่ง", "PERIOD_FINAL ยังว่าง", f"{sites.PERIOD_FINAL.isna().mean():.0%}", "กลาง", "เติมจากแผนที่หน่วยหิน (เป็นอายุหิน)"),
    ("แหล่ง", "คอลัมน์ว่างทั้งหมด", f"{len(all_empty)} คอลัมน์", "ต่ำ", "ตัดทิ้ง"),
    ("แหล่ง", "แหล่งไดโนเสาร์ที่ยุคเป็นไปไม่ได้", f"{consistency.set_index('การตรวจ').loc['แหล่งไดโนเสาร์ (ตามชื่อ) ที่ยุคสุดท้ายเก่ากว่าไทรแอสซิก', 'พบ']} แหล่ง", "สูง",
     "notebook 05 ส่วน 2b ตัดคำทายที่ขัดกับซากแล้ว (คอลัมน์ rejected_by_fossils)"),
    ("แหล่ง", "SITE_CODE ชนกันข้ามจังหวัด (เช่น 0139xxx ใช้ทั้งเพชรบุรีและหนองบัวลำภู)", f"{sites.SITE_CODE.duplicated(keep=False).sum()} แถว", "กลาง", "ใช้ OBJECTID เป็นคีย์แทน SITE_CODE"),
    ("ทะเบียน", "ชื่อ f_type / com_name สะกดหลายแบบ (เช่น ฟอแรมมินิเฟอร์/ฟอแรมินิเฟอรา)", f"{(ct_type > 1).sum()} com_name", "ต่ำ", "ทำตาราง mapping ชื่อกลุ่มซาก"),
    ("ไฟล์", "รหัสเสีย 0 นำหน้าเมื่ออ่าน CSV", "แก้แล้ว", "สูง", "อ่านด้วย dtype=str ทุกครั้ง"),
], columns=["ชุดข้อมูล", "ปัญหา", "ขนาด", "ความรุนแรง", "ข้อแนะนำ"])
summary
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (Project_Jurassic)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
