# Build notebooks/09_features_satellite.ipynb and regenerate notebooks/feature_lib.py from scripts/lib_sections/.
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_feature_lib import SECTIONS, write_lib  # noqa: E402

out = sys.argv[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


write_lib()
SECTION_TITLE = {
    "00_common.py": "ค่าคงที่และ path", "10_pangea.py": "พันเจีย (ใช้ตอนทำนายจุดใหม่ ตรรกะเดียวกับ notebook 08)",
    "20_dem.py": "ภูมิประเทศจาก DEM", "30_geology.py": "ธรณีวิทยา: หน่วยหิน, ถ้ำ, หินโผล่",
    "40_s2.py": "Sentinel-2", "50_batch.py": "ตัวรันแบบชุด (cache + รันต่อได้)", "90_build.py": "FeatureBuilder สำหรับขั้นทำนาย",
}

md(r"""
# ขั้นที่ 2: Feature ภูมิประเทศ ธรณีวิทยา และดาวเทียม

ต่อจากขั้นที่ 1 (พันเจีย) ใช้ชุดจุดเดียวกัน (`data/features/points.parquet`: แหล่ง ทธ. 518, PBDB 290, background 2 ชุด × 5,000)

| กลุ่ม | ที่มา | feature | เหตุผล (อ้างอิง) |
|---|---|---|---|
| `dem_` | Copernicus DEM GLO-30 | ความสูง, ความลาดชัน, TPI, ความขรุขระ ที่จุด / 1 กม. / 5 กม. | ความลาดชัน = การกัดเซาะที่ทำให้ซากโผล่ (Block et al. 2016, discovery model) |
| `geo_` | แผนที่หน่วยหิน 1:250,000 + ถ้ำ (ทธ.) | หน่วยหิน/ยุค ณ จุด, สัดส่วนยุคหินในรัศมี 5 กม., ระยะถึงหินแต่ละยุค, ระยะถึงถ้ำ | ซากอยู่ในหินตะกอนยุคที่ถูกต้อง (Block 2016 preservation model); ถ้ำ = หินปูน/ซากควอเทอร์นารี |
| `oc_` | หินโผล่ (ทธ.) | ระยะถึงหินโผล่/หินคาร์บอเนตโผล่, ชนิดหินโผล่ที่ใกล้สุด, อยู่ในจังหวัดที่สำรวจไหม | **แยกกลุ่ม** เพราะมาจากการสำรวจแค่ ~10 จังหวัด (ทดสอบในขั้นโมเดลว่าช่วยจริงหรือแค่ความลำเอียง) |
| `s2_` | Sentinel-2 L2A 20 ม., patch 32×32 px (640 ม.) | ค่าสะท้อน 6 แบนด์, NDVI, ดินโล่ง, SWIR ratio (ดินเหนียว/คาร์บอเนต), iron oxide | สเปกตรัมของแหล่งฟอสซิล (Emerson et al. 2015 ใช้ Landsat แบนด์ 2–7 รวม SWIR) |

**คำตอบจากการถามตอบ**: DEM หลายระดับ (จุด + 1 + 5 กม.), ธรณีวิทยาทั้ง 4 แบบ, หินโผล่ใช้แต่แยกกลุ่ม,
Sentinel-2 ความละเอียด 20 ม. ครบทุกจุด, โค้ด feature อยู่ใน notebook นี้และถูกเขียนออกเป็น `notebooks/feature_lib.py` สำหรับขั้นทำนาย

**แหล่งภาพ Sentinel-2**: Microsoft Planetary Computer (`sentinel-2-l2a`) เพราะ Earth Search (AWS สหรัฐฯ) ดาวน์โหลดได้ต่ำกว่า 0.2 MB/s จากเครือข่ายนี้
ใช้กติกาเลือกภาพเดียวกันทั้งตอน train และตอนทำนาย: ภาพหน้าแล้ง (พ.ย.–มี.ค. 2024/25–2025/26) เมฆทั้งภาพ < 40%
เรียงตามเมฆน้อย→มาก เลือกภาพแรกที่ patch มีเมฆ/เงาเมฆ ≤ 5% (ตรวจจากชั้น SCL)

**รันต่อเมื่อหยุดกลางทาง**: ทุกขั้นบันทึก cache ทีละ chunk (`data/features/sat_cache/`), patch Sentinel-2 บันทึกทีละจุด (`data/features/s2/patches/`)
รันซ้ำจะข้ามส่วนที่เสร็จแล้ว การดึงภาพครั้งแรกใช้สคริปต์ `scripts/fetch_s2_patches.py` (เรียกฟังก์ชันชุดเดียวกับ notebook นี้)
""")

code(r"""
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score

_installed = {f.name for f in font_manager.fontManager.ttflist}
THAI_FONT = next((f for f in ["Tahoma", "Thonburi", "Noto Sans Thai", "Leelawadee UI", "Loma", "Garuda"] if f in _installed), None)
plt.rcParams.update({
    "font.family": [THAI_FONT, "DejaVu Sans"] if THAI_FONT else ["DejaVu Sans"], "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
})
INK, MUTED = "#52514e", "#8a8984"
SOURCE_COLOR = {"dmr_site": "#2a78d6", "pbdb": "#eb6834", "bg_uniform": "#bdbcb6", "bg_bias": "#1baf7a"}
GROUP_COLOR = {"pg_": "#4a3aa7", "dem_": "#eb6834", "geo_": "#1baf7a", "oc_": "#eda100", "s2_": "#2a78d6"}
""")

md(r"""
## 1. โค้ดคำนวณ feature (`feature_lib`)

เซลล์ด้านล่างคือโค้ดชุดเดียวกับไฟล์ `notebooks/feature_lib.py` ทุกตัวอักษร (สร้างจาก `scripts/lib_sections/*.py`)
แก้ที่ `scripts/lib_sections/` แล้วรัน `scripts/run_satellite_nb.sh` ทั้ง notebook และไฟล์ .py จะอัปเดตพร้อมกัน
""")

for p in SECTIONS:
    md(f"### `{p.name}`: {SECTION_TITLE.get(p.name, '')}")
    code(p.read_text(encoding="utf-8"))

md("## 2. ชุดจุด")

code(r"""
points = pd.read_parquet(FEAT / "points.parquet")
print(f"{len(points):,} จุด", points.source.value_counts().to_dict())
""")

md(r"""
## 3. ภูมิประเทศ (DEM)

อ่านหน้าต่าง ±5.2 กม. รอบจุดจาก tile ที่เกี่ยวข้อง (รวม tile อัตโนมัติเมื่อจุดอยู่ใกล้ขอบ) แล้วคำนวณความลาดชันเป็นองศา
(ระยะพิกเซลแปลงเป็นเมตรตามละติจูด) TPI = ความสูงของจุด − ความสูงเฉลี่ยรอบข้าง (บวก = สันเขา, ลบ = หุบ)
""")

code(r"""
t0 = time.time()
dem_df = batch_dem(points)
print(f"DEM {dem_df.shape} ใน {time.time() - t0:.0f} วินาที | ค่าว่าง: {int(dem_df.isna().any(axis=1).sum())} จุด")
dem_df.describe().T[["mean", "50%", "min", "max"]].round(1)
""")

md("## 4. ธรณีวิทยา")

code(r"""
t0 = time.time()
geo_df = batch_geology(points)
print(f"ธรณีวิทยา {geo_df.shape} ใน {time.time() - t0:.0f} วินาที")
d = geo_df.merge(points[["point_id", "source"]], on="point_id")
display(pd.crosstab(d.geo_era_old, d.source, normalize="columns").mul(100).round(1).loc[lambda t: t.max(axis=1) > 1])
print("ถ้ำใกล้สุด (กม., มัธยฐาน):", d.groupby("source").geo_cave_dist_km.median().round(1).to_dict())
print("อยู่ในจังหวัดที่มีการสำรวจหินโผล่ (%):", d.groupby("source").oc_in_survey_prov.mean().mul(100).round(1).to_dict())
""")

md(r"""
## 5. Sentinel-2

ใช้รายการภาพและการเลือกภาพที่ cache ไว้จากการดึงครั้งแรก (`scripts/fetch_s2_patches.py`) แล้วดึงเฉพาะจุดที่ยังไม่มี patch
""")

code(r"""
S2DIR = FEAT / "s2"
items_file = S2DIR / "items.parquet"
if items_file.exists():
    items = pd.read_parquet(items_file)
else:
    x0, y0, x1, y1 = THAI_BBOX
    items = search_items({"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})
    items.to_parquet(items_file, index=False)
asg = scene_assignment(points, items, cache_file=S2DIR / "assignment.parquet", workers=12)
n_new = fetch_patches_batch(points, items, asg, S2DIR / "patches", workers=12)
print(f"ภาพหน้าแล้งที่เป็นตัวเลือก {len(items):,} ภาพ | ใช้จริง {asg.item_id.nunique()} ภาพ | ดึงเพิ่มรอบนี้ {n_new} จุด")
print(f"จุดที่หา patch ไร้เมฆ (≤5%) ไม่ได้: {(asg.cloud_frac > S2_CLEAR_OK).sum()} จุด")
s2_df = batch_s2_indices(points, S2DIR / "patches")
print("ดัชนี Sentinel-2:", s2_df.shape, "| เดือนของภาพ:", pd.to_datetime(s2_df.s2_date).dt.month.value_counts().sort_index().to_dict())
""")

code(r"""
def show_patches(ids, title, ax_row):
    for ax, pid in zip(ax_row, ids):
        z = np.load(S2DIR / "patches" / f"{pid}.npz")["patch"].astype(float) * S2_SCALE + S2_OFFSET
        rgb = np.stack([z[5], z[3], z[1]], axis=-1)           # SWIR2 / NIR / green: แยกหิน-ดิน-พืช
        rgb = np.clip((rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2) + 1e-6), 0, 1)
        ax.imshow(rgb)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    ax_row[0].set_ylabel(title, fontsize=9)


rng = np.random.default_rng(0)
have = set(s2_df.point_id)
fig, axes = plt.subplots(3, 8, figsize=(16, 6.3))
for row, (src, title) in enumerate([("dmr_site", "แหล่ง ทธ."), ("bg_bias", "bg_bias"), ("bg_uniform", "bg_uniform")]):
    ids = rng.choice([p for p in points.loc[points.source.eq(src), "point_id"] if p in have], 8, replace=False)
    show_patches(ids, title, axes[row])
fig.suptitle("ตัวอย่าง patch 640 ม. (สีผสม SWIR2-NIR-เขียว: เขียว = พืช, ชมพู/น้ำตาล = ดิน/หินโล่ง)", x=0.01, ha="left", fontweight="bold")
plt.tight_layout()
""")

md(r"""
## 6. ตรวจว่าโค้ดพันเจียใน `feature_lib` ให้ค่าเดียวกับ notebook 08

ขั้นทำนายจะคำนวณ feature พันเจียของพิกัดใหม่ด้วย `Pangea` ใน `feature_lib` จึงต้องได้ค่าเดียวกับที่ใช้ train ทุกค่า
""")

code(r"""
ref = pd.read_parquet(FEAT / "pangea_features.parquet")
sample = points.sample(60, random_state=7)
pf = Pangea().features(sample.lon, sample.lat)
r = ref.set_index("point_id").loc[sample.point_id, pf.columns].reset_index(drop=True)
num = [c for c in pf.columns if pd.api.types.is_numeric_dtype(pf[c])]
max_diff = float((pf[num].astype(float) - r[num].astype(float)).abs().max().max())
same_cat = bool((pf.pg_me21_terrane.fillna("") == r.pg_me21_terrane.fillna("")).all())
print(f"ต่างกันสูงสุด {max_diff:.2e} | terrane ตรงกันทุกจุด: {same_cat}")
assert max_diff < 1e-6 and same_cat
""")

md("## 7. รวมทุกกลุ่ม feature")

code(r"""
pg = ref.drop(columns=["source", "lon", "lat", "ref_id"])
allf = (points.merge(pg, on="point_id", how="left").merge(dem_df, on="point_id", how="left")
        .merge(geo_df, on="point_id", how="left").merge(s2_df, on="point_id", how="left"))
groups = {g: [c for c in allf.columns if c.startswith(p)] for g, p in FEATURE_GROUPS.items()}
cat_cols = [c for c in allf.columns if c[:3] in ("pg_", "geo", "oc_", "s2_", "dem") and not pd.api.types.is_numeric_dtype(allf[c])]
print({g: len(c) for g, c in groups.items()}, "| หมวดหมู่:", cat_cols)
miss = allf[[c for cs in groups.values() for c in cs]].isna().mean().mul(100)
print("คอลัมน์ที่มีค่าว่าง > 0%:", miss[miss > 0].round(2).to_dict())
allf.to_parquet(FEAT / "all_features.parquet", index=False)
(FEAT / "feature_groups.json").write_text(json.dumps({"groups": groups, "categorical": cat_cols}, ensure_ascii=False, indent=2), encoding="utf-8")
print("บันทึก all_features.parquet", allf.shape)
""")

md(r"""
## 8. feature ไหนแยกแหล่งจริงออกจาก background ได้

วัดแต่ละ feature แยกกันด้วย AUC (0.5 = แยกไม่ได้, ห่างจาก 0.5 มาก = แยกได้ดี) เทียบแหล่ง ทธ. กับ background ทั้งสองแบบ
ถ้า feature แยกได้ดีกับ `bg_uniform` แต่ไม่ดีกับ `bg_bias` แปลว่ามันจับ "พื้นที่ที่คนไปสำรวจ" มากกว่าความต่างของพื้นที่จริง
""")

code(r"""
num_cols = [c for cs in groups.values() for c in cs if pd.api.types.is_numeric_dtype(allf[c]) and allf[c].nunique() > 2
            and not c.endswith(("_plate", "_begin_ma")) and not re.search(r"_lon_\d+$", c)]
rows = []
for bg in ["bg_uniform", "bg_bias"]:
    d = allf[allf.source.isin(["dmr_site", bg])]
    y = d.source.eq("dmr_site").astype(int)
    for c in num_cols:
        m = d[c].notna()
        if m.sum() > 50 and y[m].nunique() == 2:
            rows.append({"background": bg, "feature": c, "auc": roc_auc_score(y[m], d.loc[m, c])})
auc = pd.DataFrame(rows)
auc["strength"] = (auc.auc - 0.5).abs()
wide = auc.pivot(index="feature", columns="background", values="strength")
wide["group"] = [next(p for p in GROUP_COLOR if f.startswith(p)) for f in wide.index]

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
for ax, bg in zip(axes, ["bg_uniform", "bg_bias"]):
    top = wide.sort_values(bg, ascending=False).head(25)[::-1]
    ax.barh(top.index, top[bg] + 0.5, left=0, color=[GROUP_COLOR[g] for g in top.group], height=0.7)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_xlim(0.45, max(0.85, top[bg].max() + 0.55))
    ax.set_title(f"แหล่ง ทธ. vs {bg}: 25 feature ที่แยกได้ดีสุด (max(AUC, 1−AUC))")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
axes[1].legend(handles, list(GROUP_COLOR), frameon=False, loc="lower right", title="กลุ่ม")
plt.tight_layout()
display(wide.groupby("group")[["bg_uniform", "bg_bias"]].max().add(0.5).round(3).rename(columns=lambda c: f"AUC สูงสุด vs {c}"))
""")

code(r"""
key = ["dem_slope_mean_1km", "dem_relief_5km", "geo_cave_dist_km", "s2_bare_frac", "s2_clay_ratio", "oc_dist_km"]
fig, axes = plt.subplots(1, len(key), figsize=(18, 3.6))
for ax, c in zip(axes, key):
    data = [allf.loc[allf.source.eq(s), c].dropna() for s in ["dmr_site", "pbdb", "bg_bias", "bg_uniform"]]
    lo, hi = np.nanpercentile(allf[c], [1, 99])
    for s, v in zip(["dmr_site", "pbdb", "bg_bias", "bg_uniform"], data):
        ax.hist(v.clip(lo, hi), bins=30, density=True, histtype="step", lw=1.8, color=SOURCE_COLOR[s], label=s)
    ax.set_title(c, fontsize=9)
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False, fontsize=7)
plt.tight_layout()
""")

md(r"""
## 9. สรุปขั้นที่ 2

**ข้อมูล**: feature ครบทุกจุด (10,808) — DEM ไม่มีค่าว่าง, Sentinel-2 ได้ภาพครบทุกจุด (ใช้ 188 ภาพ ส่วนใหญ่ ธ.ค.–ม.ค.),
มี 14 จุดที่หา patch เมฆ ≤ 5% ไม่ได้ (ใช้ patch ที่เมฆน้อยที่สุดแทน) และโค้ดพันเจียใน `feature_lib` ให้ค่าตรงกับ notebook 08 ทุกค่า

**feature ที่แยกแหล่งจริงออกจาก background ได้ดีสุด (ดูทีละตัว)**
- **ระยะถึงถ้ำ** (`geo_cave_dist_km`): AUC 0.80 vs `bg_uniform`, 0.72 vs `bg_bias` — แหล่งซากส่วนหนึ่งเป็นถ้ำในหินปูน
  **ข้อควรระวัง**: ทั้งรายชื่อแหล่งและข้อมูลถ้ำมาจากการสำรวจของ ทธ. ความสัมพันธ์นี้จึงอาจมีความลำเอียงของพื้นที่สำรวจปนอยู่
- **ความลาดชัน** (`dem_slope`, slope ในรัศมี 1 กม.): แหล่งอยู่บนพื้นลาดปานกลาง (5–15°) มากกว่าที่ราบลุ่ม
  ตรงกับ Block et al. 2016 ที่ใช้ slope เป็นตัวแทนการกัดเซาะที่ทำให้ซากโผล่
- **Sentinel-2**: ความแปรปรวนของพืชคลุมดินในพื้นที่ (`s2_ndvi_std`), ค่าสะท้อน NIR/SWIR ที่จุด และ SWIR1/SWIR2 ratio ช่วยได้ระดับปานกลาง
- **พันเจีย**: ช่วยระดับภูมิภาค (AUC ~0.6 กับ `bg_uniform`) แต่แทบไม่ช่วยกับ `bg_bias` ตามที่คาดในขั้นที่ 1 (เป็นข้อมูลระดับภูมิภาค)
- **หินโผล่** (`oc_`): AUC เพียง 0.53–0.57 และสัดส่วนจุดในจังหวัดที่สำรวจเท่ากันทุกกลุ่ม (33–41%) → แทบไม่มีข้อมูล ใช้ทดสอบ ablation ในขั้นโมเดลเท่านั้น

**`bg_bias` ทำให้งานยากขึ้นทุกกลุ่ม feature** (AUC สูงสุดลดลง) ซึ่งเป็นสิ่งที่ตั้งใจ: background อยู่ในพื้นที่ที่มีการสำรวจเหมือนแหล่ง
โมเดลต้องแยกด้วยลักษณะพื้นที่จริง ไม่ใช่ด้วย "ภูมิภาคที่คนไปสำรวจ"

**ข้อสังเกต**: background บางจุดตกในแหล่งน้ำ (ภาพสีน้ำเงิน) เพราะขอบเขตจังหวัดรวมพื้นที่น้ำด้วย ทำให้โมเดลเรียนว่า "น้ำ = ไม่มีซาก" ซึ่งถูกต้องตามจริง
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (Project_Jurassic)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
