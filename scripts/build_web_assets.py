# Background images for the web app (webapp/static/img/), drawn from Project_Jurassic data only (no outside artwork):
#   hero.png      potential map of Thailand (task 2, bg_bias rank)        -> หน้าแรก
#   p_map.png     zoom on the north-east (dinosaur country)              -> แผง "วิธีอ่านผล"
#   p_roc.png     ROC curve of stage A (spatial_block out-of-fold)       -> แผง "ความแม่น"
#   p_data.png    DMR sites + PBDB collections                           -> แผง "แหล่งข้อมูล"
#   p_rock.png    rock age of every 5-km cell (geology map)              -> แผง "พบซากต้องทำอย่างไร"
# Run: .venv/bin/python scripts/build_web_assets.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import model_lib as ml  # noqa: E402
import feature_lib as fl  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

OUT = ROOT / "webapp" / "static" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BG, CYAN = "#07090b", "#18d1ff"
CMAP = LinearSegmentedColormap.from_list("ak", ["#0a1418", "#0f3a4a", "#12708f", "#18b4e0", "#b8f3ff"])
COSLAT = np.cos(np.radians(13.0))

grid = pd.read_parquet(ml.MODELS / "task2" / "predictions" / "grid_5km_predictions.parquet")


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("wrote", OUT / name, f"{(OUT / name).stat().st_size / 1024:.0f} KB")


def thai_frame(ax, lon=(97.2, 105.8), lat=(5.5, 20.6)):
    ax.set_xlim(lon[0] * COSLAT, lon[1] * COSLAT)
    ax.set_ylim(*lat)
    ax.set_aspect("equal")


# ---- hero: potential map of the whole country
fig, ax = canvas(900, 1500)
ax.scatter(grid.lon * COSLAT, grid.lat, c=grid.rank_bg_bias, cmap=CMAP, s=9, marker="s", linewidths=0)
thai_frame(ax)
save(fig, "hero.png")

# ---- zoom on the north-east (Khorat plateau)
fig, ax = canvas(700, 1300)
ax.scatter(grid.lon * COSLAT, grid.lat, c=grid.rank_bg_bias, cmap=CMAP, s=60, marker="s", linewidths=0)
thai_frame(ax, lon=(101.4, 104.4), lat=(14.6, 18.2))
save(fig, "p_map.png")

# ---- ROC curve of stage A (out-of-fold, spatial_block, XGBoost) with HUD grid
o = ml.Registry(ml.MODELS / "task2" / "A_bg_bias").oof("spatial_block__xgb")
fpr, tpr, _ = roc_curve(o.y, o.p)
fig, ax = canvas(700, 1300)
ax.set_axis_on()
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.3, 1.3)
for v in np.linspace(0, 1, 11):
    ax.axhline(v, color="#ffffff", lw=0.4, alpha=0.08)
    ax.axvline(v, color="#ffffff", lw=0.4, alpha=0.08)
ax.plot([0, 1], [0, 1], color="#ffffff", lw=1, alpha=0.25, ls="--")
ax.fill_between(fpr, tpr, color=CYAN, alpha=0.12)
ax.plot(fpr, tpr, color=CYAN, lw=3)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([]), ax.set_yticks([])
save(fig, "p_roc.png")

# ---- data: DMR sites (white) + PBDB (cyan) on a dim country
pts = pd.read_parquet(ml.FEAT / "points.parquet")
fig, ax = canvas(700, 1300)
ax.scatter(grid.lon * COSLAT, grid.lat, color="#1b252b", s=10, marker="s", linewidths=0)
d, p = pts[pts.source.eq("dmr_site")], pts[pts.source.eq("pbdb")]
ax.scatter(p.lon * COSLAT, p.lat, color=CYAN, s=26, alpha=0.9, linewidths=0)
ax.scatter(d.lon * COSLAT, d.lat, color="#ffffff", s=14, alpha=0.85, linewidths=0)
thai_frame(ax)
save(fig, "p_data.png")

# ---- rock age (older = darker blue, younger = warm) of every grid cell
geo = fl.batch_geology(grid[["point_id", "lon", "lat"]], tag="grid_")
g = grid[["point_id", "lon", "lat"]].merge(geo[["point_id", "geo_era_old"]], on="point_id", how="left")
order = {p: i for i, p in enumerate(fl.PERIOD_ORDER)}
r = g.geo_era_old.map(order)
cmap = LinearSegmentedColormap.from_list("age", ["#123a52", "#1c6f8f", "#3aa7c4", "#9fd8c9", "#eeb763", "#f08a4b"])
fig, ax = canvas(700, 1300)
ax.scatter(g.lon[r.isna()] * COSLAT, g.lat[r.isna()], color="#1b252b", s=10, marker="s", linewidths=0)
ax.scatter(g.lon[r.notna()] * COSLAT, g.lat[r.notna()], c=r[r.notna()], cmap=cmap, s=10, marker="s", linewidths=0)
thai_frame(ax)
save(fig, "p_rock.png")
