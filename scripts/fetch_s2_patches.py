# Bulk-fetch Sentinel-2 patches for every point in data/features/points.parquet (resumable).
# usage: .venv/bin/python scripts/fetch_s2_patches.py [limit] [points.parquet] [patch_subdir]
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import feature_lib as fl  # noqa: E402
import pandas as pd  # noqa: E402

limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "0" else None
points_file = Path(sys.argv[2]) if len(sys.argv) > 2 else fl.FEAT / "points.parquet"
sub = sys.argv[3] if len(sys.argv) > 3 else "patches"
S2DIR = fl.FEAT / "s2"
S2DIR.mkdir(parents=True, exist_ok=True)
points = pd.read_parquet(points_file)
if limit:
    points = points.sample(limit, random_state=0)

items_file = S2DIR / "items.parquet"
if items_file.exists():
    items = pd.read_parquet(items_file)
else:
    x0, y0, x1, y1 = fl.THAI_BBOX
    bbox = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    items = fl.search_items(bbox)
    items.to_parquet(items_file, index=False)
print("items:", len(items), flush=True)

asg = fl.scene_assignment(points, items, cache_file=S2DIR / ("assignment.parquet" if sub == "patches" else f"assignment_{sub}.parquet"), workers=12)
print("assigned:", asg.item_id.notna().sum(), "| scenes:", asg.item_id.nunique(), "| cloud_frac>0.05:", (asg.cloud_frac > 0.05).sum(), flush=True)
n = fl.fetch_patches_batch(points, items, asg, S2DIR / sub, workers=12)
print("fetched:", n, flush=True)
