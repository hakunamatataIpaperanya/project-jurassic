# Build the 5 km grid (if missing) and pre-fill its feature caches: pangea / geology / DEM (Sentinel-2 via fetch_s2_patches.py).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import feature_lib as fl  # noqa: E402
import pandas as pd  # noqa: E402

GRID = fl.FEAT / "grid_5km.parquet"
if not GRID.exists():                       # กริด 5 กม. บนแผ่นดิน (ขอบเขตจังหวัด ทธ.) 20,706 จุด — make_grid ให้ผลเดิมทุกครั้ง
    fl.make_grid(5).to_parquet(GRID, index=False)
grid = pd.read_parquet(GRID)
print("pangea", fl.batch_pangea(grid, tag="grid_").shape, flush=True)
print("geology", fl.batch_geology(grid, tag="grid_").shape, flush=True)
print("dem", fl.batch_dem(grid, tag="grid_").shape, flush=True)
