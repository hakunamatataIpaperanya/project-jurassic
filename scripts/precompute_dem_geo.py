# Pre-fill the DEM / geology caches used by notebook 09 (same functions, same cache files).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import feature_lib as fl  # noqa: E402
import pandas as pd  # noqa: E402

points = pd.read_parquet(fl.FEAT / "points.parquet")
print("geology", fl.batch_geology(points).shape, flush=True)
print("dem", fl.batch_dem(points).shape, flush=True)
