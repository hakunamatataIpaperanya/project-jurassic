# Build the national 0.002° rasters used by feature_lib.extra_features (cached in data/features/extra_cache/).
# usage: .venv/bin/python scripts/build_extra_layers.py [builder ...]   (default: all except terraclimate)
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import feature_lib as fl  # noqa: E402

BUILDERS = {"structure": fl.build_structure, "rock_contacts": fl.build_rock_contacts, "osm": fl.build_osm,
            "mineral": fl.build_mineral, "worldcover": fl.build_worldcover, "gsw": fl.build_gsw,
            "lithology": fl.rock_unit_lithology, "terraclimate": fl.build_terraclimate}

if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in BUILDERS if n != "terraclimate"]
    for n in names:
        t = time.time()
        try:
            BUILDERS[n]()
            print(f"OK {n} ({time.time() - t:.0f} s)", flush=True)
        except Exception:                      # noqa: BLE001  ทำ builder อื่นต่อ แล้วรายงานรวม
            print(f"FAILED {n}", flush=True)
            traceback.print_exc()
