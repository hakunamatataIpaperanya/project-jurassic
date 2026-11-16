# Download DMR ArcGIS MapServer layers (all features, paged by objectIds) as GeoJSON in WGS84.
# usage: .venv/bin/python scripts/fetch_dmr_layers.py
import json
import sys
import time
import urllib.parse
from pathlib import Path

import geopandas as gpd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://gisportal.dmr.go.th/arcgis/rest/services"
LAYERS = {                                   # output name -> (service, layer id)
    "geol_structure_250k": ("GEOL/GEOL_STR_250K", 0),     # รอยเลื่อน / ชั้นหินโค้ง / รอยแตก
    "karst_systems": ("GEOL/KARST_SYST", 0),              # ขอบเขตระบบถ้ำ
    "mineral_potential_50k": ("MINERAL/MIN_AREA", 0),     # พื้นที่ศักยภาพแร่ (กลุ่มการค้นพบ: เหมือง/บ่อหิน)
}


def get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.post(url, data=params, timeout=180)          # POST: รายการ objectIds ยาวเกิน URL
            r.raise_for_status()
            return r.json()
        except Exception as e:                # noqa: BLE001
            if i == tries - 1:
                raise
            print("  retry", i + 1, e)
            time.sleep(5 * (i + 1))


def fetch(service, layer):
    url = f"{BASE}/{urllib.parse.quote(service)}/MapServer/{layer}/query"
    ids = sorted(get(url, {"where": "1=1", "returnIdsOnly": "true", "f": "json"})["objectIds"])
    feats = []
    for i in range(0, len(ids), 300):
        chunk = ids[i:i + 300]
        d = get(url, {"objectIds": ",".join(map(str, chunk)), "outFields": "*", "outSR": 4326, "f": "geojson"})
        feats += d["features"]
        print(f"  {service}: {len(feats)}/{len(ids)}", end="\r")
    print()
    return gpd.GeoDataFrame.from_features(feats, crs=4326)


if __name__ == "__main__":
    out = ROOT / "data" / "dmr_geology"
    names = sys.argv[1:] or list(LAYERS)
    for name in names:
        path = out / f"{name}.geojson"
        if path.exists():
            print("มีแล้ว:", path.name)
            continue
        g = fetch(*LAYERS[name])
        g.columns = [c.split(".")[-1] for c in g.columns]          # ตัด prefix ฐานข้อมูล (dmrdb2.DMRAPP...)
        g.to_file(path, driver="GeoJSON")
        (out / f"{name}_metadata.json").write_text(json.dumps(
            {"source": f"{BASE}/{LAYERS[name][0]}/MapServer/{LAYERS[name][1]}", "n_features": len(g),
             "fetched": time.strftime("%Y-%m-%d")}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("บันทึก", path.name, len(g), "features")
