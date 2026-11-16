"""Dump every feature of an ArcGIS MapServer layer to GeoJSON (WGS84) + attribute CSV.

usage: arcgis_dump.py <layer_url> <out_dir> <basename>
"""
import csv
import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request

layer_url, out_dir, base = sys.argv[1:4]
CHUNK = 200


def get(params, retries=4):
    data = urllib.parse.urlencode(params).encode()
    for i in range(retries):
        try:
            with urllib.request.urlopen(layer_url + "/query", data=data, timeout=300) as r:
                d = json.load(r)
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:
            if i == retries - 1:
                raise
            print("  retry", i + 1, e, file=sys.stderr)
            time.sleep(5 * (i + 1))


os.makedirs(out_dir, exist_ok=True)
with urllib.request.urlopen(layer_url + "?f=json", timeout=120) as r:
    meta = json.load(r)
with open(os.path.join(out_dir, base + "_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
date_fields = {f["name"] for f in meta["fields"] if f["type"] == "esriFieldTypeDate"}

ids = sorted(get({"where": "1=1", "returnIdsOnly": "true", "f": "json"})["objectIds"])
print(base, "objects:", len(ids))

features = []
for i in range(0, len(ids), CHUNK):
    chunk = ids[i:i + CHUNK]
    d = get({"objectIds": ",".join(map(str, chunk)), "outFields": "*", "outSR": 4326,
             "returnGeometry": "true", "f": "geojson"})
    features.extend(d["features"])
    print(f"  {len(features)}/{len(ids)}", flush=True)

assert len(features) == len(ids), (len(features), len(ids))
for ft in features:
    p = ft["properties"]
    for c in date_fields:
        if p.get(c) is not None:
            p[c] = datetime.datetime.fromtimestamp(p[c] / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")

with open(os.path.join(out_dir, base + ".geojson"), "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

keys = list(features[0]["properties"].keys())
with open(os.path.join(out_dir, base + "_attributes.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(keys)
    for ft in features:
        w.writerow([ft["properties"].get(k) for k in keys])
print("done", base, len(features), "no-geometry:", sum(1 for f in features if not f.get("geometry")))
