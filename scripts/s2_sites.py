"""Clip the least-cloudy dry-season Sentinel-2 L2A scene around each fossil site,
plus a matching raster of the DMR geological-structure layer (its vector geometry is not served).

usage: s2_sites.py <fossil_sites.csv> <out_dir> [box_km=10] [workers=6]

Per site (named by OBJECTID):
  <id>_visual.tif     RGB true colour, uint8, 10 m
  <id>_bands.tif      blue, green, red, nir, swir16, swir22 at 10 m (20 m bands bilinear-upsampled),
                      uint16 DN; reflectance = DN * 0.0001 - 0.1
  <id>_structure.tif  RGBA rendering of DMR GEOL_STR_250K on the same grid
and a row in sites_index.csv.
"""
import concurrent.futures as cf
import csv
import io
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5",
                  GDAL_HTTP_RETRY_DELAY="3", GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="15",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

sites_csv, out_dir = sys.argv[1], sys.argv[2]
BOX_KM = float(sys.argv[3]) if len(sys.argv) > 3 else 10
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
RES = 10
DATES = "2024-11-01/2026-03-31"
SEASON_MONTHS = {11, 12, 1, 2, 3}
BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
CLOUDY_SCL = {0, 1, 3, 8, 9, 10}  # nodata, saturated, cloud shadow, cloud med/high, cirrus
STRUCT_EXPORT = "https://gisportal.dmr.go.th/arcgis/rest/services/GEOL/GEOL_STR_250K/MapServer/export"

catalog = Client.open("https://earth-search.aws.element84.com/v1", timeout=60)


def utm_epsg(lon):
    return 32647 if lon < 102 else 32648


def site_grid(lon, lat, epsg):
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = BOX_KM * 500
    x0, y1 = round((x - half) / RES) * RES, round((y + half) / RES) * RES
    n = int(BOX_KM * 1000 / RES)
    return (x0, y1 - n * RES, x0 + n * RES, y1), from_origin(x0, y1, RES, RES), n


def local_cloud(item, bounds):
    with rasterio.open(item.assets["scl"].href) as src:
        scl = src.read(1, window=from_bounds(*bounds, src.transform), boundless=True, fill_value=0)
    return float(np.isin(scl, list(CLOUDY_SCL)).mean())


def pick_scene(lon, lat):
    """Least locally-cloudy dry-season item whose footprint covers the whole site box."""
    epsg_guess = utm_epsg(lon)
    bounds, _, _ = site_grid(lon, lat, epsg_guess)
    to_ll = Transformer.from_crs(epsg_guess, 4326, always_xy=True).transform
    aoi = shp_transform(to_ll, box(*bounds))
    items = catalog.search(collections=["sentinel-2-c1-l2a"], intersects=aoi, datetime=DATES,
                           query={"eo:cloud_cover": {"lt": 40}}, max_items=200).item_collection()
    cands = [i for i in items if i.datetime.month in SEASON_MONTHS and shape(i.geometry).buffer(1e-4).contains(aoi)]
    cands.sort(key=lambda i: i.properties["eo:cloud_cover"])
    best = None
    for item in cands[:8]:
        epsg = item.properties["proj:epsg"] if "proj:epsg" in item.properties else int(item.properties["proj:code"].split(":")[1])
        b, _, _ = site_grid(lon, lat, epsg)
        cf_ = local_cloud(item, b)
        if best is None or cf_ < best[1]:
            best = (item, cf_, epsg)
        if cf_ < 0.01:
            break
    return best, len(cands)


def read_to_grid(href, bounds, n, resampling):
    with rasterio.open(href) as src:
        win = from_bounds(*bounds, src.transform)
        return src.read(window=win, out_shape=(src.count, n, n), resampling=resampling, boundless=True, fill_value=0)


def write(path, arr, transform, epsg, **tags):
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=arr.dtype,
                crs=f"EPSG:{epsg}", transform=transform, compress="deflate", predictor=2, tiled=True,
                blockxsize=256, blockysize=256, nodata=None if arr.shape[0] == 4 else 0)
    tmp = path + ".part"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, path)


def fetch_structure(bounds, n, epsg):
    q = urllib.parse.urlencode(dict(bbox=",".join(map(str, bounds)), bboxSR=epsg, imageSR=epsg, size=f"{n},{n}",
                                    format="png32", transparent="true", layers="show:0", dpi=96, f="image"))
    for i in range(4):
        try:
            with urllib.request.urlopen(f"{STRUCT_EXPORT}?{q}", timeout=180) as r:
                data = r.read()
            with rasterio.MemoryFile(io.BytesIO(data)) as mf, mf.open() as src:
                return src.read()
        except Exception:
            if i == 3:
                raise
            time.sleep(5 * (i + 1))


def existing_row(base, row):
    """Index row for a site finished earlier, from the GeoTIFF tags (backfilling cloud stats if absent)."""
    with rasterio.open(base + "_bands.tif") as src:
        tags, epsg = src.tags(), src.crs.to_epsg()
    if "site_cloud_pct" not in tags:
        item = next(catalog.search(collections=["sentinel-2-c1-l2a"], ids=[tags["stac_item"]]).items())
        bounds, _, _ = site_grid(row["longitude"], row["latitude"], epsg)
        tags["scene_cloud_pct"] = round(item.properties["eo:cloud_cover"], 2)
        tags["site_cloud_pct"] = round(local_cloud(item, bounds) * 100, 2)
        with rasterio.open(base + "_bands.tif", "r+") as dst:
            dst.update_tags(scene_cloud_pct=tags["scene_cloud_pct"], site_cloud_pct=tags["site_cloud_pct"])
    return row | dict(status="ok", stac_item=tags["stac_item"], date=tags["datetime"][:10], epsg=epsg,
                      scene_cloud_pct=tags["scene_cloud_pct"], site_cloud_pct=tags["site_cloud_pct"],
                      candidates=tags.get("candidates", ""))


def process(site):
    oid, lon, lat = site["OBJECTID"], float(site["longitude"]), float(site["latitude"])
    base = os.path.join(out_dir, oid)
    row = dict(OBJECTID=oid, SITE_CODE=site["SITE_CODE"], NAME_TH=site["NAME_TH"], PROVINCE=site["PROVINCE"],
               longitude=lon, latitude=lat)
    if os.path.exists(base + "_structure.tif"):
        return existing_row(base, row)
    best, ncand = pick_scene(lon, lat)
    if best is None:
        return row | dict(status="no_scene", candidates=ncand)
    item, cloud, epsg = best
    bounds, transform, n = site_grid(lon, lat, epsg)
    meta = dict(stac_item=item.id, datetime=item.datetime.isoformat(), site_objectid=oid, site_code=site["SITE_CODE"],
                scene_cloud_pct=round(item.properties["eo:cloud_cover"], 2), site_cloud_pct=round(cloud * 100, 2),
                candidates=ncand)

    vis = read_to_grid(item.assets["visual"].href, bounds, n, Resampling.nearest)
    write(base + "_visual.tif", vis, transform, epsg, **meta)
    bands = np.concatenate([read_to_grid(item.assets[b].href, bounds, n, Resampling.bilinear) for b in BANDS])
    write(base + "_bands.tif", bands, transform, epsg, band_names=",".join(BANDS),
          reflectance="DN*0.0001-0.1", **meta)
    with rasterio.open(base + "_bands.tif", "r+") as dst:
        dst.descriptions = tuple(BANDS)
    write(base + "_structure.tif", fetch_structure(bounds, n, epsg), transform, epsg,
          source=STRUCT_EXPORT, **meta)
    return row | dict(status="ok", stac_item=item.id, date=item.datetime.date().isoformat(), epsg=epsg,
                      scene_cloud_pct=round(item.properties["eo:cloud_cover"], 2),
                      site_cloud_pct=round(cloud * 100, 2), candidates=ncand)


def main():
    os.makedirs(out_dir, exist_ok=True)
    with open(sites_csv, encoding="utf-8-sig") as f:
        sites = list(csv.DictReader(f))
    rows, done = [], 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(process, s): s for s in sites}
        for f in cf.as_completed(futs):
            s = futs[f]
            try:
                r = f.result()
            except Exception as e:
                r = dict(OBJECTID=s["OBJECTID"], SITE_CODE=s["SITE_CODE"], NAME_TH=s["NAME_TH"], PROVINCE=s["PROVINCE"],
                         longitude=s["longitude"], latitude=s["latitude"], status=f"error: {type(e).__name__}: {e}"[:300])
            rows.append(r)
            done += 1
            print(f"{done}/{len(sites)} {r['OBJECTID']} {r['status']} {r.get('date', '')} cloud={r.get('site_cloud_pct', '')}", flush=True)
    keys = ["OBJECTID", "SITE_CODE", "NAME_TH", "PROVINCE", "longitude", "latitude", "status", "stac_item", "date",
            "epsg", "scene_cloud_pct", "site_cloud_pct", "candidates"]
    with open(os.path.join(out_dir, "sites_index.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: int(r["OBJECTID"])))


if __name__ == "__main__":
    main()
