# Sentinel-2 L2A patches (20 m, 32x32 px = 640 m) from Microsoft Planetary Computer.
# (Earth Search on AWS us-west-2 was < 0.2 MB/s from this network; Planetary Computer serves the same L2A COGs ~20x faster.)
#
# Scene rule (identical for training and inference):
#   candidates = dry-season (Nov-Mar, 2024-11..2026-03) items with eo:cloud_cover < 40 whose footprint contains the point,
#   ordered by (eo:cloud_cover, item id); take the first whose SCL patch is <= 5 % cloud/shadow/no-data;
#   if none qualifies within the first 8 candidates, take the least cloudy patch among them.
import planetary_computer
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Point, shape

S2_DATES = "2024-11-01/2026-03-31"
S2_MONTHS = (11, 12, 1, 2, 3)
S2_MAX_SCENE_CLOUD = 40
S2_MAX_CANDIDATES = 8
S2_CLEAR_OK = 0.05
S2_PATCH = 32                        # pixels at 20 m
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = [("B02", True), ("B03", True), ("B04", True), ("B8A", False), ("B11", False), ("B12", False)]
S2_BAND_NAMES = [b for b, _ in S2_BANDS]   # (asset, read 10 m band from its 2x overview = 20 m)
S2_SCL = "SCL"
S2_CLOUDY_SCL = [0, 1, 3, 8, 9, 10]  # no-data, saturated, cloud shadow, cloud medium/high, cirrus
S2_SCALE, S2_OFFSET = 0.0001, -0.1   # reflectance = DN * scale + offset (processing baseline >= 04.00, i.e. all scenes since 2022)

_transformers = {}


def _to_crs(crs, lon, lat):
    key = str(crs)
    if key not in _transformers:
        _transformers[key] = Transformer.from_crs(4326, crs, always_xy=True)
    return _transformers[key].transform(lon, lat)


def _window(ds, lon, lat):
    x, y = _to_crs(ds.crs, lon, lat)
    r, c = ds.index(x, y)
    return Window(c - S2_PATCH // 2, r - S2_PATCH // 2, S2_PATCH, S2_PATCH)


def read_patch(ds, lon, lat):
    return ds.read(1, window=_window(ds, lon, lat), boundless=True, fill_value=0)


def open_band(item_assets, asset, overview):
    # hrefs are stored unsigned; sign at open time (SAS tokens expire, the library caches/refreshes them)
    return rasterio.open(planetary_computer.sign(item_assets[asset]), overview_level=0 if overview else None)


def search_items(geometry, max_items=None):
    # dry-season candidate items intersecting a geometry (dict GeoJSON), as a DataFrame
    from pystac_client import Client
    cat = Client.open(PC_STAC, timeout=120)
    rows = []
    for it in cat.search(collections=["sentinel-2-l2a"], intersects=geometry, datetime=S2_DATES,
                         query={"eo:cloud_cover": {"lt": S2_MAX_SCENE_CLOUD}}, max_items=max_items, limit=500).items():
        if it.datetime.month not in S2_MONTHS:
            continue
        rows.append({"id": it.id, "datetime": it.datetime.isoformat(), "cloud": it.properties["eo:cloud_cover"],
                     "geometry": json.dumps(it.geometry),
                     "baseline": it.properties.get("s2:processing_baseline"),
                     "assets": json.dumps({k: it.assets[k].href for k in S2_BAND_NAMES + [S2_SCL]})})
    return pd.DataFrame(rows)


def candidate_lists(points, items):
    # per point: candidate item ids ordered by (cloud, id) -> list
    from shapely import STRtree
    geoms = [shape(json.loads(g)) for g in items.geometry]
    tree = STRtree(geoms)
    order = items.assign(_k=range(len(items))).sort_values(["cloud", "id"])
    rank = dict(zip(order._k, range(len(order))))
    out = []
    for lon, lat in points[["lon", "lat"]].itertuples(index=False):
        hit = [int(i) for i in tree.query(Point(lon, lat), predicate="within")]
        hit = sorted(hit, key=rank.get)[:S2_MAX_CANDIDATES]
        out.append([items.id.iat[i] for i in hit])
    return out


def scene_assignment(points, items, cache_file=None, workers=8):
    # choose one scene per point using SCL (whole SCL file per scene, read once): returns DataFrame point_id -> item id
    tqdm = tqdm_auto()
    if cache_file is not None and Path(cache_file).exists():
        done = pd.read_parquet(cache_file)
    else:
        done = pd.DataFrame(columns=["point_id", "item_id", "cloud_frac", "rank"])
    todo = points[~points.point_id.isin(done.point_id)].copy()
    if todo.empty:
        return done
    todo["cands"] = candidate_lists(todo, items)
    assets = dict(zip(items.id, items.assets.map(json.loads)))
    best = {pid: (None, 2.0, -1) for pid in todo.point_id}
    pending = todo.set_index("point_id")
    for k in range(S2_MAX_CANDIDATES):
        groups = {}
        for pid, row in pending.iterrows():
            if k < len(row.cands):
                groups.setdefault(row.cands[k], []).append((pid, row.lon, row.lat))
        if not groups:
            break

        def check(item_id):
            with rasterio.open(planetary_computer.sign(assets[item_id][S2_SCL])) as ds:
                return [(pid, float(np.isin(read_patch(ds, lon, lat), S2_CLOUDY_SCL).mean())) for pid, lon, lat in groups[item_id]]

        accepted = set()
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(check, iid): iid for iid in groups}
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"SCL รอบที่ {k + 1}", leave=False):
                iid = futs[f]
                for pid, cf in f.result():
                    if cf < best[pid][1]:
                        best[pid] = (iid, cf, k)
                    if cf <= S2_CLEAR_OK:
                        accepted.add(pid)
        pending = pending.drop(index=list(accepted))
        if pending.empty:
            break
    new = pd.DataFrame([(pid, *v) for pid, v in best.items()], columns=["point_id", "item_id", "cloud_frac", "rank"])
    done = pd.concat([done, new], ignore_index=True)
    if cache_file is not None:
        done.to_parquet(cache_file, index=False)
    return done


def save_patch(path, arr, meta):
    np.savez_compressed(path, patch=arr, **{k: np.asarray(v) for k, v in meta.items()})


def fetch_patches_batch(points, items, assignment, out_dir, workers=8):
    # read 6-band patches grouped by scene (each band file opened once per scene); skips points already saved
    tqdm = tqdm_auto()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out_dir.glob("*.npz")}
    a = assignment.merge(points[["point_id", "lon", "lat"]], on="point_id")
    a = a[~a.point_id.isin(have) & a.item_id.notna()]
    if a.empty:
        return 0
    assets = dict(zip(items.id, items.assets.map(json.loads)))
    meta_items = items.set_index("id")

    import time as _time
    failed = []

    def run_scene(item_id, rows):
        # network hiccups: retry a read up to 3 times (re-opening the files), then skip the point and record it
        done = 0
        for attempt in range(3):
            todo = [r for r in rows if not (out_dir / f"{r[0]}.npz").exists()]
            if not todo:
                break
            try:
                dss = [open_band(assets[item_id], b, ov) for b, ov in S2_BANDS]
                try:
                    for pid, lon, lat, cf in todo:
                        arr = np.stack([read_patch(ds, lon, lat) for ds in dss]).astype(np.uint16)
                        save_patch(out_dir / f"{pid}.npz", arr, {"item_id": item_id, "datetime": meta_items.at[item_id, "datetime"],
                                                               "cloud_frac": cf, "lon": lon, "lat": lat})
                        done += 1
                finally:
                    for ds in dss:
                        ds.close()
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    failed.extend((r[0], item_id, repr(e)[:200]) for r in rows if not (out_dir / f"{r[0]}.npz").exists())
                _time.sleep(5 * (attempt + 1))
        return done

    groups = {iid: list(g[["point_id", "lon", "lat", "cloud_frac"]].itertuples(index=False)) for iid, g in a.groupby("item_id")}
    n = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_scene, iid, rows) for iid, rows in groups.items()]
        with tqdm(total=len(a), desc="ดึง patch Sentinel-2") as bar:
            for f in as_completed(futs):
                k = f.result()
                n += k
                bar.update(k)
    if failed:
        pd.DataFrame(failed, columns=["point_id", "item_id", "error"]).to_csv(out_dir.parent / f"{out_dir.name}_failed.csv", index=False)
        print(f"อ่านไม่สำเร็จ {len(failed)} จุด (บันทึกไว้ที่ {out_dir.name}_failed.csv) รันซ้ำเพื่อลองใหม่")
    return n


def fetch_patch_single(lon, lat):
    # inference: same scene rule for one point -> (patch uint16 [6,32,32], meta dict)
    pt = pd.DataFrame({"point_id": ["q"], "lon": [lon], "lat": [lat]})
    items = search_items({"type": "Point", "coordinates": [lon, lat]})
    if items.empty:
        return None, {"item_id": None}
    asg = scene_assignment(pt, items, cache_file=None, workers=4)
    iid = asg.item_id.iat[0]
    assets = json.loads(items.set_index("id").at[iid, "assets"])
    dss = [open_band(assets, b, ov) for b, ov in S2_BANDS]
    try:
        arr = np.stack([read_patch(ds, lon, lat) for ds in dss]).astype(np.uint16)
    finally:
        for ds in dss:
            ds.close()
    return arr, {"item_id": iid, "cloud_frac": float(asg.cloud_frac.iat[0])}


def s2_indices(patch):
    # tabular features from one patch (DN uint16 [6, H, W]) -> dict
    refl = patch.astype("float32") * S2_SCALE + S2_OFFSET
    valid = (patch > 0).all(axis=0)
    out = {}
    if valid.mean() < 0.5:
        return {f"s2_{k}": np.nan for k in S2_FEATURE_KEYS}
    b = {n: refl[i][valid] for i, n in enumerate(S2_BAND_NAMES)}
    c = S2_PATCH // 2
    center = refl[:, c - 1:c + 2, c - 1:c + 2].reshape(6, -1).mean(axis=1)
    for i, n in enumerate(S2_BAND_NAMES):
        out[f"s2_{n}_mean"] = float(b[n].mean())
        out[f"s2_{n}_center"] = float(center[i])
    eps = 1e-6
    ndvi = (b["B8A"] - b["B04"]) / (b["B8A"] + b["B04"] + eps)
    ndwi = (b["B03"] - b["B8A"]) / (b["B03"] + b["B8A"] + eps)
    bsi = ((b["B11"] + b["B04"]) - (b["B8A"] + b["B02"])) / ((b["B11"] + b["B04"]) + (b["B8A"] + b["B02"]) + eps)
    out.update({
        "s2_ndvi_mean": float(ndvi.mean()), "s2_ndvi_std": float(ndvi.std()),
        "s2_ndwi_mean": float(ndwi.mean()), "s2_bsi_mean": float(bsi.mean()),
        "s2_bare_frac": float((ndvi < 0.2).mean()),                       # สัดส่วนพื้นที่โล่ง (Block 2016: bare soil)
        "s2_clay_ratio": float((b["B11"] / (b["B12"] + eps)).mean()),       # SWIR1/SWIR2: แร่ดินเหนียว/คาร์บอเนต
        "s2_iron_ratio": float((b["B04"] / (b["B02"] + eps)).mean()),       # red/blue: เหล็กออกไซด์
        "s2_ferrous_ratio": float((b["B11"] / (b["B8A"] + eps)).mean()),
        "s2_brightness": float(np.mean([b[n].mean() for n in S2_BAND_NAMES])),
        "s2_texture_swir": float(b["B12"].std()),
    })
    return out


S2_FEATURE_KEYS = ([f"{n}_{s}" for n in S2_BAND_NAMES for s in ("mean", "center")]
                   + ["ndvi_mean", "ndvi_std", "ndwi_mean", "bsi_mean", "bare_frac", "clay_ratio", "iron_ratio",
                      "ferrous_ratio", "brightness", "texture_swir"])
