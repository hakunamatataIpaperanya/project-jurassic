# Batch runners for training data: chunked, cached, resumable (re-running skips finished chunks).
SAT_CACHE = FEAT / "sat_cache"


def batch_dem(points, chunk=500, tag=""):
    tqdm = tqdm_auto()
    SAT_CACHE.mkdir(parents=True, exist_ok=True)
    dem = Dem()
    parts = []
    starts = range(0, len(points), chunk)
    for s in tqdm(starts, desc="DEM (chunk)"):
        f = SAT_CACHE / f"{tag}dem_{s // chunk:03d}.parquet"
        sub = points.iloc[s:s + chunk]
        if f.exists() and pd.read_parquet(f, columns=["point_id"]).point_id.tolist() == sub.point_id.tolist():
            parts.append(pd.read_parquet(f))
            continue
        rows = [dem.features(lo, la) for lo, la in zip(sub.lon, sub.lat)]
        df = pd.DataFrame(rows).assign(point_id=sub.point_id.values)
        df.to_parquet(f, index=False)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def batch_geology(points, chunk=2000, tag=""):
    tqdm = tqdm_auto()
    SAT_CACHE.mkdir(parents=True, exist_ok=True)
    geo = Geology()
    parts = []
    for s in tqdm(range(0, len(points), chunk), desc="ธรณีวิทยา (chunk)"):
        f = SAT_CACHE / f"{tag}geology_{s // chunk:03d}.parquet"
        sub = points.iloc[s:s + chunk]
        if f.exists() and pd.read_parquet(f, columns=["point_id"]).point_id.tolist() == sub.point_id.tolist():
            parts.append(pd.read_parquet(f))
            continue
        df = geo.features(sub.lon.values, sub.lat.values).assign(point_id=sub.point_id.values)
        df.to_parquet(f, index=False)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def batch_s2_indices(points, patch_dir, tag=""):
    # indices for every saved patch (cache keyed by point_id; recomputes only new patches)
    tqdm = tqdm_auto()
    f = SAT_CACHE / f"{tag}s2_indices.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["point_id"])
    seen = set(old.point_id)
    rows = []
    for pid in tqdm([p for p in points.point_id if p not in seen], desc="ดัชนี Sentinel-2"):
        p = Path(patch_dir) / f"{pid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        rows.append({"point_id": pid, **s2_indices(z["patch"]), "s2_item_id": str(z["item_id"]),
                     "s2_cloud_frac": float(z["cloud_frac"]), "s2_date": str(z["datetime"])[:10]})
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    df.to_parquet(f, index=False)
    return df


def batch_pangea(points, chunk=2000, tag=""):
    tqdm = tqdm_auto()
    SAT_CACHE.mkdir(parents=True, exist_ok=True)
    pg = Pangea()
    parts = []
    for s in tqdm(range(0, len(points), chunk), desc="พันเจีย (chunk)"):
        f = SAT_CACHE / f"{tag}pangea_{s // chunk:03d}.parquet"
        sub = points.iloc[s:s + chunk]
        if f.exists() and pd.read_parquet(f, columns=["point_id"]).point_id.tolist() == sub.point_id.tolist():
            parts.append(pd.read_parquet(f))
            continue
        df = pg.features(sub.lon.values, sub.lat.values).assign(point_id=sub.point_id.values)
        df.to_parquet(f, index=False)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def make_grid(step_km=5):
    # regular grid (EPSG:32647) over Thai land (DMR province boundaries) -> DataFrame point_id, lon, lat
    import geopandas as gpd
    from pyproj import Transformer
    from shapely import contains_xy
    prov = gpd.read_file(DATA / "dmr_admin" / "provinces.geojson").to_crs(32647)
    land = prov.union_all()
    x0, y0, x1, y1 = land.bounds
    step = step_km * 1000
    xs, ys = np.meshgrid(np.arange(np.ceil(x0 / step) * step, x1, step), np.arange(np.ceil(y0 / step) * step, y1, step))
    xs, ys = xs.ravel(), ys.ravel()
    keep = contains_xy(land, xs, ys)
    lon, lat = Transformer.from_crs(32647, 4326, always_xy=True).transform(xs[keep], ys[keep])
    return pd.DataFrame({"point_id": [f"g{step_km}_{int(x)}_{int(y)}" for x, y in zip(xs[keep], ys[keep])],
                         "source": f"grid_{step_km}km", "lon": lon, "lat": lat})
