# Spatial context features (notebook 14 onwards): how much known fossil evidence surrounds a point.
# Computed from a reference set of DMR sites while ignoring reference sites that share the point's group:
#   CV       -> the scheme's group (2 km cluster / 50 km block / the point itself)
#   final model, maps, new coordinates -> the point's 50 km spatial block (the setting that was evaluated, spatial_block CV)
# so a known site never "sees" itself (no distance-0 leak) and training / prediction rows are built the same way.
SC_A = ["sc_dist_site_km", "sc_kde_10km", "sc_kde_50km", "sc_n_25km", "sc_unit_rate"]


def block_group(lons, lats, km=50.0):
    # เหมือน model_lib.assign_folds (group_spatial_block): บล็อก km × km ใน EPSG:32647
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 32647, always_xy=True).transform(np.asarray(lons, float), np.asarray(lats, float))
    return np.floor(x / (km * 1000)).astype(int) * 100000 + np.floor(y / (km * 1000)).astype(int)


def unit_areas():
    # กม.² ของแต่ละหน่วยหิน (จากราสเตอร์ gx_rock_code ของ notebook 13)
    if "__unit_area" not in _RASTERS:
        codes = json.loads((EXTRA_DIR / "rock_codes.json").read_text(encoding="utf-8"))
        cnt = np.bincount(load_raster("gx_rock_code").ravel(), minlength=max(codes.values()) + 1)
        _RASTERS["__unit_area"] = {s: cnt[c] * KM_Y * KM_X for s, c in codes.items()}
    return _RASTERS["__unit_area"]


def spatial_context(rows, ref, row_groups, ref_groups, ref_labels=None, classes=(), radius_km=100.0):
    # rows / ref: DataFrame (lon, lat, geo_rock_sym); *_groups: arrays; ref_labels: list of sets (ประเภทซาก) for sc_type_*
    from sklearn.neighbors import BallTree
    R = 6371.0
    area = unit_areas()
    rg, fg = np.asarray(row_groups), np.asarray(ref_groups)
    tree = BallTree(np.radians(ref[["lat", "lon"]].values), metric="haversine")
    ind, dist = tree.query_radius(np.radians(rows[["lat", "lon"]].values), r=radius_km / R, return_distance=True)
    L = np.array([[c in s for c in classes] for s in ref_labels], dtype=float) if ref_labels is not None else None
    ref_u = ref.geo_rock_sym.astype(str).values
    unit_n = pd.Series(ref_u).value_counts()
    unit_grp = pd.Series(1, index=pd.MultiIndex.from_arrays([ref_u, fg])).groupby(level=[0, 1]).size()
    row_u = rows.geo_rock_sym.astype(str).values
    n_extra = (1 + len(classes)) if L is not None else 0
    out = np.full((len(rows), 5 + n_extra), np.nan)
    for i, (ix, d) in enumerate(zip(ind, dist)):
        keep = fg[ix] != rg[i]
        ix, d = ix[keep], d[keep] * R
        u = row_u[i]
        n_u = unit_n.get(u, 0) - unit_grp.get((u, rg[i]), 0)
        out[i, 4] = np.log1p(1000 * n_u / area[u]) if area.get(u, 0) > 0 else np.nan
        if len(ix) == 0:
            out[i, :4] = [radius_km, 0, 0, 0]
            continue
        out[i, 0] = d.min()
        out[i, 1] = np.exp(-d ** 2 / (2 * 10 ** 2)).sum()
        out[i, 2] = np.exp(-d ** 2 / (2 * 50 ** 2)).sum()
        out[i, 3] = (d <= 25).sum()
        if L is not None:
            w = np.exp(-d ** 2 / (2 * 15 ** 2))
            lab = L[ix]
            has = lab.sum(axis=1) > 0
            out[i, 5] = has.sum()
            if has.any():
                out[i, 6:] = (w[has, None] * lab[has]).sum(0) / w[has].sum()
    cols = SC_A + ((["sc_n_labelled_100km"] + [f"sc_type_{c}" for c in classes]) if L is not None else [])
    return pd.DataFrame(out, columns=cols, index=rows.index)
