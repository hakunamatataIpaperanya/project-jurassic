# Extra feature layers for the feature-engineering experiments (notebooks 13-15).
# Every layer is a national raster on a 0.002° grid (~220 m) cached in data/features/extra_cache/*.npz and sampled at
# points, so training points, the 5 km grid and new coordinates get identical values.
#   gx_  geology+ : distance to faults / folds (DMR 1:250k, exported as images because the service returns no geometry),
#                   distance to rock-unit contacts, unit diversity, lithology / environment flags (ThaiLex)
#   pgx_ paleo    : signed distance to the paleo-coastline at each time slice (Scotese PaleoDEM, at paleo-coordinates)
#   lc_  surface  : ESA WorldCover fractions, JRC surface water, OSM rivers / streams / drainage density, TerraClimate
#   dc_  discovery: roads, quarries, DMR mineral-potential areas, reservoirs (where people dig — experiment only)
import io                 # json, re มาจาก 00_common

EXTRA_DIR = FEAT / "extra_cache"
X_RES = 0.002
X_W, X_S, X_E, X_N = THAI_BBOX
X_COLS, X_ROWS = int(round((X_E - X_W) / X_RES)), int(round((X_N - X_S) / X_RES))
KM_Y, KM_X = X_RES * 110.574, X_RES * 111.320 * np.cos(np.radians(13.0))     # ไทยอยู่ราว 6–20°N: คลาดไม่เกิน ~4%
_RASTERS = {}


def x_transform():
    from rasterio.transform import from_origin
    return from_origin(X_W, X_N, X_RES, X_RES)


def x_rc(lons, lats):
    r = np.clip(((X_N - np.asarray(lats, float)) / X_RES).astype(int), 0, X_ROWS - 1)
    c = np.clip(((np.asarray(lons, float) - X_W) / X_RES).astype(int), 0, X_COLS - 1)
    return r, c


def save_raster(name, arr):
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXTRA_DIR / f"{name}.npz", a=arr)
    _RASTERS[name] = arr


def load_raster(name):
    if name not in _RASTERS:
        _RASTERS[name] = np.load(EXTRA_DIR / f"{name}.npz")["a"]
    return _RASTERS[name]


def has_raster(name):
    return (EXTRA_DIR / f"{name}.npz").exists()


def dist_from_mask(mask):
    # km to the nearest True cell (float16 พอ: ความละเอียด ~0.1 กม. ที่ระยะ < 100 กม.)
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~mask, sampling=(KM_Y, KM_X)).astype("float16")


def rasterize_geoms(geoms, all_touched=True):
    from rasterio.features import rasterize
    return rasterize(((g, 1) for g in geoms if g is not None and not g.is_empty), out_shape=(X_ROWS, X_COLS),
                     transform=x_transform(), fill=0, all_touched=all_touched, dtype="uint8").astype(bool)


def sample_raster(name, lons, lats):
    r, c = x_rc(lons, lats)
    return load_raster(name)[r, c].astype("float32")


def window_mean(name, lons, lats, radius_px):
    # ค่าเฉลี่ยในหน้าต่าง (2r+1)^2 รอบจุด — ใช้ integral image ของทั้งประเทศ (เร็วกับหลายหมื่นจุด)
    key = f"__cum_{name}"
    if key not in _RASTERS:
        a = load_raster(name).astype("float64")
        _RASTERS[key] = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    S = _RASTERS[key]
    r, c = x_rc(lons, lats)
    r0, r1 = np.clip(r - radius_px, 0, X_ROWS), np.clip(r + radius_px + 1, 0, X_ROWS)
    c0, c1 = np.clip(c - radius_px, 0, X_COLS), np.clip(c + radius_px + 1, 0, X_COLS)
    tot = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
    return (tot / ((r1 - r0) * (c1 - c0))).astype("float32")


def window_nunique(name, lons, lats, radius_px):
    a = load_raster(name)
    r, c = x_rc(lons, lats)
    out = np.empty(len(r), "float32")
    for i, (ri, ci) in enumerate(zip(r, c)):
        w = a[max(ri - radius_px, 0):ri + radius_px + 1, max(ci - radius_px, 0):ci + radius_px + 1]
        out[i] = len(np.unique(w[w > 0]))
    return out


# ---------------------------------------------------------------- builders (ใช้ครั้งเดียว แล้ว cache)
DMR_EXPORT = "https://gisportal.dmr.go.th/arcgis/rest/services/GEOL/GEOL_STR_250K/MapServer/export"
STRUCT_GROUPS = {"fault": ("0200", "0220"), "fold": ("0110", "0120", "0140")}     # STR_GRP_ID ตาม renderer ของ ทธ.


def build_structure(tile_px=2000, workers=6):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    tqdm = tqdm_auto()
    for kind, ids in STRUCT_GROUPS.items():
        name = f"gx_dist_{kind}_km"
        if has_raster(name):
            continue
        mask = np.zeros((X_ROWS, X_COLS), bool)
        where = "dmrdb2.DMRAPP.GEOL_STR_250K.STR_GRP_ID IN (" + ",".join(f"'{i}'" for i in ids) + ")"
        tiles = [(r0, c0) for r0 in range(0, X_ROWS, tile_px) for c0 in range(0, X_COLS, tile_px)]

        def get(rc):
            r0, c0 = rc
            h, w = min(tile_px, X_ROWS - r0), min(tile_px, X_COLS - c0)
            bbox = (X_W + c0 * X_RES, X_N - (r0 + h) * X_RES, X_W + (c0 + w) * X_RES, X_N - r0 * X_RES)
            p = {"bbox": ",".join(map(str, bbox)), "bboxSR": 4326, "imageSR": 4326, "size": f"{w},{h}", "format": "png",
                 "transparent": "true", "layers": "show:0", "f": "image", "dpi": 96, "layerDefs": json.dumps({"0": where})}
            for k in range(4):
                try:
                    r = requests.get(DMR_EXPORT, params=p, timeout=180)
                    img = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
                    return r0, c0, img[..., 3] > 0
                except Exception:                                   # noqa: BLE001
                    if k == 3:
                        raise
        with ThreadPoolExecutor(workers) as ex:
            for r0, c0, m in tqdm(ex.map(get, tiles), total=len(tiles), desc=f"ทธ. {kind}"):
                mask[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] |= m
        save_raster(name, dist_from_mask(mask))


def build_rock_contacts():
    if has_raster("gx_dist_contact_km") and has_raster("gx_rock_code"):
        return
    import geopandas as gpd
    from rasterio.features import rasterize
    ru = gpd.read_file(DATA / "dmr_geology" / "rock_unit_250k.geojson")[["TEXT_SYM", "geometry"]].to_crs(4326)
    syms = sorted(ru.TEXT_SYM.dropna().unique())
    code = {s: i + 1 for i, s in enumerate(syms)}
    arr = rasterize(((g, code[s]) for g, s in zip(ru.geometry, ru.TEXT_SYM) if s in code), out_shape=(X_ROWS, X_COLS),
                    transform=x_transform(), fill=0, dtype="uint8")
    edge = np.zeros_like(arr, bool)
    edge[:, 1:] |= (arr[:, 1:] != arr[:, :-1]) & (arr[:, 1:] > 0) & (arr[:, :-1] > 0)
    edge[1:, :] |= (arr[1:, :] != arr[:-1, :]) & (arr[1:, :] > 0) & (arr[:-1, :] > 0)
    save_raster("gx_rock_code", arr)
    save_raster("gx_dist_contact_km", dist_from_mask(edge))
    (EXTRA_DIR / "rock_codes.json").write_text(json.dumps(code, ensure_ascii=False), encoding="utf-8")


OSM_SHP = DATA / "osm" / "shp"
ROAD_ALL = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "track", "service",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street",
            "track_grade1", "track_grade2", "track_grade3", "track_grade4", "track_grade5"}
ROAD_MAJOR = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}


def _osm(layer, classes):
    import pyogrio
    g = pyogrio.read_dataframe(OSM_SHP / f"gis_osm_{layer}_free_1.shp", columns=["fclass"],
                               bbox=(X_W, X_S, X_E, X_N))
    return g[g.fclass.isin(classes)].geometry


def build_osm():
    jobs = {
        "dc_dist_road_km": ("roads", ROAD_ALL), "dc_dist_major_road_km": ("roads", ROAD_MAJOR),
        "lc_dist_river_km": ("waterways", {"river"}), "lc_dist_stream_km": ("waterways", {"stream", "canal", "drain"}),
        "dc_dist_reservoir_km": ("water_a", {"reservoir"}), "dc_dist_quarry_km": ("landuse_a", {"quarry"}),
    }
    for name, (layer, classes) in tqdm_auto()(jobs.items(), desc="OSM"):
        if has_raster(name):
            continue
        mask = rasterize_geoms(_osm(layer, classes))
        save_raster(name, dist_from_mask(mask))
        if name == "lc_dist_stream_km":
            save_raster("lc_stream_mask", mask.astype("uint8"))
        if name == "dc_dist_road_km":
            save_raster("dc_road_mask", mask.astype("uint8"))


def build_mineral():
    if has_raster("dc_dist_mineral_km"):
        return
    import geopandas as gpd
    g = gpd.read_file(DATA / "dmr_geology" / "mineral_potential_50k.geojson").to_crs(4326)
    save_raster("dc_dist_mineral_km", dist_from_mask(rasterize_geoms(g.geometry)))


WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}


def _pc_items(collection, query=None):
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    return list(cat.search(collections=[collection], bbox=THAI_BBOX, query=query).items())


def _read_on_grid(href, sub=5, resampling="nearest"):
    # อ่าน COG ลงกริด 0.002° ละเอียดขึ้น sub เท่า (เช่น 5 -> 0.0004° ≈ 44 ม.) เฉพาะส่วนที่ทับประเทศไทย
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    with rasterio.open(href) as src:
        b = src.bounds
        w, s, e, n = max(b.left, X_W), max(b.bottom, X_S), min(b.right, X_E), min(b.top, X_N)
        if w >= e or s >= n:
            return None
        c0, c1 = int(round((w - X_W) / X_RES)), int(round((e - X_W) / X_RES))
        r0, r1 = int(round((X_N - n) / X_RES)), int(round((X_N - s) / X_RES))
        win = from_bounds(X_W + c0 * X_RES, X_N - r1 * X_RES, X_W + c1 * X_RES, X_N - r0 * X_RES, src.transform)
        a = src.read(1, window=win, out_shape=((r1 - r0) * sub, (c1 - c0) * sub), boundless=True, fill_value=0,
                     resampling=getattr(Resampling, resampling))
    return r0, c0, a


def build_worldcover(sub=5, workers=6):
    names = [f"lc_wc_{v}" for v in WC_CLASSES.values()]
    if all(has_raster(n) for n in names):
        return
    from concurrent.futures import ThreadPoolExecutor
    items = [it for it in _pc_items("esa-worldcover") if "2021" in it.id] or _pc_items("esa-worldcover")
    acc = {k: np.zeros((X_ROWS, X_COLS), "uint8") for k in WC_CLASSES}

    def one(it):
        return _read_on_grid(it.assets["map"].href, sub)
    with ThreadPoolExecutor(workers) as ex:
        for res in tqdm_auto()(ex.map(one, items), total=len(items), desc="ESA WorldCover"):
            if res is None:
                continue
            r0, c0, a = res
            h, w = a.shape[0] // sub, a.shape[1] // sub
            blk = a[:h * sub, :w * sub].reshape(h, sub, w, sub)
            for k in WC_CLASSES:
                frac = np.round((blk == k).mean(axis=(1, 3)) * 100).astype("uint8")       # % ของเซลล์ 0.002°
                tgt = acc[k][r0:r0 + h, c0:c0 + w]
                np.maximum(tgt, frac, out=tgt)                                         # tile ทับกันที่ขอบ
    for k, v in WC_CLASSES.items():
        save_raster(f"lc_wc_{v}", acc[k])


def build_gsw(sub=2, workers=4):
    if has_raster("lc_gsw_occ") and has_raster("lc_dist_water_km"):
        return
    from concurrent.futures import ThreadPoolExecutor
    items = _pc_items("jrc-gsw")
    occ = np.zeros((X_ROWS, X_COLS), "float32")

    def one(it):
        return _read_on_grid(it.assets["occurrence"].href, sub)
    with ThreadPoolExecutor(workers) as ex:
        for res in tqdm_auto()(ex.map(one, items), total=len(items), desc="JRC surface water"):
            if res is None:
                continue
            r0, c0, a = res
            h, w = a.shape[0] // sub, a.shape[1] // sub
            a = np.where(a > 100, 0, a).astype("float32")                               # 255 = ไม่มีข้อมูล
            m = a[:h * sub, :w * sub].reshape(h, sub, w, sub).mean(axis=(1, 3))
            tgt = occ[r0:r0 + h, c0:c0 + w]
            np.maximum(tgt, m, out=tgt)
    save_raster("lc_gsw_occ", occ.astype("uint8"))
    save_raster("lc_dist_water_km", dist_from_mask(occ >= 75))


TC_VARS = ["ppt", "q", "def", "aet"]          # ฝน, น้ำท่า (การกัดเซาะ), การขาดน้ำ, การคายระเหยจริง


def build_terraclimate(years=("2012", "2021")):
    path = EXTRA_DIR / "terraclimate.nc"
    if path.exists():
        return
    import planetary_computer
    import pystac_client
    import xarray as xr
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    asset = cat.get_collection("terraclimate").assets["zarr-abfs"]
    kw = dict(asset.extra_fields["xarray:open_kwargs"])
    kw["chunks"] = None
    ds = xr.open_dataset(asset.href, **kw)
    out = {}
    for v in tqdm_auto()(TC_VARS, desc="TerraClimate"):
        part = []
        for y in range(int(years[0]), int(years[1]) + 1):                    # ทีละปี (chunk ของ zarr เป็นรายปีทั่วโลก)
            sub = ds[v].sel(lat=slice(X_N + 0.1, X_S - 0.1), lon=slice(X_W - 0.1, X_E + 0.1), time=str(y))
            part.append(sub.sum("time").load())
        out[v] = xr.concat(part, "year").mean("year")
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    xr.Dataset(out).to_netcdf(path)


def sample_terraclimate(lons, lats):
    # ค่าที่เซลล์บกใกล้สุด: เซลล์ทะเลของ TerraClimate เป็น NaN และผลรวมรายปีกลายเป็น 0 — ถ้าใช้ตรง ๆ จุดริมฝั่ง (ส่วนใหญ่เป็น
    # background) จะได้ "ฝน = 0" ซึ่งเป็นสัญญาณปลอม จึงเลือกเฉพาะเซลล์ที่มีค่า (ppt > 0) แล้วหาเซลล์ใกล้สุด
    import xarray as xr
    from scipy.spatial import cKDTree
    if "__tc" not in _RASTERS:
        ds = xr.open_dataset(EXTRA_DIR / "terraclimate.nc").load()
        LA, LO = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
        ok = ds.ppt.values > 0
        _RASTERS["__tc"] = (cKDTree(np.c_[LA[ok], LO[ok]]), {v: ds[v].values[ok] for v in TC_VARS})
    tree, vals = _RASTERS["__tc"]
    _, i = tree.query(np.c_[np.asarray(lats, float), np.asarray(lons, float)])
    return pd.DataFrame({f"lc_tc_{v}": vals[v][i].astype("float32") for v in TC_VARS})


# ---------------------------------------------------------------- paleo-coastline (Scotese PaleoDEM)
def paleo_coast_km(plons, plats, t):
    # ระยะ (กม.) จากตำแหน่งโบราณถึงแนวชายฝั่งโบราณ ณ เวลา t: บวก = บนบก, ลบ = ในทะเล
    from sklearn.neighbors import BallTree
    key = f"__coast_{t}"
    if key not in _RASTERS:
        f = sorted((DATA / "paleo" / "paleodem_0.2deg" / "paleoDEM_0.2Deg_grids").glob(f"*_{65 if t == 66 else t}Ma.nc"))[0]
        import xarray as xr
        z = xr.open_dataset(f)["z"]
        latn, lonn = [d for d in z.dims if "lat" in d][0], [d for d in z.dims if "lon" in d][0]
        land = z.values > 0
        coast = np.zeros_like(land)
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        coast[1:, :] |= land[1:, :] != land[:-1, :]
        LA, LO = np.meshgrid(z[latn].values, z[lonn].values, indexing="ij")
        _RASTERS[key] = (BallTree(np.radians(np.c_[LA[coast], LO[coast]]), metric="haversine"), z, latn, lonn)
    tree, z, latn, lonn = _RASTERS[key]
    ok = np.isfinite(plats) & np.isfinite(plons)
    out = np.full(len(plats), np.nan, "float32")
    if ok.any():
        d = tree.query(np.radians(np.c_[plats[ok], plons[ok]]), k=1)[0][:, 0] * 6371.0
        zz = z.sel({latn: xr_da(plats[ok]), lonn: xr_da(plons[ok])}, method="nearest").values
        out[ok] = np.where(zz > 0, d, -d)
    return out


def xr_da(v):
    import xarray as xr
    return xr.DataArray(np.asarray(v), dims="p")


# ---------------------------------------------------------------- lithology (ThaiLex)
LITH_KW = {
    "sandstone": r"sandstone|quartzite|arenite|greywacke|graywacke",
    "fine_clastic": r"siltstone|mudstone|claystone|shale|slate|argillite|marl",
    "carbonate": r"limestone|dolomite|dolostone|marl|calcarenite",
    "conglomerate": r"conglomerate|breccia",
    "chert": r"chert|radiolarite",
    "evaporite": r"evaporite|salt|halite|gypsum|anhydrite|potash",
    "coal": r"coal|lignite",
    "volcanic": r"volcanic|tuff|andesite|basalt|rhyolite|ash",
    "plutonic": r"granite|granodiorite|diorite|plutonic",
    "metamorphic": r"schist|gneiss|phyllite|metamorphic|migmatite|marble",
}
ENV_KW = {"marine": r"marine|reef|shelf|carbonate platform|pelagic|turbidite", "continental": r"fluvial|lacustrine|river|alluvial|floodplain|continental|red bed"}
# ชื่อไทยที่ตรวจด้วยมือแล้ว: ค่า = ชื่อใน ThaiLex (fuzzy match ไม่ได้ เช่น RTGS ต่างจากการสะกดใน ThaiLex)
# หรือ None = ไม่มีใน ThaiLex (fuzzy ให้คะแนนผ่านเกณฑ์แต่เป็นคนละหน่วย)
#   ซำบอน (พิษณุโลก กลุ่มหินโคราช จูแรสซิก) ได้ 83.3 กับ Sai Bon Fm = หมวดหินไสบอน (ไทรแอสซิก คาบสมุทรภาคใต้)
THAILEX_MANUAL = {"อุ้งผาง": "Umphang Gr", "ทุ่งสง": "Thung Song Limestone Fm", "น้ำพอง": "Lower Nam Phong Fm", "อีเลิศ": "E-Lert Fm",
                  "ซำบอน": None}


def thailex_match(th_name, min_score=83):
    # ชื่อหน่วยหินภาษาไทย (มีหรือไม่มี "หมวดหิน/กลุ่มหิน") -> แถวของ ThaiLex หรือ None
    from pythainlp.transliterate import romanize
    from rapidfuzz import fuzz, process
    if "__thailex" not in _RASTERS:
        d = pd.read_csv(DATA / "thailex" / "formations.csv", encoding="utf-8-sig")
        d["key"] = (d.name_en.str.replace(r"\b(Fm|Gr|Formation|Group|Limestone|Quartzite|Conglomerate|Sandstone|Shale|Member|Mb)\b", "", regex=True)
                    .str.lower().str.replace(r"[^a-z]", "", regex=True))
        _RASTERS["__thailex"] = d
    d = _RASTERS["__thailex"]
    t = re.sub(r"^(หมวดหิน|กลุ่มหิน)", "", str(th_name)).strip(" ?")
    if not t or t == "nan":
        return None
    if t in THAILEX_MANUAL:
        return None if THAILEX_MANUAL[t] is None else d[d.name_en.eq(THAILEX_MANUAL[t])].iloc[0]
    hit = d[d.name_th.fillna("").str.replace(r"^(หมวดหิน|กลุ่มหิน)", "", regex=True).eq(t)]
    if len(hit):
        return hit.iloc[0]
    best = process.extractOne(romanize(t, engine="royin").lower().replace(" ", ""), d.key.tolist(), scorer=fuzz.ratio)
    return d.iloc[best[2]] if best and best[1] >= min_score else None


def lithology_flags(text):
    t = str(text).lower()
    out = {f"lith_{k}": int(bool(re.search(p, t))) for k, p in LITH_KW.items()}
    out.update({f"env_{k}": int(bool(re.search(p, t))) for k, p in ENV_KW.items()})
    return out


def rock_unit_lithology():
    # ตาราง TEXT_SYM -> lithology flags: ชื่อหน่วยหินจาก legend ของ ทธ. -> ThaiLex; หน่วยไม่มีชื่อใช้ตัวอักษรของสัญลักษณ์
    path = EXTRA_DIR / "rock_unit_lithology.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    leg = pd.read_csv(DATA / "dmr_geology" / "rock_unit_250k_legend.csv", encoding="utf-8-sig")
    rows = []
    for sym, lab in zip(leg.TEXT_SYM, leg.label_th):
        m = thailex_match(lab) if isinstance(lab, str) and re.search("หมวดหิน|กลุ่มหิน", lab) else None
        if m is not None:
            text, src = f"{m.lithology} {m.lithology_pattern}", f"ThaiLex: {m.name_en}"
        elif isinstance(lab, str) and lab.startswith("ตะกอน"):
            text, src = "unconsolidated alluvial sediment", "legend: ตะกอน"
        elif re.search(r"gr$", str(sym)):
            text, src = "granite", "symbol: gr"
        elif re.search(r"(v|bs)$", str(sym)):
            text, src = "volcanic basalt", "symbol: v/bs"
        else:
            text, src = "", "unknown"
        rows.append({"TEXT_SYM": sym, "source": src, "unconsolidated": int(str(sym).startswith("Q")), **lithology_flags(text)})
    df = pd.DataFrame(rows)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------- features at points
GX_TIMES = [480, 400, 300, 280, 270, 260, 250, 240, 230, 200, 150, 120, 66]


def extra_features(points, pangea=None, geology=None):
    # points: DataFrame(lon, lat [, point_id]); pangea: แถวเดียวกันที่มี pg_sw18_lat_t / pg_sw18_lon_t;
    # geology: แถวเดียวกันที่มี geo_rock_sym (จาก batch_geology) — ถ้าไม่ให้มา จะไม่มี lith_* / pgx_*
    lo, la = points.lon.values, points.lat.values
    f = pd.DataFrame(index=points.index)
    for n in ["gx_dist_fault_km", "gx_dist_fold_km", "gx_dist_contact_km"]:
        f[n] = sample_raster(n, lo, la)
    f["gx_units_2km"] = window_nunique("gx_rock_code", lo, la, 9)
    if geology is not None:
        lith = rock_unit_lithology().set_index("TEXT_SYM").drop(columns="source")
        f = f.join(lith.add_prefix("gx_").reindex(geology.geo_rock_sym.astype(str).values).set_index(f.index))
    if pangea is not None:
        for t in GX_TIMES:
            if f"pg_sw18_lat_{t}" in pangea:
                f[f"pgx_coast_km_{t}"] = paleo_coast_km(pangea[f"pg_sw18_lon_{t}"].values.astype(float),
                                                        pangea[f"pg_sw18_lat_{t}"].values.astype(float), t)
    for v in WC_CLASSES.values():
        f[f"lc_wc_{v}_1km"] = window_mean(f"lc_wc_{v}", lo, la, 2)
    f["lc_gsw_occ_1km"] = window_mean("lc_gsw_occ", lo, la, 2)
    for n in ["lc_dist_water_km", "lc_dist_river_km", "lc_dist_stream_km"]:
        f[n] = sample_raster(n, lo, la)
    f["lc_drainage_2km"] = window_mean("lc_stream_mask", lo, la, 9)
    f = f.join(sample_terraclimate(lo, la).set_index(f.index))
    for n in ["dc_dist_road_km", "dc_dist_major_road_km", "dc_dist_quarry_km", "dc_dist_mineral_km", "dc_dist_reservoir_km"]:
        f[n] = sample_raster(n, lo, la)
    f["dc_road_density_2km"] = window_mean("dc_road_mask", lo, la, 9)
    return f


EXTRA_GROUPS = {"geo_plus": ("gx_", "pgx_"), "surface": ("lc_",), "discovery": ("dc_",)}
