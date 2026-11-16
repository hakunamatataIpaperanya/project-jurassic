# Geology features: DMR 1:250,000 rock units (at point, era share within 5 km, distance to each era),
# caves (CAVE_SURVEY, nationwide) and outcrops (OUT_CROP, only a few surveyed provinces -> separate "oc_" group).
GEO_DIR = DATA / "dmr_geology"
GEO_CACHE = FEAT / "geology_cache"
GEO_RES_M = 500                      # raster resolution for era share / distance (EPSG:32647)
GEO_SHARE_KM = 5
ERAS = ["Precambrian", "Cambrian", "Ordovician", "Silurian", "Devonian", "Carboniferous", "Permian",
        "Triassic", "Jurassic", "Cretaceous", "Tertiary", "Quaternary"]
_ERA_PREFIX = {"PE": ("Precambrian", "Precambrian"), "EO": ("Cambrian", "Ordovician"), "E": ("Cambrian", "Cambrian"),
               "O": ("Ordovician", "Ordovician"), "SDC": ("Silurian", "Carboniferous"), "SD": ("Silurian", "Devonian"),
               "DC": ("Devonian", "Carboniferous"), "D": ("Devonian", "Devonian"), "CP": ("Carboniferous", "Permian"),
               "C": ("Carboniferous", "Carboniferous"), "PTr": ("Permian", "Triassic"), "P": ("Permian", "Permian"),
               "TrJ": ("Triassic", "Jurassic"), "Tr": ("Triassic", "Triassic"), "JK": ("Jurassic", "Cretaceous"),
               "J": ("Jurassic", "Jurassic"), "KT": ("Cretaceous", "Tertiary"), "K": ("Cretaceous", "Cretaceous"),
               "T": ("Tertiary", "Tertiary"), "Q": ("Quaternary", "Quaternary")}
CARBONATE = ("limestone", "dolomite", "marble")


def symbol_era(sym):
    for p in sorted(_ERA_PREFIX, key=len, reverse=True):
        if isinstance(sym, str) and sym.startswith(p):
            return _ERA_PREFIX[p]
    return (None, None)


class Geology:
    def __init__(self):
        import geopandas as gpd
        from sklearn.neighbors import BallTree
        GEO_CACHE.mkdir(parents=True, exist_ok=True)
        legend = pd.read_csv(GEO_DIR / "rock_unit_250k_legend.csv", encoding="utf-8-sig")
        igneous_from = legend.loc[legend.TEXT_SYM.eq("Qgy"), "order"].iat[0]
        legend["rock_kind"] = np.where(legend.TEXT_SYM.eq("Water body"), "water",
                                       np.where(legend.order >= igneous_from, "igneous", "sedimentary/metamorphic"))
        rock = gpd.read_file(GEO_DIR / "rock_unit_250k.geojson")[["TEXT_SYM", "geometry"]].merge(
            legend[["TEXT_SYM", "rock_kind"]], on="TEXT_SYM", how="left")
        eras = rock.TEXT_SYM.map(symbol_era)
        rock["era_old"] = [e[0] for e in eras]
        rock["era_young"] = [e[1] for e in eras]
        self.rock = rock.to_crs(32647)
        self.rock_sindex = self.rock.sindex
        self._build_rasters()
        caves = gpd.read_file(GEO_DIR / "caves.geojson")
        self.cave_tree = BallTree(np.radians(np.c_[caves.geometry.y, caves.geometry.x]), metric="haversine")
        oc = gpd.read_file(GEO_DIR / "outcrops.geojson")
        oc = oc[oc.geometry.notna()]
        self.oc_rock = oc.RCKNAME.fillna("").str.lower().values
        self.oc_tree = BallTree(np.radians(np.c_[oc.geometry.y, oc.geometry.x]), metric="haversine")
        carb = np.array([any(k in r for k in CARBONATE) for r in self.oc_rock])
        self.oc_carb_tree = BallTree(np.radians(np.c_[oc.geometry.y[carb], oc.geometry.x[carb]]), metric="haversine")
        self.oc_provinces = set(oc.PROVINCE.dropna().str.strip()) - {""}
        prov = gpd.read_file(DATA / "dmr_admin" / "provinces.geojson")[["prov_nam_t", "geometry"]]
        self.prov = prov

    def _build_rasters(self):
        # era-of-unit raster (oldest era of each unit) and distance-to-era rasters, cached as .npz
        f = GEO_CACHE / f"era_raster_{GEO_RES_M}m.npz"
        if f.exists():
            z = np.load(f)
            self.era_grid, self.dist_grid, self.grid_origin = z["era"], z["dist"], tuple(z["origin"])
            return
        from rasterio import features as rfeat
        from rasterio.transform import from_origin
        from scipy.ndimage import distance_transform_edt
        x0, y0, x1, y1 = self.rock.total_bounds
        x0, y1 = np.floor(x0 / GEO_RES_M) * GEO_RES_M, np.ceil(y1 / GEO_RES_M) * GEO_RES_M
        w, h = int(np.ceil((x1 - x0) / GEO_RES_M)), int(np.ceil((y1 - y0) / GEO_RES_M))
        tr = from_origin(x0, y1, GEO_RES_M, GEO_RES_M)
        code = self.rock.era_old.map({e: i + 1 for i, e in enumerate(ERAS)}).fillna(0).astype(int)
        era = rfeat.rasterize(zip(self.rock.geometry, code), out_shape=(h, w), transform=tr, fill=0, dtype="uint8")
        dist = np.zeros((len(ERAS), h, w), dtype="float16")
        for i in range(len(ERAS)):
            mask = era == i + 1
            dist[i] = (distance_transform_edt(~mask) * GEO_RES_M / 1000).astype("float16") if mask.any() else np.float16(np.inf)
        np.savez_compressed(f, era=era, dist=dist, origin=np.array([x0, y1]))
        self.era_grid, self.dist_grid, self.grid_origin = era, dist, (x0, y1)

    def _rc(self, x, y):
        return int((self.grid_origin[1] - y) // GEO_RES_M), int((x - self.grid_origin[0]) // GEO_RES_M)

    def features(self, lons, lats):
        import geopandas as gpd
        from pyproj import Transformer
        lons, lats = np.asarray(lons, float), np.asarray(lats, float)
        xs, ys = Transformer.from_crs(4326, 32647, always_xy=True).transform(lons, lats)
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=32647)
        j = gpd.sjoin(pts, self.rock, how="left", predicate="within")
        j = j[~j.index.duplicated()]
        out = pd.DataFrame({"geo_rock_sym": j.TEXT_SYM.fillna("none").values, "geo_era_old": j.era_old.fillna("none").values,
                            "geo_era_young": j.era_young.fillna("none").values, "geo_rock_kind": j.rock_kind.fillna("none").values})
        k = int(round(GEO_SHARE_KM * 1000 / GEO_RES_M))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = (yy ** 2 + xx ** 2) <= k * k
        H, W = self.era_grid.shape
        share = np.full((len(lons), len(ERAS)), np.nan)
        dist = np.full((len(lons), len(ERAS)), np.nan)
        for i, (x, y) in enumerate(zip(xs, ys)):
            r, c = self._rc(x, y)
            if not (0 <= r < H and 0 <= c < W):
                continue
            dist[i] = self.dist_grid[:, r, c].astype(float)
            r0, r1, c0, c1 = max(r - k, 0), min(r + k + 1, H), max(c - k, 0), min(c + k + 1, W)
            win = self.era_grid[r0:r1, c0:c1]
            m = disk[(r0 - r + k):(r1 - r + k), (c0 - c + k):(c1 - c + k)] & (win > 0)
            if m.any():
                share[i] = np.bincount(win[m], minlength=len(ERAS) + 1)[1:] / m.sum()
        for e_i, e in enumerate(ERAS):
            out[f"geo_share5_{e}"] = share[:, e_i]
            out[f"geo_dist_{e}_km"] = np.where(np.isinf(dist[:, e_i]), np.nan, dist[:, e_i])
        R = 6371.0
        q = np.radians(np.c_[lats, lons])
        out["geo_cave_dist_km"] = self.cave_tree.query(q, k=1)[0][:, 0] * R
        out["geo_cave_n5"] = self.cave_tree.query_radius(q, r=5 / R, count_only=True)
        d, idx = self.oc_tree.query(q, k=1)
        out["oc_dist_km"] = d[:, 0] * R
        out["oc_n5"] = self.oc_tree.query_radius(q, r=5 / R, count_only=True)
        out["oc_nearest_rock"] = self.oc_rock[idx[:, 0]]
        out["oc_carbonate_dist_km"] = self.oc_carb_tree.query(q, k=1)[0][:, 0] * R
        pj = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs=4326), self.prov, how="left", predicate="within")
        pj = pj[~pj.index.duplicated()]
        out["oc_in_survey_prov"] = pj.prov_nam_t.fillna("").str.strip().isin(self.oc_provinces).astype(int).values
        return out
