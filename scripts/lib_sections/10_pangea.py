# Paleogeography (Pangea) features for arbitrary points: same logic as notebook 08
# (plate partition -> finite rotation per time -> PaleoDEM at the Scotese & Wright reconstructed position).
import glob as _glob

PG_TIMES = [480, 400, 300, 280, 270, 260, 250, 240, 230, 200, 150, 120, 66, 0]
PG_MODELS = {"sw18": "scotese_and_wright2018", "me21": "merdith2021"}


def plate_model_files(name):
    # ไฟล์ StaticPolygons / Rotations ของแบบจำลองแผ่นเปลือกโลก: ใช้ไฟล์ที่ดาวน์โหลดไว้ใน data/paleo ก่อน
    # (PlateModelManager จะเช็กเซิร์ฟเวอร์ GPlates ทุกครั้งแม้มีไฟล์แล้ว — ถ้าเซิร์ฟเวอร์ล่ม การทำนายพิกัดใหม่จะล้มทั้งที่ข้อมูลครบ)
    d = DATA / "paleo" / name
    static = sorted(_glob.glob(str(d / "StaticPolygons" / "*.gpml*")))
    rot = sorted(_glob.glob(str(d / "Rotations" / "*.rot")))
    if static and rot:
        return static, rot
    from plate_model_manager import PlateModelManager
    m = PlateModelManager().get_model(name, data_dir=str(DATA / "paleo"))     # ยังไม่มีไฟล์ -> ดาวน์โหลด
    return m.get_layer("StaticPolygons"), m.get_rotation_model()


class Pangea:
    def __init__(self):
        import pygplates
        self.pg = pygplates
        self.models = {}
        for key, name in PG_MODELS.items():
            static_files, rot_files = plate_model_files(name)
            static = pygplates.FeatureCollection()
            for f in static_files:
                static.add(pygplates.FeatureCollection(f))
            rot = pygplates.RotationModel(rot_files)
            self.models[key] = {"rot": rot, "part": pygplates.PlatePartitioner(static, rot)}
        self.dem_files = {}
        for f in _glob.glob(str(DATA / "paleo" / "paleodem_0.2deg" / "paleoDEM_0.2Deg_grids" / "*.nc")):
            self.dem_files[float(re.search(r"_([\d.]+)Ma\.nc$", f).group(1))] = f
        spec = json.loads((FEAT / "pangea_feature_list.json").read_text(encoding="utf-8"))
        self.columns = spec["numeric"] + [c for c in spec["categorical"] if c not in spec["numeric"]]

    def _dem_file(self, t):
        return self.dem_files[min(self.dem_files, key=lambda m: abs(m - t))]

    def features(self, lons, lats):
        import xarray as xr
        pg = self.pg
        lons, lats = np.asarray(lons, float), np.asarray(lats, float)
        out = pd.DataFrame(index=range(len(lons)))
        for key, mdl in self.models.items():
            plate, name, begin = [], [], []
            for lo, la in zip(lons, lats):
                poly = mdl["part"].partition_point(pg.PointOnSphere(la, lo))
                if poly is None:
                    plate.append(np.nan), name.append(None), begin.append(np.nan)
                else:
                    f = poly.get_feature()
                    plate.append(f.get_reconstruction_plate_id()), name.append(f.get_name()), begin.append(f.get_valid_time()[0])
            out[f"pg_{key}_plate"], out[f"pg_{key}_terrane"], out[f"pg_{key}_begin_ma"] = plate, name, begin
            plate_arr, begin_arr = np.asarray(plate, float), np.asarray(begin, float)
            pts = [pg.PointOnSphere(la, lo) for lo, la in zip(lons, lats)]
            for t in PG_TIMES:
                la_t, lo_t = np.full(len(lons), np.nan), np.full(len(lons), np.nan)
                for pid in np.unique(plate_arr[~np.isnan(plate_arr)]):
                    rot = mdl["rot"].get_rotation(float(t), int(pid))      # หนึ่ง rotation ต่อ (แผ่น, เวลา)
                    for i in np.nonzero((plate_arr == pid) & (begin_arr >= t))[0]:
                        la_t[i], lo_t[i] = (rot * pts[i]).to_lat_lon()
                out[f"pg_{key}_lat_{t}"], out[f"pg_{key}_lon_{t}"] = la_t, lo_t
                out[f"pg_{key}_abslat_{t}"] = np.abs(la_t)
            out[f"pg_{key}_drift_lat_250_0"] = out[f"pg_{key}_lat_0"] - out[f"pg_{key}_lat_250"]
        for t in PG_TIMES:
            z = np.full(len(lons), np.nan)
            ok = out[f"pg_sw18_lat_{t}"].notna().values
            if ok.any():
                with xr.open_dataset(self._dem_file(t)) as ds:
                    z[ok] = ds.z.interp(lat=xr.DataArray(out.loc[ok, f"pg_sw18_lat_{t}"].values, dims="p"),
                                        lon=xr.DataArray(out.loc[ok, f"pg_sw18_lon_{t}"].values, dims="p")).values
            out[f"pg_dem_{t}"] = z
            out[f"pg_marine_{t}"] = np.where(np.isnan(z), np.nan, (z < 0).astype(float))
        marine = out[[f"pg_marine_{t}" for t in PG_TIMES]]
        out["pg_n_marine"] = marine.sum(axis=1, min_count=1)
        out["pg_youngest_marine_ma"] = marine.apply(lambda r: min([t for t, v in zip(PG_TIMES, r.values) if v == 1], default=np.nan), axis=1)
        return out[[c for c in self.columns if c in out.columns]]
