# Inference with the saved final models (used by notebook 12, the web app in webapp/ and the tests — one implementation).
#   Predictor().predict_location(lat, lon)   task 2: potential (both backgrounds) + national percentile + fossil types + AOA flags
#   Predictor().predict_specimen(record)      task 1: genus top-k (+ species guess) routed like notebook 11
#   Predictor().predict_specimen_appearance(record, card, option)   task 1 when the fossil type is unknown: appearance answers
#                                             -> type weights from the registry -> genus model per type, mixed by the weights
# Features for new coordinates come from feature_lib (same code as training); Sentinel-2 is fetched live (internet).
import model_lib as ml  # noqa: E402  (import before torch users: OpenMP order)
import feature_lib as fl  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

T1 = ml.MODELS / "task1" / "final"
T2 = ml.MODELS / "task2" / "final"
CAT = ["pg_sw18_plate", "pg_me21_plate", "pg_me21_terrane", "geo_rock_sym", "geo_era_old", "geo_era_young", "geo_rock_kind", "oc_nearest_rock"]


class Predictor:
    def __init__(self):
        self.meta2 = json.loads((T2 / "final_meta.json").read_text(encoding="utf-8"))
        self.meta1 = json.loads((T1 / "final_meta.json").read_text(encoding="utf-8"))
        self.builder = fl.FeatureBuilder()
        self.deploy = self.meta2["deploy"]
        self.aoas = {k: ml.joblib.load(T2 / v) for k, v in self.meta2["aoa"].items()}
        sites = (pd.read_parquet(ml.FEAT / "all_features.parquet").merge(pd.read_parquet(ml.FEAT / "folds.parquet"), on="point_id")
                 .query("source == 'dmr_site'").reset_index(drop=True))
        sites["geo_rock_sym"] = sites.geo_rock_sym.astype(str)
        # ขั้น A แบบ feature engineering (notebook 14) ถ้า deploy เลือก "fe": + ธรณี+ / ผิวดิน + บริบทเชิงพื้นที่
        self.stage_a_fe = self.deploy.get("stage_A_variant") == "fe"
        self.stage_a = self.meta2["stage_A"]
        if self.stage_a_fe:
            meta_fe = json.loads((T2 / "final_meta_fe.json").read_text(encoding="utf-8"))
            self.stage_a = meta_fe["stage_A"]
            self.aoas.update({f"A_{bg}": ml.joblib.load(T2 / self.stage_a[bg]["aoa"]) for bg in self.deploy["backgrounds"]})
            sites = sites.merge(pd.read_parquet(ml.FEAT / "extra_features.parquet"), on="point_id", how="left")
            sites = pd.concat([sites, fl.spatial_context(sites, sites, sites.group_spatial_block.values, sites.group_spatial_block.values)], axis=1)
        self.sites = sites
        raw = gpd.read_file(fl.DATA / "dmr_fossil_sites" / "fossil_sites.geojson")[["OBJECTID", "NAME_TH", "PROVINCE"]]
        raw = raw.set_index(raw.OBJECTID.astype(int))
        ids = sites.point_id.str[4:].astype(int)
        self.site_names = (ids.map(raw.NAME_TH) + " (" + ids.map(raw.PROVINCE) + ")").values
        self.aoa0 = self.aoas[f"A_{self.deploy['default_background']}"]
        self.site_nn = NearestNeighbors(n_neighbors=3).fit(self.aoa0._z(sites))       # แหล่งที่ feature คล้ายที่สุด
        self.grid = pd.read_parquet(ml.MODELS / "task2" / "predictions" / "grid_5km_predictions.parquet")   # notebook 10b
        # ---- งานที่ 1
        self.genus = self.meta1["genus"]
        self.site_t1 = pd.read_parquet(ml.FEAT / "site_task1_features.parquet")       # ประเภทซากในแหล่ง + CNN PCA (notebook 11)
        self.text = ml.joblib.load(T1 / self.genus["text"])                            # TF-IDF (แหล่งเดิม) + WangchanBERTa PCA (แหล่งใหม่)
        self.site_fm = fl.site_formation_table()
        self.known_sites = set(self.genus["known_sites"])
        self.group_map = {g: set(c) for g, c in self.genus["group_map"].items()}
        self.prov = gpd.read_file(fl.DATA / "dmr_admin" / "provinces.geojson")[["prov_nam_t", "geometry"]]
        # ชนิดที่น่าจะเป็นของแต่ละสกุล: ชนิดที่พบบ่อยสุดในแหล่งนั้น -> ทั้งประเทศ (แบบเดียวกับ notebook 11)
        spec = pd.read_csv(fl.DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str, "site_OBJECTID": "Int64"},
                           encoding="utf-8-sig", low_memory=False)
        taxon = spec.sci_name.map(fl.parse_taxon)
        lab = spec.assign(genus_true=taxon.map(lambda t: t[0] if t else None), species_true=taxon.map(lambda t: t[1] if t else None))
        lab = lab[lab.genus_true.notna()].dropna(subset=["species_true"])
        self.species_nat = lab.groupby("genus_true").species_true.agg(lambda s: s.value_counts().idxmax())
        self.species_loc = lab.groupby(["site_OBJECTID", "genus_true"]).species_true.agg(lambda s: s.value_counts().idxmax())
        # ทายจาก "หน้าตา" (ไม่รู้ประเภทซาก): น้ำหนักประเภทและคำบรรยายส่วนของซากมาจากทะเบียนทั้งหมด (notebook 15 วัดผลแบบใช้เฉพาะ fold train)
        self.spec = spec[["f_type", "f_part", "f_group"]].copy()
        self.leaf_types = fl.appearance_leaf_types(self.spec)
        self.type_group = self.spec.dropna(subset=["f_type", "f_group"]).groupby("f_type").f_group.agg(lambda s: s.value_counts().idxmax()).to_dict()

    # ---------------------------------------------------------------- งานที่ 2
    def location_features(self, lat, lon, with_s2=True):
        df, patches = self.builder.tabular([lon], [lat], with_s2=with_s2)
        if not with_s2:                                   # ไม่ดึงภาพ: feature ดาวเทียมเป็นค่าว่าง (แบบเดียวกับตอนดึงภาพไม่ได้)
            for k in fl.S2_FEATURE_KEYS:
                df[f"s2_{k}"] = np.nan
            df["s2_item_id"], df["s2_cloud_frac"] = None, np.nan
        for c in CAT:
            if c in df:
                df[c] = df[c].astype(str)
        patch = patches[0] if patches and patches[0] is not None else None
        P = None if patch is None else (patch.astype("float32") * fl.S2_SCALE + fl.S2_OFFSET)[None]
        if self.stage_a_fe:
            df = df.join(fl.extra_features(df[["lon", "lat"]], pangea=df, geology=df))
            df = df.join(fl.spatial_context(df, self.sites, fl.block_group(df.lon, df.lat), self.sites.group_spatial_block.values))
        return df, P

    def _predict(self, stage, df, P):
        p = ml.predict_bundle(T2 / stage["tabular"], df)[0]
        if self.deploy["use_cnn"] and P is not None:
            p = (p + ml.predict_bundle(T2 / stage["cnn"], P)[0]) / 2
        return p

    def predict_location(self, lat, lon, top=3, with_s2=True):
        # -> (ผล dict, feature DataFrame, patch); ผลเป็นชนิดข้อมูลพื้นฐาน (ส่งเป็น JSON ได้)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df, P = self.location_features(lat, lon, with_s2=with_s2)
        out = {"lat": float(lat), "lon": float(lon)}
        for bg in self.deploy["backgrounds"]:
            p = float(self._predict(self.stage_a[bg], df, P))
            out[f"potential_{bg}"] = round(p, 3)
            out[f"percentile_{bg}"] = round(100 * float((self.grid[f"potential_{bg}"] < p).mean()), 1)   # เทียบกริด 5 กม. ทั้งประเทศ
        for bg in self.deploy["backgrounds"]:
            out[f"inside_{bg}"] = bool(self.aoas[f"A_{bg}"].inside(df)[0])
        out["inside_types"] = bool(self.aoas["B"].inside(df)[0])
        km = np.hypot((self.sites.lat - lat) * 111.0, (self.sites.lon - lon) * 111.0 * np.cos(np.radians(lat)))
        out["nearest_site_km"] = round(float(km.min()), 1)
        _, idx = self.site_nn.kneighbors(self.aoa0._z(df), n_neighbors=3)
        out["similar_sites"] = [str(self.site_names[i]) for i in idx[0]]
        pb = self._predict(self.meta2["stage_B"], df, P)
        cls = self.meta2["classes_B"]
        order = np.argsort(-pb)[:top]
        item = df.get("s2_item_id", pd.Series([None])).iat[0]
        out.update({
            "top_types": [(cls[i], round(float(pb[i]), 3)) for i in order],
            "rock_unit": str(df.geo_rock_sym.iat[0]), "rock_era": str(df.geo_era_old.iat[0]), "terrane": str(df.pg_me21_terrane.iat[0]),
            "s2_scene": None if item is None or pd.isna(item) else str(item),
            "warnings": [str(w.message) for w in caught if "Sentinel-2" in str(w.message)],
            "seconds": round(time.time() - t0, 1),
        })
        if out["s2_scene"] is None:                       # ขั้น A พึ่ง feature ดาวเทียมมาก (เช่น ภูเวียง 0.972 -> 0.137 เมื่อไม่มีภาพ)
            out["warnings"].append("ไม่มีภาพ Sentinel-2: โมเดลขั้น A พึ่ง feature ดาวเทียมมาก คะแนนโอกาสพบซากอาจคลาดเคลื่อนมาก — ควรดึงภาพเมื่อใช้งานจริง")
        return out, df, P

    # ---------------------------------------------------------------- งานที่ 1
    def specimen_features(self, record):
        # สลับตามกรณีแบบเดียวกับ notebook 11:
        #   แหล่งที่มีตัวอย่างระบุสกุลแล้ว -> feature ของแหล่ง (ตาราง ทธ. + ประเภทซาก + CNN PCA) + ข้อมูลตัวอย่าง (โมเดล known)
        #   แหล่งใหม่ / พิกัดใหม่ / ไม่ระบุแหล่ง -> ข้อมูลตัวอย่าง + หมวดหิน + ข้อความ (โมเดล new) ไม่ต้องดึงภาพดาวเทียม
        rec = dict(record)
        sid = rec.get("site_OBJECTID")
        known = sid is not None and pd.notna(sid) and int(sid) in self.known_sites
        if known:
            df = self.sites[self.sites.point_id.eq(f"dmr_{int(sid)}")].reset_index(drop=True).copy()
            df = df.merge(self.site_t1.drop(columns=["oid"]), on="point_id", how="left")
            for c in CAT:
                if c in df:
                    df[c] = df[c].astype(str)
        else:
            df = pd.DataFrame(index=[0])
        if not rec.get("province") and rec.get("lat") is not None:                 # มีแต่พิกัด -> หาจังหวัดจากขอบเขต ทธ.
            hit = self.prov[self.prov.contains(gpd.points_from_xy([rec["lon"]], [rec["lat"]])[0])]
            rec["province"] = hit.prov_nam_t.iat[0] if len(hit) else None
        for c in ["com_name", "f_type", "f_group", "f_part", "province", "district"]:
            df[c] = str(rec.get(c) if rec.get(c) is not None else "nan")
        # กัน data leak แบบเดียวกับตอน train: ชื่อสามัญที่ไม่ใช่ "ชื่อทั่วไป" (เช่นชื่อไทยของชนิด) แทนด้วย f_type
        df["com_name_generic"] = df.com_name.where(df.com_name.isin(self.genus["com_vocab"]), df.f_type)
        txt = fl.specimen_text(df)
        if known:                                          # แหล่งเดิม: + TF-IDF ของข้อความ
            df = df.join(self.text["tfidf"].transform(txt))
        else:                                              # แหล่งใหม่: + หมวดหิน (+ ThaiLex) + WangchanBERTa
            if rec.get("formation"):
                fm, gg = str(rec["formation"]), str(rec.get("geo_group") or "Unknown")
            elif sid is not None and pd.notna(sid):
                fm, gg = self.site_fm.FORMATION.get(int(sid), "Unknown"), self.site_fm.GEO_GROUP.get(int(sid), "Unknown")
            elif rec.get("lat") is not None:                # พิกัดใหม่: หมวดหินจากแผนที่ธรณี ทธ. ที่ตำแหน่งนั้น
                fm, gg = fl.map_formation_name(self.builder.geology.features([rec["lon"]], [rec["lat"]]).geo_rock_sym.iat[0]), "Unknown"
            else:
                fm, gg = "Unknown", "Unknown"
            df["fm_FORMATION"], df["fm_GEO_GROUP"] = str(fm), str(gg)
            df = df.join(fl.formation_features([fm])).join(self.text["bert"].transform(txt))
        return df, known

    def species_guess(self, genus, site_id=None):
        if genus == "อื่น ๆ":
            return None
        if site_id is not None and pd.notna(site_id) and (int(site_id), genus) in self.species_loc.index:
            return self.species_loc[(int(site_id), genus)]
        return self.species_nat.get(genus)

    def _genus_probs(self, record):
        df, known = self.specimen_features(record)
        P = ml.predict_bundle(T1 / self.genus["models"]["known" if known else "new"], df)
        groups = [fl.specimen_classes(df.f_type.iat[0], df.com_name_generic.iat[0])]
        P = ml.consistency_filter(P, self.genus["classes"], groups, self.group_map)[0]       # ตัดสกุลที่ไม่อยู่กลุ่มซากเดียวกัน
        return P, df, known, groups

    def predict_specimen(self, record, top=3):
        # top = [(สกุล, ความน่าจะเป็น, กลุ่ม)]: กลุ่มมีค่าเฉพาะ "อื่น ๆ" (= สกุลพบน้อยของประเภทซากนั้น) นอกนั้นเป็น None
        P, df, known, groups = self._genus_probs(record)
        order = np.argsort(-P)[:top]
        sid = record.get("site_OBJECTID")
        oth = fl.other_group(record.get("f_type"))
        top_g = [(self.genus["classes"][i], round(float(P[i]), 3), oth if self.genus["classes"][i] == "อื่น ๆ" else None) for i in order]
        fm = df.get("fm_FORMATION", pd.Series([None])).iat[0]
        return {"route": "แหล่งเดิม" if known else "แหล่งใหม่", "formation": None if fm is None else str(fm), "groups": sorted(groups[0]),
                "top": top_g, "species": [self.species_guess(g, sid) for g, _, _ in top_g]}

    def predict_specimen_appearance(self, record, card, option=None, top=3):
        # ไม่รู้ประเภทซาก: คำตอบ "หน้าตา" -> น้ำหนักประเภท (ทะเบียน) -> ทายสกุลทีละประเภทที่เป็นไปได้ แล้วรวมตามน้ำหนัก
        #   P(สกุล) = Σ_ประเภท w(ประเภท | คำตอบ) · P(สกุล | ประเภท, ข้อมูลอื่น)   — ส่วนของซากใช้คำกลางของการ์ด (notebook 15)
        #   "อื่น ๆ" แยกตามกลุ่มซาก (fl.other_group): Σ w(ประเภท) · P("อื่น ๆ" | ประเภท) ของประเภทในกลุ่มเดียวกัน เป็นรายการของตัวเอง
        #   ไม่รวมข้ามกลุ่ม — ไม่งั้น "อื่น ๆ" สะสมคะแนนจากทุกกลุ่มจนขึ้นอันดับ 1 ทั้งที่บอกไม่ได้ว่าเป็นกลุ่มไหน
        leaf = fl.appearance_leaf_id(card, option)
        w = fl.appearance_type_weights(leaf, self.spec, leaf_types=self.leaf_types)
        classes = self.genus["classes"]
        oth_i = classes.index("อื่น ๆ")
        P, groups, others = 0.0, set(), {}
        for t, wt in w.items():
            rec = dict(record, f_type=t, f_group=self.type_group.get(t),
                       f_part=record.get("f_part") or fl.appearance_part(t, leaf, self.spec))
            Pt, df, known, g = self._genus_probs(rec)
            P, groups = P + wt * Pt, groups | set(g[0])
            grp = fl.other_group(t)
            others[grp] = others.get(grp, 0.0) + float(wt * Pt[oth_i])
        entries = [(classes[i], float(P[i]), None) for i in range(len(classes)) if i != oth_i] + [("อื่น ๆ", v, g) for g, v in others.items()]
        entries.sort(key=lambda e: -e[1])
        sid = record.get("site_OBJECTID")
        top_g = [(g, round(p, 3), grp) for g, p, grp in entries[:top]]
        fm = df.get("fm_FORMATION", pd.Series([None])).iat[0]
        return {"route": "แหล่งเดิม" if known else "แหล่งใหม่", "formation": None if fm is None else str(fm), "groups": sorted(groups),
                "top": top_g, "species": [self.species_guess(g, sid) for g, _, _ in top_g],
                "mode": "appearance", "leaf": leaf, "types": [(t, round(float(v), 3)) for t, v in w.items()]}
