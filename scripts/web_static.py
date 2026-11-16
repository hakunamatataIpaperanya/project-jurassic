# ข้อมูลของเว็บแบบไม่มีเซิร์ฟเวอร์ (static site บน Cloudflare Pages) — ทุกอย่างคำนวณจากโค้ดชุดเดียวกับ webapp/server.py / predict_lib
#   webapp/data/<endpoint>.json : ผลของทุก GET /api/<endpoint> (ไบต์เดียวกับที่ server ส่ง; ขอบเขตจังหวัด / ยุคของหิน server ก็อ่านจากไฟล์นี้)
#   webapp/data/cells.json      : ผลวิเคราะห์จุดของทุกช่องกริด 5 กม. (= predict_location ที่ศูนย์กลางช่อง) — เว็บเปิดดูช่องที่ใกล้จุดที่คลิกที่สุด
#   models/web/specimen.json + specimen.bin : ตารางของขั้นทายสกุลที่เบราว์เซอร์ต้องใช้ (webapp/static/local.js)
#       ระดับของคอลัมน์หมวดหมู่, feature ของแหล่งเดิม, หมวดหิน + ThaiLex, TF-IDF + SVD, PCA ของ WangchanBERTa,
#       embedding ที่คำนวณไว้แล้ว (ข้อความในทะเบียน + ข้อความของโหมด "หน้าตา") ตัวกรองกลุ่มซาก, ชนิด, คำถามหน้าตา
#   โมเดล ONNX อยู่ใน scripts/web_models.py
# รัน:  .venv/bin/python scripts/web_static.py            (หลังเปลี่ยนโมเดล / ข้อมูล แล้วรัน scripts/build_static_site.py)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp import server  # noqa: E402  (ตั้ง sys.path ของ notebooks/ และลำดับ import ของ OpenMP ให้)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ml, fl, pl = server.ml, server.fl, server.pl

DATA_OUT = server.LAYER_DIR
MODEL_OUT = ml.MODELS / "web"
GET_ENDPOINTS = {"meta": server.meta, "grid": server.grid, "pbdb": server.pbdb, "options": server.options, "showcase": server.showcase,
                 "period_profile": server.period_profile, "provinces": server.provinces, "geology": server.geology}
SPEC_COLS = ["com_name", "f_type", "f_group", "f_part", "province", "district", "com_name_generic"]
TINY = 1.0000000180025095e-35                     # kZeroThreshold ของ LightGBM (float32 ของ 1e-35): |x| ≤ ค่านี้ LightGBM นับเป็น 0


def _num(v):
    v = float(v)
    return None if np.isnan(v) else v


# ---------------------------------------------------------------- ผลวิเคราะห์จุดของทุกช่องกริด
def grid_features(P):
    # feature ของกริด 5 กม. จาก cache ของ notebook 10b (ไม่ดึงภาพใหม่: patch ของทุกช่องอยู่ใน data/features/s2/grid_patches)
    grid = pd.read_parquet(fl.FEAT / "grid_5km.parquet")
    if grid.point_id.tolist() != P.grid.point_id.tolist():
        raise ValueError("grid_5km.parquet กับ grid_5km_predictions.parquet เรียงไม่ตรงกัน — รัน notebook 10b ใหม่")
    G = (grid.merge(fl.batch_pangea(grid, tag="grid_"), on="point_id").merge(fl.batch_dem(grid, tag="grid_"), on="point_id")
         .merge(fl.batch_geology(grid, tag="grid_"), on="point_id")
         .merge(fl.batch_s2_indices(grid, fl.FEAT / "s2" / "grid_patches", tag="grid_"), on="point_id", how="left"))
    for c in pl.CAT:
        if c in G:
            G[c] = G[c].astype(str)
    return G


def build_cells(P):
    # predict_location ที่ศูนย์กลางทุกช่อง (ค่าเดียวกับ server เมื่อคลิกตรงศูนย์กลางช่อง) แบบ column: ตำแหน่งที่ i = ช่องที่ i ของ grid.json
    #   ยกเว้นจังหวัด และระยะถึงแหล่ง ทธ. ที่ใกล้ที่สุด ซึ่งเว็บคำนวณจากจุดที่คลิกจริง (จึงเก็บพิกัดแหล่งไว้ด้วย)
    if P.stage_a_fe or P.deploy["use_cnn"]:
        raise NotImplementedError("ขั้น A แบบ fe / CNN ต้องมี feature เพิ่ม — ยังไม่รองรับในเว็บแบบ static")
    G = grid_features(P)
    bgs = P.deploy["backgrounds"]
    out = {"n": len(G), "lon": np.round(G.lon.values, 4).tolist(), "lat": np.round(G.lat.values, 4).tolist(),
           "backgrounds": bgs, "potential": {}, "percentile": {}, "inside": {}}
    for bg in bgs:
        p = ml.predict_bundle(pl.T2 / P.stage_a[bg]["tabular"], G)          # = Predictor._predict ทีละแถว (ไม่ใช้ CNN)
        if not np.array_equal(p, P.grid[f"potential_{bg}"].values):
            raise ValueError(f"โมเดลขั้น A ({bg}) ให้ผลต่างจาก grid_5km_predictions.parquet — รัน notebook 10b ใหม่")
        srt = np.sort(P.grid[f"potential_{bg}"].values)
        below = np.searchsorted(srt, p, side="left")                    # = (กริด < p).sum() แบบเดียวกับ predict_location
        out["potential"][bg] = [round(float(v), 3) for v in p]
        out["percentile"][bg] = [round(100 * float(k / len(srt)), 1) for k in below]
        out["inside"][bg] = P.aoas[f"A_{bg}"].inside(G).astype(int).tolist()
    out["inside"]["types"] = P.aoas["B"].inside(G).astype(int).tolist()
    # ขั้น B (random forest): ใช้ค่าที่บันทึกไว้ (notebook 10b = ค่าที่แผนที่แสดง) — คำนวณซ้ำได้ต่างกันราว 1e-16 ตามลำดับการรวมของหลายเธรด
    #   ซึ่งพอทำให้ค่าที่อยู่กึ่งกลางการปัดทศนิยม 3 ตำแหน่งพอดีสลับไปมา (ไฟล์ไม่เหมือนเดิมทุกครั้งที่ export)
    saved = P.grid[[f"type_{c}" for c in P.meta2["classes_B"]]].values
    if not np.allclose(ml.predict_bundle(pl.T2 / P.meta2["stage_B"]["tabular"], G), saved, rtol=0, atol=1e-9):
        raise ValueError("โมเดลขั้น B ให้ผลต่างจาก grid_5km_predictions.parquet — รัน notebook 10b ใหม่")
    pb = saved
    order = np.argsort(-pb, axis=1)[:, :3]
    out["classes_B"] = P.meta2["classes_B"]
    out["top_types"] = [[[int(i), round(float(pb[r, i]), 3)] for i in row] for r, row in enumerate(order)]
    _, nn = P.site_nn.kneighbors(P.aoa0._z(G), n_neighbors=3)
    out["site_names"] = [str(n) for n in P.site_names]
    out["sites_lonlat"] = [[float(a), float(b)] for a, b in zip(P.sites.lon, P.sites.lat)]
    out["similar"] = nn.tolist()
    for key, col in [("rock_unit", "geo_rock_sym"), ("rock_era", "geo_era_old"), ("terrane", "pg_me21_terrane"), ("s2_scene", "s2_item_id")]:
        vals = G[col].where(G[col].notna(), None)
        names = sorted({v for v in vals if v is not None})
        idx = {v: i for i, v in enumerate(names)}
        out[f"{key}_names"] = names
        out[key] = [-1 if v is None else idx[v] for v in vals]
    out["no_s2_warning"] = ("ไม่มีภาพ Sentinel-2: โมเดลขั้น A พึ่ง feature ดาวเทียมมาก คะแนนโอกาสพบซากอาจคลาดเคลื่อนมาก — ควรดึงภาพเมื่อใช้งานจริง")
    return out


# ---------------------------------------------------------------- ตารางของขั้นทายสกุล
class _Bin:
    # ตัวเลขชุดใหญ่เก็บเป็นไบนารี (float64 / float32 little-endian) — JSON เก็บแค่ตำแหน่ง
    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr, dtype="<f8"):
        a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
        pad = (-self.size) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        ref = {"offset": self.size, "dtype": "f8" if dtype == "<f8" else "f4", "shape": list(a.shape)}
        self.parts.append(a.tobytes())
        self.size += a.nbytes
        return ref

    def bytes(self):
        return b"".join(self.parts)


def _appearance_texts(P):
    # ข้อความทั้งหมดที่โหมด "หน้าตา" ป้อน WangchanBERTa เมื่อผู้ใช้ไม่ได้พิมพ์ส่วนของซากเอง: ชื่อทั่วไป (หรือประเภท) | ประเภท | คำกลางของการ์ด
    texts = set()
    for lid in fl.appearance_leaves():
        for t in fl.appearance_type_weights(lid, P.spec, leaf_types=P.leaf_types).index:
            part = fl.appearance_part(t, lid, P.spec)
            for com in [None, *P.genus["com_vocab"]]:
                df = pd.DataFrame({"com_name": [str(com) if com else "nan"], "f_type": [t], "f_part": [part if part else "nan"]})
                df["com_name_generic"] = df.com_name.where(df.com_name.isin(P.genus["com_vocab"]), df.f_type)
                texts.add(fl.specimen_text(df).iat[0])
    return texts


def embedding_table(P, extra_texts):
    # PCA-32 ของ WangchanBERTa ที่คำนวณแบบเดียวกับ server (cache ของ text_embed; ข้อความใหม่เข้ารหัสทีละข้อความเหมือนคำขอบนเว็บ)
    #   ไม่เขียนเพิ่มลง cache ของโปรเจกต์ — ข้อความใหม่เก็บแยกใน models/web/text_emb_extra.parquet
    cache = pd.read_parquet(fl.FEAT / "text_emb" / "wangchanberta.parquet").set_index("text")
    extra_f = MODEL_OUT / "text_emb_extra.parquet"
    extra = pd.read_parquet(extra_f).set_index("text") if extra_f.exists() else pd.DataFrame(columns=cache.columns)
    todo = sorted(set(extra_texts) - set(cache.index) - set(extra.index))
    if todo:
        emb = np.vstack([fl.encode_texts([t], "wangchanberta") for t in ml.tqdm_auto()(todo, desc="WangchanBERTa")])
        new = pd.DataFrame(emb, index=pd.Index(todo, name="text"), columns=cache.columns)
        extra = pd.concat([extra, new]) if len(extra) else new
        extra.reset_index().to_parquet(extra_f, index=False)
    allv = pd.concat([cache, extra.astype(cache.dtypes.iloc[0])])
    texts = sorted(set(allv.index))
    E = allv.loc[texts].values
    pca = P.text["bert"].pca_
    Z = np.vstack([pca.transform(E[i:i + 1]) for i in range(len(E))])      # ทีละแถวเหมือนคำขอบนเว็บ (ผลรวมเรียงลำดับเดียวกัน)
    return texts, Z.astype("float32")                     # ค่าที่ป้อนโมเดลถูกแปลงเป็น float32 อยู่แล้ว (TabularPreprocessor)


def build_specimen(P):
    g, b = P.genus, _Bin()
    classes = list(g["classes"])
    models = {}
    for route in ("known", "new"):
        bundle = ml.joblib.load(pl.T1 / g["models"][route])
        pre, m = bundle["pre"], bundle["model"]
        if pre.mode != "tree" or pre.fill_nan or list(bundle["classes"]) != classes or list(m.present_) != list(range(len(classes))):
            raise ValueError(f"โมเดล {route}: รูปแบบไม่ตรงกับที่ local.js รองรับ")
        models[route] = {"file": f"genus_{route}.onnx", "num_cols": pre.num_cols, "cat_cols": pre.cat_cols, "levels": pre.levels_,
                         "zero_tiny": type(m).__name__.startswith("LGBM")}
    # แหล่งเดิม (มีตัวอย่างระบุสกุลแล้ว): feature ของแหล่งทุกคอลัมน์ที่โมเดล known ใช้ ยกเว้นข้อมูลตัวอย่างและ TF-IDF
    kpre = models["known"]
    site_cols = [c for c in kpre["num_cols"] + kpre["cat_cols"] if c not in SPEC_COLS and not c.startswith("tfidf_")]
    known = {}
    for sid in sorted(P.known_sites):
        df = P.sites[P.sites.point_id.eq(f"dmr_{int(sid)}")].reset_index(drop=True).copy()
        if len(df) != 1:
            raise ValueError(f"ไม่พบ feature ของแหล่ง {sid}")
        df = df.merge(P.site_t1.drop(columns=["oid"]), on="point_id", how="left")
        for c in pl.CAT:
            if c in df:
                df[c] = df[c].astype(str)
        known[str(int(sid))] = [str(df[c].iat[0]) if c in kpre["cat_cols"] else _num(df[c].iat[0]) for c in site_cols]
    # หมวดหิน: ชื่อที่เป็นไปได้ทั้งหมด (ตารางแหล่ง notebook 07 + ชื่อจากแผนที่ธรณี) -> feature ThaiLex
    legend = pd.read_csv(fl.GEO_DIR / "rock_unit_250k_legend.csv", encoding="utf-8-sig")
    sym_fm = {str(s): fl.map_formation_name(s) for s in legend.TEXT_SYM.dropna()}
    names = sorted(set(P.site_fm.FORMATION.astype(str)) | set(sym_fm.values()) | {"Unknown"})
    ff = fl.formation_features(names)
    formations = {n: [_num(v) for v in row] for n, row in zip(names, ff.values)}
    site_fm = {str(int(i)): [str(a), str(c)] for i, a, c in zip(P.site_fm.index, P.site_fm.FORMATION, P.site_fm.GEO_GROUP)}
    # ข้อความ: TF-IDF (char_wb 2–4) + SVD สำหรับแหล่งเดิม, PCA ของ WangchanBERTa สำหรับแหล่งใหม่
    vec, svd, pca = P.text["tfidf"].vec_, P.text["tfidf"].svd_, P.text["bert"].pca_
    if not (vec.analyzer == "char_wb" and vec.lowercase and vec.sublinear_tf and vec.norm == "l2" and vec.use_idf and not pca.whiten):
        raise ValueError("ตั้งค่า TF-IDF / PCA ไม่ตรงกับที่ local.js รองรับ")
    vocab = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    texts, Z = embedding_table(P, _appearance_texts(P))
    gen = {t[0] for t in pd.read_csv(fl.DATA / "processed" / "fossil_list_final.csv", usecols=["sci_name"], encoding="utf-8-sig",
                                     low_memory=False).sci_name.map(fl.parse_taxon) if t}
    if fl.genus_name_pattern().pattern != _genus_pattern_py(sorted(gen, key=len, reverse=True)):
        raise ValueError("genus_name_pattern เปลี่ยนรูปแบบ — แก้ local.js (cleanPart) และ _genus_pattern_py ให้ตรงกัน")
    # ยาวก่อนสั้น (ชื่อยาวที่ขึ้นต้นด้วยชื่อสั้นต้องจับก่อน); ชื่อยาวเท่ากันเรียงตามตัวอักษร — ลำดับในกลุ่มนี้ไม่มีผลต่อการจับ
    gen = sorted(gen, key=lambda g: (-len(g), g))
    leaves = {lid: [[t, float(w), fl.appearance_part(t, lid, P.spec)]                  # [ประเภท, น้ำหนัก, ส่วนของซากที่ป้อนโมเดล]
                    for t, w in fl.appearance_type_weights(lid, P.spec, leaf_types=P.leaf_types).items()] for lid in fl.appearance_leaves()}
    spec = {
        "classes": classes, "other": ml.OTHER_GENUS, "models": models, "site_cols": site_cols, "known_sites": known,
        "com_vocab": list(g["com_vocab"]), "group_map": {k: sorted(v) for k, v in P.group_map.items()},
        "fossil_classes": fl.FOSSIL_CLASSES, "not_fossil": fl.NOT_FOSSIL_RE.pattern, "group_names": fl._GROUP_NAME,
        "formation_cols": list(ff.columns), "formations": formations, "site_formation": site_fm, "symbol_formation": sym_fm,
        "species_site": {f"{int(s)}|{k}": v for (s, k), v in P.species_loc.items()}, "species_nat": P.species_nat.to_dict(),
        "type_group": P.type_group, "appearance": {"cards": {c["id"]: [o["id"] for o in c.get("options", [])] for c in fl.APPEARANCE},
                                                   "leaves": leaves},
        "genus_names": gen, "whitespace": _whitespace_chars(),
        "tfidf": {"vocab": vocab, "idf": b.add(vec.idf_), "svd": b.add(svd.components_), "ngram": list(vec.ngram_range)},
        "bert": {"pca_mean": b.add(pca.mean_), "pca_components": b.add(pca.components_), "max_length": 64,
                 "texts": texts, "table": b.add(Z, "<f4")},
    }
    return spec, b.bytes()


_KEEP = r"(?!(?:teeth|tooth|bone|bones|shell|skull|jaw|vertebra|vertebrae|neural|spine|fragment|fragments|and|or|with)\b)"


def _genus_pattern_py(gen):
    # รูปแบบเดียวกับ feature_lib.genus_name_pattern — local.js สร้าง regex แบบเดียวกันจากรายชื่อสกุล (ตรวจตอน export)
    import re
    return r"\b(?:" + "|".join(map(re.escape, gen)) + r")\b(?:\s+(?:cf\.|aff\.)?\s*(?:sp\.|spp\.|" + _KEEP + r"[a-z][a-z\-]{2,}))?"


def _whitespace_chars():
    # อักขระที่ Python นับเป็นช่องว่าง (\s ของ re / str.split / str.strip) — JS ใช้ชุดเดียวกันแทน \s ของตัวเอง
    return "".join(chr(c) for c in range(0x110000) if chr(c).isspace())


# ---------------------------------------------------------------- เขียนไฟล์
def write_json(path, obj):
    path.write_bytes(server._dumps(obj))
    return path


def main():
    P = server.predictor()
    DATA_OUT.mkdir(exist_ok=True)
    MODEL_OUT.mkdir(exist_ok=True)
    for name, fn in GET_ENDPOINTS.items():
        f = DATA_OUT / f"{name}.json"
        # provinces / geology: server ส่งไฟล์นี้เอง จึงคำนวณใหม่จากข้อมูลดิบ (server.STATIC_LAYERS) แทนการอ่านไฟล์เดิม
        f.write_bytes(server._dumps(server.STATIC_LAYERS[name]()) if name in server.STATIC_LAYERS else fn().body)
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.2f} MB")
    f = write_json(DATA_OUT / "cells.json", build_cells(P))
    print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.2f} MB")
    spec, blob = build_specimen(P)
    write_json(MODEL_OUT / "specimen.json", spec)
    (MODEL_OUT / "specimen.bin").write_bytes(blob)
    print(f"models/web/specimen.json {len(server._dumps(spec)) / 1e6:.2f} MB + specimen.bin {len(blob) / 1e6:.2f} MB "
          f"({len(spec['bert']['texts']):,} ข้อความที่คำนวณ embedding ไว้แล้ว)")


if __name__ == "__main__":
    main()
