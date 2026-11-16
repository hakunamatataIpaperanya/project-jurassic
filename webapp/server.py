# Project Jurassic web app: map of the task-2 predictions + the final models run live (FastAPI).
#   sh scripts/run_webapp.sh   ->  http://127.0.0.1:8050  (PORT=8051 sh scripts/run_webapp.sh to change)
# All predictions go through notebooks/predict_lib.py (same code as notebook 12 and the tests), so the web app gives exactly
# what the notebooks give. Live coordinates fetch Sentinel-2 from Microsoft Planetary Computer (needs internet).
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import model_lib as ml  # noqa: E402  (import before torch users: OpenMP order)
import feature_lib as fl  # noqa: E402
import predict_lib as pl  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

STATIC = Path(__file__).parent / "static"
LAYER_DIR = Path(__file__).parent / "data"          # ชั้นข้อมูลที่ไม่เปลี่ยน (ขอบเขตจังหวัด, ยุคของหิน) export ไว้ด้วย scripts/web_static.py
# ข้อมูล / โมเดลทั้งหมดต้องมาจากโฟลเดอร์ของโปรเจกต์นี้เท่านั้น
assert all(Path(d).resolve().is_relative_to(ROOT) for d in (ml.DATA, ml.MODELS, fl.DATA, pl.T1, pl.T2)), "data path outside project"
THAI_BOX = (97.3, 105.7, 5.6, 20.5)                 # lon min/max, lat min/max (ขอบเขตเดียวกับที่ตรวจใน notebook 06)

app = FastAPI(title="Project Jurassic — แผนที่ซากดึกดำบรรพ์ประเทศไทย")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def no_stale_pages(request, call_next):
    # หน้าเว็บ / ไฟล์ static: ให้เบราว์เซอร์ถามเซิร์ฟเวอร์ทุกครั้ง (ใช้ ETag) — ไม่งั้นแก้โค้ดแล้วผู้ใช้ยังเห็นเวอร์ชันเก่า
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------- รับมือคนเข้าเยอะ
# การทำนายสดทำทีละคำขอ (_LOCK) และใช้เวลาราว 10–20 วินาที: ถ้าคิวยาวเกิน Cloudflare Tunnel จะตัดที่ 100 วินาที (error 524)
# จึงรับคิวทำนายได้ไม่เกิน PJ_MAX_PREDICT คำขอ และคำขอทั้งหมดพร้อมกันไม่เกิน PJ_MAX_INFLIGHT; เกินแล้วตอบ 503 ทันที
# พร้อม Retry-After: API ตอบ JSON {"busy": true} (หน้าเว็บแสดงการ์ดขอโทษ), หน้าเว็บตอบหน้า busy.html; ไฟล์ static ไม่จำกัด
MAX_INFLIGHT = int(os.environ.get("PJ_MAX_INFLIGHT", "48"))
MAX_PREDICT = int(os.environ.get("PJ_MAX_PREDICT", "4"))
HEAVY = {"/api/predict_location", "/api/predict_specimen"}
BUSY_HTML = STATIC / "busy.html"
BUSY_TEXT = "ขออภัย ตอนนี้มีผู้ใช้งานพร้อมกันจำนวนมาก กรุณาลองใหม่อีกครั้งในอีกสักครู่"
_LOAD = {"all": 0, "predict": 0}                      # นับในลูป asyncio เดียว (middleware) จึงไม่ต้องใช้ lock


def _busy(request, retry):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"busy": True, "retry_after": retry, "detail": BUSY_TEXT}, status_code=503,
                            headers={"Retry-After": str(retry), "Cache-Control": "no-store"})
    return Response(BUSY_HTML.read_bytes(), status_code=503, media_type="text/html; charset=utf-8",
                    headers={"Retry-After": str(retry), "Cache-Control": "no-store"})


@app.middleware("http")
async def limit_load(request, call_next):
    path = request.url.path
    if path.startswith("/static/") or path == "/busy":
        return await call_next(request)
    heavy = path in HEAVY
    if _LOAD["all"] >= MAX_INFLIGHT or (heavy and _LOAD["predict"] >= MAX_PREDICT):
        return _busy(request, 20 if heavy else 10)
    _LOAD["all"] += 1
    _LOAD["predict"] += heavy
    try:
        return await call_next(request)
    finally:
        _LOAD["all"] -= 1
        _LOAD["predict"] -= heavy


@app.get("/busy")
def busy_preview():
    # ดูหน้าขอโทษได้ตลอด (หน้าเดียวกับที่ส่งเมื่อคนเข้าเยอะ แต่ตอบ 200)
    return Response(BUSY_HTML.read_bytes(), media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")

_STATE = {}
_LOCK = threading.Lock()                             # feature_lib เปิดไฟล์ราสเตอร์ร่วมกัน: ทำนายทีละคำขอ


def predictor():
    with _LOCK:
        if "pred" not in _STATE:
            _STATE["pred"] = pl.Predictor()
        return _STATE["pred"]


def _cached(key, build):
    if key not in _STATE:
        _STATE[key] = build()
    return _STATE[key]


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_default).encode("utf-8")


def _json(obj):
    return Response(_dumps(obj), media_type="application/json")


def _static_layer(name, build):
    # ชั้นข้อมูลที่ไม่เปลี่ยน: ส่งไฟล์ webapp/data/<name>.json ที่ export ไว้ — ไม่มีไฟล์จึงคำนวณจากข้อมูลดิบ (ผลเหมือนกันทุกไบต์)
    def load():
        f = LAYER_DIR / f"{name}.json"
        return f.read_bytes() if f.exists() else _dumps(build())
    return Response(_cached(name, load), media_type="application/json")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


@app.get("/")
def index():
    # อ้าง css / js พร้อมเลขเวอร์ชันตามเวลาแก้ไฟล์ (แก้ไฟล์ = URL ใหม่ เบราว์เซอร์โหลดใหม่แน่นอน)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in ["style.css", "i18n.js", "local.js", "holo.js", "hero.js", "app.js"]:
        html = html.replace(f'"/static/{name}"', f'"/static/{name}?v={int((STATIC / name).stat().st_mtime)}"')
    return Response(html, media_type="text/html; charset=utf-8")


# ---------------------------------------------------------------- ข้อมูลประกอบแผนที่
CLICK = ml.MODELS / "web_click_stats.json"


@app.get("/api/meta")
def meta():
    def build():
        p = predictor()
        crit = pd.read_csv(ml.MODELS / "success_criteria.csv", encoding="utf-8-sig") if (ml.MODELS / "success_criteria.csv").exists() else None
        return {"project": ROOT.name, "data_dirs": [str(d.relative_to(ROOT)) for d in (ml.DATA, ml.MODELS)],
                "classes_B": p.meta2["classes_B"], "backgrounds": p.deploy["backgrounds"], "default_background": p.deploy["default_background"],
                "models": {"stage_A": {bg: p.stage_a[bg]["tabular"] for bg in p.deploy["backgrounds"]}, "stage_B": p.meta2["stage_B"]["tabular"],
                           "genus": p.genus["models"]},
                "criteria": [] if crit is None else crit.to_dict(orient="records"), "look_eval": _look_eval(), "data_dates": _data_dates(),
                "click": json.loads(CLICK.read_text(encoding="utf-8")) if CLICK.exists() else None}   # notebook 16: ความแม่นเมื่อกดแผนที่
    return _json(_cached("meta", build))


def _data_dates():
    # วันที่ของข้อมูลต้นทาง (หน้าเว็บแสดง "ข้อมูล ณ"): data/source_dates.json + ช่วงวันที่ของภาพ Sentinel-2 ที่แผนที่กริด 5 กม. ใช้จริง
    f = ml.DATA / "source_dates.json"
    out = {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not k.startswith("_")} if f.exists() else {}
    items, asg = ml.FEAT / "s2" / "items.parquet", ml.FEAT / "s2" / "assignment_grid_patches.parquet"
    if items.exists() and asg.exists():
        d = pd.to_datetime(pd.read_parquet(items).set_index("id").loc[pd.read_parquet(asg).item_id.dropna().unique(), "datetime"])
        out["sentinel2"] = {"from": d.min().date().isoformat(), "to": d.max().date().isoformat(),
                            "th": "ภาพถ่ายดาวเทียม Sentinel-2 ที่แผนที่ใช้", "en": "Sentinel-2 imagery used by the map"}
    return out


def _look_eval():
    # ความแม่นเมื่อผู้ใช้บอก "หน้าตา" แทนประเภทซาก (notebook 15 หัวข้อ 7): {cv: {แบบ: {acc, top3, type_top1}}}
    f = ml.MODELS / "task1" / "FE" / "appearance_eval.parquet"
    if not f.exists():
        return {}
    t = pd.read_parquet(f)
    t = t[t.deployed]                                # วิธีเติมส่วนของซากที่ใช้จริง (feature_lib.APPEARANCE_PART_POLICY)
    return {cv: {r.scenario: {"acc": float(r.acc), "top3": float(r.top3), "type_top1": None if pd.isna(r.type_top1) else float(r.type_top1)}
                 for r in g.itertuples()} for cv, g in t.groupby("cv")}


def _provinces():
    # ขอบเขตจังหวัดของ ทธ. (data/dmr_admin) — ลำดับนี้คือรหัสจังหวัดที่ใช้ใน /api/grid และ /api/provinces
    def build():
        return gpd.read_file(ml.DATA / "dmr_admin" / "provinces.geojson")[["prov_nam_t", "geometry"]].reset_index(drop=True)
    return _cached("prov_gdf", build)


def _cell_province(g):
    # ช่องกริด -> จังหวัด (อยู่ในขอบเขต; ช่องริมฝั่งที่ตกนอกขอบเขตใช้จังหวัดที่ใกล้ที่สุด)
    prov = _provinces()
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(g.lon, g.lat), crs=prov.crs)
    hit = gpd.sjoin(pts, prov[["geometry"]], predicate="within", how="left")
    hit = hit[~hit.index.duplicated()]
    out = hit["index_right"].reindex(pts.index)
    miss = out.isna()
    if miss.any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                      # ระยะในองศา: พอสำหรับหาจังหวัดที่ใกล้สุด
            near = gpd.sjoin_nearest(pts[miss], prov[["geometry"]], how="left")
        out[miss] = near[~near.index.duplicated()]["index_right"].reindex(pts[miss].index)
    return out.fillna(-1).astype(int).tolist()


def build_provinces():
    # ขอบเขตจังหวัดแบบย่อ (เส้นจังหวัดที่ซ้อนบนแผนที่ / หาจังหวัดตอนชี้เมาส์) + จุดวางชื่อจังหวัด
    #   + ขอบเขตประเทศ (รวมทุกจังหวัด ทำเส้นให้เรียบ ~220 ม. เท่าชั้นยุคของหิน): ตัดภาพโอกาสพบซากตามแนวชายแดน / ชายฝั่ง และวาดเส้นชายแดน
    import shapely
    prov = _provinces()
    simp = shapely.set_precision(prov.geometry.simplify(0.008, preserve_topology=True).values, 0.0005)
    polys, labels = [], []
    for i, (name, geom, gs) in enumerate(zip(prov.prov_nam_t, prov.geometry, simp)):
        bx = [round(v, 3) for v in geom.bounds]
        polys.append({"type": "Feature", "id": i, "properties": {"id": i, "name": name, "bbox": bx}, "geometry": shapely.geometry.mapping(gs)})
        rp = geom.representative_point()
        labels.append({"type": "Feature", "properties": {"id": i, "name": name},
                       "geometry": {"type": "Point", "coordinates": [round(rp.x, 4), round(rp.y, 4)]}})
    land = shapely.set_precision(shapely.union_all(prov.geometry.values).simplify(0.002, preserve_topology=True), 0.0002)
    return {"polygons": {"type": "FeatureCollection", "features": polys}, "labels": {"type": "FeatureCollection", "features": labels},
            "country": {"type": "Feature", "properties": {}, "geometry": shapely.geometry.mapping(land)}}


@app.get("/api/provinces")
def provinces():
    return _static_layer("provinces", build_provinces)


@app.get("/api/grid")
def grid():
    # กริด 5 กม. ทั้งประเทศ (notebook 10b) แบบ column: ค่า 0–1 คูณ 1000 เป็นจำนวนเต็มเพื่อให้ไฟล์เล็ก
    #   ix / iy = ตำแหน่งช่องใน EPSG:32647 (หาร 5 กม.) สำหรับเฉลี่ยกับช่องข้างเคียง; prov = รหัสจังหวัด (ลำดับใน /api/provinces)
    def build():
        g = predictor().grid
        def q(s):
            return np.round(s.values * 1000).astype(int).tolist()
        classes = predictor().meta2["classes_B"]
        xy = g.point_id.str.extract(r"g5_(\d+)_(\d+)").astype(int)
        return {"n": len(g), "lon": np.round(g.lon.values, 4).tolist(), "lat": np.round(g.lat.values, 4).tolist(),
                "ix": (xy[0] // 5000).tolist(), "iy": (xy[1] // 5000).tolist(), "prov": _cell_province(g),
                "rank": {bg: q(g[f"rank_{bg}"]) for bg in ["bg_bias", "bg_uniform"]},
                "potential": {bg: q(g[f"potential_{bg}"]) for bg in ["bg_bias", "bg_uniform"]},
                "inside": {k: g[f"inside_{k}"].astype(int).tolist() for k in ["bg_bias", "bg_uniform", "B"]},
                "types": {c: q(g[f"type_{c}"]) for c in classes}}
    return _json(_cached("grid", build))


@app.get("/api/pbdb")
def pbdb():
    # collection ของ PBDB ในไทย (ไม่เคยใช้ train: ชุดทดสอบอิสระ) + กลุ่มซากตาม fl.pbdb_class
    def build():
        pts = pd.read_parquet(ml.FEAT / "points.parquet").query("source == 'pbdb'")
        col = pd.read_csv(ml.DATA / "pbdb_thailand" / "collections.csv", low_memory=False, usecols=["collection_no", "collection_name", "early_interval"])
        col = col.drop_duplicates("collection_no").set_index(col.drop_duplicates("collection_no").collection_no.astype(str))
        occ = pd.read_csv(ml.DATA / "pbdb_thailand" / "occurrences.csv", low_memory=False)
        occ["cls"] = [fl.pbdb_class(a, b, c) for a, b, c in zip(occ.phylum, occ["class"], occ.order)]
        cls = occ.dropna(subset=["cls"]).groupby(occ.collection_no.astype(str)).cls.agg(lambda s: sorted(set(s)))
        out = []
        for pid, lo, la in zip(pts.point_id, pts.lon, pts.lat):
            cno = pid[5:]
            out.append({"lon": round(float(lo), 4), "lat": round(float(la), 4), "name": str(col.collection_name.get(cno, "")),
                        "age": str(col.early_interval.get(cno, "")), "types": cls.get(cno, [])})
        return out
    return _json(_cached("pbdb", build))


@app.get("/api/options")
def options():
    # ตัวเลือกของฟอร์มทายสกุล: ค่าจากทะเบียนตัวอย่าง (เรียงตามความถี่) + แหล่ง ทธ. 518 แหล่ง
    def build():
        p = predictor()
        spec = pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str}, encoding="utf-8-sig", low_memory=False)
        def vc(c):
            return [str(x) for x in spec[c].dropna().astype(str).value_counts().index]
        ids = p.sites.point_id.str[4:].astype(int)
        raw = gpd.read_file(ml.DATA / "dmr_fossil_sites" / "fossil_sites.geojson")[["OBJECTID", "PROVINCE"]]
        prov = dict(zip(raw.OBJECTID.astype(int), raw.PROVINCE.astype(str)))
        sites = [{"id": int(i), "name": str(n), "province": prov.get(int(i)), "lon": round(float(lo), 5), "lat": round(float(la), 5),
                  "known": int(i) in p.known_sites} for i, n, lo, la in zip(ids, p.site_names, p.sites.lon, p.sites.lat)]
        # ชื่อที่ทะเบียนใช้เรียกแต่ละสกุล (ชื่อสามัญที่พบบ่อย ≥ 10% ของตัวอย่างสกุลนั้น, ไม่เกิน 2 ชื่อ) — ช่วยให้คนทั่วไปอ่านผลเข้าใจ
        taxon = spec.sci_name.map(fl.parse_taxon)
        lab = spec.assign(genus=taxon.map(lambda t: t[0] if t else None)).dropna(subset=["genus", "com_name"])
        names = {}
        for g, sub in lab.groupby("genus"):
            vcn = sub.com_name.value_counts(normalize=True)
            names[g] = " / ".join(list(vcn[vcn >= 0.1].index[:2]))
        type_group = spec.dropna(subset=["f_type", "f_group"]).groupby("f_type").f_group.agg(lambda s: s.value_counts().idxmax())
        return {"f_type": vc("f_type"), "f_group": vc("f_group"), "com_name": sorted(p.genus["com_vocab"]), "province": vc("province"),
                "district": vc("district"), "parts": vc("f_part")[:40], "n_specimens": int(len(spec)), "type_group": type_group.to_dict(),
                "genus_names": {g: names.get(g, "") for g in p.genus["classes"]}, "sites": sorted(sites, key=lambda s: s["name"]),
                "appearance": _appearance_public()}
    return _json(_cached("options", build))


SHOWCASE = ["Armenoceras", "Plagiolaria", "Reticulatia", "Isanosaurus", "Thaiichthys", "Phuwiangosaurus", "Stegolophodon"]   # เรียงจากเก่าไปใหม่
_CONF_RANK = {"recorded": 0, "high": 1, "medium": 2, "low": 3}


@app.get("/api/showcase")
def showcase():
    # หน้าแรก: ซากตัวอย่างข้ามยุคจากทะเบียน — แหล่งที่ยุคเชื่อถือได้มากสุด (ทธ. บันทึก > high > medium > low) แล้วจึงดูจำนวนตัวอย่าง
    #   + จำนวนแหล่ง ทธ. ต่อยุค (ยุคสุดท้ายของ notebook 05; ยุคเก่าสุดของช่วง) สำหรับชั้นหินในภาพ
    def build():
        spec = pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str, "site_OBJECTID": "Int64"},
                           encoding="utf-8-sig", low_memory=False)
        taxon = spec.sci_name.map(fl.parse_taxon)
        spec["genus"], spec["species"] = taxon.map(lambda t: t[0] if t else None), taxon.map(lambda t: t[1] if t else None)
        items = []
        for g in SHOWCASE:
            s = spec[spec.genus == g]
            linked = s.dropna(subset=["site_OBJECTID"])
            if linked.empty:
                continue
            cnt = linked.groupby("site_OBJECTID").agg(n=("fossil_id", "size"), conf=("site_final_confidence", "first"))
            cnt["rank"] = cnt.conf.map(_CONF_RANK).fillna(9)
            top = cnt.sort_values(["rank", "n"], ascending=[True, False]).index[0]
            r = linked[linked.site_OBJECTID == top].iloc[0]
            com = s.com_name.dropna().value_counts()
            items.append({"genus": g, "species": s.species.value_counts().idxmax() if s.species.notna().any() else None,
                          "com_name": str(com.index[0]) if len(com) else None, "f_type": str(s.f_type.mode().iat[0]), "n": int(len(s)),
                          "site": {"id": int(top), "name": str(r.site_NAME_TH), "province": str(r.site_PROVINCE),
                                   "lat": round(float(r.site_latitude), 4), "lon": round(float(r.site_longitude), 4),
                                   "n": int((linked.site_OBJECTID == top).sum())},
                          "period": str(r.site_final_old), "period_young": str(r.site_final_young), "confidence": str(r.site_final_confidence)})
        sites = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", encoding="utf-8-sig")
        per = sites.final_old.value_counts()
        periods = [{"period": p, "th": fl.PERIOD_TH_NAME[p], "sites": int(per.get(p, 0))} for p in fl.PERIOD_ORDER if p != "Precambrian"]
        return {"items": items, "periods": periods, "n_sites": int(len(sites)), "unknown_period": int(sites.final_old.isna().sum())}
    return _json(_cached("showcase", build))


# ป้ายภาษาอังกฤษของหน่วยหิน (สลับภาษาบนหน้าเว็บ): ชื่อหมวด / กลุ่มหินสะกดตาม ThaiLex (data/thailex; fl.thailex_match ตัวเดียวกับ feature
#   ธรณีของโมเดล) + ลำดับชั้น (หมวดหิน = Formation, กลุ่มหิน = Group) ตามป้ายภาษาไทยของ ทธ.; ป้ายที่ไม่ใช่ชื่อหน่วย และชื่อที่ ThaiLex ไม่มี แปลไว้ตรงนี้
_GEO_EN = {
    "หินอัคนี": "Igneous rock", "หินตะกอน / หินแปร": "Sedimentary / metamorphic rock",
    "ตะกอนที่ราบลุ่มน้ำ": "Alluvial plain deposits", "ตะกอนที่ราบน้ำท่วมถึง": "Floodplain deposits", "ตะกอนลากูน": "Lagoon deposits",
    "ตะกอนชายฝั่งทะเลโดยอิทธิพลของน้ำขึ้น": "Coastal deposits (high-tide influence)",
    "ตะกอนชายฝั่งทะเลโดยอิทธิพลของน้ำขึ้นน้ำลง": "Coastal deposits (tidal influence)", "เนินตะกอนน้ำพารูปพัด": "Alluvial fan deposits",
    "ตะกอนเศษหินเชิงเขา และตะกอนหินผุพัง": "Colluvial and residual deposits", "ตะกอนตะพักลำน้ำ": "River terrace deposits",
    "มายอ": "Mayo", "เขาย่าปุก": "Khao Ya Puk", "ภูขัด": "Phu Khat", "ตะนาวศรี": "Tanaosi", "ทองผาภูมิ": "Thong Pha Phum", "แม่เมาะ": "Mae Moh",
    "ชัยบุรี": "Chai Buri",
}


def _geo_label_en(th):
    if th in _GEO_EN:
        return _GEO_EN[th]
    m = re.match(r"^(หมวดหิน|กลุ่มหิน)(.+)$", th)
    if not m:
        return th
    name = _GEO_EN.get(m[2].strip())
    if name is None:
        row = fl.thailex_match(th)
        if row is None:
            from pythainlp.transliterate import romanize
            name = romanize(m[2].strip(), engine="royin").title()
        else:
            name = re.sub(r"\s+", " ", re.sub(r"\b(Lower|Upper|Fm|Gr|Formation|Group|Limestone)\b", "", row.name_en)).strip()
    return f"{name} {'Formation' if m[1] == 'หมวดหิน' else 'Group'}"


def build_geology():
    # ชั้นแผนที่ "ยุคของหิน": หน่วยหินจากแผนที่ธรณี 1:250,000 ของ ทธ. (data/dmr_geology) รวมตามสัญลักษณ์ ทำเส้นให้เรียบ (~220 ม.) สำหรับแสดงผล
    #   ยุคของหน่วยหินจากสัญลักษณ์ (fl.symbol_era — ตัวเดียวกับ feature ธรณีที่โมเดลใช้); สีและตัวกรองตามยุคที่เก่าที่สุดของหน่วย
    #   + ต่อยุค: สัดส่วนพื้นที่ประเทศ และจำนวนแหล่ง ทธ. ที่ยุคของแหล่ง (notebook 05) ตรงกับยุคนั้น
    import shapely
    rock = gpd.read_file(fl.GEO_DIR / "rock_unit_250k.geojson")[["TEXT_SYM", "geometry"]]
    rock = rock[rock.TEXT_SYM.notna() & rock.TEXT_SYM.ne("Water body")]
    legend = pd.read_csv(fl.GEO_DIR / "rock_unit_250k_legend.csv", encoding="utf-8-sig")
    igneous_from = legend.loc[legend.TEXT_SYM.eq("Qgy"), "order"].iat[0]
    label = legend.set_index("TEXT_SYM").label_th
    igneous = set(legend.loc[legend.order >= igneous_from, "TEXT_SYM"])
    g = rock.dissolve("TEXT_SYM").reset_index()
    area = g.to_crs(32647).area
    total = float(area.sum())
    g["geometry"] = shapely.set_precision(g.geometry.simplify(0.002, preserve_topology=True).values, 0.0002)
    feats, by_period = [], {}
    for (sym, geom), a in zip(zip(g.TEXT_SYM, g.geometry), area):
        old, young = fl.symbol_era(sym)
        if old is None or geom.is_empty:
            continue
        lab = label.get(sym)
        kind = "หินอัคนี" if sym in igneous else "หินตะกอน / หินแปร"
        lab = lab if isinstance(lab, str) else kind
        feats.append({"type": "Feature", "properties": {"sym": sym, "label": lab, "kind": kind, "label_en": _geo_label_en(lab), "kind_en": _GEO_EN[kind],
                                                        "old": old, "young": young, "pct": round(100 * float(a) / total, 2)},
                      "geometry": shapely.geometry.mapping(geom)})
        by_period[old] = by_period.get(old, 0.0) + float(a)
    sites = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", encoding="utf-8-sig").final_old.value_counts()
    periods = [{"period": p, "th": fl.PERIOD_TH_NAME[p], "pct": round(100 * by_period[p] / total, 1), "sites": int(sites.get(p, 0)),
                "units": sum(f["properties"]["old"] == p for f in feats)} for p in fl.PERIOD_ORDER if p in by_period]
    return {"type": "FeatureCollection", "features": feats, "periods": periods}


@app.get("/api/geology")
def geology():
    return _static_layer("geology", build_geology)


# ชั้นข้อมูลที่ export เป็น webapp/data/<ชื่อ>.json (scripts/web_static.py; tests/test_webapp.py ตรวจว่ายังตรงกับข้อมูลดิบ)
STATIC_LAYERS = {"provinces": build_provinces, "geology": build_geology}


_PERIOD_BASE = [(541, "Precambrian"), (485.4, "Cambrian"), (443.8, "Ordovician"), (419.2, "Silurian"), (358.9, "Devonian"),
                (298.9, "Carboniferous"), (251.9, "Permian"), (201.4, "Triassic"), (145, "Jurassic"), (66, "Cretaceous"), (2.58, "Tertiary"), (0, "Quaternary")]


def _period_of_ma(ma):
    # อายุ (ล้านปี) -> ยุค (ฐานของยุคตามตารางธรณีกาลสากล ICS; เทอร์เชียรี = พาลีโอจีน + นีโอจีน)
    return next(p for base, p in _PERIOD_BASE if ma >= base)


@app.get("/api/period_profile")
def period_profile():
    # หน้าทายชื่อซาก: ซากแต่ละสกุล / กลุ่ม "ส่วนใหญ่พบในชั้นหินยุคไหน" — นับเป็นจำนวนแหล่ง (แหล่งเดียวมีหลายชิ้นนับครั้งเดียว)
    #   ทะเบียน ทธ.: ตัวอย่างที่เชื่อมแหล่งระดับ high / medium กับแหล่งที่ยุคเชื่อถือได้ (ทธ. บันทึก / เติมระดับ high / medium, notebook 05) — ยุคเก่าสุดของแหล่ง
    #   PBDB: collection ที่มีสกุล / กลุ่มนั้น — ยุคจากกึ่งกลางช่วงอายุ (max_ma + min_ma) / 2
    #   สกุลที่ไม่มีข้อมูลระดับนี้เลย: ใช้ตัวอย่างที่เชื่อมแหล่ง / ยุคของแหล่งความเชื่อมั่นต่ำ และติดธง weak (หน้าเว็บบอกว่า "ข้อมูลไม่แน่นอน")
    def build():
        spec = pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", dtype={"site_OBJECTID": "Int64"}, encoding="utf-8-sig", low_memory=False)
        spec["genus"] = spec.sci_name.map(lambda n: (fl.parse_taxon(n) or (None,))[0])
        spec["group"] = spec.f_type.map(fl.other_group)
        ok = spec[spec.confidence.isin(["high", "medium"]) & spec.site_final_confidence.isin(["recorded", "high", "medium"]) & spec.site_final_old.notna()]
        loose = spec[spec.site_OBJECTID.notna() & spec.site_final_old.notna()]
        occ = pd.read_csv(ml.DATA / "pbdb_thailand" / "occurrences.csv", low_memory=False).dropna(subset=["max_ma", "min_ma"])
        occ["genus"] = [(fl.parse_taxon(a) or fl.parse_taxon(b) or (None,))[0] for a, b in zip(occ.accepted_name, occ.identified_name)]
        cls = [fl.pbdb_class(a, b, c) for a, b, c in zip(occ.phylum, occ["class"], occ.order)]
        occ["group"] = [fl.class_group(c) if isinstance(c, str) else None for c in cls]
        occ["period"] = [_period_of_ma((a + b) / 2) for a, b in zip(occ.max_ma, occ.min_ma)]
        keep = set(predictor().genus["classes"]) - {"อื่น ๆ"}

        def profile(key, names=None, reg=ok):
            a = reg.dropna(subset=[key]).groupby([key, "site_final_old"]).site_OBJECTID.nunique()
            b = occ.dropna(subset=[key]).groupby([key, "period"]).collection_no.nunique()
            out = {}
            for name in sorted(set(a.index.get_level_values(0)) | set(b.index.get_level_values(0))):   # เรียงชื่อ: ไฟล์ที่ export เหมือนเดิมทุกครั้ง
                if names is not None and name not in names:
                    continue
                da = a[name].to_dict() if name in a.index.get_level_values(0) else {}
                db = b[name].to_dict() if name in b.index.get_level_values(0) else {}
                cnt = {p: da.get(p, 0) + db.get(p, 0) for p in set(da) | set(db)}
                order = sorted(cnt, key=lambda p: (-cnt[p], fl.PERIOD_ORDER.index(p)))
                out[name] = {"p": [[p, int(cnt[p])] for p in order], "n": int(sum(cnt.values())), "dmr": int(sum(da.values())), "pbdb": int(sum(db.values()))}
            return out

        gg = spec.dropna(subset=["genus", "group"]).groupby("genus").group.agg(lambda s: s.value_counts().idxmax())
        genus = profile("genus", keep)
        weak = profile("genus", keep - set(genus), reg=loose)
        genus.update({g: {**v, "weak": True} for g, v in weak.items()})
        return {"genus": genus, "group": profile("group"), "genus_group": {g: gg[g] for g in sorted(keep) if g in gg.index}}
    return _json(_cached("period_profile", build))


def _appearance_public():
    # คำถาม "หน้าตา" สำหรับหน้าเว็บ (ข้อความ / ไอคอน) — ชุดประเภทและคำค้นอยู่ฝั่ง server (feature_lib.APPEARANCE)
    keep = ("id", "group", "label", "hint", "icon", "ask")
    return [{**{k: c[k] for k in keep if k in c},
             "options": [{k: o[k] for k in ("id", "label", "hint", "icon") if k in o} for o in c.get("options", [])]} for c in fl.APPEARANCE]


# ---------------------------------------------------------------- ทำนายสด
class LocationIn(BaseModel):
    lat: float = Field(..., description="ละติจูด")
    lon: float = Field(..., description="ลองจิจูด")
    with_s2: bool = Field(True, description="ดึงภาพ Sentinel-2 (ต้องต่ออินเทอร์เน็ต)")


@app.post("/api/predict_location")
def predict_location(body: LocationIn):
    lo0, lo1, la0, la1 = THAI_BOX
    if not (lo0 <= body.lon <= lo1 and la0 <= body.lat <= la1):
        raise HTTPException(422, "พิกัดอยู่นอกประเทศไทย (โมเดล train เฉพาะในไทย)")
    p = predictor()
    with _LOCK:
        out, _, _ = p.predict_location(body.lat, body.lon, with_s2=body.with_s2)
    hit = p.prov[p.prov.contains(gpd.points_from_xy([body.lon], [body.lat])[0])]
    out["province"] = str(hit.prov_nam_t.iat[0]) if len(hit) else None              # จังหวัดของจุด (ขอบเขต ทธ.)
    g = p.grid
    i = int(np.argmin((g.lon.values - body.lon) ** 2 + ((g.lat.values - body.lat) * 1.0) ** 2))
    out["grid_cell"] = {"lon": float(g.lon.iat[i]), "lat": float(g.lat.iat[i]), "rank_bg_bias": float(g.rank_bg_bias.iat[i]),
                        "rank_bg_uniform": float(g.rank_bg_uniform.iat[i]), "top_type": str(g.top_type.iat[i])}
    return _json(out)


class SpecimenIn(BaseModel):
    com_name: str | None = None
    f_type: str | None = None
    f_group: str | None = None
    f_part: str | None = None
    province: str | None = None
    district: str | None = None
    site_OBJECTID: int | None = None
    lat: float | None = None
    lon: float | None = None
    look_card: str | None = Field(None, description="การ์ดหน้าตาซาก (ไม่รู้ประเภท) เช่น tooth, bone, twoshell")
    look_option: str | None = Field(None, description="คำตอบคำถามต่อของการ์ด เช่น blade (ไม่ระบุ = ไม่แน่ใจ)")


@app.post("/api/predict_specimen")
def predict_specimen(body: SpecimenIn):
    rec = {k: (v.strip() if isinstance(v, str) else v) for k, v in body.model_dump().items()}
    rec = {k: (None if v == "" else v) for k, v in rec.items()}
    card, option = rec.pop("look_card"), rec.pop("look_option")
    if card and rec.get("f_type"):
        raise HTTPException(422, "เลือกอย่างใดอย่างหนึ่ง: หน้าตาซาก หรือประเภทซาก")
    if card:
        try:
            fl.appearance_leaf_id(card, option)
        except KeyError:
            raise HTTPException(422, "ไม่รู้จักคำตอบเรื่องหน้าตาซาก") from None
    elif not rec.get("f_type") and not rec.get("com_name"):
        raise HTTPException(422, "ต้องบอกหน้าตาซาก ประเภทซาก หรือชื่อสามัญ อย่างน้อยหนึ่งอย่าง")
    if rec.get("lat") is not None:
        lo0, lo1, la0, la1 = THAI_BOX
        if rec.get("lon") is None or not (lo0 <= rec["lon"] <= lo1 and la0 <= rec["lat"] <= la1):
            raise HTTPException(422, "พิกัดอยู่นอกประเทศไทย")
    p = predictor()
    with _LOCK:
        out = p.predict_specimen_appearance(rec, card, option) if card else p.predict_specimen(rec)
    return _json(out)
