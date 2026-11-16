# Web app (webapp/server.py): every endpoint answers, and its predictions are the same as the notebooks / predict_lib.
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, ml

sys_path_ok = (ROOT / "webapp" / "server.py").exists()
pytestmark = pytest.mark.skipif(not sys_path_ok, reason="ไม่มี webapp/")


@pytest.fixture(scope="module")
def client():
    import sys
    sys.path.insert(0, str(ROOT))
    from webapp import server
    return TestClient(server.app)


def test_pages_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "แผนที่ซากดึกดำบรรพ์ประเทศไทย" in r.text
    for f in ["app.js", "style.css", "holo.js", "hero.js", "i18n.js"]:
        assert client.get(f"/static/{f}").status_code == 200
        assert f'/static/{f}?v=' in r.text                         # เลขเวอร์ชัน: แก้ไฟล์แล้วเบราว์เซอร์ต้องโหลดใหม่
    assert r.headers["cache-control"] == "no-cache" and client.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_grid_matches_predictions_file(client):
    g = client.get("/api/grid").json()
    pred = pd.read_parquet(ml.MODELS / "task2" / "predictions" / "grid_5km_predictions.parquet")
    assert g["n"] == len(pred) == len(g["lon"]) == len(g["rank"]["bg_bias"])
    assert np.allclose(np.array(g["rank"]["bg_bias"]) / 1000, pred.rank_bg_bias.values, atol=6e-4)
    assert (np.array(g["inside"]["B"]) == pred.inside_B.astype(int).values).all()
    meta = client.get("/api/meta").json()
    assert set(g["types"]) == set(meta["classes_B"]) and len(meta["classes_B"]) == 16


def test_options_and_pbdb(client):
    o = client.get("/api/options").json()
    assert len(o["sites"]) == 518 and {"id", "name", "lon", "lat", "known"} <= set(o["sites"][0])
    assert "ไดโนเสาร์" in o["f_type"] and len(o["com_name"]) > 10
    pb = client.get("/api/pbdb").json()
    assert len(pb) == 290 and any(p["types"] for p in pb)


def test_predict_specimen_matches_notebook11(client):
    # ทุกเส้นทางต้องให้สกุลอันดับ 1 และชนิดที่เดา เหมือนไฟล์ผลของ notebook 11
    reg = pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str, "site_OBJECTID": "Int64"},
                      encoding="utf-8-sig", low_memory=False).set_index("fossil_id")
    pred = pd.read_csv(ml.DATA / "processed" / "task1_genus_predictions.csv", dtype={"fossil_id": str}, encoding="utf-8-sig")
    smp = pred.groupby("route", group_keys=False).apply(lambda g: g.sample(min(len(g), 6), random_state=1))
    for r in smp.itertuples():
        rec = reg.loc[r.fossil_id, ["com_name", "f_type", "f_group", "f_part", "province", "district", "site_OBJECTID"]].to_dict()
        body = {k: (None if pd.isna(v) else (int(v) if k == "site_OBJECTID" else str(v))) for k, v in rec.items()}
        out = client.post("/api/predict_specimen", json=body).json()
        assert out["top"][0][0] == r.genus_top1, (r.fossil_id, out)
        sp = None if pd.isna(r.species_guess) else r.species_guess
        assert out["species"][0] == sp, (r.fossil_id, out["species"][0], sp)


def test_predict_location_offline_matches_predict_lib(client):
    # ไม่ดึงภาพ (ทดสอบได้โดยไม่ใช้เน็ต): ต้องเท่ากับ predict_lib ตรง ๆ และมีคำเตือนเรื่องภาพ
    import predict_lib as pl
    out = client.post("/api/predict_location", json={"lat": 16.6803, "lon": 102.2357, "with_s2": False}).json()
    ref, _, _ = pl.Predictor().predict_location(16.6803, 102.2357, with_s2=False)
    for k in ["potential_bg_bias", "potential_bg_uniform", "percentile_bg_bias", "inside_bg_bias", "inside_types"]:
        assert out[k] == ref[k], k
    assert [t[0] for t in out["top_types"]] == [t[0] for t in ref["top_types"]]
    assert out["s2_scene"] is None and any("Sentinel-2" in w for w in out["warnings"])
    assert "grid_cell" in out


def test_other_class_carries_group(client):
    # "อื่น ๆ" ต้องบอกกลุ่ม: ทายแบบรู้ประเภท = ประเภทที่ให้มา; บอกหน้าตา = แยกเป็นรายการต่อประเภท (ไม่รวมข้ามกลุ่ม)
    out = client.post("/api/predict_specimen", json={"f_type": "เอไคโนเดิร์ม", "f_part": "ก้านไครนอยด์"}).json()
    import feature_lib as fl
    assert out["top"][0][0] == "อื่น ๆ" and out["top"][0][2] == fl.other_group("เอไคโนเดิร์ม") == fl.other_group("ไครนอยด์") == "เอไคโนเดิร์ม"
    assert all(len(e) == 3 and (e[2] is None) == (e[0] != "อื่น ๆ") for e in out["top"])
    look = client.post("/api/predict_specimen", json={"look_card": "tooth"}).json()
    others = [e for e in look["top"] if e[0] == "อื่น ๆ"]
    assert all(e[2] in {fl.other_group(t) for t, _ in look["types"]} for e in others) and len({e[2] for e in others}) == len(others)
    assert fl.other_group("ซอโรพอด") == fl.other_group("ไดโนเสาร์") and fl.other_group("ฟอแรมินิเฟอรา") == fl.other_group("ฟอแรมมินิเฟอร์")


def test_input_validation(client):
    assert client.post("/api/predict_location", json={"lat": 35.0, "lon": 139.0}).status_code == 422
    assert client.post("/api/predict_specimen", json={}).status_code == 422
    assert client.post("/api/predict_specimen", json={"f_type": "ไดโนเสาร์", "lat": 35.0, "lon": 139.0}).status_code == 422


@pytest.mark.slow
def test_predict_location_live_sentinel2(client):
    # ใช้เน็ต: ภูเวียง (แหล่ง train) ต้องได้อันดับสูงและประเภทอันดับ 1 = ไดโนเสาร์ เหมือน notebook 12
    out = client.post("/api/predict_location", json={"lat": 16.6803, "lon": 102.2357, "with_s2": True}).json()
    assert out["s2_scene"] and out["percentile_bg_bias"] > 95 and out["top_types"][0][0] == "ไดโนเสาร์"


def test_data_only_from_this_project(client):
    # เว็บต้องอ่านข้อมูล / โมเดลจาก Project_Jurassic เท่านั้น (ไม่ใช่โฟลเดอร์อื่นในเครื่อง)
    import predict_lib as pl
    meta = client.get("/api/meta").json()
    assert meta["project"] == ROOT.name and meta["data_dirs"] == ["data", "models"]
    for d in [ml.DATA, ml.MODELS, pl.T1, pl.T2]:
        assert d.resolve().is_relative_to(ROOT.resolve()), d
    src = "\n".join(p.read_text(encoding="utf-8") for p in [ROOT / "webapp" / "server.py", ROOT / "notebooks" / "predict_lib.py",
                                                            ROOT / "webapp" / "static" / "app.js", ROOT / "webapp" / "static" / "holo.js",
                                                            ROOT / "webapp" / "static" / "hero.js", ROOT / "webapp" / "static" / "i18n.js",
                                                            ROOT / "scripts" / "run_webapp.sh"])
    assert "/Users/" not in src and "Project_Y4" not in src
    assert "แผนที่ซากดึกดำบรรพ์ประเทศไทย" in client.get("/").text and "Project Jurassic" in client.get("/").text


def test_plain_language_helpers(client):
    # ข้อมูลที่หน้าเว็บใช้แปลผลให้คนทั่วไป: ชื่อไทยของสกุล, กลุ่มซากอัตโนมัติจากประเภท, จังหวัดของจุดที่เลือก
    o = client.get("/api/options").json()
    meta = client.get("/api/meta").json()
    assert o["n_specimens"] == 8521 and o["type_group"]["ไดโนเสาร์"] == "Vertebrate"
    assert "Phuwiangosaurus" in o["genus_names"] and o["genus_names"]["Phuwiangosaurus"]
    assert all(s["province"] for s in o["sites"])
    assert {c["รหัส"] for c in meta["criteria"]} >= {"A1", "B2", "G1", "G2", "P2"}   # หน้า MORE แสดง B2 G1 G2 P2
    assert meta["click"] and set(meta["click"]["share_at_or_above"]) == {"vhigh", "high", "mid"}   # หน้าแรก / MORE: ความแม่นเมื่อกดแผนที่
    out = client.post("/api/predict_location", json={"lat": 13.75, "lon": 100.5, "with_s2": False}).json()
    assert out["province"] == "กรุงเทพมหานคร"          # (ธง AOA ขึ้นกับภาพดาวเทียม: ไม่ดึงภาพ = ตัดสินขอบเขตไม่ได้ดี จึงไม่ตรวจที่นี่)


def test_province_overview_data(client):
    # ภาพรวมรายจังหวัด (ซูมออก): 77 จังหวัดพร้อมชื่อ / กรอบ / จุดวางชื่อ; ทุกช่องกริดมีจังหวัด และกรอบจังหวัดครอบช่องนั้น
    pv = client.get("/api/provinces").json()
    feats = pv["polygons"]["features"]
    names = [f["properties"]["name"] for f in feats]
    assert len(feats) == len(pv["labels"]["features"]) == 77 and len(set(names)) == 77
    assert [f["properties"]["id"] for f in feats] == list(range(77))
    assert {"กรุงเทพมหานคร", "ขอนแก่น", "บึงกาฬ"} <= set(names)
    g = client.get("/api/grid").json()
    prov = np.array(g["prov"])
    assert len(prov) == g["n"] and prov.min() >= 0 and prov.max() < 77
    assert len(np.unique(prov)) == 77                                   # ทุกจังหวัดมีช่องอย่างน้อย 1 ช่อง
    lon, lat = np.array(g["lon"]), np.array(g["lat"])
    bb = np.array([feats[p]["properties"]["bbox"] for p in prov])
    near = (lon >= bb[:, 0] - 0.05) & (lon <= bb[:, 2] + 0.05) & (lat >= bb[:, 1] - 0.05) & (lat <= bb[:, 3] + 0.05)
    assert near.mean() > 0.999, (~near).sum()
    for (la, lo), name in [((13.75, 100.5), "กรุงเทพมหานคร"), ((16.6803, 102.2357), "ขอนแก่น")]:
        i = int(np.argmin((lat - la) ** 2 + (lon - lo) ** 2))
        assert names[prov[i]] == name


def test_grid_neighbour_index(client):
    # ix / iy ใช้หาช่องข้างเคียงเพื่อทำให้แผนที่เรียบ: ต้องไม่ซ้ำ และช่องข้างกันห่างกันราว 5 กม. จริง
    g = client.get("/api/grid").json()
    ix, iy = np.array(g["ix"]), np.array(g["iy"])
    assert len({(a, b) for a, b in zip(ix, iy)}) == g["n"]
    assert ix.max() < 10000 and iy.max() < 10000                         # app.js ใช้ ix * 10000 + iy เป็นกุญแจ
    key = {(a, b): i for i, (a, b) in enumerate(zip(ix, iy))}
    lon, lat = np.array(g["lon"]), np.array(g["lat"])
    d = [np.hypot((lon[j] - lon[i]) * 111 * np.cos(np.radians(lat[i])), (lat[j] - lat[i]) * 111)
         for i, (a, b) in enumerate(zip(ix, iy)) if (j := key.get((a + 1, b))) is not None]
    assert len(d) > 0.9 * g["n"] and 4.5 < np.median(d) < 5.5 and max(d) < 6.0


def test_hologram_covers_every_genus_and_type(client):
    # หน้า IDENTIFY แสดง hologram ตามสกุลที่ทาย: ทุกสกุลของโมเดลต้องมีรูปร่าง, ทุกประเภทซากในฟอร์มต้องมีรูปร่างสำรอง (ใช้กับ "อื่น ๆ")
    import re
    js = (ROOT / "webapp" / "static" / "holo.js").read_text(encoding="utf-8")
    assert client.get("/static/holo.js").status_code == 200 and "/static/holo.js" in client.get("/").text
    def block(name):
        return js[js.index(f"const {name} = {{"):].split("\n  };", 1)[0]
    shapes = set(re.findall(r"^    (\w+): \{ th:", block("SHAPES"), re.M))
    genus = {k: v.split() for k, v in re.findall(r'(\w+): "([^"]+)"', block("HOLO_GENUS").replace('"\n      + "', ""))}
    types = {k: v.split("|") for k, v in re.findall(r'(\w+): "([^"]+)"', block("HOLO_TYPE"))}
    assert set(genus) <= shapes and set(types) <= shapes and len(shapes) >= 40
    mapped = [g for v in genus.values() for g in v]
    assert len(mapped) == len(set(mapped))                                     # สกุลหนึ่งมีรูปร่างเดียว
    o = client.get("/api/options").json()
    assert set(o["genus_names"]) - {"อื่น ๆ"} == set(mapped)
    tmapped = [t for v in types.values() for t in v]
    assert len(tmapped) == len(set(tmapped)) and set(o["f_type"]) <= set(tmapped)
    # "อื่น ๆ" (ยังระบุชื่อไม่ได้) แสดงเป็นเครื่องหมาย ? ทั้งภาพใหญ่และภาพเล็ก
    assert "question" in shapes
    app = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'key: other ? "question"' in app and '"ยังระบุชื่อไม่ได้"' in app
    # ไดโนเสาร์ไทย 6 สกุลมีภาพเฉพาะตัว (ไม่ใช้ร่วมกับสกุลอื่น) และมีขาข้างใกล้ / ไกลสำหรับมุมมอง 3D
    dino = ["Phuwiangosaurus", "Isanosaurus", "Siamraptor", "Phuwiangvenator", "Siamosaurus", "Kinnareemimus"]
    for g in dino:
        assert genus[g.lower()] == [g], g
        part = block("DINO").split(f"\n    {g.lower()}: {{", 1)[1].split("\n    },", 1)[0]
        assert "far:" in part and "near:" in part and "body:" in part, g


def test_showcase_matches_registry(client):
    # หน้าแรก: ตัวอย่างข้ามยุค — จำนวน / แหล่ง / ยุค ต้องตรงกับทะเบียนและตารางยุคของ notebook 05, เรียงจากเก่าไปใหม่, มีรูปร่าง hologram
    import feature_lib as fl
    d = client.get("/api/showcase").json()
    spec = pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", dtype={"site_OBJECTID": "Int64"}, encoding="utf-8-sig", low_memory=False)
    genus = spec.sci_name.map(fl.parse_taxon).map(lambda t: t[0] if t else None)
    sites = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", encoding="utf-8-sig").set_index("OBJECTID")
    js = (ROOT / "webapp" / "static" / "holo.js").read_text(encoding="utf-8")
    order = [fl.PERIOD_ORDER.index(it["period"]) for it in d["items"]]
    assert len(d["items"]) >= 5 and order == sorted(order) and len(set(order)) == len(order)
    for it in d["items"]:
        s = spec[genus == it["genus"]]
        assert it["n"] == len(s) and it["site"]["n"] == int((s.site_OBJECTID == it["site"]["id"]).sum()) > 0
        assert sites.loc[it["site"]["id"], "final_old"] == it["period"] and sites.loc[it["site"]["id"], "final_confidence"] == it["confidence"]
        assert 97.3 <= it["site"]["lon"] <= 105.7 and 5.6 <= it["site"]["lat"] <= 20.5
        assert it["genus"] in js or it["f_type"] in js, it               # มีรูปร่างของสกุลหรือของประเภท
    assert sum(p["sites"] for p in d["periods"]) + d["unknown_period"] == d["n_sites"] == len(sites)
    assert [p["period"] for p in d["periods"]] == [p for p in fl.PERIOD_ORDER if p != "Precambrian"]


def test_static_layers_match_raw_data(client):
    # ชั้นข้อมูลที่ export ไว้ใน webapp/data/ (scripts/web_static.py) ต้องตรงกับที่คำนวณใหม่จากข้อมูลดิบทุกไบต์ และเว็บส่งไฟล์นั้นตรง ๆ
    from webapp import server
    for name, build in server.STATIC_LAYERS.items():
        f = server.LAYER_DIR / f"{name}.json"
        assert f.exists(), f"ไม่มี {f.name} — รัน scripts/web_static.py"
        assert f.read_bytes() == server._dumps(build()), f"{f.name} ไม่ตรงกับข้อมูลดิบ — รัน scripts/web_static.py ใหม่"
        assert client.get(f"/api/{name}").content == f.read_bytes()


def test_geology_layer_matches_source_map(client):
    # ชั้น "ยุคของหิน": ทุกหน่วยหินมียุค (ตัวเดียวกับ feature ของโมเดล), สัดส่วนพื้นที่รวม ≈ 100%, จำนวนแหล่งต่อยุคตรงกับตารางยุคของ notebook 05,
    # ทุกยุคมีสีในหน้าเว็บ และรูปร่างที่ทำให้เรียบยังให้หน่วยหินตรงกับแผนที่ต้นฉบับเกือบทุกจุด
    import re
    import feature_lib as fl
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import shape
    d = client.get("/api/geology").json()
    feats, periods = d["features"], d["periods"]
    assert len(feats) >= 80 and all(f["properties"]["old"] in fl.PERIOD_ORDER and f["properties"]["young"] in fl.PERIOD_ORDER for f in feats)
    assert all((f["properties"]["old"], f["properties"]["young"]) == fl.symbol_era(f["properties"]["sym"]) for f in feats)
    assert [p["period"] for p in periods] == [p for p in fl.PERIOD_ORDER if p in {p["period"] for p in periods}]
    assert abs(sum(p["pct"] for p in periods) - 100) < 0.5
    sites = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", encoding="utf-8-sig").final_old.value_counts()
    assert all(p["sites"] == int(sites.get(p["period"], 0)) for p in periods)
    app = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    colors = dict(re.findall(r'(\w+): "(#[0-9a-f]{6})"', app[app.index("const PERIOD_COLOR"):app.index("const PERIOD_MA")]))
    assert {p["period"] for p in periods} <= set(colors)
    # สุ่มจุดในแผนที่ต้นฉบับ: หน่วยหินในชั้นที่ทำให้เรียบต้องตรง ≥ 95%
    rock = gpd.read_file(fl.GEO_DIR / "rock_unit_250k.geojson")[["TEXT_SYM", "geometry"]]
    rock = rock[rock.TEXT_SYM.ne("Water body")].reset_index(drop=True)
    simp = gpd.GeoDataFrame({"sym": [f["properties"]["sym"] for f in feats]}, geometry=[shape(f["geometry"]) for f in feats], crs=4326)
    rng = np.random.default_rng(0)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(rng.uniform(97.3, 105.7, 3000), rng.uniform(5.6, 20.5, 3000)), crs=4326)
    want = gpd.sjoin(pts, rock, predicate="within", how="inner")
    want = want[~want.index.duplicated()].TEXT_SYM.iloc[:400]                       # จุดสุ่มสม่ำเสมอบนแผ่นดิน (ตามพื้นที่จริง)
    got = gpd.sjoin(pts.loc[want.index], simp, predicate="within", how="left")
    got = got[~got.index.duplicated()].sym.reindex(want.index)
    assert len(want) == 400 and (got.values == want.values).mean() >= 0.95
    assert np.isclose(sum(f["properties"]["pct"] for f in feats), 100, atol=1)


def test_period_profile_for_identify_results(client):
    # หน้าทายชื่อซาก "ส่วนใหญ่พบในชั้นหินยุคไหน": จำนวนแหล่งต่อยุคต้องรวมกันได้เท่าจำนวนแหล่ง (ทะเบียน + PBDB), ยุคถูกต้อง,
    # ทุกสกุลของโมเดลมีข้อมูลของสกุลหรือของกลุ่มซาก, ค่าจริงของสกุลที่รู้แน่ (ภูเวียงโกซอรัส = ครีเทเชียส), ทุกยุคมีสีในหน้าเว็บ
    import re
    import feature_lib as fl
    d = client.get("/api/period_profile").json()
    for part in ("genus", "group"):
        for name, v in d[part].items():
            assert v["n"] == v["dmr"] + v["pbdb"] == sum(c for _, c in v["p"]) > 0, name
            assert all(p in fl.PERIOD_ORDER for p, _ in v["p"]) and [c for _, c in v["p"]] == sorted((c for _, c in v["p"]), reverse=True), name
    classes = set(client.get("/api/options").json()["genus_names"]) - {"อื่น ๆ"}
    assert all(g in d["genus"] or d["group"].get(d["genus_group"].get(g)) for g in classes)
    assert sum(not v.get("weak") for g, v in d["genus"].items()) >= 100
    assert d["genus"]["Phuwiangosaurus"]["p"][0][0] == "Cretaceous" and d["group"]["ไดโนเสาร์"]["p"][0][0] == "Cretaceous"
    assert d["group"]["แบรคิโอพอด"]["p"][0][0] == "Permian"
    app = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    colors = set(re.findall(r'(\w+): "#[0-9a-f]{6}"', app[app.index("const PERIOD_COLOR"):app.index("const PERIOD_MA")]))
    assert {p for v in list(d["genus"].values()) + list(d["group"].values()) for p, _ in v["p"]} <= colors
    assert "function ageLine" in app and "/api/period_profile" in app


def test_language_and_theme_switch(client):
    # ปุ่มภาษาไทย / อังกฤษ และโหมดมืด / สว่าง: ทุกข้อความคงที่ในหน้ามีคำแปล, ทุกค่าข้อมูลที่หน้าเว็บแสดงมีชื่ออังกฤษ (i18n.js),
    # ป้ายหน่วยหิน / hologram มีภาษาอังกฤษ, ค่าที่ส่งให้โมเดลแปลงกลับเป็นไทยได้, โหมดสว่างกำหนดสีครบทุกตัวแปร
    import re
    from html.parser import HTMLParser
    import feature_lib as fl
    thai = re.compile(r"[฀-๿]")
    static = ROOT / "webapp" / "static"
    page = client.get("/").text
    head = page.split("</head>", 1)[0]
    assert "/static/i18n.js?v=" in head and client.get("/static/i18n.js").status_code == 200   # โหลดใน <head>: ใส่ธีมก่อนหน้าแสดง
    assert 'id="theme-btn"' in page and 'id="lang-btn"' in page

    class Walk(HTMLParser):
        # ข้อความไทยทุกชิ้นต้องอยู่ใต้ element ที่มี data-en (placeholder / aria-label ไทยต้องมี data-en-ph / data-en-aria)
        def __init__(self):
            super().__init__()
            self.stack, self.missing = [], []
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            for attr, alt in (("placeholder", "data-en-ph"), ("aria-label", "data-en-aria"), ("alt", "data-en-aria")):
                if thai.search(a.get(attr) or "") and alt not in a:
                    self.missing.append(f"{tag} {attr}={a[attr]}")
            if tag not in ("meta", "link", "input", "br", "img"):
                self.stack.append((tag, a))
        def handle_endtag(self, tag):
            while self.stack and self.stack.pop()[0] != tag:
                pass
        def handle_data(self, data):
            if not thai.search(data) or any(t in ("title", "script") for t, _ in self.stack):
                return                                                   # <title>: I18N.apply ตั้งเอง
            if not any("data-en" in a or a.get("id") == "rank-title" for _, a in self.stack):   # rank-title: app.js เขียนทับตามภาษา
                self.missing.append(data.strip()[:60])
    w = Walk()
    w.feed((static / "index.html").read_text(encoding="utf-8"))
    assert not w.missing, w.missing

    js = (static / "i18n.js").read_text(encoding="utf-8")
    def block(src, name):
        return src[src.index(f"const {name} = {{"):].split("\n  };", 1)[0]
    data = dict(re.findall(r'"([^"]+)": "([^"]+)"', block(js, "DATA")))
    assert not any(thai.search(v) for v in data.values())
    o, meta = client.get("/api/options").json(), client.get("/api/meta").json()
    pp, pbdb = client.get("/api/period_profile").json(), client.get("/api/pbdb").json()
    provinces = [f["properties"]["name"] for f in client.get("/api/provinces").json()["polygons"]["features"]]
    shown = (set(o["f_type"]) | set(meta["classes_B"]) | set(pp["group"]) | set(pp["genus_group"].values())
             | {t for p in pbdb for t in p["types"]} | set(provinces) | set(o["province"]))
    assert shown <= set(data), shown - set(data)
    en_prov = [data[p] for p in provinces]
    assert len(set(en_prov)) == len(provinces) == 77
    assert all(sum(v.lower() == e.lower() for v in data.values()) == 1 for e in en_prov)   # แปลงกลับ (toThai) ได้จังหวัดเดียว
    app = (static / "app.js").read_text(encoding="utf-8")
    picks = re.findall(r'"([^"]+)"', app[app.index("const PART_PICKS"):].split("\n", 1)[0])
    assert set(picks) <= set(data)
    # คำถาม "หน้าตา": ทุกการ์ด / ตัวเลือก / กลุ่มมีคำแปล
    look = set(re.findall(r"(\w+): \[", block(js, "LOOK")))
    look_opt = set(re.findall(r'"(\w+/\w+)": \[', block(js, "LOOK_OPT")))
    groups = dict(re.findall(r'"([^"]+)": "([^"]+)"', block(js, "LOOK_GROUP")))
    for c in o["appearance"]:
        assert c["id"] in look and c["group"] in groups, c["id"]
        assert {f"{c['id']}/{op['id']}" for op in c["options"] if op["id"] != "unsure"} <= look_opt, c["id"]
    assert set(re.findall(r"(\w+): \[", block(js, "PERIOD"))) >= set(fl.PERIOD_ORDER)
    # ป้ายหน่วยหิน (แผนที่ยุคของหิน): ภาษาอังกฤษครบทุกหน่วย ชื่อหมวด / กลุ่มหินตามลำดับชั้นของป้ายไทย
    feats = client.get("/api/geology").json()["features"]
    for f in feats:
        q = f["properties"]
        assert q["label_en"] and q["kind_en"] and not thai.search(q["label_en"] + q["kind_en"]), q["sym"]
        if q["label"].startswith("หมวดหิน"):
            assert q["label_en"].endswith(" Formation"), q
        if q["label"].startswith("กลุ่มหิน"):
            assert q["label_en"].endswith(" Group"), q
    byname = {f["properties"]["label"]: f["properties"]["label_en"] for f in feats}
    assert byname["หมวดหินเสาขัว"] == "Sao Khua Formation" and byname["หมวดหินน้ำพอง"] == "Nam Phong Formation"
    # hologram: ป้ายรูปร่างและคำอธิบายใต้ภาพมีภาษาอังกฤษครบ
    holo = (static / "holo.js").read_text(encoding="utf-8")
    shapes = set(re.findall(r"^    (\w+): \{ th:", block(holo, "SHAPES"), re.M))
    assert set(re.findall(r"(\w+): \"", block(holo, "EN"))) == shapes
    notes = set(re.findall(r'note: "([^"]+)"', holo))
    assert notes <= set(re.findall(r'^    "([^"]+)":$', block(holo, "NOTE_EN"), re.M)) and len(notes) == 2
    # ธีม: โหมดสว่างกำหนดสีใหม่ครบทุกตัวแปรสีของโหมดมืด; แผนที่ฐานเปลี่ยนตามธีมและย้ายชั้นข้อมูลของเว็บไปด้วย
    css = (static / "style.css").read_text(encoding="utf-8")
    dark = css[css.index(":root {"):].split("}", 1)[0]
    light = css[css.index(':root[data-theme="light"] {'):].split("}", 1)[0]
    colour_vars = {v for v, val in re.findall(r"(--[\w-]+):\s*([^;]+);", dark) if not v.startswith(("--f-", "--top", "--side"))}
    assert colour_vars <= set(re.findall(r"(--[\w-]+):", light)), colour_vars - set(re.findall(r"(--[\w-]+):", light))
    assert "styles/positron" in app and "transformStyle" in app and "THEME.onChange(retheme)" in app and "I18N.onChange(relabel)" in app
    # ภาพที่วาดแบบแสงบนพื้นมืดต้องแปลงสีในโหมดสว่าง ไม่ให้เป็นจอมืดกลางหน้าสว่าง: โฮโลแกรม และภาพพื้นหลังแผงหน้า MORE
    #   -> ฟิลเตอร์ #mono (หมึกสีตามตัวแปร --holo-*), แผงหน้า MORE พื้นโปร่ง
    html = (static / "index.html").read_text(encoding="utf-8")
    mono = re.search(r'([^}\n]*)\{ filter: url\(#mono\); \}', css.replace(",\n:root", ", :root"))
    assert mono and all(f':root[data-theme="light"] {c}' in mono.group(1) for c in (".scan-cv", ".holo-cv", ".holo-float", ".holo-mini svg"))
    assert ':root[data-theme="light"] .col .bg { filter: url(#mono);' in css and ':root[data-theme="light"] .col { background: transparent; }' in css
    assert re.search(r'<filter id="mono"[^>]*>.*<feFlood class="hf-ink"', html)                 # สีหมึกมาจากตัวแปร CSS
    assert re.search(r'\.hf-ink \{ flood-color: var\(--holo-ink', css) and '--holo-ink:' in css
    # โหมดมืด: โฮโลแกรมคงชุดสีเดิม (ฟ้า #18d1ff) แม้สีเน้นของหน้าเปลี่ยน; แผงภาพหน้า MORE ใช้ฟิลเตอร์ #night
    assert re.search(r'\n:root:not\(\[data-theme="light"\]\) \.cols, :root:not\(\[data-theme="light"\]\) \.hero-scan, :root:not\(\[data-theme="light"\]\) \.holo \{[^}]*--cyan: #18d1ff;', css)
    assert ':root:not([data-theme="light"]) .cols { filter: url(#night); }' in css and '<filter id="night"' in html
    # ค่าที่ส่งให้โมเดลเป็นภาษาไทยเสมอ (จังหวัด / ส่วนของซากที่พิมพ์เป็นอังกฤษแปลงกลับ)
    assert 'province: I18N.toThai(val("province"))' in app and 'f_part: I18N.toThai(val("f_part"))' in app


def test_busy_when_overloaded(client, monkeypatch):
    # คนเข้าเยอะ: คิวทำนายเต็ม -> API ตอบ 503 JSON {"busy": true} + Retry-After ทันที (ไม่รอจน Cloudflare ตัดที่ 100 วินาที),
    # คำขอทั้งหมดเต็ม -> หน้าเว็บตอบหน้าขอโทษ busy.html (ไฟล์เดียว ไม่โหลดอะไรจากภายนอก) แต่ไฟล์ static ยังโหลดได้
    import re
    from webapp import server
    monkeypatch.setattr(server, "MAX_PREDICT", 0)
    r = client.post("/api/predict_location", json={"lat": 16.68, "lon": 102.23, "with_s2": False})
    assert r.status_code == 503 and r.json()["busy"] is True and int(r.headers["Retry-After"]) > 0
    r = client.post("/api/predict_specimen", json={"f_type": "ไดโนเสาร์"})
    assert r.status_code == 503 and r.json()["busy"] is True
    assert client.get("/api/options").status_code == 200                      # คำขอเบาไม่โดนจำกัดคิวทำนาย
    monkeypatch.setattr(server, "MAX_INFLIGHT", 0)
    r = client.get("/")
    assert r.status_code == 503 and "ขอโทษ" in r.text and "Retry-After" in r.headers
    assert not re.search(r'(src|href)="https?://', r.text)                      # ไม่พึ่งไฟล์ภายนอก
    assert client.get("/static/style.css").status_code == 200 and client.get("/busy").status_code == 200
    monkeypatch.undo()
    assert client.get("/").status_code == 200 and server._LOAD == {"all": 0, "predict": 0}   # ตัวนับคืนค่าหลังคำขอจบ
    app = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    assert "class BusyError" in app and all(c in app for c in ("429", "502", "503", "504", "524"))   # หน้าเว็บแสดงการ์ดขอโทษแทนข้อความผิดพลาด
