# เว็บแบบไม่มีเซิร์ฟเวอร์ (static site): ข้อมูลที่ export ยังตรงกับ server และโค้ดของเบราว์เซอร์ (webapp/static/local.js) ให้ผลเดียวกับ Python
#   โมเดล ONNX       : scripts/web_models.py     ตาราง / ผลกริด : scripts/web_static.py     หน้าเว็บ : scripts/build_static_site.py
#   เทสต์ที่รัน local.js ต้องมี Node + tools/web/node_modules (cd tools/web && npm ci) — ไม่มีจะข้าม
import datetime
import functools
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import shapely

from conftest import ROOT, fl, ml

WEB = ml.MODELS / "web"
NODE = shutil.which("node")
HAVE_NODE = NODE is not None and (ROOT / "tools" / "web" / "node_modules" / "onnxruntime-web").exists()
pytestmark = pytest.mark.skipif(not (WEB / "specimen.json").exists(), reason="ยังไม่ได้รัน scripts/web_models.py / web_static.py")
needs_node = pytest.mark.skipif(not HAVE_NODE, reason="ไม่มี Node หรือ tools/web/node_modules")


@pytest.fixture(scope="module")
def ws():
    sys.path.insert(0, str(ROOT / "scripts"))
    import web_static
    return web_static


@pytest.fixture(scope="module")
def P(ws):
    return ws.server.predictor()


@pytest.fixture(scope="module")
def py(P, ws):
    # Predictor ที่โหลดโมเดลครั้งเดียว (predict_bundle เปิดไฟล์ทุกครั้ง) และไม่เขียนข้อความใหม่ลง cache ของโปรเจกต์
    mp = pytest.MonkeyPatch()
    mp.setattr(ml.joblib, "load", functools.lru_cache(maxsize=None)(ml.joblib.load))
    cache = pd.read_parquet(fl.FEAT / "text_emb" / "wangchanberta.parquet").set_index("text")
    extra_f = WEB / "text_emb_extra.parquet"
    if extra_f.exists():
        cache = pd.concat([cache, pd.read_parquet(extra_f).set_index("text").astype(cache.dtypes.iloc[0])])

    def text_embed(texts, kind="multilingual"):
        assert kind == "wangchanberta"
        texts = list(texts)
        new = [t for t in dict.fromkeys(texts) if t not in cache.index]
        got = {t: fl.encode_texts([t], kind)[0] for t in new}
        return np.vstack([got[t] if t in got else cache.loc[t].values for t in texts])

    mp.setattr(fl, "text_embed", text_embed)
    yield P
    mp.undo()


def run_node(tmp_path, cases):
    f, out = tmp_path / "cases.jsonl", tmp_path / "out.jsonl"
    f.write_text("\n".join(json.dumps({"id": i, **c}, ensure_ascii=False) for i, c in enumerate(cases)), encoding="utf-8")
    subprocess.run([NODE, str(ROOT / "tools" / "web" / "parity.mjs"), str(f), str(out)], check=True, timeout=1800)
    res = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        res[r["id"]] = r
    assert len(res) == len(cases)
    return [res[i] for i in range(len(cases))]


def registry_bodies(spec):
    # ข้อมูลตัวอย่างทั้งทะเบียนในรูปที่หน้าเว็บส่ง (เลือกแหล่งจากรายการ / ไม่ทราบแหล่ง)
    out = []
    for r in spec.itertuples():
        b = {k: (None if pd.isna(getattr(r, k)) else str(getattr(r, k))) for k in ["com_name", "f_type", "f_group", "f_part", "province", "district"]}
        b["site_OBJECTID"] = None if pd.isna(r.site_OBJECTID) else int(r.site_OBJECTID)
        out.append(b)
    return out


def _clean(body):
    return {k: v for k, v in body.items() if v is not None}


# ---------------------------------------------------------------- ไฟล์ที่ export
def test_onnx_genus_models_match_python(P, spec):
    # ONNX (Python onnxruntime) = predict_proba ของโมเดลเดิม บนข้อมูลจริงทั้งทะเบียน (ทั้งสองเส้นทาง)
    import onnxruntime as ort
    bodies = registry_bodies(spec)[::4]
    for route in ("known", "new"):
        bundle = ml.joblib.load(ml.MODELS / "task1" / "final" / P.genus["models"][route])
        rows = []
        for b in bodies:
            rec = dict(b)
            if route == "new":
                rec["site_OBJECTID"] = None
            elif rec["site_OBJECTID"] not in P.known_sites:
                continue
            rows.append(P.specimen_features(rec)[0])
        X = bundle["pre"].transform(pd.concat(rows, ignore_index=True)).values
        p0 = np.asarray(bundle["model"].predict_proba(X))
        s = ort.InferenceSession(str(WEB / f"genus_{route}.onnx"), providers=["CPUExecutionProvider"])
        if route == "new":
            X = np.where(np.abs(X) <= 1.0000000180025095e-35, np.float32(0), X)
        p1 = s.run(None, {"X": X})[0]
        assert np.abs(p0 - p1).max() < 1e-5 and (p0.argmax(1) == p1.argmax(1)).all(), route


def test_exported_data_up_to_date(ws, P):
    # webapp/data/*.json = สิ่งที่ server ส่งตอนนี้ (รัน scripts/web_static.py ใหม่เมื่อข้อมูลหรือโมเดลเปลี่ยน)
    for name, fn in ws.GET_ENDPOINTS.items():
        assert (ws.DATA_OUT / f"{name}.json").read_bytes() == fn().body, f"{name}.json เก่า — รัน scripts/web_static.py"
    assert (ws.DATA_OUT / "cells.json").read_bytes() == ws.server._dumps(ws.build_cells(P)), "cells.json เก่า"
    spec, blob = ws.build_specimen(P)
    assert (WEB / "specimen.json").read_bytes() == ws.server._dumps(spec) and (WEB / "specimen.bin").read_bytes() == blob


def test_cells_match_live_prediction_at_cell_centres(P):
    # ผลที่เก็บของแต่ละช่อง = predict_location ที่ศูนย์กลางช่อง (ไม่ดึงภาพ: ใช้ feature ภาพของช่องจาก cache แบบเดียวกับ notebook 10b)
    C = json.loads((ROOT / "webapp" / "data" / "cells.json").read_text(encoding="utf-8"))
    g = P.grid
    for i in np.random.default_rng(0).choice(C["n"], 40, replace=False):
        assert C["potential"]["bg_bias"][i] == round(float(g.potential_bg_bias.iat[i]), 3)
        assert C["top_types"][i][0][0] == C["classes_B"].index(g.top_type.iat[i])
        assert bool(C["inside"]["types"][i]) == bool(g.inside_B.iat[i])
    assert sum(s < 0 for s in C["s2_scene"]) == 0                    # ทุกช่องมีภาพ Sentinel-2


def test_static_files_fit_cloudflare_pages():
    # Cloudflare Pages: ไฟล์ละไม่เกิน 25 MiB
    files = [p for d in (ROOT / "webapp" / "data", WEB) for p in d.rglob("*") if p.is_file() and p.suffix != ".parquet"]
    assert files and max(p.stat().st_size for p in files) <= 25 * 1024 * 1024
    man = json.loads((WEB / "bert" / "manifest.json").read_text())
    assert sum((WEB / "bert" / p).stat().st_size for p in man["parts"]) == man["bytes"]


# ---------------------------------------------------------------- local.js เทียบกับ Python
@needs_node
def test_js_text_steps_match_python(tmp_path, P, spec):
    # ล้างชื่อสกุลใน f_part, TF-IDF + SVD, และตัดโทเค็นของ WangchanBERTa
    from transformers import AutoTokenizer
    parts = sorted(set(spec.f_part.dropna().astype(str))) + ["Siamosaurus teeth  ฟันSiamosaurus sp. ชิ้นส่วน", "  Phuwiangosaurus cf. sirindhornae  "]
    S = json.loads((WEB / "specimen.json").read_text(encoding="utf-8"))
    texts = S["bert"]["texts"][::7] + [f"ไดโนเสาร์ | ไดโนเสาร์ | {p}" for p in parts[::5]] + ["ก" * 300]
    cases = ([{"kind": "clean", "body": {"text": p}} for p in parts] + [{"kind": "tfidf", "body": {"text": t}} for t in texts]
             + [{"kind": "tokens", "body": {"text": t}} for t in texts])
    res = run_node(tmp_path, cases)
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]][:3]
    df = pd.DataFrame({"f_part": parts, "com_name_generic": "x", "f_type": "y"})
    want = [t.split(" | ", 2)[2] for t in fl.specimen_text(df)]
    got = [r["result"] for r in res[:len(parts)]]
    assert got == want
    Z = P.text["tfidf"].transform(texts).values
    assert np.abs(np.array([r["result"] for r in res[len(parts):len(parts) + len(texts)]]) - Z).max() < 1e-12
    tok = AutoTokenizer.from_pretrained(fl.EMB_MODELS["wangchanberta"])
    assert [r["result"] for r in res[len(parts) + len(texts):]] == [tok(t, truncation=True, max_length=64)["input_ids"] for t in texts]


@needs_node
def test_js_specimen_predictions_match_python(tmp_path, py, spec):
    # ทุกตัวอย่างในทะเบียน (8,521) แบบเลือกแหล่ง / ไม่ทราบแหล่ง + โหมดหน้าตาทุกคำตอบ: สกุล 3 อันดับและความน่าจะเป็นตรงกับ server
    bodies = registry_bodies(spec)
    looks = [{"look_card": lid.split("/")[0], "look_option": lid.split("/")[1] if "/" in lid else None, **extra}
             for lid in fl.appearance_leaves() for extra in ({}, {"site_OBJECTID": min(py.known_sites)}, {"site_OBJECTID": 1})]
    # ข้อความที่ไม่มีในตาราง (ประเภทกับส่วนของซากคนละแถว) -> หน้าเว็บใช้ WangchanBERTa float16
    rng = np.random.default_rng(2)
    typed = spec.dropna(subset=["f_type", "f_part"])
    novel = [{"f_type": typed.f_type.iat[i], "f_part": typed.f_part.iat[j] + " (พิมพ์เอง)"}
             for i, j in zip(rng.choice(len(typed), 150), rng.choice(len(typed), 150))]
    cases = [{"kind": "specimen", "body": b} for b in bodies + looks + novel]
    res = run_node(tmp_path, cases)
    bad, checked, model_path, n_model_same = [], 0, 0, 0
    for case, r in zip(cases, res):
        b = dict(case["body"])
        card, opt = b.pop("look_card", None), b.pop("look_option", None)
        if not b.get("f_type") and not b.get("com_name") and not card:
            assert not r["ok"]                                            # server ตอบ 422 แบบเดียวกัน
            continue
        assert r["ok"], (b, r)
        want = py.predict_specimen_appearance(b, card, opt) if card else py.predict_specimen(b)
        got = r["result"]
        if "model" in got["_text_source"]:                            # ข้อความที่ไม่มีในตาราง: WangchanBERTa float16 — ใกล้เคียง ไม่ใช่ตรงทุกตัว
            model_path += 1
            n_model_same += got["top"][0][0] == want["top"][0][0]
            continue
        # อันดับที่ความน่าจะเป็นเป็น 0 ทั้งสองฝั่ง (สกุลถูกตัดด้วยตัวกรองกลุ่มซาก) เสมอกันหลายตัว: numpy เลือกตัวไหนก็ได้ จึงไม่เทียบชื่อ
        keep = [k for k, (a, w) in enumerate(zip(got["top"], want["top"])) if not (a[1] == 0 and w[1] == 0)]
        same = (got["route"] == want["route"] and got["formation"] == want["formation"] and got["groups"] == want["groups"]
                and all(got["top"][k][0] == want["top"][k][0] and got["top"][k][2] == want["top"][k][2]
                        and got["species"][k] == want["species"][k] for k in keep)
                and all(abs(a[1] - w[1]) <= 0.0011 for a, w in zip(got["top"], want["top"])))
        checked += 1
        if not same:
            bad.append((b, card, opt, got["top"], want["top"]))
    print(f"\nตรงกับ Python: {checked - len(bad):,} / {checked:,} กรณีที่ข้อความมีในตาราง; "
          f"ข้อความใหม่ (WangchanBERTa float16) {model_path} กรณี สกุลอันดับ 1 ตรง {n_model_same}")
    assert not bad, bad[:5]
    if model_path:
        assert n_model_same / model_path >= 0.9


@needs_node
def test_js_point_and_coordinate_routes(tmp_path, py, spec):
    # วิเคราะห์จุด = ผลของช่องกริดที่ใกล้ที่สุด + จังหวัด / ระยะแหล่งของจุดที่คลิก; ทายสกุลจากพิกัด = หมวดหินและจังหวัดจากแผนที่แบบย่อ
    rng = np.random.default_rng(1)
    C = json.loads((ROOT / "webapp" / "data" / "cells.json").read_text(encoding="utf-8"))
    idx = rng.choice(C["n"], 300, replace=False)
    pts = [(C["lat"][i] + rng.normal(0, 0.01), C["lon"][i] + rng.normal(0, 0.01)) for i in idx]
    rows = spec.dropna(subset=["f_type"]).sample(300, random_state=1)
    coord_bodies = [{"f_type": r.f_type, "f_part": None if pd.isna(r.f_part) else r.f_part, "lat": la, "lon": lo}
                    for r, (la, lo) in zip(rows.itertuples(), pts)]
    cases = [{"kind": "location", "body": {"lat": la, "lon": lo}} for la, lo in pts] + [{"kind": "specimen", "body": b} for b in coord_bodies]
    res = run_node(tmp_path, cases)
    assert all(r["ok"] for r in res)
    prov_same = site_same = 0
    for (la, lo), r in zip(pts, res[:len(pts)]):
        got = r["result"]
        cos = np.cos(np.radians(la))
        i = int(np.argmin((np.array(C["lat"]) - la) ** 2 + ((np.array(C["lon"]) - lo) * cos) ** 2))
        assert got["potential_bg_bias"] == C["potential"]["bg_bias"][i] and got["similar_sites"] == [C["site_names"][k] for k in C["similar"][i]]
        hit = py.prov[py.prov.contains(shapely.geometry.Point(lo, la))]
        prov_same += got["province"] == (str(hit.prov_nam_t.iat[0]) if len(hit) else None)
        km = np.hypot((py.sites.lat - la) * 111.0, (py.sites.lon - lo) * 111.0 * cos)
        site_same += got["nearest_site_km"] == round(float(km.min()), 1)
    top_same = sum(r["result"]["top"][0][0] == py.predict_specimen(b)["top"][0][0] for b, r in zip(coord_bodies, res[len(pts):]))
    print(f"\nจังหวัดตรงกับ server {prov_same}/300 | ระยะแหล่งตรง {site_same}/300 | ทายสกุลจากพิกัด อันดับ 1 ตรง {top_same}/300")
    assert prov_same >= 294 and site_same == 300 and top_same >= 285


def test_build_static_site(tmp_path):
    # dist: หน้าเว็บโหมด static + ไฟล์ข้อมูล / โมเดลที่หน้าเว็บอ้างถึงครบ, ไม่เกินขีดจำกัดของ Cloudflare Pages
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static_site as b
    dist = tmp_path / "dist"
    dirs, files = b.build(dist)
    html = (dist / "index.html").read_text(encoding="utf-8")
    cfg = json.loads(html.split('name="jurassic-static" content=\'')[1].split("'")[0])
    assert cfg.pop("built") == datetime.date.today().isoformat()          # วันที่ build (หน้าแรกบอก "ปรับปรุงเว็บ")
    assert {k: v.strip("/") for k, v in cfg.items()} == {k: d.name for k, d in dirs.items()}
    for name in ["local.js", "app.js", "style.css"]:
        assert f'/static/{name}?v=' in html and (dist / "static" / name).exists()
    assert {p.name for p in (dist / cfg["data"].strip("/")).iterdir()} == {f"{n}.json" for n in b.DATA_FILES}
    man = json.loads((dist / cfg["bert"].strip("/") / "manifest.json").read_text())
    assert all((dist / cfg["bert"].strip("/") / p).exists() for p in man["parts"])
    assert max(p.stat().st_size for p in files) <= b.MAX_FILE and len(files) <= b.MAX_FILES
    assert "immutable" in (dist / "_headers").read_text()
    og = html.split('property="og:image" content="')[1].split('"')[0]                    # ภาพตัวอย่างลิงก์: URL เต็ม + มีไฟล์จริง
    assert og.startswith(b.SITE_URL + "/static/img/og.jpg?v=") and (dist / "static" / "img" / "og.jpg").exists()


def test_local_js_versions_pinned():
    # local.js โหลด onnxruntime-web / tokenizers จาก CDN รุ่นเดียวกับที่เทสต์ใช้ (tools/web/package.json)
    js = (ROOT / "webapp" / "static" / "local.js").read_text(encoding="utf-8")
    pkg = json.loads((ROOT / "tools" / "web" / "package.json").read_text())["dependencies"]
    assert f'ORT_VERSION = "{pkg["onnxruntime-web"]}"' in js and f'TOKENIZERS_VERSION = "{pkg["@huggingface/tokenizers"]}"' in js
