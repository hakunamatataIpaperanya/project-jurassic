# Training and inference agree: features computed live = stored features; saved prediction files = what the saved models give.
import nbformat
import numpy as np
import pandas as pd
import pytest

from conftest import CAT, ROOT, fl, ml

T1, T2 = ml.MODELS / "task1" / "final", ml.MODELS / "task2" / "final"


@pytest.mark.slow
def test_live_features_match_training(allf, meta2):
    # FeatureBuilder (ใช้ตอนทำนายพิกัดใหม่) ต้องให้ค่าเท่ากับตารางที่ใช้ train ทุก feature รวมทั้งการเลือกภาพ Sentinel-2
    smp = pd.concat([allf[allf.source.eq(s)].sample(2, random_state=11) for s in ["dmr_site", "pbdb", "bg_bias", "bg_uniform"]])
    live, _ = fl.FeatureBuilder().tabular(smp.lon.tolist(), smp.lat.tolist(), with_s2=True)
    live.index = smp.index
    bad = []
    for c in meta2["features"]["num"] + meta2["features"]["cat"] + ["s2_item_id"]:
        a, b = smp[c], live[c]
        if c in CAT or c == "s2_item_id":
            if (a.astype(str) != b.astype(str)).any():
                bad.append(c)
        elif not np.allclose(a.astype(float), b.astype(float), rtol=1e-4, atol=1e-6, equal_nan=True):
            bad.append(c)
    assert not bad, bad[:10]


def test_extra_features_recompute(allf):
    stored = pd.read_parquet(ml.FEAT / "extra_features.parquet").set_index("point_id")
    pg = pd.read_parquet(ml.FEAT / "pangea_features.parquet")
    smp = allf.sample(80, random_state=3)[["point_id", "lon", "lat", "geo_rock_sym"]].merge(
        pg[["point_id"] + [c for c in pg.columns if c.startswith("pg_sw18_l")]], on="point_id")
    new = fl.extra_features(smp, pangea=smp, geology=smp).set_index(smp.point_id)
    old = stored.loc[new.index, new.columns]
    assert np.allclose(new.astype(float).values, old.astype(float).values, rtol=1e-3, atol=1e-3, equal_nan=True)


def _grid_features():
    grid = pd.read_parquet(ml.FEAT / "grid_5km.parquet")
    G = (grid.merge(fl.batch_pangea(grid, tag="grid_"), on="point_id").merge(fl.batch_dem(grid, tag="grid_"), on="point_id")
         .merge(fl.batch_geology(grid, tag="grid_"), on="point_id")
         .merge(fl.batch_s2_indices(grid, ml.FEAT / "s2" / "grid_patches", tag="grid_"), on="point_id", how="left"))
    for c in CAT:
        G[c] = G[c].astype(str)
    return G


def test_grid_predictions_match_models(meta2):
    pred = pd.read_parquet(ml.MODELS / "task2" / "predictions" / "grid_5km_predictions.parquet").set_index("point_id")
    G = _grid_features().sample(600, random_state=5).set_index("point_id", drop=False)
    for bg in meta2["deploy"]["backgrounds"]:
        p = ml.predict_bundle(T2 / meta2["stage_A"][bg]["tabular"], G)
        assert np.allclose(p, pred.loc[G.index, f"potential_{bg}"].values, atol=1e-6), bg
        aoa = ml.joblib.load(T2 / meta2["aoa"][f"A_{bg}"])
        assert (aoa.inside(G) == pred.loc[G.index, f"inside_{bg}"].values).all(), f"AOA {bg}"
    PB = ml.predict_bundle(T2 / meta2["stage_B"]["tabular"], G)
    cols = [f"type_{c}" for c in meta2["classes_B"]]
    assert np.allclose(PB, pred.loc[G.index, cols].values, atol=1e-6)
    assert (pred.loc[G.index, "top_type"].values == np.array(meta2["classes_B"])[PB.argmax(1)]).all()


def test_period_predictions_match_model(allf, meta1):
    pred = pd.read_csv(ml.DATA / "processed" / "task1_period_predictions.csv")
    ex = pd.read_parquet(ml.FEAT / "extra_features.parquet")
    s = allf[allf.source.eq("dmr_site")].merge(ex, on="point_id", how="left")
    s["oid"] = s.point_id.str[4:].astype(int)
    X = s.set_index("oid").loc[pred.OBJECTID].reset_index()
    P = ml.predict_bundle(T1 / meta1["period"]["model"], X)
    assert (np.array(meta1["period"]["classes"])[P.argmax(1)] == pred.pred_period.values).all()
    assert np.allclose(P.max(1), pred.p.values, atol=1e-6)


def test_inference_notebook_matches_task1():
    nb = nbformat.read(ROOT / "notebooks" / "12_inference.ipynb", 4)
    outs = [o.text for c in nb.cells if c.cell_type == "code" and "chk = pd.DataFrame" in c.source
            for o in c.outputs if o.output_type == "stream" and o.name == "stdout"]
    assert outs and "ตรงกัน 100%" in outs[0], outs


def test_genus_predictions_file(meta1, spec):
    g = pd.read_csv(ml.DATA / "processed" / "task1_genus_predictions.csv", dtype={"fossil_id": str})
    todo = spec[spec.sci_name.map(lambda n: fl.parse_taxon(n) is None)]
    assert set(g.fossil_id) == set(todo.fossil_id), "ต้องทำนายทุกตัวอย่างที่ไม่มีสกุล (ตาม parse_taxon ปัจจุบัน)"
    assert set(g.genus_top1) <= set(meta1["genus"]["classes"])
    assert (g.p_top1 >= g.p_top2).all() and g.p_top1.between(0, 1).all()
    gm = {k: set(v) for k, v in meta1["genus"]["group_map"].items()}
    vocab = set(meta1["genus"]["com_vocab"])
    grp = {f: fl.specimen_classes(t, c if c in vocab else t) for f, t, c in zip(spec.fossil_id, spec.f_type, spec.com_name)}
    cross = [r.fossil_id for r in g.itertuples() if grp[r.fossil_id] and r.genus_top1 != "อื่น ๆ" and gm.get(r.genus_top1) and not (gm[r.genus_top1] & grp[r.fossil_id])]
    assert not cross, f"ทายข้ามกลุ่มซาก {len(cross)} ตัวอย่าง"


def test_pangea_offline_matches_notebook08():
    # Pangea โหลดแบบจำลองแผ่นเปลือกโลกจากไฟล์ในเครื่อง (ไม่ต้องพึ่งเซิร์ฟเวอร์ GPlates) และต้องได้ค่าเท่ากับ notebook 08
    ref = pd.read_parquet(ml.FEAT / "pangea_features.parquet")
    pts = (pd.read_parquet(ml.FEAT / "points.parquet")[["point_id", "lon", "lat"]]
           .merge(ref.drop(columns=[c for c in ["lon", "lat", "source"] if c in ref]), on="point_id").sample(120, random_state=2).reset_index(drop=True))
    new = fl.Pangea().features(pts.lon.values, pts.lat.values)
    num = [c for c in new.columns if c in pts.columns and pd.api.types.is_numeric_dtype(pts[c])]
    assert np.allclose(new[num].astype(float).values, pts[num].astype(float).values, equal_nan=True)
    assert all(fl.plate_model_files(n)[0] for n in fl.PG_MODELS.values())


def test_inference_survives_sentinel2_outage(monkeypatch):
    # ถ้า Planetary Computer ล่ม ต้องได้ feature ที่เหลือครบ, s2_* เป็นค่าว่าง และมีคำเตือน (ไม่ล้มทั้งการทำนาย)
    def boom(lon, lat):
        raise ConnectionError("simulated outage")
    monkeypatch.setattr(fl, "fetch_patch_single", boom)
    with pytest.warns(UserWarning, match="Sentinel-2"):
        df, patches = fl.FeatureBuilder().tabular([102.2357], [16.6803], with_s2=True)
    assert patches == [None] and df.s2_item_id.isna().all()
    assert df[[c for c in df.columns if c.startswith("s2_") and c != "s2_item_id"]].isna().all().all()
    assert df[[c for c in df.columns if c.startswith(("pg_", "dem_", "geo_"))]].notna().any(axis=1).all()


@pytest.mark.parametrize("kind", ["multilingual", "wangchanberta"])
def test_text_embedding_cache_matches_fresh(kind):
    # embedding ที่ cache ต้องเท่ากับค่าที่คำนวณใหม่จากตัวข้อความ (จับ cache ค้าง / เลื่อนแถว — รอบก่อน notebook 15 ใช้ .npy ตามลำดับแถว
    # ซึ่งเลื่อนผิดข้อความหลังข้อความเปลี่ยน)
    f = ml.FEAT / "text_emb" / f"{kind}.parquet"
    if not f.exists():
        pytest.skip(f"ยังไม่มี cache {kind}")
    smp = pd.read_parquet(f).set_index("text").sample(32, random_state=1)
    fresh = fl.encode_texts(list(smp.index), kind)
    A = smp.values.astype(float)
    cos = (A * fresh).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(fresh, axis=1)
    assert (cos > 0.999).all(), f"{(cos <= 0.999).sum()} ข้อความไม่ตรง"


def test_no_positional_embedding_cache():
    # embedding ต้องดึงด้วยตัวข้อความเท่านั้น: ห้ามมีไฟล์ .npy ตามลำดับแถวที่ notebook ใช้อยู่
    assert not list((ml.FEAT / "text_emb").glob("*.npy"))
    for b in (ROOT / "scripts").glob("build_*.py"):
        assert "TEXT_EMB[kind][" not in b.read_text(encoding="utf-8"), b.name


def test_final_genus_text_features(meta1):
    # เส้นทางแหล่งใหม่ใช้ WangchanBERTa (กติกา: BERT ก่อน ถ้าไม่ดีกว่าใช้ multilingual — notebook 15: BERT ดีกว่า); แหล่งเดิมใช้ TF-IDF
    text = ml.joblib.load(T1 / meta1["genus"]["text"])
    assert set(text) - {"fp"} == {"tfidf", "bert"} and isinstance(text["bert"], fl.TextBert)
    new_num = meta1["genus"]["features"]["new"][0]
    assert any(c.startswith("wangchanberta_") for c in new_num) and not any(c.startswith("multilingual_") for c in new_num)
    assert not any(c.startswith(("multilingual_", "wangchanberta_")) for c in meta1["genus"]["features"]["known"][0])
