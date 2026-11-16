# Every cache that results depend on must be traceable to the current data:
#   CV registries / final models / derived feature tables carry a fingerprint of their inputs (see model_lib.fingerprint),
#   and cached point features equal a fresh computation.
import json

import numpy as np
import pandas as pd
import pytest

from conftest import CAT, fl, ml

# registries the current builders run experiments in (older ones, e.g. task1/genus, are kept only for comparison tables)
ACTIVE = ["task2/A_bg_bias", "task2/A_bg_uniform", "task2/B_v3", "task2/B_v3_labels/strict", "task2/B_v3_labels/text_only",
          "task2/FE/A_bg_bias", "task2/FE/A_bg_uniform", "task2/FE/B_v2",
          "task1/genus_v5", "task1/period_v5_rec", "task1/period_v5_rec_fill", "task1/FE/genus_v2", "task1/FE/period"]


@pytest.mark.parametrize("name", ACTIVE)
def test_registry_rows_fingerprinted(name):
    # ทุกผล (การทดลอง, fold) ต้องมี fingerprint = คำนวณ/ตรวจซ้ำแล้วกับข้อมูลปัจจุบัน (ไม่มีผลค้างจากข้อมูลรุ่นเก่า)
    r = pd.read_parquet(ml.MODELS / name / "results.parquet")
    assert "fp" in r and r.fp.notna().all(), f"{r.fp.isna().sum() if 'fp' in r else len(r)} แถวไม่มี fingerprint"


@pytest.mark.parametrize("name", ACTIVE)
def test_registry_audit_resolved(name):
    # ถ้าเคยพบผลเก่าที่ไม่ตรง (DIFFERENT) แถวนั้นต้องถูกแทนที่แล้ว (fp ปัจจุบันมาจากการคำนวณใหม่)
    f = ml.MODELS / name / "cache_audit.parquet"
    if not f.exists():
        pytest.skip("ไม่มีการตรวจผลรุ่นก่อน")
    a = pd.read_parquet(f)
    r = pd.read_parquet(ml.MODELS / name / "results.parquet").set_index(["exp", "fold"])
    bad = a[a.result.str.contains("DIFFERENT")]
    for row in bad.itertuples():
        assert pd.notna(r.loc[(row.exp, row.fold), "fp"]), (row.exp, row.fold)


def test_registry_oof_consistent_with_results():
    # ไฟล์ OOF ของทุกแถวต้องมีจริง และจำนวนแถวตรงกับ n_test
    for name in ACTIVE:
        reg = ml.Registry(ml.MODELS / name)
        r = reg.results()
        for row in r.itertuples():
            o = reg.oof_path(row.exp, row.fold)
            assert o.exists(), o
        smp = r.sample(min(40, len(r)), random_state=0)
        for row in smp.itertuples():
            assert len(pd.read_parquet(reg.oof_path(row.exp, row.fold))) == row.n_test, (name, row.exp, row.fold)


def test_final_tabular_models_fingerprinted(meta1, meta2):
    names = [meta2["stage_A"][bg]["tabular"] for bg in meta2["deploy"]["backgrounds"]] + [meta2["stage_B"]["tabular"]]
    paths = [ml.MODELS / "task2" / "final" / n for n in names]
    paths += [ml.MODELS / "task1" / "final" / n for n in list(meta1["genus"]["models"].values()) + [meta1["period"]["model"]]]
    for p in paths:
        if p.suffix == ".joblib":
            assert ml.joblib.load(p).get("fp"), p.name


def test_derived_feature_caches_fingerprinted():
    for d in [ml.MODELS / "task1" / "emb", ml.MODELS / "task2" / "FE" / "sc_v2", ml.MODELS / "task1" / "FE" / "ctx_v2"]:
        files = sorted(d.glob("*.parquet"))
        assert files, d
        missing = [f.name for f in files if not f.with_name(f.name + ".fp").exists()]
        assert not missing, (d.name, missing)
    assert (ml.MODELS / "task1" / "final" / "cnn_pca.joblib.fp").exists()
    assert ml.joblib.load(ml.MODELS / "task1" / "final" / "genus_v7_text.joblib").get("fp")


def test_folds_cover_all_points_and_reproduce():
    allf = pd.read_parquet(ml.FEAT / "all_features.parquet")
    folds = pd.read_parquet(ml.FEAT / "folds.parquet")
    assert set(allf.point_id) == set(folds.point_id) and folds.point_id.is_unique
    new = ml.assign_folds(allf[["point_id", "lon", "lat"]]).set_index("point_id").loc[folds.point_id]
    for c in folds.columns.drop("point_id"):
        assert (new[c].astype(str).values == folds[c].astype(str).values).all(), c


def test_grid_cached_features_match_live():
    # feature ของตาราง 5 กม. (cache ของ notebook 10b) = คำนวณสดด้วย FeatureBuilder (ส่วนที่ไม่ใช้เน็ต)
    grid = pd.read_parquet(ml.FEAT / "grid_5km.parquet").sample(12, random_state=4).reset_index(drop=True)
    cached = (grid[["point_id"]].merge(fl.batch_pangea(grid, tag="grid_"), on="point_id")
              .merge(fl.batch_dem(grid, tag="grid_"), on="point_id").merge(fl.batch_geology(grid, tag="grid_"), on="point_id"))
    live, _ = fl.FeatureBuilder().tabular(grid.lon.tolist(), grid.lat.tolist(), with_s2=False)
    bad = []
    for c in [c for c in live.columns if c in cached.columns and c != "point_id"]:
        a, b = cached[c], live[c]
        if c in CAT or not pd.api.types.is_numeric_dtype(a):
            if (a.astype(str).values != b.astype(str).values).any():
                bad.append(c)
        elif not np.allclose(a.astype(float), b.astype(float), rtol=1e-4, atol=1e-6, equal_nan=True):
            bad.append(c)
    assert not bad, bad[:10]


def test_cnn_pca_matches_final_cnn():
    # PCA ของ CNN (site_task1_features) ต้อง fit จาก CNN ตัวจริงรุ่นปัจจุบัน: PCA ที่ fit ใหม่ต้องให้ค่าเดียวกัน (ไม่นับเครื่องหมาย)
    from sklearn.decomposition import PCA
    meta2 = json.loads((ml.MODELS / "task2" / "final" / "final_meta.json").read_text(encoding="utf-8"))
    st = pd.read_parquet(ml.FEAT / "site_task1_features.parquet")
    patches, ok = ml.load_patches(st.point_id.values, ml.FEAT / "s2" / "patches")
    emb = ml.embed_bundle(ml.MODELS / "task2" / "final" / meta2["stage_A"]["bg_bias"]["cnn"], patches)
    fresh = PCA(n_components=16, random_state=0).fit(emb[ok]).transform(emb[ok])
    stored = st.loc[ok, [f"cnn_pc{i}" for i in range(16)]].values
    assert np.allclose(np.abs(fresh), np.abs(stored), atol=1e-2)


def test_final_cnns_trained_on_current_patches(allf, meta2):
    # CNN ตัวจริง (train ซ้ำได้ไม่ตรงทุกบิตบน MPS) ตรวจทางอ้อม: ค่า normalise ที่เก็บในโมเดล = ค่าจาก patch ของชุด train ปัจจุบัน
    import torch
    T2 = ml.MODELS / "task2" / "final"
    sets = {meta2["stage_A"][bg]["cnn"]: allf[allf.source.isin(["dmr_site", bg])].point_id.values for bg in ["bg_bias", "bg_uniform"]}
    sets[meta2["stage_B"]["cnn"]] = ml.Registry(ml.MODELS / "task2" / "B_v3").oof("spatial_block__cnn").point_id.values
    for name, ids in sets.items():
        b = torch.load(T2 / name, map_location="cpu", weights_only=False)
        fresh = ml.PatchNorm().fit(ml.load_patches(ids, ml.FEAT / "s2" / "patches")[0])
        assert np.allclose(fresh.mean_, b["pre"].mean_, atol=1e-5) and np.allclose(fresh.std_, b["pre"].std_, atol=1e-5), name
    assert torch.load(T2 / meta2["stage_B"]["cnn"], map_location="cpu", weights_only=False)["classes"] == meta2["classes_B"]


def test_active_registry_list_is_complete():
    # registry ที่ builder ปัจจุบันใช้ (มี fingerprint แล้ว) ต้องอยู่ใน ACTIVE ทั้งหมด — เพิ่ม registry ใหม่แล้วลืมใส่ ACTIVE จะไม่ถูกตรวจ
    used = set()
    for f in ml.MODELS.glob("**/results.parquet"):
        if "_stale" in str(f):
            continue
        r = pd.read_parquet(f)
        if "fp" in r and r.fp.notna().any():
            used.add(str(f.parent.relative_to(ml.MODELS)))
    assert used == set(ACTIVE), (sorted(used - set(ACTIVE)), sorted(set(ACTIVE) - used))
