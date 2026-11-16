# Metric / helper functions (unit tests) and re-computation of every saved CV result from its out-of-fold predictions.
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from conftest import fl, ml


def test_multiclass_metrics():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1], [0.2, 0.5, 0.3]])
    y = np.array([0, 2, 1, 0])
    order = ["Permian", "Triassic", "Jurassic"]
    m = ml.multiclass_metrics(y, P, order, order)
    assert m["acc"] == 0.5 and m["top3"] == 1.0
    assert m["within1"] == 1.0           # ผิด 2 แถว แต่คลาดแค่ 1 ยุคทั้งคู่ (Permian↔Triassic)


def test_within1_exact():
    order = ["Cambrian", "Ordovician", "Silurian", "Devonian"]
    P = np.eye(4)[[0, 3, 1]]                       # ทาย Cambrian, Devonian, Ordovician
    y = np.array([0, 1, 1])                        # จริง Cambrian, Ordovician, Ordovician
    m = ml.multiclass_metrics(y, P, order, order)
    assert m["acc"] == pytest.approx(2 / 3) and m["within1"] == pytest.approx(2 / 3)


def test_multilabel_and_binary_metrics():
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 1], [0, 0]])
    P = Y * 0.8 + 0.1
    m = ml.multilabel_metrics(Y, P, min_pos=2)
    assert m["macro_auc"] == 1.0 and m["top1"] == 1.0
    b = ml.bin_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert b["auc"] == pytest.approx(0.75)
    assert ml.ks_vs_random(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0


def test_consistency_filter_and_routing():
    P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    gm = ml.genus_group_map(["A", "B"], [{"x"}, {"y"}])
    Q = ml.consistency_filter(P, ["A", "B", "อื่น ๆ"], [{"y"}, set()], gm)
    assert np.allclose(Q.sum(1), 1) and Q[0, 0] == 0 and np.allclose(Q[1], P[1])
    assert list(ml.route_known_site(pd.array([5, None, 7], dtype="Int64"), [5])) == [True, False, False]


def test_spatial_context_excludes_own_group(allf):
    s = allf[allf.source.eq("dmr_site")].head(200).reset_index(drop=True)
    sc = fl.spatial_context(s, s, s.group_spatial_block.values, s.group_spatial_block.values)
    from sklearn.neighbors import BallTree
    t = BallTree(np.radians(s[["lat", "lon"]].values), metric="haversine")
    for i in range(0, 200, 25):
        d, j = t.query(np.radians(s.loc[[i], ["lat", "lon"]].values), k=len(s))
        other = [dd * 6371 for dd, jj in zip(d[0], j[0]) if s.group_spatial_block[jj] != s.group_spatial_block[i]]
        expect = min(min(other), 100.0) if other else 100.0
        assert sc.sc_dist_site_km[i] == pytest.approx(min(expect, 100.0), rel=1e-6) or (expect > 100 and sc.sc_dist_site_km[i] == 100)


def test_aoa_training_points_mostly_inside(allf, meta2):
    a = ml.joblib.load(ml.MODELS / "task2" / "final" / meta2["aoa"]["A_bg_bias"])
    d = allf[allf.source.isin(["dmr_site", "bg_bias"])]
    assert a.inside(d).mean() > 0.97


# ---- re-compute every registry's saved fold metric from its OOF file
REGISTRIES = [("task2", "A_bg_bias", "auc"), ("task2", "A_bg_uniform", "auc"), ("task2", "B_v2", "macro_auc"),
              ("task2/FE", "A_bg_bias", "auc"), ("task2/FE", "A_bg_uniform", "auc"), ("task2/FE", "B", "macro_auc")]


@pytest.mark.parametrize("folder,name,metric", REGISTRIES)
def test_registry_metrics_match_oof(folder, name, metric):
    reg = ml.Registry(ml.MODELS / folder / name)
    r = reg.results()
    assert len(r) and not r.duplicated(["exp", "fold"]).any()
    bad = []
    for exp, g in r.groupby("exp"):
        o = reg.oof(exp)
        assert o is not None and o.point_id.is_unique, exp
        assert sorted(g.fold) == sorted(o.fold.unique()), exp
        for _, row in g.iterrows():
            f = o[o.fold.eq(row.fold)]
            if metric == "auc":
                v = roc_auc_score(f.y, f.p)
            else:
                yc = [c for c in f.columns if c.startswith("y_")]
                v = ml.multilabel_metrics(f[yc].values, f[[c.replace("y_", "p_") for c in yc]].values)["macro_auc"]
            if not np.isclose(v, row[metric], atol=1e-9):
                bad.append((exp, row.fold, v, row[metric]))
    assert not bad, bad[:5]


@pytest.mark.parametrize("name", ["A_bg_bias", "A_bg_uniform"])
def test_oof_covers_each_point_once(allf, name):
    reg = ml.Registry(ml.MODELS / "task2" / name)
    bg = name[2:]
    expect = set(allf[allf.source.isin(["dmr_site", bg])].point_id)
    for exp in ["spatial_block__xgb", "random__lgbm", "site_cluster__rf"]:
        o = reg.oof(exp)
        assert set(o.point_id) == expect and len(o) == len(expect), exp


def test_final_meta_cv_scores_match_registries(meta2):
    # คะแนน CV ที่บันทึกใน final_meta.json ต้องเท่ากับผลใน registry ปัจจุบัน (ไม่ค้างจากรอบก่อนที่คำนวณใหม่)
    def spatial(reg, metric):
        r = pd.read_parquet(ml.MODELS / "task2" / reg / "results.parquet")
        r[["cv", "model"]] = r.exp.str.split("__", expand=True)
        return r[r.cv.eq("spatial_block")].groupby("model")[metric].mean()
    for bg in ["bg_bias", "bg_uniform"]:
        t = spatial(f"A_{bg}", "auc")
        assert all(abs(v - t[k]) < 1e-4 for k, v in meta2["stage_A"][bg]["cv_auc_spatial_block"].items()), bg
    t = spatial("B_v3", "macro_auc")
    assert all(abs(v - t[k]) < 1e-4 for k, v in meta2["stage_B"]["cv_macro_auc_spatial_block"].items())
