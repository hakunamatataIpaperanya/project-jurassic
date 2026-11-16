# No final model uses a feature that is derived from the label, from the data-linking process, or that encodes location /
# survey effort directly. Checks the feature lists stored INSIDE the saved bundles (what the models really use).
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from conftest import fl, ml

T2, T1 = ml.MODELS / "task2" / "final", ml.MODELS / "task1" / "final"
TASK2_FORBIDDEN = re.compile(r"(_lat_0$|_abslat_0$|_lon_\d+$|^s2_item_id$|^s2_date$|^s2_cloud_frac$|^lon$|^lat$|^point_id$|^source$|"
                             r"^fold_|^group_|^dc_|^sc_|^type_|^cnn_pc|_begin_ma$)")
TASK1_FORBIDDEN = re.compile(r"^(com_name|dis_name|match_level|confidence|merge_confidence|regis_code|sci_name|genus|genus_true|"
                             r"species_true|taxon|genus_cls|fossil_id|point_id|tied_site_ids|score|kw|grp|adm|fuzzy_note|n_candidates|"
                             r"n_top)$|^target_|^ref_|^fold_|^group_|^site_(PERIODFROM|PERIOD_FINAL|FOS_DES_TH|NAME_TH|IMPORTA_TH)$")


def bundle_cols(path):
    b = joblib.load(path)
    return list(b["pre"].num_cols) + list(b["pre"].cat_cols)


def test_task2_final_features(meta2):
    for stage in [meta2["stage_A"]["bg_bias"], meta2["stage_A"]["bg_uniform"], meta2["stage_B"]]:
        cols = bundle_cols(T2 / stage["tabular"])
        bad = [c for c in cols if TASK2_FORBIDDEN.search(c)]
        assert not bad, f"{stage['tabular']}: {bad}"
        assert set(cols) == set(meta2["features"]["num"]) | set(meta2["features"]["cat"])


def test_task1_final_features(meta1):
    for name in list(meta1["genus"]["models"].values()) + [meta1["period"]["model"]]:
        cols = bundle_cols(T1 / name)
        bad = [c for c in cols if TASK1_FORBIDDEN.search(c)]
        assert not bad, f"{name}: {bad}"
    period = bundle_cols(T1 / meta1["period"]["model"])
    assert not [c for c in period if c in ("FORMATION", "GEO_GROUP") or c.startswith(("type_", "cnn_pc", "period_", "fm_"))], \
        "period model must not use recorded FORMATION / type flags / CNN"


def test_deploy_uses_base_stage_a(meta2):
    assert meta2["deploy"]["stage_A_variant"] == "base" and meta2["deploy"]["use_cnn"] is False


def test_no_single_feature_too_good(allf, meta2):
    # feature เดี่ยวที่แยกแหล่งจาก background ได้เกือบสมบูรณ์ = สงสัยว่ารั่ว
    d = allf[allf.source.isin(["dmr_site", "bg_bias"])]
    y = d.source.eq("dmr_site")
    worst = {}
    for c in meta2["features"]["num"]:
        v = d[c]
        ok = v.notna()
        if v[ok].nunique() > 1:
            a = roc_auc_score(y[ok], v[ok])
            worst[c] = max(a, 1 - a)
    top = max(worst, key=worst.get)
    assert worst[top] < 0.9, (top, round(worst[top], 3))


def test_com_name_generic_has_no_species_names(spec, meta1):
    # ค่า com_name_generic / f_type ที่โมเดลมองเห็นได้ (≥ 5 ครั้ง; ค่าที่น้อยกว่านั้นตัวแปลงรวมเป็น "อื่น") ต้องไม่ใช่ชื่อเฉพาะชนิด
    # ที่มีแต่ตัวอย่างที่ระบุชื่อแล้วใช้ — ยอมให้เป็นหมวดซากจาก f_type (เช่น วาฬ) ที่บังเอิญมีแต่ฝั่งมีชื่อ
    vocab = set(meta1["genus"]["com_vocab"])
    named_mask = spec.sci_name.map(lambda n: fl.parse_taxon(n) is not None)
    generic = spec.com_name.where(spec.com_name.isin(vocab), spec.f_type)
    vc = generic[named_mask].value_counts()
    visible = set(vc[vc >= 5].index)
    from_com_name = set(spec.loc[named_mask & spec.com_name.isin(vocab), "com_name"])
    only_named = visible - set(generic[~named_mask]) - set(spec.f_type.dropna())
    assert not (only_named & from_com_name), only_named & from_com_name


def test_specimen_text_has_no_genus_names(spec):
    d = spec.assign(com_name_generic=spec.f_type)
    pat = fl.genus_name_pattern()
    left = [t for t in fl.specimen_text(d) if pat.search(t)]
    assert not left, left[:3]


def test_period_fills_used_for_training_are_pbdb_backed():
    fin = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", dtype={"SITE_CODE": str}, encoding="utf-8-sig")
    ok = fin.final_confidence.isin(["high", "medium"]) & fin.final_source.ne("DMR record") & fin.final_source.astype(str).str.contains("PBDB")
    assert not fin[ok].final_source.eq("04 taxon hybrid").any()
    assert ok.sum() == 129          # 128 + รอยตีนไดโนเสาร์ลำน้ำหมันแดง (notebook 05 ส่วน 2b: เพอร์เมียน -> ครีเทเชียสจาก PBDB)


def test_cnn_embeddings_are_per_fold():
    # embedding ที่ใช้ใน CV ของงานที่ 1 ต้องมาจาก CNN ของ fold นั้น (ไม่ใช่ตัวจริงที่เห็นทุกแหล่ง)
    final = pd.read_parquet(ml.FEAT / "site_task1_features.parquet").set_index("oid")
    for cv in ml.CV_SCHEMES:
        for k in range(ml.N_FOLDS):
            e = pd.read_parquet(ml.MODELS / "task1" / "emb" / f"{cv}__f{k}.parquet")
            assert len(e) == 518 and not np.allclose(e.values, final[e.columns].reindex(e.index).values)
