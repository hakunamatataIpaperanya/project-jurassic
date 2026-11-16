# Success criteria (notebook 16 -> models/success_criteria.csv): statuses follow the thresholds, key values re-computed
# independently, and the table in README_FULL.md (kept locally, not in git) matches the CSV.
import re

import pandas as pd
import pytest

from conftest import ROOT, ml

CSV = ml.MODELS / "success_criteria.csv"


@pytest.fixture(scope="module")
def crit():
    if not CSV.exists():
        pytest.skip("ยังไม่ได้รัน notebook 16")
    return pd.read_csv(CSV, encoding="utf-8-sig").set_index("รหัส")


def test_status_follows_thresholds(crit):
    for cid, r in crit.iterrows():
        ok = (lambda x, t: x >= t - 1e-9) if r["ทิศทาง"] == ">=" else (lambda x, t: x <= t + 1e-9)
        expect = "ถึงเป้าหมาย" if ok(r["ผล"], r["เป้าหมาย"]) else "ผ่านขั้นต่ำ" if ok(r["ผล"], r["ขั้นต่ำ"]) else "ไม่ผ่าน"
        assert r["สถานะ"] == expect, cid


def test_values_recomputed(crit):
    r = pd.read_parquet(ml.MODELS / "task2" / "A_bg_bias" / "results.parquet")
    a1 = r[r.exp.eq("spatial_block__xgb")].auc.mean()
    knn = r[r.exp.eq("spatial_block__knn_coords")].auc.mean()
    assert abs(crit.loc["A1", "ผล"] - a1) < 6e-4 and abs(crit.loc["A4", "ผล"] - (a1 - knn)) < 6e-4
    rb = pd.read_parquet(ml.MODELS / "task2" / "B_v3" / "results.parquet")
    assert abs(crit.loc["B1", "ผล"] - rb[rb.exp.eq("spatial_block__rf")].macro_auc.mean()) < 6e-4
    grid = pd.read_parquet(ml.MODELS / "task2" / "predictions" / "grid_5km_predictions.parquet")
    assert abs(crit.loc["B3", "ผล"] - grid.inside_B.mean()) < 6e-4
    sysG = pd.read_parquet(ml.MODELS / "task1" / "FE" / "genus_systems.parquet").set_index(["cv", "ระบบ"])
    assert abs(crit.loc["G2", "ผล"] - sysG.loc[("spatial_block", "ชุดรวมทั้งสองเส้นทาง"), "top3"]) < 6e-4


def test_readme_table_matches_csv(crit):
    f = ROOT / "README_FULL.md"                     # เอกสารฉบับเต็มเก็บไว้ในเครื่องผู้จัดทำ (ไม่ขึ้น git)
    if not f.exists():
        pytest.skip("ไม่มี README_FULL.md")
    text = f.read_text(encoding="utf-8")
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| ([A-Z]\d) \|.*$", text, flags=re.M)}
    assert set(crit.index) <= set(rows), sorted(set(crit.index) - set(rows))
    for cid, r in crit.iterrows():
        line = rows[cid]
        assert r["สถานะ"] in line, (cid, line)
        if not cid.startswith("E"):
            assert f"{r['ผล']:.3f}" in line, (cid, line)


def test_web_click_stats(crit):
    # ความแม่นเมื่อกดแผนที่ (หน้าเว็บ): ระดับ "สูงมาก" ขึ้นไปต้องเท่ากับ A3, ระดับต่ำลงครอบคลุมมากขึ้น และดีกว่าการสุ่ม
    import json
    k = json.loads((ml.MODELS / "web_click_stats.json").read_text(encoding="utf-8"))
    sh = k["share_at_or_above"]
    assert abs(sh["vhigh"] - crit.loc["A3", "ผล"]) < 6e-4
    assert sh["vhigh"] <= sh["high"] <= sh["mid"] and all(sh[lvl] > k["random"][lvl] for lvl in sh)
