# Integrity of the processed data, points, folds and feature tables.
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from conftest import fl, ml


def test_points_counts_and_ids():
    p = pd.read_parquet(ml.FEAT / "points.parquet")
    assert p.point_id.is_unique
    assert p.source.value_counts().to_dict() == {"bg_uniform": 5000, "bg_bias": 5000, "dmr_site": 518, "pbdb": 290}
    w, s, e, n = fl.THAI_BBOX
    assert p.lon.between(w, e).all() and p.lat.between(s, n).all()


def test_background_far_from_known_sites(allf):
    known = allf[allf.source.isin(["dmr_site", "pbdb"])]
    tree = BallTree(np.radians(known[["lat", "lon"]].values), metric="haversine")
    for bg in ["bg_bias", "bg_uniform"]:
        b = allf[allf.source.eq(bg)]
        d = tree.query(np.radians(b[["lat", "lon"]].values), k=1)[0][:, 0] * 6371
        assert d.min() >= 1.99, f"{bg}: background {d.min():.2f} km from a known site"


def test_all_features_matches_points(allf):
    p = pd.read_parquet(ml.FEAT / "points.parquet")
    assert allf.point_id.is_unique and set(allf.point_id) == set(p.point_id)


def test_missingness_not_a_label_proxy(allf, meta2):
    # ค่าว่างต้องไม่ต่างกันระหว่างแหล่งกับ background (ไม่งั้นโมเดลเรียนจาก "ว่างหรือไม่")
    num = meta2["features"]["num"]
    s, b = allf[allf.source.eq("dmr_site")], allf[allf.source.eq("bg_bias")]
    diff = (s[num].isna().mean() - b[num].isna().mean()).abs()
    assert diff.max() < 0.02, diff.sort_values().tail(3).to_dict()


def test_folds_groups_not_split(allf):
    for sc in ["site_cluster", "spatial_block"]:
        assert (allf.groupby(f"group_{sc}")[f"fold_{sc}"].nunique() == 1).all(), sc
    for sc in ml.CV_SCHEMES:
        sizes = allf[allf.source.eq("dmr_site")][f"fold_{sc}"].value_counts()
        assert len(sizes) == ml.N_FOLDS and sizes.min() > 0.5 * sizes.mean(), (sc, sizes.to_dict())


def test_block_group_matches_folds(allf):
    assert (fl.block_group(allf.lon, allf.lat) == allf.group_spatial_block.values).all()


def test_specimen_table(spec, raw_sites):
    assert spec.fossil_id.is_unique
    linked = spec.site_OBJECTID.dropna().astype(int)
    assert set(linked) <= set(raw_sites.OBJECTID.astype(int))
    tied = spec.dropna(subset=["site_OBJECTID"])
    ok = [str(int(s)) in str(t).replace(".0", "").split(";") for s, t in zip(tied.site_OBJECTID, tied.tied_site_ids)]
    assert np.mean(ok) == 1.0, "site_OBJECTID must be one of tied_site_ids"
    assert set(spec.confidence.dropna()) <= {"high", "medium", "low", "none"}          # none = ไม่ได้เชื่อมแหล่ง
    assert spec.loc[spec.confidence.eq("none"), "site_OBJECTID"].isna().all()


def test_period_table(raw_sites):
    fin = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", dtype={"SITE_CODE": str}, encoding="utf-8-sig")
    assert fin.OBJECTID.is_unique and len(fin) == 518
    assert set(fin.final_old.dropna()) <= set(fl.PERIOD_ORDER)
    rec = fin[fin.final_source.eq("DMR record")]
    parsed = rec.PERIODFROM.map(lambda t: fl.parse_period_th(t)[0])
    assert (parsed == rec.final_old).all(), "DMR-recorded sites must keep their recorded (oldest) period"


def test_extra_features(allf):
    x = pd.read_parquet(ml.FEAT / "extra_features.parquet")
    assert set(x.point_id) == set(allf.point_id) and x.point_id.is_unique
    assert not [c for c in x.columns if c != "point_id" and x[c].isna().all()]
    assert (x.lc_tc_ppt > 0).all(), "TerraClimate rainfall must never be 0 (sea-cell artefact)"
    wc = x[[c for c in x.columns if c.startswith("lc_wc_")]].sum(axis=1)
    assert (wc > 50).mean() > 0.99, "WorldCover class shares should sum to ~100% on land"
    g = pd.read_parquet(ml.FEAT / "extra_grid.parquet")
    assert len(g) == len(pd.read_parquet(ml.FEAT / "grid_5km.parquet"))


def test_constant_features_not_used(allf, meta2):
    const = [c for c in meta2["features"]["num"] if allf[c].nunique(dropna=True) <= 1]
    # ค่าคงที่ไม่ทำให้ผิด แต่ไม่มีประโยชน์ — รายงานไว้ (ยอมให้ผ่านถ้าเป็น pg_marine ของช่วงที่ไทยเป็นบกทั้งหมด)
    assert all(c.startswith("pg_marine_") for c in const), const


def test_period_fills_consistent_with_fossils(raw_sites):
    # notebook 05 ส่วน 2b: ยุคที่ "เติม" ต้องไม่ขัดกับช่วงอายุของซากที่ชื่อ/คำบรรยายแหล่งระบุ (เช่น ไดโนเสาร์ไม่อยู่ในเพอร์เมียน)
    fin = pd.read_csv(ml.DATA / "processed" / "sites_period_final.csv", dtype={"SITE_CODE": str}, encoding="utf-8-sig")
    R = {p: i for i, p in enumerate(fl.PERIOD_ORDER)}
    AGE = {"ไดโนเสาร์": ("Triassic", "Cretaceous"), "สัตว์เลี้ยงลูกด้วยนม": ("Triassic", "Quaternary"),
           "ไทรโลไบต์": ("Cambrian", "Permian"), "แกรปโตไลต์": ("Cambrian", "Carboniferous")}
    text = {int(o): fl.classes_of(f"{n or ''} {d or ''}") for o, n, d in zip(raw_sites.OBJECTID, raw_sites.NAME_TH, raw_sites.FOS_DES_TH)}
    filled = fin[fin.final_source.ne("DMR record") & fin.final_old.notna()]
    bad = [(r.NAME_TH, r.PERIOD_FINAL, c) for r in filled.itertuples() for c in text.get(r.OBJECTID, set()) & set(AGE)
           if R[r.final_young] < R[AGE[c][0]] or R[r.final_old] > R[AGE[c][1]]]
    assert not bad, bad
    assert "rejected_by_fossils" in fin and fin.rejected_by_fossils.notna().any()
