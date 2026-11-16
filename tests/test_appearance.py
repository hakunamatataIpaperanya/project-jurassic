# ทายสกุลจาก "หน้าตา" (feature_lib.APPEARANCE + predict_lib.predict_specimen_appearance + notebook 15 หัวข้อ 7):
# คำตอบครอบคลุมทุกประเภทในทะเบียน, ชื่อประเภทสะกดตรงกับทะเบียน, น้ำหนักเป็นการแจกแจงที่ถูกต้อง, ผลวัดของ notebook 15 ตรงกับระบบที่ใช้จริง
import re

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, ml

fl = pytest.importorskip("feature_lib")


@pytest.fixture(scope="module")
def spec():
    return pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", usecols=["f_type", "f_part", "f_group"], dtype=str, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def leaf_types(spec):
    return fl.appearance_leaf_types(spec)


def test_types_spelled_like_registry_and_all_reachable(spec, leaf_types):
    # ทุกชื่อประเภทในคำถามต้องมีจริงในทะเบียน (สะกดผิด = ประเภทนั้นไม่มีวันถูกเลือก) และทุกประเภทในทะเบียนต้องตอบถึงได้โดยไม่ต้องเลือก "ไม่ตรงกับแบบไหน"
    registry = set(spec.f_type.dropna())
    for c in fl.APPEARANCE:
        for item in [c] + c.get("options", []):
            if isinstance(item.get("types"), list):
                assert set(item["types"]) <= registry, (c["id"], item.get("id"), set(item["types"]) - registry)
            if item.get("part"):
                re.compile(item["part"])
    reach = set().union(*(set(t) for lid, t in leaf_types.items() if not lid.startswith("unknown")))
    assert registry <= reach, registry - reach
    ids = [c["id"] for c in fl.APPEARANCE]
    assert len(ids) == len(set(ids)) and "unknown" in ids


def test_icons_exist_in_hologram_shapes():
    js = (ROOT / "webapp" / "static" / "holo.js").read_text(encoding="utf-8")
    shapes = set(re.findall(r"^    (\w+): \{ th:", js[js.index("const SHAPES = {"):].split("\n  };", 1)[0], re.M))
    icons = {c["icon"] for c in fl.APPEARANCE} | {o["icon"] for c in fl.APPEARANCE for o in c.get("options", []) if "icon" in o}
    assert icons <= shapes, icons - shapes


def test_follow_up_answer_reaches_the_model(spec, leaf_types):
    # ใบที่ได้ชุดประเภทเดียวกันแต่ความหมายต่างกัน ต้องส่งคำบรรยายส่วนของซากต่างกัน ไม่งั้นโมเดลไม่รู้คำตอบคำถามต่อ
    same_meaning = {frozenset({"coil/spire", "cone/spiral"})}           # หอยฝาเดียวทั้งคู่
    leaves = {lid: lv for lid, lv in fl.appearance_leaves().items() if lv["option"] not in (None, "unsure")}
    for a in leaves:
        for b in leaves:
            if a < b and set(leaf_types[a]) == set(leaf_types[b]) and frozenset({a, b}) not in same_meaning:
                assert fl.appearance_part(leaf_types[a][0], a, spec) != fl.appearance_part(leaf_types[b][0], b, spec), (a, b)
    assert fl.appearance_part("เซฟาโลพอด", "coil/flat", spec) == "เปลือกหอยแอมโมนอยด์"
    assert fl.appearance_part("เซฟาโลพอด", "cone/chambers", spec) == "โครงร่างแข็งของนอติลอยด์"


def test_type_weights_are_distributions(spec, leaf_types):
    for lid, types in leaf_types.items():
        w = fl.appearance_type_weights(lid, spec, leaf_types=leaf_types)
        assert 1 <= len(w) <= 5 and abs(w.sum() - 1) < 1e-9 and (w > 0).all(), lid
        assert set(w.index) <= set(types) and w.is_monotonic_decreasing, lid
    # คำตอบที่เจาะจงต้องชี้ถูกกลุ่ม (ค่าจากสถิติทะเบียน)
    assert fl.appearance_type_weights("tooth/molar", spec, leaf_types=leaf_types).index[0] == "สัตว์กีบคู่"
    assert fl.appearance_type_weights("plate", spec, leaf_types=leaf_types).index[0] == "เต่า"
    assert list(fl.appearance_type_weights("twoshell/sym", spec, leaf_types=leaf_types).index) == ["แบรคิโอพอด"]


def test_truth_mapping_keeps_true_type(spec, leaf_types):
    # ผู้ใช้ที่ "มองซากถูก" ต้องได้ชุดประเภทที่มีประเภทจริงเกือบทุกครั้ง (ตัดหางแล้ว)
    kept = [t in fl.appearance_type_weights(fl.appearance_truth(t, p, spec, leaf_types)[0], spec, leaf_types=leaf_types).index
            for t, p in zip(spec.f_type, spec.f_part)]
    assert np.mean(kept) > 0.95
    assert fl.appearance_truth("ไดโนเสาร์", "ฟัน", spec, leaf_types)[0].startswith("tooth/")
    assert fl.appearance_truth("ไดโนเสาร์", "เศษกระดูก", spec, leaf_types) == ("bone/frag", "bone/unsure")
    assert fl.appearance_truth("ไดโนเสาร์", None, spec, leaf_types) == ("bone/unsure", "bone/unsure")     # ทะเบียนไม่บอกส่วน = ตอบ "ไม่แน่ใจ"
    assert fl.appearance_truth("ปลากระดูกแข็ง", None, spec, leaf_types)[0] == "fish/unsure"
    assert fl.appearance_truth("ปลากระดูกแข็ง", "ตัวปลา", spec, leaf_types)[0] == "fish/body"
    assert fl.appearance_truth("เซฟาโลพอด", "โครงร่างแข็งของนอติลอยด์", spec, leaf_types)[0] == "cone/chambers"


def test_mixture_prediction_matches_direct_when_one_type():
    # ใบที่มีประเภทเดียว = ทายแบบรู้ประเภท (ส่วนของซากใช้คำที่ทะเบียนเขียนบ่อยสุด)
    import predict_lib as pl
    p = pl.Predictor()
    a = p.predict_specimen_appearance({"province": "ขอนแก่น"}, "trilobite")
    b = p.predict_specimen({"province": "ขอนแก่น", "f_type": "ไทรโลไบต์", "f_group": p.type_group["ไทรโลไบต์"],
                            "f_part": fl.appearance_part("ไทรโลไบต์", "trilobite", p.spec)})
    assert a["top"] == b["top"] and a["types"] == [("ไทรโลไบต์", 1.0)]
    m = p.predict_specimen_appearance({"province": "ขอนแก่น"}, "tooth", "blade")
    assert abs(sum(w for _, w in m["types"]) - 1) < 2e-3 and m["leaf"] == "tooth/blade" and len(m["top"]) == 3


def test_notebook15_appearance_eval():
    f = ml.MODELS / "task1" / "FE" / "appearance_eval.parquet"
    if not f.exists():
        pytest.skip("ยังไม่ได้รัน notebook 15 หัวข้อ 7")
    t = pd.read_parquet(f)
    # วิธีเติมส่วนของซากที่ใช้จริงต้องเป็นวิธีที่ CV ดีสุด และแถวที่เว็บใช้คือของวิธีนั้น
    assert (t.best_policy == fl.APPEARANCE_PART_POLICY).all()
    t = t[t.deployed].set_index(["cv", "scenario"])
    assert not t.index.duplicated().any()
    sysG = pd.read_parquet(ml.MODELS / "task1" / "FE" / "genus_systems.parquet").set_index(["cv", "ระบบ"])
    for cv in ["random", "spatial_block"]:
        ref = sysG.loc[(cv, "ชุดรวมทั้งสองเส้นทาง")]
        assert abs(t.loc[(cv, "base"), "top3"] - ref.top3) < 1e-9 and abs(t.loc[(cv, "base"), "acc"] - ref.acc) < 1e-9
        for s in ["type_nocom", "look_full", "look_card", "look_unknown"]:
            for sfx in ["", "_noloc"]:
                assert 0 < t.loc[(cv, s + sfx), "top3"] <= 1 and t.loc[(cv, s + sfx), "acc"] <= t.loc[(cv, s + sfx), "top3"]
        assert (cv, "base_noloc") in t.index
        assert t.loc[(cv, "look_full"), "type_kept"] > 0.95
