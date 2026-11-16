# โมเดลสำหรับเว็บแบบไม่มีเซิร์ฟเวอร์ (static site): แปลงโมเดลทายสกุลเป็น ONNX ให้เบราว์เซอร์รันด้วย onnxruntime-web
#   แปลงต้นไม้เองจากโครงสร้างของ XGBoost / LightGBM (ไม่ใช้ onnxmltools) เพื่อให้การเทียบค่าที่จุดแบ่งตรงกับ Python ทุกกรณี:
#     XGBoost  : x < เกณฑ์ (float32)                        -> BRANCH_LT  ค่าว่างไปทาง default_left
#     LightGBM : x <= เกณฑ์ (double)                        -> BRANCH_LEQ เกณฑ์ปัดลงเป็น float32 ตัวที่ใหญ่สุดที่ ≤ เกณฑ์
#                (ข้อมูลเข้าเป็น float32 จึงได้ผลเหมือนเทียบกับ double ทุกค่า); missing_type None = ค่าว่างนับเป็น 0
#   ผลลัพธ์ = softmax ของผลรวมใบไม้ต่อคลาส (+ ค่าเริ่มต้นต่อคลาสของ XGBoost) = predict_proba ของโมเดลเดิม
import json

import numpy as np
import onnx
from onnx import TensorProto, helper

IR_VERSION, OPSET, ML_OPSET = 8, 17, 3          # รุ่นที่ onnxruntime-web รุ่นปัจจุบันและรุ่นก่อน ๆ อ่านได้


def _float32_floor(t):
    # float32 ที่ใหญ่ที่สุดที่ ≤ t (t เป็น double)
    f = np.float32(t)
    return float(np.nextafter(f, np.float32(-np.inf)) if float(f) > t else f)


def _ensemble(nodes, leaves, n_features, n_classes, base, name):
    # nodes: [(tree, node, feature, threshold, mode, true_id, false_id, missing_true)], leaves: [(tree, node, class, value)]
    tr, nd, ft, th, md, ti, fi, ms = (list(x) for x in zip(*nodes))
    lt, ln, lc, lv = (list(x) for x in zip(*leaves))
    node = helper.make_node(
        "TreeEnsembleRegressor", ["X"], ["P"], domain="ai.onnx.ml", n_targets=n_classes,
        nodes_treeids=tr, nodes_nodeids=nd, nodes_featureids=ft, nodes_values=th, nodes_modes=md,
        nodes_truenodeids=ti, nodes_falsenodeids=fi, nodes_missing_value_tracks_true=ms,
        target_treeids=lt, target_nodeids=ln, target_ids=lc, target_weights=lv,
        base_values=[float(b) for b in base], aggregate_function="SUM", post_transform="SOFTMAX")
    graph = helper.make_graph([node], name, [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
                              [helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, n_classes])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid("ai.onnx.ml", ML_OPSET)],
                              producer_name="project-jurassic")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def xgb_to_onnx(clf):
    # XGBClassifier (multi:softprob, จุดแบ่งตัวเลขเท่านั้น) -> ONNX
    j = json.loads(clf.get_booster().save_raw("json"))
    lmp = j["learner"]["learner_model_param"]
    n_classes, n_features = int(lmp["num_class"]), int(lmp["num_feature"])
    base = json.loads(lmp["base_score"]) if lmp["base_score"].startswith("[") else [float(lmp["base_score"])] * n_classes
    gb = j["learner"]["gradient_booster"]["model"]
    nodes, leaves = [], []
    for t, (tree, cls) in enumerate(zip(gb["trees"], gb["tree_info"])):
        if any(tree["split_type"]):
            raise ValueError("ยังไม่รองรับจุดแบ่งแบบหมวดหมู่ของ XGBoost")
        L, R, F, C, D = tree["left_children"], tree["right_children"], tree["split_indices"], tree["split_conditions"], tree["default_left"]
        for i in range(len(L)):
            if L[i] == -1:
                nodes.append((t, i, 0, 0.0, "LEAF", 0, 0, 0))
                leaves.append((t, i, int(cls), float(np.float32(C[i]))))
            else:
                nodes.append((t, i, int(F[i]), float(np.float32(C[i])), "BRANCH_LT", int(L[i]), int(R[i]), int(D[i])))
    return _ensemble(nodes, leaves, n_features, n_classes, base, "genus_known_xgb")


def lgbm_to_onnx(clf):
    # LGBMClassifier (multiclass, จุดแบ่ง "<=" เท่านั้น) -> ONNX
    d = clf.booster_.dump_model()
    n_classes, n_features = int(d["num_class"]), len(d["feature_names"])
    nodes, leaves = [], []
    for t, info in enumerate(d["tree_info"]):
        cls, counter = t % d["num_tree_per_iteration"], [0]

        def add(n, t=t, cls=cls, counter=counter):
            i = counter[0]
            counter[0] += 1
            if "split_feature" not in n:
                nodes.append((t, i, 0, 0.0, "LEAF", 0, 0, 0))
                leaves.append((t, i, cls, float(n["leaf_value"])))
                return i
            if n["decision_type"] != "<=" or n["missing_type"] not in ("None", "NaN"):
                raise ValueError(f"ยังไม่รองรับจุดแบ่ง {n['decision_type']} / {n['missing_type']}")
            slot = len(nodes)
            nodes.append(None)
            thr = float(n["threshold"])
            # ค่าว่าง: missing_type NaN -> ตาม default_left; None -> LightGBM แทนด้วย 0 แล้วเทียบตามปกติ
            miss = bool(n["default_left"]) if n["missing_type"] == "NaN" else (thr >= 0.0)
            left, right = add(n["left_child"]), add(n["right_child"])
            nodes[slot] = (t, i, int(n["split_feature"]), _float32_floor(thr), "BRANCH_LEQ", left, right, int(miss))
            return i

        add(info["tree_structure"])
    return _ensemble(nodes, leaves, n_features, n_classes, [0.0] * n_classes, "genus_new_lgbm")


# ---------------------------------------------------------------- WangchanBERTa
PART_BYTES = 24 * 1024 * 1024                    # Cloudflare Pages รับไฟล์ละไม่เกิน 25 MiB -> แบ่งไฟล์ละ 24 MiB


def _fp16_weights(model):
    # น้ำหนักเก็บเป็น float16 (ไฟล์เล็กลงครึ่งหนึ่ง) + Cast กลับเป็น float32 ตอนโหลด -> คำนวณด้วย float32 เหมือนเดิม
    #   (int8 ทำให้ embedding ของ WangchanBERTa เพี้ยนจนผลทายสกุลต่างจาก Python ครึ่งหนึ่ง — สกุลอันดับ 1 ตรง: float32 100%, float16 97.3%, int8 50.3%)
    from onnx import numpy_helper
    g = model.graph
    keep, casts = [], []
    for t in g.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype == np.float32 and a.size >= 1024:
            keep.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_fp16"))
            casts.append(helper.make_node("Cast", [t.name + "_fp16"], [t.name], to=TensorProto.FLOAT, name=t.name + "_cast"))
        else:
            keep.append(t)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return model


def export_bert(out, name):
    # WangchanBERTa + mean pooling (แบบเดียวกับ feature_lib.encode_texts: เฉลี่ยเวกเตอร์ของโทเค็นจริง, ตัดที่ 64 โทเค็น)
    #   -> bert/model.part-NN (fp16) + bert/manifest.json + bert/tokenizer.json / tokenizer_config.json
    import hashlib

    import torch
    from transformers import AutoModel, AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(name)
    tok.save_pretrained(out)
    mdl = AutoModel.from_pretrained(name, attn_implementation="eager").eval()

    class Pooled(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            k = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * k).sum(1) / k.sum(1)

    b = tok(["ไดโนเสาร์ | ไดโนเสาร์ | ฟัน", "หอย | หอยสองฝา | nan"], padding=True, return_tensors="pt")
    fp32 = out / "_fp32.onnx"
    torch.onnx.export(Pooled(mdl), (b["input_ids"], b["attention_mask"]), str(fp32), input_names=["input_ids", "attention_mask"],
                      output_names=["pooled"], opset_version=OPSET, dynamo=False,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "pooled": {0: "batch"}})
    model = _fp16_weights(onnx.load(str(fp32)))
    model.ir_version = IR_VERSION
    fp32.unlink()
    blob = model.SerializeToString()
    for old in out.glob("model.part-*"):
        old.unlink()
    parts = []
    for i in range(0, len(blob), PART_BYTES):
        f = out / f"model.part-{i // PART_BYTES:02d}"
        f.write_bytes(blob[i:i + PART_BYTES])
        parts.append(f.name)
    (out / "manifest.json").write_text(json.dumps({"parts": parts, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                                                   "source": name, "weights": "float16", "pooling": "mean", "max_length": 64}, indent=1))
    return len(blob), parts


def main():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "notebooks"))
    import feature_lib as fl
    import model_lib as ml
    t1 = ml.MODELS / "task1" / "final"
    out = ml.MODELS / "web"
    out.mkdir(exist_ok=True)
    meta = json.loads((t1 / "final_meta.json").read_text(encoding="utf-8"))["genus"]
    for route, conv in [("known", xgb_to_onnx), ("new", lgbm_to_onnx)]:
        f = out / f"genus_{route}.onnx"
        onnx.save(conv(ml.joblib.load(t1 / meta["models"][route])["model"]), f)
        print(f"{f.relative_to(root)}  {f.stat().st_size / 1e6:.1f} MB  ({meta['models'][route]})")
    if "--no-bert" not in sys.argv:
        n, parts = export_bert(out / "bert", fl.EMB_MODELS["wangchanberta"])
        print(f"models/web/bert/model.part-*  {n / 1e6:.0f} MB ใน {len(parts)} ไฟล์ (น้ำหนัก float16)")


if __name__ == "__main__":
    main()
