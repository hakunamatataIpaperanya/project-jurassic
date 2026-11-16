# Concatenate scripts/lib_sections/*.py -> notebooks/feature_lib.py and scripts/model_sections/*.py -> notebooks/model_lib.py
# (single source for notebooks 08-12).
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = sorted((ROOT / "scripts" / "lib_sections").glob("*.py"))
MODEL_SECTIONS = sorted((ROOT / "scripts" / "model_sections").glob("*.py"))


def _write(sections, out_name, src):
    header = (f'"""Generated from scripts/{src}/*.py by scripts/make_feature_lib.py.\n'
              "Do not edit by hand: edit the section files and rebuild.\n\"\"\"\n")
    body = "\n\n".join(f"# {'=' * 70}\n# section: {p.name}\n# {'=' * 70}\n" + p.read_text(encoding="utf-8") for p in sections)
    out = ROOT / "notebooks" / out_name
    out.write_text(header + body, encoding="utf-8")
    return out


def write_lib():
    return _write(sorted((ROOT / "scripts" / "lib_sections").glob("*.py")), "feature_lib.py", "lib_sections")


def write_model_lib():
    return _write(sorted((ROOT / "scripts" / "model_sections").glob("*.py")), "model_lib.py", "model_sections")


def write_predict_lib():
    # inference ด้วยโมเดลตัวจริง (notebook 12, webapp/, tests) — import model_lib / feature_lib
    return _write(sorted((ROOT / "scripts" / "predict_sections").glob("*.py")), "predict_lib.py", "predict_sections")


if __name__ == "__main__":
    print("wrote", write_lib())
    if MODEL_SECTIONS:
        print("wrote", write_model_lib())
    print("wrote", write_predict_lib())
