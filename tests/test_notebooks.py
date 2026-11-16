# Every notebook ran cleanly and is up to date with its builder script.
import re
import subprocess
import sys

import nbformat
import pytest

from conftest import ROOT

NOTEBOOKS = {
    "01_eda_fossils": "build_nb.py", "02_merge_fossils": "build_merge_nb.py", "03_fill_period_pbdb": "build_period_nb.py",
    "04_fill_period_taxon": "build_taxon_fill_nb.py", "05_final_period": "build_final_period_nb.py", "06_eda_final": "build_eda_final_nb.py",
    "07_impute_for_model": "build_impute_nb.py", "08_features_pangea": "build_pangea_nb.py", "09_features_satellite": "build_satellite_nb.py",
    "10_task2_models": "build_task2_nb.py", "10b_task2_map": "build_task2_map_nb.py", "11_task1_models": "build_task1_nb.py",
    "12_inference": "build_inference_nb.py", "13_features_extra": "build_extra_nb.py", "14_task2_fe": "build_task2_fe_nb.py",
    "15_task1_fe": "build_task1_fe_nb.py",
}
# คำเตือนที่มักบอก bug จริง (ไม่ใช่แค่ความรำคาญ)
RISKY = re.compile(r"SettingWithCopyWarning|invalid value encountered|divide by zero|Mean of empty slice|ConvergenceWarning|"
                   r"overflow encountered|DataConversionWarning|PerformanceWarning: DataFrame is highly fragmented|"
                   r"The behavior of .* is deprecated|Downcasting .* is deprecated|chained assignment")


def _nb(name):
    return nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", 4)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_executed_without_errors(name):
    nb = _nb(name)
    errors = [o.get("ename") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    unexecuted = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.strip() and c.get("execution_count") is None]
    assert not errors, f"error outputs: {errors}"
    assert not unexecuted, f"code cells never executed: {unexecuted}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_up_to_date_with_builder(name, tmp_path):
    # notebook ที่บันทึกต้องมีโค้ด/ข้อความตรงกับที่ builder สร้างตอนนี้ (ไม่มี notebook ค้างจากโค้ดเก่า)
    out = tmp_path / f"{name}.ipynb"
    subprocess.run([sys.executable, str(ROOT / "scripts" / NOTEBOOKS[name]), str(out)], check=True, cwd=ROOT, capture_output=True)
    fresh, saved = nbformat.read(out, 4), _nb(name)
    a = [(c.cell_type, c.source) for c in fresh.cells]
    b = [(c.cell_type, c.source) for c in saved.cells]
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and not diff, f"stale cells {diff[:5]} (builder {len(a)} cells, saved {len(b)}) — rerun scripts/run_*.sh"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_risky_warnings(name):
    nb = _nb(name)
    hits = []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []) if c.cell_type == "code" else []:
            if o.output_type == "stream" and o.name == "stderr":
                hits += [(i, m.group(0)) for m in RISKY.finditer(o.text)]
    assert not hits, f"risky warnings: {hits[:8]}"


@pytest.mark.parametrize("name", [n for n in NOTEBOOKS if n >= "09"])
def test_has_summary(name):
    nb = _nb(name)
    last = nb.cells[-1]
    assert last.cell_type == "markdown" and "สรุป" in last.source and "@@SUMMARY@@" not in last.source
