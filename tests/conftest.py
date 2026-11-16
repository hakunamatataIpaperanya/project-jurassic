# Shared fixtures for the project test-suite: `.venv/bin/python -m pytest tests -q`
# (slow tests that call the network: add `-m slow`; everything else: `-m "not slow"`)
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import model_lib as ml  # noqa: E402  (import before torch users: OpenMP order)
import feature_lib as fl  # noqa: E402,F401  (tests import fl from here)

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

CAT = ["pg_sw18_plate", "pg_me21_plate", "pg_me21_terrane", "geo_rock_sym", "geo_era_old", "geo_era_young", "geo_rock_kind", "oc_nearest_rock"]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs network (Planetary Computer) or takes minutes")


@pytest.fixture(scope="session")
def allf():
    a = pd.read_parquet(ml.FEAT / "all_features.parquet").merge(pd.read_parquet(ml.FEAT / "folds.parquet"), on="point_id")
    for c in CAT:
        a[c] = a[c].astype(str)
    return a


@pytest.fixture(scope="session")
def raw_sites():
    return gpd.read_file(ml.DATA / "dmr_fossil_sites" / "fossil_sites.geojson")


@pytest.fixture(scope="session")
def spec():
    return pd.read_csv(ml.DATA / "processed" / "fossil_list_final.csv", dtype={"fossil_id": str, "site_OBJECTID": "Int64"},
                       encoding="utf-8-sig", low_memory=False)


@pytest.fixture(scope="session")
def meta2():
    return json.loads((ml.MODELS / "task2" / "final" / "final_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def meta1():
    return json.loads((ml.MODELS / "task1" / "final" / "final_meta.json").read_text(encoding="utf-8"))
