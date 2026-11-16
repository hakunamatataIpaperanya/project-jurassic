# Core imports, device, seeds, output folders for model experiments.
# OpenMP: LightGBM / XGBoost and PyTorch must share ONE libomp in a process (two copies -> segfault/deadlock on macOS).
# scripts/fix_openmp.sh repoints LightGBM/XGBoost to torch's bundled libomp; re-run it after reinstalling those packages.
import lightgbm  # noqa: F401,E402
import xgboost  # noqa: F401,E402
import logging
import os
import random
import time
import warnings
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
DATA = ROOT / "data"
FEAT = DATA / "features"
MODELS = ROOT / "models"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def torch_device(prefer_gpu=True):
    import torch
    if not prefer_gpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tqdm_auto():
    from tqdm.auto import tqdm
    return tqdm
