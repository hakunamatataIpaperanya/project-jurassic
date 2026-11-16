"""Generated from scripts/model_sections/*.py by scripts/make_feature_lib.py.
Do not edit by hand: edit the section files and rebuild.
"""
# ======================================================================
# section: 00_core.py
# ======================================================================
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


# ======================================================================
# section: 10_cv.py
# ======================================================================
# Cross-validation schemes. Fold ids are assigned per point_id once, so every stage / task uses the same split
#   random        : plain random K-fold
#   site_cluster  : points within CLUSTER_KM of each other (e.g. dig pits of one quarry) always share a fold
#   spatial_block : 50 km x 50 km blocks (EPSG:32647) share a fold -> no near neighbour on the other side
CV_SCHEMES = ["random", "site_cluster", "spatial_block"]
CLUSTER_KM = 2.0
BLOCK_KM = 50.0
N_FOLDS = 5


def _clusters(lon, lat, km):
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import BallTree
    X = np.radians(np.c_[lat, lon])
    tree = BallTree(X, metric="haversine")
    nb = tree.query_radius(X, r=km / 6371.0)
    rows = np.concatenate([np.full(len(n), i) for i, n in enumerate(nb)])
    cols = np.concatenate(nb)
    from scipy.sparse import coo_matrix
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(X), len(X)))
    return connected_components(g, directed=False)[1]


def assign_folds(points, seed=42):
    # DataFrame point_id + fold_<scheme> for all schemes (groups shuffled, then balanced greedily by size)
    from pyproj import Transformer
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"point_id": points.point_id.values})
    groups = {
        "random": np.arange(len(points)),
        "site_cluster": _clusters(points.lon.values, points.lat.values, CLUSTER_KM),
    }
    x, y = Transformer.from_crs(4326, 32647, always_xy=True).transform(points.lon.values, points.lat.values)
    groups["spatial_block"] = (np.floor(x / (BLOCK_KM * 1000)).astype(int) * 100000 + np.floor(y / (BLOCK_KM * 1000)).astype(int))
    for scheme, g in groups.items():
        uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
        order = rng.permutation(len(uniq))
        order = order[np.argsort(-counts[order], kind="stable")]      # ใหญ่ก่อน แล้วใส่ fold ที่เล็กสุด
        load, gfold = np.zeros(N_FOLDS), np.empty(len(uniq), int)
        for gi in order:
            k = int(np.argmin(load))
            gfold[gi] = k
            load[k] += counts[gi]
        out[f"fold_{scheme}"] = gfold[inv]
        out[f"group_{scheme}"] = g
    return out


# ======================================================================
# section: 20_preprocess.py
# ======================================================================
# Tabular preprocessing fitted on the training fold only (no leakage), saved together with each model.
#   mode="tree": numeric as-is (NaN kept for LightGBM/XGBoost, median-filled for RF/kNN), categoricals -> ordinal codes
#   mode="dense": numeric median-fill + standardise, categoricals -> one-hot (rare levels -> "__other__")
class TabularPreprocessor:
    def __init__(self, num_cols, cat_cols, mode="tree", fill_nan=False, min_count=5):
        self.num_cols, self.cat_cols, self.mode, self.fill_nan, self.min_count = list(num_cols), list(cat_cols), mode, fill_nan, min_count

    def fit(self, df):
        X = df[self.num_cols].astype(float)
        self.median_ = X.median()
        self.mean_ = X.fillna(self.median_).mean()
        self.std_ = X.fillna(self.median_).std().replace(0, 1).fillna(1)
        self.levels_ = {}
        for c in self.cat_cols:
            vc = df[c].astype(str).value_counts()
            self.levels_[c] = sorted(vc[vc >= self.min_count].index.tolist())
        return self

    def transform(self, df):
        X = df[self.num_cols].astype(float).copy()
        if self.mode == "dense":
            X = (X.fillna(self.median_) - self.mean_) / self.std_
            parts = [X.reset_index(drop=True)]
            for c in self.cat_cols:
                v = df[c].astype(str).where(df[c].astype(str).isin(self.levels_[c]), "__other__").reset_index(drop=True)
                oh = pd.get_dummies(pd.Categorical(v, categories=self.levels_[c] + ["__other__"]), prefix=c, dtype=float)
                parts.append(oh)
            return pd.concat(parts, axis=1).astype("float32")
        if self.fill_nan:
            X = X.fillna(self.median_)
        for c in self.cat_cols:
            lv = {v: i for i, v in enumerate(self.levels_[c])}
            X[c] = df[c].astype(str).map(lv).fillna(-1).astype(float).values
        return X.reset_index(drop=True).astype("float32")

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# ======================================================================
# section: 30_registry.py
# ======================================================================
# Experiment registry: one row per (experiment, fold) in <dir>/results.parquet + out-of-fold predictions per experiment.
# Re-running an experiment skips folds that are already recorded (resume after a crash) — but only when the stored
# fingerprint (rows, folds, labels, feature values, model settings and fitting code) equals the current one; otherwise the
# fold is recomputed. Rows written before fingerprints existed ("legacy") are trusted, or with ML_VERIFY_CACHE=1 checked by
# recomputing (tabular) / comparing rows + labels (torch); every check is logged in <dir>/cache_audit.parquet.
import hashlib
import inspect
import io
import tokenize

VERIFY_CACHE = os.environ.get("ML_VERIFY_CACHE") == "1"


def fingerprint(*parts):
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, pd.DataFrame):
            h.update(f"df:{len(p)}:".encode() + "|".join(map(str, p.columns)).encode())
            if p.shape[1]:
                h.update(pd.util.hash_pandas_object(p, index=False).values.tobytes())
        elif isinstance(p, pd.Series):
            h.update(pd.util.hash_pandas_object(p, index=False).values.tobytes())
        elif isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(f"nd:{a.dtype}:{a.shape}".encode())
            h.update(repr(a.tolist()).encode() if a.dtype == object else a.tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:20]


def code_hash(*objs):
    # source of the fitting code without comments / blank lines (editing a comment does not invalidate caches)
    toks = []
    for o in objs:
        for t in tokenize.generate_tokens(io.StringIO(inspect.getsource(o)).readline):
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                toks.append(t.string)
    return fingerprint(" ".join(toks))


class Registry:
    def __init__(self, folder):
        self.dir = Path(folder)
        (self.dir / "oof").mkdir(parents=True, exist_ok=True)
        (self.dir / "fold_models").mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "results.parquet"

    def results(self):
        return pd.read_parquet(self.file) if self.file.exists() else pd.DataFrame(columns=["exp", "fold"])

    def done(self, exp, fold):
        r = self.results()
        return bool(((r.exp == exp) & (r.fold == fold)).any())

    def status(self, exp, fold, fp):
        # "missing" | "fresh" (fingerprint matches) | "stale" (data / features / settings changed) | "legacy" (no fingerprint)
        r = self.results()
        row = r[(r.exp == exp) & (r.fold == fold)]
        if row.empty:
            return "missing"
        old = row["fp"].iat[0] if "fp" in row else None
        if old is None or pd.isna(old):
            return "legacy"
        return "fresh" if old == fp else "stale"

    def set_fp(self, exp, fold, fp):
        r = self.results()
        if "fp" not in r:
            r["fp"] = pd.Series([None] * len(r), dtype=object)
        r["fp"] = r["fp"].astype(object)
        r.loc[(r.exp == exp) & (r.fold == fold), "fp"] = fp
        r.to_parquet(self.file, index=False)

    def audit(self, **row):
        f = self.dir / "cache_audit.parquet"
        a = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        row["time"] = pd.Timestamp.now().isoformat(timespec="seconds")
        pd.concat([a, pd.DataFrame([row])], ignore_index=True).to_parquet(f, index=False)

    def oof_diff(self, exp, fold, ids, y, p):
        # largest |p_stored - p_new| on the same rows; inf if rows / labels differ
        f = self.oof_path(exp, fold)
        if not f.exists():
            return float("inf")
        o = pd.read_parquet(f)
        if len(o) != len(ids) or (o.point_id.astype(str).values != np.asarray(ids).astype(str)).any():
            return float("inf")
        y = np.asarray(y)
        ycols = ["y"] if "y" in o else [c for c in o.columns if c.startswith("y_")]
        if not np.allclose(o[ycols].values.astype(float).reshape(y.shape), y.astype(float)):
            return float("inf")
        if p is None:
            return 0.0
        pcols = ["p"] if "p" in o else [c for c in o.columns if c.startswith("p_")]
        return float(np.abs(o[pcols].values.reshape(np.shape(p)) - p).max()) if len(o) else 0.0

    def add(self, row):
        r = self.results()
        r = pd.concat([r[~((r.exp == row["exp"]) & (r.fold == row["fold"]))], pd.DataFrame([row])], ignore_index=True)
        r.to_parquet(self.file, index=False)

    def oof_path(self, exp, fold):
        return self.dir / "oof" / f"{exp}__f{fold}.parquet"

    def model_path(self, exp, fold, ext="joblib"):
        return self.dir / "fold_models" / f"{exp}__f{fold}.{ext}"

    def oof(self, exp):
        fs = sorted((self.dir / "oof").glob(f"{exp}__f*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:20]


def cached_frame(path, fp, compute, index=False, atol=0.0):
    # parquet cache of a derived feature table, guarded by a fingerprint of its inputs in "<file>.fp".
    # Missing / different fingerprint -> recompute (legacy cache without .fp: trusted, or with ML_VERIFY_CACHE=1 recomputed
    # and compared; numeric columns may differ by `atol`, e.g. CNN embeddings on MPS). Audit -> models/file_cache_audit.parquet
    path = Path(path)
    side = path.with_name(path.name + ".fp")
    old_fp = side.read_text().strip() if side.exists() else None
    if path.exists() and (old_fp == fp or (old_fp is None and not VERIFY_CACHE)):
        return pd.read_parquet(path)
    new = compute()
    if path.exists():
        cur = pd.read_parquet(path)
        same = cur.shape == new.shape and list(map(str, cur.columns)) == list(map(str, new.columns))
        diff = float("inf")
        if same:
            num = [c for c in new.columns if pd.api.types.is_numeric_dtype(new[c])]
            oth = [c for c in new.columns if c not in num]
            diff = float(np.nanmax(np.abs(cur[num].values.astype(float) - new[num].values.astype(float)))) if num and len(new) else 0.0
            same = (diff <= atol and np.array_equal(np.isnan(cur[num].values.astype(float)), np.isnan(new[num].values.astype(float)))
                    and all((cur[c].astype(str).values == new[c].astype(str).values).all() for c in oth)
                    and (not index or (cur.index.astype(str) == new.index.astype(str)).all()))
        f = MODELS / "file_cache_audit.parquet"
        a = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        row = {"path": str(path.relative_to(ROOT)), "kind": "legacy" if old_fp is None else "stale",
               "result": "recomputed-match" if same else "recomputed-DIFFERENT", "max_diff": diff,
               "time": pd.Timestamp.now().isoformat(timespec="seconds")}
        pd.concat([a, pd.DataFrame([row])], ignore_index=True).to_parquet(f, index=False)
        if same:
            side.write_text(fp)
            return cur
        print(f"[cache] {path.name}: ค่าเดิมไม่ตรงกับที่คำนวณใหม่ (ต่างสูงสุด {diff:.3g}) -> แทนที่")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=index)
    side.write_text(fp)
    return new


# ======================================================================
# section: 40_torch.py
# ======================================================================
# PyTorch models (MLP for tables, small CNN for 6-band 32x32 Sentinel-2 patches) and a training loop
# that saves a checkpoint every epoch and resumes from it (model + optimizer + epoch + early-stopping state).
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        return self.head(self.body(x))


class PatchCNN(nn.Module):
    # 6x32x32 -> 128-d embedding -> head
    def __init__(self, n_out, n_bands=6, width=32, dropout=0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.features = nn.Sequential(block(n_bands, width), nn.MaxPool2d(2), block(width, width * 2), nn.MaxPool2d(2),
                                      block(width * 2, width * 4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * 4, n_out)

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.head(self.drop(self.features(x)))


def load_patches(point_ids, patch_dir):
    # uint16 DN -> float32 reflectance [N, 6, 32, 32]; missing patch -> zeros + flag
    arr = np.zeros((len(point_ids), 6, 32, 32), dtype="float32")
    ok = np.zeros(len(point_ids), dtype=bool)
    for i, pid in enumerate(point_ids):
        p = Path(patch_dir) / f"{pid}.npz"
        if p.exists():
            arr[i] = np.load(p)["patch"].astype("float32") * 0.0001 - 0.1
            ok[i] = True
    return arr, ok


class PatchNorm:
    def fit(self, x):
        self.mean_ = x.mean(axis=(0, 2, 3), keepdims=True)
        self.std_ = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        return self

    def __call__(self, x):
        return (x - self.mean_) / self.std_


def _augment(xb):
    # random flips / 90-degree rotations (valid for nadir imagery)
    k = int(torch.randint(0, 4, (1,)))
    xb = torch.rot90(xb, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        xb = torch.flip(xb, dims=(3,))
    return xb


def train_torch(model, Xtr, ytr, Xva, yva, ckpt, loss_fn, score_fn, epochs=40, lr=1e-3, batch=128,
                patience=8, augment=False, weight_decay=1e-4, desc="train", seed=42, gpu=True):
    # Trains with early stopping on score_fn(yva, pred) (higher = better). Checkpoint after every epoch at `ckpt`;
    # if it exists, training resumes from the saved epoch. Returns (best_state_dict, best_score, history).
    tqdm = tqdm_auto()
    dev = torch_device(gpu)
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    start, best, best_state, bad, hist = 0, -np.inf, None, 0, []
    ckpt = Path(ckpt)
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        start, best, best_state, bad, hist = st["epoch"] + 1, st["best"], st["best_state"], st["bad"], st["hist"]
        if st.get("finished"):
            return best_state, best, hist
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    Xva_t = torch.as_tensor(Xva)
    bar = tqdm(range(start, epochs), desc=desc, leave=False, initial=start, total=epochs)
    for ep in bar:
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            if augment:
                xb = _augment(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        pred = predict_torch(model, Xva_t, gpu=gpu)
        sc = float(score_fn(yva, pred))
        hist.append({"epoch": ep, "loss": tot / len(perm), "val": sc})
        if sc > best:
            best, bad = sc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        finished = bad >= patience or ep == epochs - 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "best_state": best_state, "bad": bad, "hist": hist, "finished": finished}, ckpt)
        bar.set_postfix(loss=round(tot / len(perm), 4), val=round(sc, 4), best=round(best, 4))
        if finished:
            break
    return best_state, best, hist


@torch.no_grad()
def predict_torch(model, X, batch=512, embed=False, gpu=True):
    dev = torch_device(gpu)
    model = model.to(dev).eval()
    X = torch.as_tensor(X)
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(dev)
        out.append((model.embed(xb) if embed else model(xb)).float().cpu())
    return torch.cat(out).numpy()


# ======================================================================
# section: 50_metrics.py
# ======================================================================
# Metrics.
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def bin_metrics(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "pr_auc": np.nan}
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}


def ks_vs_random(p_presence, p_random):
    # Block et al. 2016: how much higher predicted suitability is at independent fossil sites than at random places (0..1)
    return float(ks_2samp(p_presence, p_random).statistic)


def multilabel_metrics(Y, P, min_pos=5):
    # Y, P: [n, k]. macro AUC over classes with >= min_pos positives; top-1/top-3 hit (any true label in the top-k)
    Y, P = np.asarray(Y), np.asarray(P)
    aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if min_pos <= Y[:, j].sum() < len(Y)]
    has = Y.sum(axis=1) > 0
    order = np.argsort(-P[has], axis=1)
    hit = lambda k: float(np.mean([Y[has][i, order[i, :k]].any() for i in range(has.sum())]))
    return {"macro_auc": float(np.mean(aucs)) if aucs else np.nan, "top1": hit(1), "top3": hit(3), "n_classes_eval": len(aucs)}


def multiclass_metrics(y, P, classes=None, ordinal=None):
    # y: int class index [n]; P: probabilities [n, K]. ordinal: list of class names in temporal order -> "within 1" accuracy
    from sklearn.metrics import f1_score
    y = np.asarray(y)
    pred = P.argmax(axis=1)
    top3 = np.argsort(-P, axis=1)[:, :3]
    out = {"acc": float((pred == y).mean()), "top3": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y)))}
    if ordinal is not None and classes is not None:
        rank = np.array([ordinal.index(c) if c in ordinal else -99 for c in classes])
        out["within1"] = float(np.mean(np.abs(rank[pred] - rank[y]) <= 1))
    return out


# ======================================================================
# section: 60_runners.py
# ======================================================================
# CV runners. Binary (fossil potential) and multi-label (fossil types) share the same code path.
# Every fold's fitted preprocessor + model is saved; OOF predictions are saved; finished folds are skipped on re-run.
MULTICLASS_K = 0          # set by set_multiclass(K) before running a multiclass experiment


def set_multiclass(k):
    global MULTICLASS_K
    MULTICLASS_K = int(k)


def make_model(name, task, pos_weight=1.0, seed=42):
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed,
                                      class_weight="balanced" if task == "binary" else None)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                  scale_pos_weight=pos_weight, random_state=seed, verbose=-1, n_jobs=-1)
    if name == "lgbm_fast":                       # คัดกรอง feature หลายชุดเร็ว ๆ (notebook 15): ต้นไม้น้อยกว่า / ใบน้อยกว่า
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.06, num_leaves=15, min_child_samples=10, max_bin=63,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                  scale_pos_weight=pos_weight, random_state=seed, verbose=-1, n_jobs=-1)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.7,
                                 min_child_weight=2, reg_lambda=1.0, scale_pos_weight=pos_weight, tree_method="hist",
                                 random_state=seed, n_jobs=-1, eval_metric="logloss")
    raise ValueError(name)


class PriorModel:
    # baseline: predict the training-set frequency of each class for every point
    def fit(self, X, y):
        self.p_ = np.asarray(y, float).mean(axis=0)
        return self

    def predict_proba_matrix(self, X):
        return np.tile(self.p_, (len(X), 1)) if np.ndim(self.p_) else np.full(len(X), float(self.p_))


def _fit_predict(name, task, Xtr, ytr, Xte, seed=42):
    # returns probabilities [n] (binary) or [n, k] (multilabel) and the fitted model(s)
    if name == "prior":
        m = PriorModel().fit(Xtr, ytr)
        return m.predict_proba_matrix(Xte), m
    if task == "binary":
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        m = make_model(name, task, pw, seed).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1], m
    if task == "multiclass":
        # ytr: int index into the global class list; returns [n, K_global] (classes absent from train get 0)
        K = int(MULTICLASS_K)
        present = np.unique(ytr)
        remap = {c: i for i, c in enumerate(present)}
        m = make_model(name, "multiclass", 1.0, seed)
        if name == "lgbm":
            m.set_params(objective="multiclass", scale_pos_weight=None, class_weight="balanced", n_estimators=300)
        if name == "lgbm_fast":
            m.set_params(objective="multiclass", scale_pos_weight=None, class_weight="balanced")
        if name == "xgb":
            m.set_params(objective="multi:softprob", scale_pos_weight=None, eval_metric="mlogloss", n_estimators=300)
        if name == "rf":
            m.set_params(class_weight="balanced")
        m.fit(Xtr, np.array([remap[c] for c in ytr]))
        P = np.zeros((len(Xte), K))
        P[:, present] = m.predict_proba(Xte)
        m.present_ = present
        return P, m
    if name in ("knn", "rf"):
        m = make_model(name, task, 1.0, seed).fit(Xtr, ytr)
        pr = m.predict_proba(Xte)
        return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(Xte)) for p in pr]), m
    ms, cols = [], []
    for j in range(ytr.shape[1]):
        yj = ytr[:, j]
        if yj.sum() == 0 or yj.sum() == len(yj):
            ms.append(None), cols.append(np.full(len(Xte), yj.mean()))
            continue
        pw = float((yj == 0).sum() / max(yj.sum(), 1))
        m = make_model(name, "binary", pw, seed).fit(Xtr, yj)
        ms.append(m), cols.append(m.predict_proba(Xte)[:, 1])
    return np.column_stack(cols), ms


MC_CONTEXT = {}          # optional {"classes": [...], "ordinal": [...]} for multiclass metrics


def _metrics(task, y, p):
    if task == "binary":
        return bin_metrics(y, p)
    if task == "multiclass":
        return multiclass_metrics(y, p, MC_CONTEXT.get("classes"), MC_CONTEXT.get("ordinal"))
    return multilabel_metrics(y, p)


def _oof_frame(ids, fold, y, p, task, classes):
    df = pd.DataFrame({"point_id": ids, "fold": fold})
    if task == "binary":
        df["y"], df["p"] = y, p
    elif task == "multiclass":
        df["y"] = y
        for j in range(p.shape[1]):
            df[f"p_{j}"] = p[:, j]
    else:
        for j in range(len(classes)):
            df[f"y_{j}"], df[f"p_{j}"] = y[:, j], p[:, j]
    return df


_VERIFIED = {}           # (registry folder, exp) -> True / False: result of recomputing the first legacy fold in this session


def _tab_cfg(model, task, num_cols, cat_cols, classes, metrics=True):
    # everything besides the data that decides a tabular fold's predictions (and, for CV, the stored metrics)
    mc = (int(MULTICLASS_K),) + ((MC_CONTEXT.get("classes"), MC_CONTEXT.get("ordinal")) if metrics else ()) if task == "multiclass" else None
    return (model, task, list(num_cols), list(cat_cols), None if classes is None else list(classes), mc,
            code_hash(make_model, _fit_predict, PriorModel, TabularPreprocessor))


def run_tabular_cv(reg, exp, df, Y, fold_col, model, num_cols, cat_cols, task="binary", classes=None, desc=None, only_folds=None):
    # only_folds: run just these folds (used when a feature depends on the fold, e.g. fold-specific CNN embeddings)
    tqdm = tqdm_auto()
    folds = df[fold_col].values
    mode = "dense" if model in ("knn", "prior") else "tree"
    cfg = _tab_cfg(model, task, num_cols, cat_cols, classes)
    data_fp = fingerprint(df.point_id, df[fold_col], df[list(num_cols) + list(cat_cols)], np.asarray(Y))
    key = (str(reg.dir), exp)
    for k in tqdm([k for k in sorted(np.unique(folds)) if only_folds is None or k in only_folds], desc=desc or exp, leave=False):
        fp = fingerprint(cfg, data_fp, int(k))
        st = reg.status(exp, k, fp)
        if st == "fresh" or (st == "legacy" and not VERIFY_CACHE):
            continue
        tr, te = folds != k, folds == k
        if st == "legacy" and _VERIFIED.get(key) is True and reg.oof_diff(exp, k, df.point_id.values[te], Y[te], None) == 0:
            reg.set_fp(exp, k, fp)                      # same experiment already reproduced exactly; rows + labels match
            reg.audit(exp=exp, fold=int(k), result="rows-match", max_diff=0.0)
            continue
        if st == "stale":
            print(f"[cache] {reg.dir.name}/{exp} fold {k}: ข้อมูล/feature/ค่าตั้งเปลี่ยน -> คำนวณใหม่")
        pre = TabularPreprocessor(num_cols, cat_cols, mode=mode, fill_nan=(model in ("rf", "knn")))
        Xtr, Xte = pre.fit_transform(df[tr]), pre.transform(df[te])
        t0 = time.time()
        p, m = _fit_predict(model, task, Xtr.values, Y[tr], Xte.values)
        if st == "legacy":
            diff = reg.oof_diff(exp, k, df.point_id.values[te], Y[te], p)
            _VERIFIED[key] = bool(diff <= 1e-6)
            reg.audit(exp=exp, fold=int(k), result="recomputed-match" if diff <= 1e-6 else "recomputed-DIFFERENT", max_diff=diff)
            if diff <= 1e-6:
                reg.set_fp(exp, k, fp)
                continue
            print(f"[cache] {reg.dir.name}/{exp} fold {k}: ผลเดิมไม่ตรงกับที่คำนวณใหม่ (ต่างสูงสุด {diff:.3g}) -> แทนที่")
        joblib.dump({"pre": pre, "model": m, "classes": classes, "task": task}, reg.model_path(exp, k), compress=3)
        _oof_frame(df.point_id.values[te], k, Y[te], p, task, classes).to_parquet(reg.oof_path(exp, k), index=False)
        reg.add({"exp": exp, "fold": k, "n_test": int(te.sum()), "seconds": round(time.time() - t0, 1), **_metrics(task, Y[te], p),
                 "fp": fp})
    return summarize(reg, exp, task)


def _torch_fit(kind, task, n_out, df_tr, Ytr, patches_tr, num_cols, cat_cols, ckpt, epochs, lr, init_state=None, seed=42):
    # fits one MLP / CNN on a training set (15 % held out for early stopping), checkpointing every epoch at `ckpt`
    gpu = kind == "cnn"                       # MLP บนตารางเล็ก: CPU เร็วกว่า MPS
    if task == "multiclass":
        ytr = np.asarray(Ytr).astype("int64")
        cnt = np.bincount(ytr, minlength=n_out).astype("float32")
        w = torch.as_tensor(np.where(cnt > 0, len(ytr) / (n_out * np.maximum(cnt, 1)), 0), dtype=torch.float32).to(torch_device(gpu))
        lossf = nn.CrossEntropyLoss(weight=w)
        loss_fn = lambda out, yb: lossf(out, yb)
        score = lambda yv, pr: float((pr.argmax(axis=1) == yv).mean())
    else:
        ytr = np.asarray(Ytr).reshape(-1, n_out).astype("float32")
        pos = ytr.sum(axis=0)
        pw = torch.as_tensor(np.clip((len(ytr) - pos) / np.maximum(pos, 1), 1, 50), dtype=torch.float32).to(torch_device(gpu))
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        loss_fn = lambda out, yb: lossf(out, yb)
        score = (lambda yv, pr: bin_metrics(yv, pr[:, 0])["auc"]) if task == "binary" else \
                (lambda yv, pr: multilabel_metrics(yv, pr)["macro_auc"])
    if kind == "mlp":
        pre = TabularPreprocessor(num_cols, cat_cols, mode="dense")
        Xtr = pre.fit_transform(df_tr).values
        net, aug = MLP(Xtr.shape[1], n_out), False
    else:
        pre = PatchNorm().fit(patches_tr)
        Xtr = pre(patches_tr)
        net, aug = PatchCNN(n_out), True
        if init_state is not None:                                 # transfer: backbone จากโมเดลอื่น (ไม่เอา head)
            net.load_state_dict({kk: v for kk, v in init_state.items() if not kk.startswith("head.")}, strict=False)
    va = np.random.default_rng(seed).random(len(Xtr)) < 0.15
    yva = ytr[va] if (n_out > 1 or task == "multiclass") else ytr[va][:, 0]
    best_state, best, hist = train_torch(net, Xtr[~va], ytr[~va], Xtr[va], yva,
                                         ckpt, loss_fn, score, epochs=epochs, lr=lr, augment=aug,
                                         desc=Path(ckpt).stem, gpu=gpu, seed=seed)
    net.load_state_dict(best_state)
    return net, pre, best_state, hist, gpu


def _torch_bundle(kind, net, pre, best_state, hist, n_out, classes, num_cols=(), cat_cols=(), task="binary"):
    return {"state": best_state, "pre": pre, "classes": classes, "kind": kind, "n_out": n_out, "task": task,
            "n_in": net.head.in_features if kind == "cnn" else net.body[0].in_features,
            "num_cols": list(num_cols), "cat_cols": list(cat_cols), "hist": hist}


def run_torch_cv(reg, exp, df, Y, fold_col, kind, task="binary", classes=None, num_cols=(), cat_cols=(), patches=None,
                 epochs=40, lr=1e-3, desc=None, init_from=None, only_folds=None):
    # kind = "mlp" (tabular) or "cnn" (Sentinel-2 patches). init_from(k) -> path of a saved .pt bundle whose CNN backbone
    # initialises fold k (transfer from stage A; the source must have been trained without fold k).
    # Torch training is not bit-reproducible on MPS, so legacy folds are verified on rows + labels only (their tabular inputs
    # are the same feature sets as the tree models, which are verified by recomputing).
    tqdm = tqdm_auto()
    folds = df[fold_col].values
    (reg.dir / "ckpt").mkdir(exist_ok=True)
    n_out = 1 if task == "binary" else (MULTICLASS_K if task == "multiclass" else Y.shape[1])
    mc = (MC_CONTEXT.get("classes"), MC_CONTEXT.get("ordinal")) if task == "multiclass" else None
    cfg = (kind, task, list(num_cols), list(cat_cols), None if classes is None else list(classes), int(n_out), epochs, lr, mc,
           code_hash(_torch_fit, train_torch, MLP, PatchCNN, PatchNorm, TabularPreprocessor))
    data_fp = fingerprint(df.point_id, df[fold_col], df[list(num_cols) + list(cat_cols)], np.asarray(Y),
                          np.zeros(0) if patches is None else patches)
    for k in tqdm([k for k in sorted(np.unique(folds)) if only_folds is None or k in only_folds], desc=desc or exp, leave=False):
        src = str(init_from(k)) if init_from else None
        fp = fingerprint(cfg, data_fp, int(k), src)
        st = reg.status(exp, k, fp)
        if st == "fresh" or (st == "legacy" and not VERIFY_CACHE):
            continue
        tr, te = folds != k, folds == k
        yte = Y[te] if task != "binary" else Y[te].ravel()
        ckpt = reg.dir / "ckpt" / f"{exp}__f{k}.pt"
        if st == "legacy":
            ok = reg.oof_diff(exp, k, df.point_id.values[te], yte, None) == 0
            reg.audit(exp=exp, fold=int(k), result="rows-match (torch)" if ok else "rows-DIFFERENT (torch)", max_diff=0.0 if ok else float("inf"))
            if ok:
                reg.set_fp(exp, k, fp)
                continue
        if st != "missing" and ckpt.exists():
            ckpt.unlink()                               # never resume a checkpoint trained on other data
        if st == "stale":
            print(f"[cache] {reg.dir.name}/{exp} fold {k}: ข้อมูล/feature/ค่าตั้งเปลี่ยน -> train ใหม่")
        init = torch.load(init_from(k), map_location="cpu", weights_only=False)["state"] if init_from else None
        t0 = time.time()
        net, pre, best_state, hist, gpu = _torch_fit(kind, task, n_out, df[tr], Y[tr], None if patches is None else patches[tr],
                                                     num_cols, cat_cols, ckpt, epochs, lr, init)
        Xte = pre.transform(df[te]).values if kind == "mlp" else pre(patches[te])
        logits = predict_torch(net, Xte, gpu=gpu)
        p = sigmoid(logits[:, 0]) if task == "binary" else (softmax(logits) if task == "multiclass" else sigmoid(logits))
        torch.save(_torch_bundle(kind, net, pre, best_state, hist, n_out, classes, num_cols, cat_cols, task), reg.model_path(exp, k, "pt"))
        _oof_frame(df.point_id.values[te], k, yte, p, task, classes).to_parquet(reg.oof_path(exp, k), index=False)
        reg.add({"exp": exp, "fold": k, "n_test": int(te.sum()), "seconds": round(time.time() - t0, 1),
                 "epochs": len(hist), **_metrics(task, yte, p), "fp": fp})
    return summarize(reg, exp, task)


def run_fusion(reg, exp, parts, task="binary", weights=None):
    # late fusion: weighted mean of OOF probabilities of finished experiments (same folds)
    oofs = [reg.oof(e) for e in parts]
    base = oofs[0].sort_values("point_id").reset_index(drop=True)
    w = np.ones(len(parts)) / len(parts) if weights is None else np.asarray(weights) / np.sum(weights)
    pcols = [c for c in base.columns if c == "p" or c.startswith("p_")]
    r = reg.results()
    part_fp = [r.loc[r.exp.eq(e), "fp"].tolist() if "fp" in r else None for e in parts]
    fp = fingerprint(list(parts), w, part_fp, task)            # fused folds inherit the freshness of their parts
    fused = sum(wi * o.sort_values("point_id").reset_index(drop=True)[pcols].values for wi, o in zip(w, oofs))
    out = base.copy()
    out[pcols] = fused
    for k, g in out.groupby("fold"):
        if task == "binary":
            m = bin_metrics(g.y.values, g.p.values)
        else:
            ycols = [c for c in g.columns if c.startswith("y_")]
            m = multilabel_metrics(g[ycols].values, g[pcols].values)
        g.to_parquet(reg.oof_path(exp, k), index=False)
        reg.add({"exp": exp, "fold": int(k), "n_test": len(g), "seconds": 0.0, **m, "fp": fp})
    return summarize(reg, exp, task)


def summarize(reg, exp, task):
    r = reg.results()
    r = r[r.exp == exp]
    cols = {"binary": ["auc", "pr_auc"], "multiclass": ["acc", "top3", "macro_f1"]}.get(task, ["macro_auc", "top1", "top3"])
    cols = [c for c in cols if c in r]
    return r[cols].mean().to_dict() | {"folds": len(r)}


# ---- prediction with saved bundles (CV fold models or final models) ----
def predict_bundle(path, X, task=None):
    # X: DataFrame (tabular bundles) or float32 patches [n, 6, 32, 32] (CNN bundles)
    path = Path(path)
    if path.suffix == ".joblib":
        b = joblib.load(path)
        Xt = b["pre"].transform(X).values
        m = b["model"]
        if isinstance(m, PriorModel):
            return m.predict_proba_matrix(Xt)
        if b.get("task") == "multiclass":
            P = np.zeros((len(Xt), len(b["classes"])))
            P[:, m.present_] = m.predict_proba(Xt)
            return P
        if isinstance(m, list):
            return np.column_stack([mm.predict_proba(Xt)[:, 1] if mm is not None else np.zeros(len(Xt)) for mm in m])
        pr = m.predict_proba(Xt)
        if isinstance(pr, list):
            return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(Xt)) for p in pr])
        return pr[:, 1]
    b = torch.load(path, map_location="cpu", weights_only=False)
    net = PatchCNN(b["n_out"]) if b["kind"] == "cnn" else MLP(b["n_in"], b["n_out"])
    net.load_state_dict(b["state"])
    Xt = b["pre"](X) if b["kind"] == "cnn" else b["pre"].transform(X).values
    lo = predict_torch(net, Xt, gpu=b["kind"] == "cnn")
    if b.get("task") == "multiclass":
        return softmax(lo)
    return sigmoid(lo[:, 0]) if b["n_out"] == 1 else sigmoid(lo)


def predict_with_fold_models(reg, exp, X):
    # mean prediction of the K fold models of an experiment (for points never used in training, e.g. PBDB)
    paths = sorted(p for p in (reg.dir / "fold_models").glob(f"{exp}__f*.*"))
    return np.mean([predict_bundle(p, X) for p in paths], axis=0)


def embed_bundle(path, patches):
    # 128-d CNN embedding (penultimate layer) for patches
    b = torch.load(path, map_location="cpu", weights_only=False)
    net = PatchCNN(b["n_out"])
    net.load_state_dict(b["state"])
    return predict_torch(net, b["pre"](patches), embed=True, gpu=True)


# ---- final models trained on all labelled data ----
def _final_audit(**row):
    f = MODELS / "final_cache_audit.parquet"
    a = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    row["time"] = pd.Timestamp.now().isoformat(timespec="seconds")
    pd.concat([a, pd.DataFrame([row])], ignore_index=True).to_parquet(f, index=False)


def fit_final_tabular(model, df, Y, num_cols, cat_cols, path, task="binary", classes=None):
    # reuses `path` only if its fingerprint matches the current data / features / settings (legacy files without a
    # fingerprint: trusted, or with ML_VERIFY_CACHE=1 refit and compared on the training rows)
    path = Path(path)
    fp = fingerprint(_tab_cfg(model, task, num_cols, cat_cols, classes, metrics=False), df[list(num_cols) + list(cat_cols)], np.asarray(Y))
    old = joblib.load(path) if path.exists() else None
    if old is not None and (old.get("fp") == fp or (old.get("fp") is None and not VERIFY_CACHE)):
        return path
    mode = "dense" if model in ("knn", "prior") else "tree"
    pre = TabularPreprocessor(num_cols, cat_cols, mode=mode, fill_nan=model in ("rf", "knn"))
    X = pre.fit_transform(df).values
    _, m = _fit_predict(model, task, X, Y, X[:1])
    b = {"pre": pre, "model": m, "classes": classes, "num_cols": list(num_cols), "cat_cols": list(cat_cols),
         "task": task, "model_name": model, "fp": fp}
    if old is not None:
        tmp = path.with_suffix(".verify.joblib")
        joblib.dump(b, tmp, compress=3)
        try:
            diff = float(np.abs(predict_bundle(tmp, df) - predict_bundle(path, df)).max())
        except Exception:                              # e.g. different columns -> cannot be the same model
            diff = float("inf")
        tmp.unlink()
        kind = "legacy" if old.get("fp") is None else "stale"
        _final_audit(path=path.name, kind=kind, result="refit-match" if diff <= 1e-6 else "refit-DIFFERENT", max_diff=diff)
        if diff <= 1e-6:
            old["fp"] = fp
            joblib.dump(old, path, compress=3)
            return path
        print(f"[cache] {path.name}: โมเดลเดิมไม่ตรงกับข้อมูลปัจจุบัน ({kind}, ต่างสูงสุด {diff:.3g}) -> แทนที่")
    joblib.dump(b, path, compress=3)
    return path


def fit_final_torch(kind, df, Y, path, task="binary", classes=None, num_cols=(), cat_cols=(), patches=None,
                    epochs=40, lr=1e-3, init_state=None):
    # same fingerprint rule as fit_final_tabular; legacy torch files are kept (not bit-reproducible) and logged
    path = Path(path)
    n_out = 1 if task == "binary" else (len(classes) if task == "multiclass" else Y.shape[1])
    fp = fingerprint(kind, task, list(num_cols), list(cat_cols), None if classes is None else list(classes), n_out, epochs, lr,
                     code_hash(_torch_fit, train_torch, MLP, PatchCNN, PatchNorm, TabularPreprocessor),
                     df.point_id, df[list(num_cols) + list(cat_cols)], np.asarray(Y), np.zeros(0) if patches is None else patches,
                     None if init_state is None else fingerprint(*[v.cpu().numpy() for v in init_state.values()]))
    ckpt = path.with_suffix(".ckpt.pt")
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False).get("fp")
        if old == fp or (old is None and not VERIFY_CACHE):
            return path
        if old is None:
            _final_audit(path=path.name, kind="legacy", result="torch-kept-unverified", max_diff=float("nan"))
            return path
        print(f"[cache] {path.name}: ข้อมูลเปลี่ยน -> train ใหม่")
        _final_audit(path=path.name, kind="stale", result="retrained", max_diff=float("nan"))
        if ckpt.exists():
            ckpt.unlink()
    net, pre, best_state, hist, _ = _torch_fit(kind, task, n_out, df, Y, patches, num_cols, cat_cols, ckpt, epochs, lr, init_state)
    b = _torch_bundle(kind, net, pre, best_state, hist, n_out, classes, num_cols, cat_cols, task)
    b["fp"] = fp
    torch.save(b, path)
    return path


# ======================================================================
# section: 70_task1.py
# ======================================================================
# Task-1 helpers shared by notebooks 11-12: taxonomic consistency filter and known/new-site routing for the genus model.
OTHER_GENUS = "อื่น ๆ"


def genus_group_map(genus_labels, spec_groups):
    # {genus: set of fossil groups seen among its training specimens}; spec_groups = fl.specimen_classes(...) per specimen
    out = {}
    for g, c in zip(genus_labels, spec_groups):
        out.setdefault(g, set()).update(c)
    return out


def consistency_filter(P, classes, spec_groups, gmap):
    # zero genera whose training specimens never belong to the specimen's fossil group, then renormalise.
    # Left untouched: specimens with no group, genera with no group, the catch-all class, rows where everything is masked.
    P = np.asarray(P, float)
    M = np.ones_like(P)
    for r, c in enumerate(spec_groups):
        if not c:
            continue
        for j, g in enumerate(classes):
            gc = gmap.get(g)
            if g != OTHER_GENUS and gc and not (gc & set(c)):
                M[r, j] = 0
    Q = P * M
    s = Q.sum(axis=1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1), P)


def route_known_site(site_ids, known_sites):
    # True -> the site already has specimens with a genus (use the model with site features); False -> new site / no site
    known = set(known_sites)
    return np.array([pd.notna(s) and int(s) in known for s in site_ids], dtype=bool)


# ======================================================================
# section: 80_aoa.py
# ======================================================================
# Area of applicability (Meyer & Pebesma 2021, Methods Ecol Evol 12:1620): is a new point similar enough to the training
# data for the model's prediction to be trusted? DI = distance to the nearest training point in standardized,
# importance-weighted predictor space / mean distance between training points. Threshold = upper whisker of the
# cross-validated DI of the training points themselves (each point vs training points of the other folds).
class AOA:
    def fit(self, df, num_cols, importance, folds, seed=0, n_pairs=3000):
        from sklearn.neighbors import NearestNeighbors
        w = pd.Series(importance, index=num_cols).clip(lower=0)
        self.cols = [c for c in num_cols if w[c] > 0]
        self.w_ = (w[self.cols] / w[self.cols].sum()).values
        X = df[self.cols].astype(float)
        self.mean_, self.std_ = X.mean(), X.std().replace(0, 1).fillna(1)
        Z = self._z(df)
        rng = np.random.default_rng(seed)
        i, j = rng.integers(0, len(Z), n_pairs), rng.integers(0, len(Z), n_pairs)
        keep = i != j
        self.dbar_ = float(np.linalg.norm(Z[i[keep]] - Z[j[keep]], axis=1).mean())
        folds = np.asarray(folds)
        di = np.empty(len(Z))
        for k in np.unique(folds):
            nn = NearestNeighbors(n_neighbors=1).fit(Z[folds != k])
            di[folds == k] = nn.kneighbors(Z[folds == k])[0][:, 0] / self.dbar_
        q1, q3 = np.percentile(di, [25, 75])
        self.threshold_ = float(q3 + 1.5 * (q3 - q1))
        self.train_di_ = di
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(Z)
        return self

    def _z(self, df):
        X = df.reindex(columns=self.cols).astype(float)
        return (((X - self.mean_) / self.std_).fillna(0).values * self.w_).astype("float32")

    def di(self, df):
        return self.nn_.kneighbors(self._z(df))[0][:, 0] / self.dbar_

    def inside(self, df):
        return self.di(df) <= self.threshold_


def tree_importance(bundle_path):
    # mean feature importance of a saved tree bundle (single model or one model per class), aligned to pre.num_cols
    if Path(bundle_path).suffix != ".joblib":                   # MLP: ไม่มี importance -> น้ำหนักเท่ากัน
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        return pd.Series(1.0, index=b["num_cols"])
    b = joblib.load(bundle_path)
    ms = b["model"] if isinstance(b["model"], list) else [b["model"]]
    imp = np.mean([m.feature_importances_ for m in ms if m is not None and hasattr(m, "feature_importances_")], axis=0)
    cols = b["pre"].num_cols + b["pre"].cat_cols
    return pd.Series(imp, index=cols)[b["pre"].num_cols]
