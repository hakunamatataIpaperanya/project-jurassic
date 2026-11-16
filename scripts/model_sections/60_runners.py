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
