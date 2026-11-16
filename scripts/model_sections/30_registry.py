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
