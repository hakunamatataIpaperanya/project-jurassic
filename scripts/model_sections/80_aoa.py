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
