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
