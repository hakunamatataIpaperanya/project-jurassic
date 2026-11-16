# One entry point for inference: all feature groups for new coordinates, computed exactly as in training.
FEATURE_GROUPS = {"pangea": "pg_", "dem": "dem_", "geology": "geo_", "outcrop": "oc_", "s2": "s2_"}


class FeatureBuilder:
    def __init__(self):
        self._pangea = self._dem = self._geo = None

    @property
    def pangea(self):
        if self._pangea is None:
            self._pangea = Pangea()
        return self._pangea

    @property
    def dem(self):
        if self._dem is None:
            self._dem = Dem()
        return self._dem

    @property
    def geology(self):
        if self._geo is None:
            self._geo = Geology()
        return self._geo

    def tabular(self, lons, lats, with_s2=True):
        # DataFrame with pg_*, dem_*, geo_*, oc_* (and s2_* fetched live) for the given coordinates
        lons, lats = np.atleast_1d(lons).astype(float), np.atleast_1d(lats).astype(float)
        parts = [self.pangea.features(lons, lats).reset_index(drop=True),
                 pd.DataFrame([self.dem.features(lo, la) for lo, la in zip(lons, lats)]),
                 self.geology.features(lons, lats).reset_index(drop=True)]
        patches = []
        if with_s2:
            rows = []
            for lo, la in zip(lons, lats):
                try:
                    patch, meta = fetch_patch_single(lo, la)
                except Exception as e:                       # noqa: BLE001  เน็ต/เซิร์ฟเวอร์ล่ม: ทำต่อโดยไม่มีภาพ แต่แจ้งชัดเจน
                    warnings.warn(f"Sentinel-2 ดึงไม่ได้ที่ ({lo:.4f}, {la:.4f}): {type(e).__name__}: {e} -> feature ดาวเทียมเป็นค่าว่าง")
                    patch, meta = None, {"item_id": None}
                patches.append(patch)
                rows.append({**(s2_indices(patch) if patch is not None else {f"s2_{k}": np.nan for k in S2_FEATURE_KEYS}),
                             "s2_item_id": meta.get("item_id"), "s2_cloud_frac": meta.get("cloud_frac", np.nan)})
            parts.append(pd.DataFrame(rows))
        df = pd.concat(parts, axis=1)
        df.insert(0, "lat", lats)
        df.insert(0, "lon", lons)
        return df, patches
