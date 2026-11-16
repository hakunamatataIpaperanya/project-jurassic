# Terrain features from Copernicus DEM GLO-30 (1 arc-second, EPSG:4326) at the point, 1 km and 5 km.
# Block et al. 2016 use slope as an erosion / exposure proxy ("discovery" model).
DEM_DIR = DATA / "dem" / "cop_dem_glo30"
DEM_RADII_KM = (1, 5)


class Dem:
    def __init__(self):
        self.tiles = {}
        for p in DEM_DIR.glob("*.tif"):
            m = re.search(r"_([NS])(\d+)_00_([EW])(\d+)_00_DEM", p.name)
            lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
            lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
            self.tiles[(lat, lon)] = p
        self._open = {}

    def _ds(self, key):
        import rasterio
        if key not in self._open:
            self._open[key] = rasterio.open(self.tiles[key])
        return self._open[key]

    def window(self, lon, lat, radius_km):
        # elevation array (float32, NaN outside tiles) covering a square of +-radius around the point, and its transform
        from rasterio.merge import merge
        dlat = radius_km / 110.574
        dlon = radius_km / (111.320 * np.cos(np.radians(lat)))
        b = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        keys = [(la, lo) for la in range(int(np.floor(b[1])), int(np.floor(b[3])) + 1)
                for lo in range(int(np.floor(b[0])), int(np.floor(b[2])) + 1) if (la, lo) in self.tiles]
        if not keys:
            return None, None
        arr, tr = merge([self._ds(k) for k in keys], bounds=b, nodata=np.nan, dtype="float32")
        return arr[0], tr

    def features(self, lon, lat):
        z, tr = self.window(lon, lat, max(DEM_RADII_KM) + 0.2)
        keys = ["elev", "slope", "tpi_300m"] + [f"{s}_{r}km" for r in DEM_RADII_KM for s in ("mean", "std", "relief", "slope_mean", "slope_p90", "tpi")]
        if z is None or np.isnan(z).all():
            return {f"dem_{k}": np.nan for k in keys}
        res_x = tr.a * 111320 * np.cos(np.radians(lat))   # metres per pixel
        res_y = -tr.e * 110574
        gy, gx = np.gradient(np.where(np.isnan(z), np.nanmean(z), z), res_y, res_x)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        rows, cols = np.indices(z.shape)
        r0, c0 = (lat - tr.f) / tr.e, (lon - tr.c) / tr.a      # fractional pixel of the point
        dist_km = np.hypot((rows - r0) * res_y, (cols - c0) * res_x) / 1000
        ci, cj = int(np.clip(round(r0), 0, z.shape[0] - 1)), int(np.clip(round(c0), 0, z.shape[1] - 1))
        elev = float(z[ci, cj])
        near = dist_km <= 0.045                                  # ~3x3 px รอบจุด
        out = {"dem_elev": elev, "dem_slope": float(np.nanmean(slope[near])),
               "dem_tpi_300m": elev - float(np.nanmean(z[dist_km <= 0.3]))}
        for r in DEM_RADII_KM:
            m = dist_km <= r
            zz, ss = z[m], slope[m]
            out.update({f"dem_mean_{r}km": float(np.nanmean(zz)), f"dem_std_{r}km": float(np.nanstd(zz)),
                        f"dem_relief_{r}km": float(np.nanmax(zz) - np.nanmin(zz)),
                        f"dem_slope_mean_{r}km": float(np.nanmean(ss)), f"dem_slope_p90_{r}km": float(np.nanpercentile(ss, 90)),
                        f"dem_tpi_{r}km": elev - float(np.nanmean(zz))})
        return out
