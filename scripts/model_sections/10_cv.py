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
