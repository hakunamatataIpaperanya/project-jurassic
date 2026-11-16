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
