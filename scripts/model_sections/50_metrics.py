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
