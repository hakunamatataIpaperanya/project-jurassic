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
