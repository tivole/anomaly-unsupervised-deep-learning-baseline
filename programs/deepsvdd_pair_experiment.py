import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, confusion_matrix

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

HEATMAP_RE = re.compile(r'^mouse_heatmap_r(\d+)_c(\d+)$')

MOUSE_TIMING_COLS = [
    'mouse_avg_speed',
    'mouse_click_pause_mean',
    'mouse_click_pause_std',
    'mouse_left_hold_mean',
    'mouse_left_hold_std',
    'mouse_right_hold_mean',
    'mouse_right_hold_std'
]

KEYSTROKE_COLS = [
    'key_press_count',
    'key_avg_hold',
    'key_std_hold',
    'key_avg_dd',
    'key_std_dd',
    'key_avg_rp',
    'key_std_rp',
    'key_avg_rr',
    'key_cpm'
]

GUI_COLS = [
    'gui_focus_time',
    'gui_switch_count',
    'gui_unique_apps',
    'gui_window_event_count'
]

TIME_COLS = ['window_start_s', 'window_end_s']
ALL_MODALITIES = ["mouse_heatmap", "mouse_timing", "keystrokes", "gui"]

def find_heatmap_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if HEATMAP_RE.match(c)]
    def key(c):
        m = HEATMAP_RE.match(c)
        return (int(m.group(1)), int(m.group(2)))
    return sorted(cols, key=key)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    heat_cols = find_heatmap_cols(df)
    if not heat_cols:
        raise ValueError(f"{path} has no mouse heatmap columns.")
    for req in MOUSE_TIMING_COLS + KEYSTROKE_COLS + GUI_COLS + TIME_COLS:
        if req not in df.columns:
            raise ValueError(f"{path} missing required column: {req}")
    df = df.sort_values('window_start_s', kind="stable").reset_index(drop=True)
    return df

def split_modalities(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    heat_cols = find_heatmap_cols(df)
    X_heat = df[heat_cols].values.astype(np.float32)
    X_mouse_t = df[MOUSE_TIMING_COLS].values.astype(np.float32)
    X_keys = df[KEYSTROKE_COLS].values.astype(np.float32)
    X_gui = df[GUI_COLS].values.astype(np.float32)
    return {
        "mouse_heatmap": X_heat,
        "mouse_timing": X_mouse_t,
        "keystrokes": X_keys,
        "gui": X_gui,
    }

def standardize(train: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    Z = (X - mu)/sd
    return Z, {"mu": mu, "sd": sd}

class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: Optional[int] = None):
        super().__init__()
        if latent_dim is None:
            if input_dim >= 512: latent_dim = 64
            elif input_dim >= 256: latent_dim = 48
            elif input_dim >= 128: latent_dim = 32
            elif input_dim >= 64:  latent_dim = 16
            else:                  latent_dim = max(8, input_dim // 2)

        if input_dim >= 512:
            hidden = [256, 128]
        elif input_dim >= 256:
            hidden = [128, 64]
        elif input_dim >= 128:
            hidden = [64, 32]
        elif input_dim >= 64:
            hidden = [32, 16]
        else:
            hidden = [max(16, input_dim)]

        layers: List[nn.Module] = []
        dims = [input_dim] + hidden + [latent_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def init_center_c(model: nn.Module, X: np.ndarray, device: str = "cpu", eps: float = 1e-3) -> torch.Tensor:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    z = model(xb)
    c = z.mean(dim=0)
    c[(c.abs() < eps)] = eps * torch.sign(c[(c.abs() < eps)] + 1e-12)
    return c.detach()

def deepsvdd_train(
    model: DeepSVDDNet,
    X_train: np.ndarray,
    X_val: np.ndarray,
    objective: str = "soft",
    nu: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
):
    torch.manual_seed(42); np.random.seed(42)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    c = init_center_c(model, X_train, device=device)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val,   dtype=torch.float32, device=device)

    R = torch.tensor(0.0, device=device)
    best_val = float("inf")
    best_state = None
    best_R2 = 0.0

    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xtr.index_select(0, idx)
            z = model(xb)
            dist2 = ((z - c)**2).sum(dim=1)

            if objective == "hard":
                loss = dist2.mean()
            else:
                loss = R**2 + (1.0 / nu) * torch.relu(dist2 - R**2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            ztr = model(Xtr); dtr = ((ztr - c)**2).sum(dim=1)
            if objective == "soft":
                R = dtr.sqrt().quantile(1 - nu)
            zva = model(Xva); dva = ((zva - c)**2).mean().item()
            val_obj = float(dva + (R**2).item() if objective == "soft" else dva)

        if val_obj < best_val - 1e-9:
            best_val = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_R2 = float((R**2).item() if objective == "soft" else 0.0)

    if best_state is not None:
        model.load_state_dict(best_state)
        R = torch.tensor(np.sqrt(best_R2), device=device) if objective == "soft" else torch.tensor(0.0, device=device)

    return {"c": c.detach(), "R": float(R.item()), "best_val": float(best_val)}

@torch.no_grad()
def deepsvdd_scores(model: DeepSVDDNet, c: torch.Tensor, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    z = model(xb)
    d2 = ((z - c)**2).sum(dim=1).cpu().numpy()
    return d2

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def probs_from_distances(dist2: np.ndarray, cal_dist2: np.ndarray) -> np.ndarray:
    mu = cal_dist2.mean()
    sd = cal_dist2.std() + 1e-9
    z = (dist2 - mu)/sd
    return sigmoid(z)

def frr_far_at_tau(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Tuple[float, float]:
    y_pred = (scores >= tau).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    frr = fp / max(tn + fp, 1)
    far = fn / max(tp + fn, 1)
    return float(frr), float(far)

def eer_and_tau(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, th = roc_curve(y_true, scores)
    frr = fpr
    far = 1.0 - tpr
    i = int(np.argmin(np.abs(far - frr)))
    eer = 0.5 * (far[i] + frr[i])
    tau_eer = float(th[i]) if i < len(th) else float(th[-1])
    return float(eer), tau_eer

def metrics_at_tau(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Dict[str, float]:
    y_pred = (scores >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2*prec*rec / max(prec + rec, 1e-9)
    frr, far = frr_far_at_tau(y_true, scores, tau)
    eer, tau_e = eer_and_tau(scores, y_true)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "frr": float(frr), "far": float(far), "eer": float(eer), "tau_eer": float(tau_e)
    }

def _prepare_data(user1_path: str, user2_path: str,
                  n_per_user: int, keep_u2: int,
                  train_n: int, val_n: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray],
                                                     np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df1 = load_csv(user1_path)
    df2 = load_csv(user2_path)

    n1 = min(n_per_user, len(df1))
    n2 = min(n_per_user, len(df2))
    df1 = df1.iloc[:n1].reset_index(drop=True)
    df2 = df2.iloc[:n2].reset_index(drop=True)
    df2k = df2.iloc[:min(keep_u2, len(df2))].reset_index(drop=True)

    data = pd.concat([df1, df2k], ignore_index=True)
    X_by = split_modalities(data)

    tr = np.arange(0, train_n)
    va = np.arange(train_n, train_n + val_n)
    te_norm = np.arange(train_n + val_n, n1)
    te_anom = np.arange(n1, n1 + len(df2k))
    return data, X_by, tr, va, te_norm, te_anom

def _standardize_all(X_by: Dict[str, np.ndarray], tr: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    Z = {}
    std_params = {}
    for k, X in X_by.items():
        Z_all, p = standardize(X[tr], X)
        Z[k] = Z_all
        std_params[k] = p
    return Z, std_params

def fuse_scores(P: np.ndarray, fusion: str, k: Optional[int] = None) -> np.ndarray:
    f = fusion.lower()
    if f == "mean":
        return P.mean(axis=1)
    if f == "max":
        return P.max(axis=1)
    if f == "kofm":
        if k is None:
            raise ValueError("k must be provided for kofm fusion")
        return (P >= 0.5).sum(axis=1) / P.shape[1]
    raise ValueError(f"Unknown fusion: {fusion}")

def default_k_list(M: int) -> List[int]:
    return list(range(int(np.ceil(M/2)), M+1))

def pick_best_fusion(P_tr: np.ndarray, P_va: np.ndarray, q: float,
                     k_list: Optional[List[int]] = None) -> Tuple[str, Optional[int]]:
    M = P_tr.shape[1]
    cands: List[Tuple[str, Optional[int]]] = [("mean", None), ("max", None)]
    ks = k_list if k_list else default_k_list(M)
    cands.extend([("kofm", k) for k in ks])

    best = None
    best_frr = float("inf")
    for fu, kk in cands:
        s_tr = fuse_scores(P_tr, fu, kk)
        tau = float(np.quantile(s_tr, q))
        s_va = fuse_scores(P_va, fu, kk)
        frr = float((s_va >= tau).mean())
        if (frr < best_frr) or (abs(frr - best_frr) < 1e-12 and
                                ((best and best[0] != "mean" and fu == "mean") or
                                 (best and best[0] not in ("mean",) and fu == "max") or
                                 (best and best[0] == "kofm" and fu == "kofm" and (kk or 0) > (best[1] or 0)))):
            best = (fu, kk); best_frr = frr
    return best[0], best[1]

def run_unimodal_all_svdd(user1: str, user2: str,
                          n_per_user: int = 60, keep_u2: int = 30,
                          train_n: int = 40, val_n: int = 10,
                          final_q: float = 0.99,
                          device: str = "cpu",
                          objective: str = "soft", nu: float = 0.1,
                          epochs: int = 120, batch_size: int = 64, lr: float = 1e-3,
                          weight_decay: float = 1e-6) -> Dict[str, float]:
    data, X_by, tr, va, te_norm, te_anom = _prepare_data(user1, user2, n_per_user, keep_u2, train_n, val_n)
    cal_idx = np.concatenate([tr, va])
    test_idx = np.concatenate([te_norm, te_anom])
    y_true = np.concatenate([np.zeros(len(te_norm), dtype=int), np.ones(len(te_anom), dtype=int)])

    X_all = np.hstack([X_by["mouse_heatmap"], X_by["mouse_timing"], X_by["keystrokes"], X_by["gui"]]).astype(np.float32)
    Z_all, _ = standardize(X_all[tr], X_all)

    model = DeepSVDDNet(Z_all.shape[1])
    tr_info = deepsvdd_train(model, Z_all[tr], Z_all[va], objective=objective, nu=nu,
                             lr=lr, weight_decay=weight_decay, epochs=epochs,
                             batch_size=batch_size, device=device)
    c = tr_info["c"]

    d2_cal  = deepsvdd_scores(model, c, Z_all[cal_idx],  device=device)
    d2_test = deepsvdd_scores(model, c, Z_all[test_idx], device=device)
    p_cal   = probs_from_distances(d2_cal, d2_cal)
    p_test  = probs_from_distances(d2_test, d2_cal)

    tau = float(np.quantile(p_cal, final_q))
    metrics = metrics_at_tau(y_true, p_test, tau)
    return {
        "mode": "unimodal",
        "fusion": "",
        "k": "",
        "threshold": tau,
        **metrics
    }

def run_multimodal_svdd(user1: str, user2: str,
                        n_per_user: int = 60, keep_u2: int = 30,
                        train_n: int = 40, val_n: int = 10,
                        final_q: float = 0.99,
                        device: str = "cpu",
                        objective: str = "soft", nu: float = 0.1,
                        epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
                        weight_decay: float = 1e-6,
                        fusion_choice: str = "auto", k_list: Optional[List[int]] = None) -> Dict[str, float]:
    data, X_by, tr, va, te_norm, te_anom = _prepare_data(user1, user2, n_per_user, keep_u2, train_n, val_n)
    cal_idx = np.concatenate([tr, va])
    test_idx = np.concatenate([te_norm, te_anom])
    y_true = np.concatenate([np.zeros(len(te_norm), dtype=int), np.ones(len(te_anom), dtype=int)])

    Z_by, _ = _standardize_all(X_by, tr)

    models: Dict[str, DeepSVDDNet] = {}
    centers: Dict[str, torch.Tensor] = {}
    for m in ALL_MODALITIES:
        Xm = Z_by[m]
        net = DeepSVDDNet(Xm.shape[1])
        tr_info = deepsvdd_train(net, Xm[tr], Xm[va], objective=objective, nu=nu,
                                 lr=lr, weight_decay=weight_decay, epochs=epochs,
                                 batch_size=batch_size, device=device)
        models[m] = net
        centers[m] = tr_info["c"]

    P_tr, P_va, P_cal, P_test = [], [], [], []
    for m in ALL_MODALITIES:
        net, c = models[m], centers[m]
        p_tr  = probs_from_distances(deepsvdd_scores(net, c, Z_by[m][tr],  device=device),
                                     deepsvdd_scores(net, c, Z_by[m][tr],  device=device))
        p_va  = probs_from_distances(deepsvdd_scores(net, c, Z_by[m][va],  device=device),
                                     deepsvdd_scores(net, c, Z_by[m][tr],  device=device))
        p_cal = probs_from_distances(deepsvdd_scores(net, c, Z_by[m][cal_idx], device=device),
                                     deepsvdd_scores(net, c, Z_by[m][cal_idx], device=device))
        p_te  = probs_from_distances(deepsvdd_scores(net, c, Z_by[m][test_idx], device=device),
                                     deepsvdd_scores(net, c, Z_by[m][cal_idx], device=device))
        P_tr.append(p_tr); P_va.append(p_va); P_cal.append(p_cal); P_test.append(p_te)

    M_tr   = np.vstack(P_tr).T
    M_va   = np.vstack(P_va).T
    M_cal  = np.vstack(P_cal).T
    M_test = np.vstack(P_test).T

    if fusion_choice == "auto":
        fu, kk = pick_best_fusion(M_tr, M_va, final_q, k_list)
    elif fusion_choice in ("mean","max","kofm"):
        fu = fusion_choice
        kk = (k_list[0] if fu == "kofm" and k_list else default_k_list(M_tr.shape[1])[0]) if fu == "kofm" else None
    else:
        raise ValueError(f"Unsupported fusion: {fusion_choice}")

    s_cal  = fuse_scores(M_cal,  fu, kk)
    s_test = fuse_scores(M_test, fu, kk)
    tau = float(np.quantile(s_cal, final_q))

    metrics = metrics_at_tau(y_true, s_test, tau)
    return {
        "mode": "multimodal",
        "fusion": fu,
        "k": (kk if kk is not None else ""),
        "threshold": tau,
        **metrics
    }

CSV_HEADER = [
    "base_user","imp_user","direction","mode","fusion","k",
    "threshold","tau_eer",
    "tn","fp","fn","tp",
    "accuracy","precision","recall","f1",
    "frr","far","eer"
]

def append_row(csv_path: Path, row: Dict[str, object]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()

def append_mean_row(csv_path: Path):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[~df["base_user"].isin(["MEAN"])]
    num_cols = ["threshold","tau_eer","tn","fp","fn","tp","accuracy","precision","recall","f1","frr","far","eer"]
    num_cols = [c for c in num_cols if c in df.columns]
    mean_vals = df[num_cols].mean(numeric_only=True)

    mean_row = {k: "" for k in CSV_HEADER}
    mean_row["base_user"] = "MEAN"
    for c in num_cols:
        mean_row[c] = float(mean_vals[c])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writerow(mean_row)
        f.flush()

def build_arg_parser():
    p = argparse.ArgumentParser(description="Deep SVDD pair experiments (unimodal & multimodal)")

    # Single pair mode
    p.add_argument("--user1", help="CSV for baseline user")
    p.add_argument("--user2", help="CSV for impostor user")
    p.add_argument("--out_dir", help="Where to write metrics.json (single pair mode)")

    # Folder mode
    p.add_argument("--pairs_dir", help="Folder with user*.csv for all pairs")
    p.add_argument("--pairs_pattern", default="*.csv", help="Glob pattern inside --pairs_dir")
    p.add_argument("--pairs_out", help="Output CSV path (streaming append); default: <pairs_dir>/svdd_pairs.csv")
    p.add_argument("--undirected", action="store_true", help="Each unordered pair once (a as baseline). If not set, run both directions.")

    # Modes
    p.add_argument("--mode", choices=["unimodal","multimodal"], default="unimodal",
                   help="Unimodal = one model over ALL features; Multimodal = per-modality + fusion")
    p.add_argument("--fusion", choices=["auto","mean","max","kofm"], default="auto",
                   help="Fusion rule for multimodal; 'auto' optimises FRR on validation")
    p.add_argument("--k_list", type=int, nargs="*", help="Candidate k for kofm; default ceil(M/2)..M")

    # Split
    p.add_argument("--n_per_user", type=int, default=60)
    p.add_argument("--keep_u2", type=int, default=30)
    p.add_argument("--train_n", type=int, default=40)
    p.add_argument("--val_n", type=int, default=10)
    p.add_argument("--final_q", type=float, default=0.99, help="Calibration quantile on baseline probabilities")

    # Optimisation
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=100, help="Epochs for unimodal")
    p.add_argument("--epochs_multimodal", type=int, default=100, help="Epochs for each modality in multimodal")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)

    # Deep SVDD objective
    p.add_argument("--objective", choices=["soft", "hard"], default="soft",
                   help="soft = soft-boundary Deep SVDD; hard = one-class objective")
    p.add_argument("--nu", type=float, default=0.1, help="Outlier fraction for soft-boundary objective")

    return p

def main():
    args = build_arg_parser().parse_args()

    if args.pairs_dir:
        data_dir = Path(args.pairs_dir)
        files = sorted(data_dir.glob(args.pairs_pattern))
        if not files:
            raise FileNotFoundError(f"No CSVs in {data_dir} matching {args.pairs_pattern}")
        out_csv = Path(args.pairs_out or (data_dir / "svdd_pairs.csv"))

        names = [f.stem for f in files]
        paths = {f.stem: f for f in files}
        if args.undirected:
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
        else:
            pairs = [(a, b) for a in names for b in names if a != b]

        total = len(pairs)
        for idx, (a, b) in enumerate(pairs, start=1):
            print(f"[{idx}/{total}] DeepSVDD {args.mode} :: {a} -> {b} (fusion={args.fusion})")
            u1 = paths[a].as_posix()
            u2 = paths[b].as_posix()
            try:
                if args.mode == "unimodal":
                    res = run_unimodal_all_svdd(
                        u1, u2,
                        n_per_user=args.n_per_user, keep_u2=args.keep_u2,
                        train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
                        device=args.device, objective=args.objective, nu=args.nu,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        weight_decay=args.weight_decay
                    )
                else:
                    res = run_multimodal_svdd(
                        u1, u2,
                        n_per_user=args.n_per_user, keep_u2=args.keep_u2,
                        train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
                        device=args.device, objective=args.objective, nu=args.nu,
                        epochs=args.epochs_multimodal, batch_size=args.batch_size, lr=args.lr,
                        weight_decay=args.weight_decay,
                        fusion_choice=args.fusion, k_list=args.k_list
                    )
                row = {
                    "base_user": a,
                    "imp_user": b,
                    "direction": f"{a}->{b}",
                    "mode": res.get("mode",""),
                    "fusion": res.get("fusion",""),
                    "k": res.get("k",""),
                    "threshold": float(res.get("threshold", np.nan)),
                    "tau_eer": float(res.get("tau_eer", np.nan)),
                    "tn": int(res.get("tn", 0)),
                    "fp": int(res.get("fp", 0)),
                    "fn": int(res.get("fn", 0)),
                    "tp": int(res.get("tp", 0)),
                    "accuracy": float(res.get("accuracy", np.nan)),
                    "precision": float(res.get("precision", np.nan)),
                    "recall": float(res.get("recall", np.nan)),
                    "f1": float(res.get("f1", np.nan)),
                    "frr": float(res.get("frr", np.nan)),
                    "far": float(res.get("far", np.nan)),
                    "eer": float(res.get("eer", np.nan)),
                }
                append_row(out_csv, row)
            except Exception as e:
                print(f"[{idx}/{total}] ERROR {a}->{b}: {e}")

        append_mean_row(out_csv)
        print(f"[DONE] Results written to {out_csv} (+ MEAN row)")
        return

    if not (args.user1 and args.user2 and args.out_dir):
        raise SystemExit("Provide --user1, --user2 and --out_dir for single-pair mode, or use --pairs_dir.")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "unimodal":
        res = run_unimodal_all_svdd(
            args.user1, args.user2,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
            device=args.device, objective=args.objective, nu=args.nu,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        res = run_multimodal_svdd(
            args.user1, args.user2,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
            device=args.device, objective=args.objective, nu=args.nu,
            epochs=args.epochs_multimodal, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay,
            fusion_choice=args.fusion, k_list=args.k_list
        )

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"[OUT] metrics.json -> {args.out_dir}")

if __name__ == "__main__":
    main()
