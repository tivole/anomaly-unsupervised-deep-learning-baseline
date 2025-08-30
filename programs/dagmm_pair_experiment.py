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
import torch.nn.functional as F
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TIME_COLS = ["window_start_s", "window_end_s"]
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

CANON_MODAL_ORDER = ["mouse_heatmap", "mouse_timing", "keystrokes", "gui"]

def load_csv_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "window_start_s" in df.columns:
        return df.sort_values("window_start_s", kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)

def find_heatmap_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if HEATMAP_RE.match(c)]
    def key(c):
        m = HEATMAP_RE.match(c)
        return (int(m.group(1)), int(m.group(2)))
    return sorted(cols, key=key)

def modality_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    heat = find_heatmap_cols(df)
    def present_numeric(cols):
        return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return {
        "mouse_heatmap": present_numeric(heat),
        "mouse_timing": present_numeric(MOUSE_TIMING_COLS),
        "keystrokes": present_numeric(KEYSTROKE_COLS),
        "gui": present_numeric(GUI_COLS),
    }

def select_all_numeric(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in df.columns if c not in TIME_COLS]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[num_cols].astype(np.float32).values
    return X, num_cols

def fit_preprocess(X_train: np.ndarray):
    med = np.nanmedian(X_train, axis=0)
    def impute(X):
        return np.where(np.isnan(X), med, X)
    scaler = StandardScaler().fit(impute(X_train))
    def transform(X):
        return scaler.transform(impute(X)).astype(np.float32)
    return transform

class CompAE(nn.Module):
    def __init__(self, d_in, h1=128, h2=64, z=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, z)
        )
        self.dec = nn.Sequential(
            nn.Linear(z, h2), nn.Tanh(),
            nn.Linear(h2, h1), nn.Tanh(),
            nn.Linear(h1, d_in)
        )
    def forward(self, x):
        z = self.enc(x)
        xr = self.dec(z)
        return z, xr

class EstNet(nn.Module):
    def __init__(self, d_in, h=32, K=4, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(h, K),
            nn.Softmax(dim=1)
        )
    def forward(self, v):
        return self.net(v)

def relative_euclid(x, xr, eps=1e-9):
    num = torch.sqrt(((x - xr) ** 2).sum(dim=1) + eps)
    den = torch.sqrt((x ** 2).sum(dim=1) + eps)
    return (num / den).unsqueeze(1)

def cosine_sim(x, xr, eps=1e-9):
    ab = (x * xr).sum(dim=1)
    an = torch.sqrt((x ** 2).sum(dim=1) + eps)
    bn = torch.sqrt((xr ** 2).sum(dim=1) + eps)
    c = ab / (an * bn + eps)
    return c.unsqueeze(1)

def compute_gmm_params(v, gamma, eps=1e-6):
    N, Dv = v.shape
    K = gamma.shape[1]
    Nk = gamma.sum(dim=0) + 1e-9
    phi = Nk / N
    mu = (gamma.t() @ v) / Nk.unsqueeze(1)
    diffs = v.unsqueeze(1) - mu.unsqueeze(0)
    cov = torch.einsum('nk,nkd,nke->kde', gamma, diffs, diffs) / Nk.view(K, 1, 1)
    eye = torch.eye(Dv, device=v.device).unsqueeze(0).repeat(K, 1, 1)
    cov = cov + eps * eye
    return phi, mu, cov

def gmm_energy(v, phi, mu, cov):
    N, Dv = v.shape
    K = mu.shape[0]
    inv = torch.inverse(cov)
    sign, logdet = torch.slogdet(cov)
    const = -0.5 * Dv * np.log(2 * np.pi)
    comps = []
    for k in range(K):
        diff = v - mu[k]
        maha = (diff @ inv[k] * diff).sum(dim=1)
        logp = torch.log(phi[k] + 1e-12) + const - 0.5 * logdet[k] - 0.5 * maha
        comps.append(logp)
    logp_all = torch.stack(comps, dim=1)
    log_sum = torch.logsumexp(logp_all, dim=1)
    return -log_sum

class DAGMM(nn.Module):
    def __init__(self, d_in, z=16, ae_h1=128, ae_h2=64, est_h=32, K=4, drop=0.5):
        super().__init__()
        self.comp = CompAE(d_in, ae_h1, ae_h2, z)
        self.est  = EstNet(z + 2, est_h, K, drop)
        self._phi = None
        self._mu  = None
        self._cov = None
    def forward(self, x):
        z, xr = self.comp(x)
        e1 = relative_euclid(x, xr)
        e2 = cosine_sim(x, xr)
        v = torch.cat([z, e1, e2], dim=1)
        gamma = self.est(v)
        return z, xr, v, gamma

def train_dagmm(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    *,
    z: int = 16, K: int = 4,
    ae_h1: int = 128, ae_h2: int = 64, est_h: int = 32,
    drop: float = 0.5, lr: float = 1e-3, weight_decay: float = 0.0,
    epochs: int = 300, patience: int = 30,
    lam_energy: float = 0.1, lam_cov: float = 0.005,
    device: str = "cpu"
) -> DAGMM:
    N, D = X_train.shape
    model = DAGMM(D, z=z, ae_h1=ae_h1, ae_h2=ae_h2, est_h=est_h, K=K, drop=drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    xtr = torch.from_numpy(X_train).to(device)
    xva = torch.from_numpy(X_val).to(device) if X_val is not None and len(X_val) > 0 else None

    best = float("inf")
    best_state = None
    best_params = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z, xr, v, gamma = model(xtr)
        phi, mu, cov = compute_gmm_params(v, gamma)
        E = gmm_energy(v, phi, mu, cov)
        rec = F.mse_loss(xr, xtr, reduction="mean")
        cov_reg = (1.0 / torch.diagonal(cov, dim1=1, dim2=2)).sum()
        loss = rec + lam_energy * E.mean() + lam_cov * cov_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            if xva is not None:
                z_v, xr_v, v_v, _ = model(xva)
                E_v = gmm_energy(v_v, phi, mu, cov)
                rec_v = F.mse_loss(xr_v, xva, reduction="mean")
                val_obj = rec_v + lam_energy * E_v.mean()
            else:
                val_obj = loss.detach()

        if val_obj.item() < best - 1e-6:
            best = val_obj.item()
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            best_params = (phi.detach().cpu(), mu.detach().cpu(), cov.detach().cpu())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    phi, mu, cov = best_params
    model._phi = phi.to(device)
    model._mu  = mu.to(device)
    model._cov = cov.to(device)
    return model

@torch.no_grad()
def energy(model: DAGMM, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    x = torch.from_numpy(X).to(device)
    model.eval()
    z, xr, v, _ = model(x)
    E = gmm_energy(v, model._phi, model._mu, model._cov)
    return E.detach().cpu().numpy()

class DAGMMWrapper:
    def __init__(self, *, z=16, K=4, ae_h1=128, ae_h2=64, est_h=32,
                 drop=0.5, lr=1e-3, weight_decay=0.0, epochs=300, patience=30,
                 lam_energy=0.1, lam_cov=0.005, device="cpu"):
        self.hparams = dict(z=z, K=K, ae_h1=ae_h1, ae_h2=ae_h2, est_h=est_h,
                            drop=drop, lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience,
                            lam_energy=lam_energy, lam_cov=lam_cov)
        self.device = device
        self.model: Optional[DAGMM] = None
        self.transform = None
        self.cal_mean = None
        self.cal_std = None
        self.cal_tau_prob = None

    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray]):
        self.transform = fit_preprocess(X_train)
        Xt = self.transform(X_train)
        Xv = self.transform(X_val) if X_val is not None and len(X_val) > 0 else None
        self.model = train_dagmm(Xt, Xv, device=self.device, **self.hparams)

    def _energies(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transform(X)
        return energy(self.model, Xt, device=self.device)

    def probs(self, X: np.ndarray, E_cal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        E = self._energies(X)
        mu = self.cal_mean if self.cal_mean is not None else float(E.mean())
        sd = self.cal_std  if self.cal_std  is not None else float(E.std() + 1e-9)
        if self.cal_tau_prob is None and E_cal is not None:
            # temporary: z_tau from E_cal's q=0.99 will imply p~0.5 median; final τ set elsewhere
            pass
        z = (E - mu) / sd
        p = 1.0 / (1.0 + np.exp(-z))
        return p, E

    def calibrate_prob(self, X_cal: np.ndarray, q: float) -> None:
        E_cal = self._energies(X_cal)
        self.cal_mean = float(E_cal.mean())
        self.cal_std  = float(E_cal.std() + 1e-9)
        z = (E_cal - self.cal_mean) / self.cal_std
        p_cal = 1.0 / (1.0 + np.exp(-z))
        self.cal_tau_prob = float(np.quantile(p_cal, q))

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

def confusion_metrics(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Dict[str, float]:
    y_pred = (scores >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    frr, far = frr_far_at_tau(y_true, scores, tau)
    eer, tau_e = eer_and_tau(scores, y_true)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "frr": float(frr), "far": float(far), "eer": float(eer), "tau_eer": float(tau_e)
    }

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

def pick_best_fusion(M_tr: np.ndarray, M_va: np.ndarray, q: float,
                     k_list: Optional[List[int]] = None) -> Tuple[str, Optional[int]]:
    cands: List[Tuple[str, Optional[int]]] = [("mean", None), ("max", None)]
    ks = k_list if k_list else default_k_list(M_tr.shape[1])
    cands.extend([("kofm", k) for k in ks])

    best = None
    best_frr = float("inf")
    for fu, kk in cands:
        s_tr = fuse_scores(M_tr, fu, kk)
        tau_tr = float(np.quantile(s_tr, q))
        s_va = fuse_scores(M_va, fu, kk)
        frr = float((s_va >= tau_tr).mean())
        if (frr < best_frr) or (abs(frr - best_frr) < 1e-12 and
                                ((best and best[0] != "mean" and fu == "mean") or
                                 (best and best[0] not in ("mean",) and fu == "max") or
                                 (best and best[0] == "kofm" and fu == "kofm" and (kk or 0) > (best[1] or 0)))):
            best = (fu, kk)
            best_frr = frr
    return best[0], best[1]

def prepare_indices(n1: int, keep2: int, train_n: int, val_n: int):
    tr = np.arange(0, train_n)
    va = np.arange(train_n, train_n + val_n)
    te_norm = np.arange(train_n + val_n, n1)
    te_anom = np.arange(n1, n1 + keep2)
    cal = np.concatenate([tr, va])
    test = np.concatenate([te_norm, te_anom])
    y_true = np.concatenate([
        np.zeros(len(te_norm), dtype=int),
        np.ones(len(te_anom), dtype=int)
    ])
    return tr, va, cal, test, y_true

def run_unimodal(user1: str, user2: str, *, n_per_user=60, keep_u2=30, train_n=40, val_n=10,
                 q=0.99, device="cpu",
                 z=16, K=4, ae_h1=128, ae_h2=64, est_h=32,
                 drop=0.5, lr=1e-3, weight_decay=0.0, epochs=300, patience=30,
                 lam_energy=0.1, lam_cov=0.005) -> Dict[str, float]:
    df1 = ensure_sorted(load_csv_df(user1))
    df2 = ensure_sorted(load_csv_df(user2))
    df1b = df1.iloc[:min(n_per_user, len(df1))].reset_index(drop=True)
    df2b = df2.iloc[:min(n_per_user, len(df2))].reset_index(drop=True)
    df2k = df2b.iloc[:min(keep_u2, len(df2b))].reset_index(drop=True)
    data = pd.concat([df1b, df2k], ignore_index=True)

    assert train_n + val_n <= len(df1b), "Train+Val exceed available user1 rows."
    tr, va, cal, test, y_true = prepare_indices(len(df1b), len(df2k), train_n, val_n)

    X_all, _ = select_all_numeric(data)
    wrap = DAGMMWrapper(z=z, K=K, ae_h1=ae_h1, ae_h2=ae_h2, est_h=est_h,
                        drop=drop, lr=lr, weight_decay=weight_decay,
                        epochs=epochs, patience=patience,
                        lam_energy=lam_energy, lam_cov=lam_cov,
                        device=device)
    wrap.fit(X_all[tr], X_all[va])
    wrap.calibrate_prob(X_all[cal], q)

    p_cal, _ = wrap.probs(X_all[cal])
    p_test, _ = wrap.probs(X_all[test])
    tau = float(np.quantile(p_cal, q))

    metrics = confusion_metrics(y_true, p_test, tau)
    return {
        "mode": "unimodal",
        "fusion": "",
        "k": "",
        "threshold": tau,
        **metrics
    }

def run_multimodal(user1: str, user2: str, *, n_per_user=60, keep_u2=30, train_n=40, val_n=10,
                   q=0.99, fusion_choice="auto", k_list: Optional[List[int]] = None, device="cpu",
                   z=16, K=4, ae_h1=128, ae_h2=64, est_h=32,
                   drop=0.5, lr=1e-3, weight_decay=0.0, epochs=300, patience=30,
                   lam_energy=0.1, lam_cov=0.005) -> Dict[str, float]:
    df1 = ensure_sorted(load_csv_df(user1))
    df2 = ensure_sorted(load_csv_df(user2))
    df1b = df1.iloc[:min(n_per_user, len(df1))].reset_index(drop=True)
    df2b = df2.iloc[:min(n_per_user, len(df2))].reset_index(drop=True)
    df2k = df2b.iloc[:min(keep_u2, len(df2b))].reset_index(drop=True)
    data = pd.concat([df1b, df2k], ignore_index=True)

    assert train_n + val_n <= len(df1b), "Train+Val exceed available user1 rows."
    tr, va, cal, test, y_true = prepare_indices(len(df1b), len(df2k), train_n, val_n)

    mod_cols = modality_columns(data)
    X_by = {}
    for m in CANON_MODAL_ORDER:
        cols = mod_cols[m]
        if not cols:
            raise ValueError(f"Modality '{m}' has no numeric columns.")
        X_by[m] = data[cols].astype(np.float32).values

    models: Dict[str, DAGMMWrapper] = {}
    for m in CANON_MODAL_ORDER:
        X = X_by[m]
        w = DAGMMWrapper(z=z, K=K, ae_h1=ae_h1, ae_h2=ae_h2, est_h=est_h,
                         drop=drop, lr=lr, weight_decay=weight_decay,
                         epochs=epochs, patience=patience,
                         lam_energy=lam_energy, lam_cov=lam_cov,
                         device=device)
        w.fit(X[tr], X[va])
        w.calibrate_prob(X[cal], q)
        models[m] = w

    P_tr, P_va, P_cal, P_test = [], [], [], []
    for m in CANON_MODAL_ORDER:
        p_tr, _ = models[m].probs(X_by[m][tr])
        p_va, _ = models[m].probs(X_by[m][va])
        p_cal, _ = models[m].probs(X_by[m][cal])
        p_te,  _ = models[m].probs(X_by[m][test])
        P_tr.append(p_tr); P_va.append(p_va)
        P_cal.append(p_cal); P_test.append(p_te)

    M_tr   = np.vstack(P_tr).T
    M_va   = np.vstack(P_va).T
    M_cal  = np.vstack(P_cal).T
    M_test = np.vstack(P_test).T

    if fusion_choice == "auto":
        fu, kk = pick_best_fusion(M_tr, M_va, q, k_list)
    elif fusion_choice in ("mean","max","kofm"):
        fu = fusion_choice
        if fu == "kofm":
            kk = (k_list[0] if k_list else default_k_list(M_tr.shape[1])[0])
        else:
            kk = None
    else:
        raise ValueError(f"Unsupported fusion: {fusion_choice}")

    s_cal  = fuse_scores(M_cal,  fu, kk)
    s_test = fuse_scores(M_test, fu, kk)
    tau = float(np.quantile(s_cal, q))

    metrics = confusion_metrics(y_true, s_test, tau)
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
    p = argparse.ArgumentParser(description="DAGMM pair experiment (unimodal & multimodal)")

    # Single pair mode
    p.add_argument("--user1", help="Path to baseline user CSV")
    p.add_argument("--user2", help="Path to impostor user CSV")
    p.add_argument("--out_dir", help="Where to write per-pair artifacts (single pair mode)")

    # Folder mode
    p.add_argument("--pairs_dir", help="Folder with CSVs to run all pairs")
    p.add_argument("--pairs_pattern", default="*.csv", help="Glob pattern inside --pairs_dir")
    p.add_argument("--pairs_out", help="Output CSV path (streaming append). Default: <pairs_dir>/dagmm_pairs.csv")
    p.add_argument("--undirected", action="store_true", help="Each unordered pair once (a as baseline). If not set, run both directions.")

    # Common knobs
    p.add_argument("--mode", choices=["unimodal","multimodal"], default="unimodal")
    p.add_argument("--fusion", choices=["auto","mean","max","kofm"], default="auto",
                   help="Multimodal fusion rule; 'auto' optimises on validation FRR")
    p.add_argument("--k_list", type=int, nargs="*", help="Candidate k for kofm; default ceil(M/2)..M")
    p.add_argument("--n_per_user", type=int, default=60)
    p.add_argument("--keep_u2", type=int, default=30)
    p.add_argument("--train_n", type=int, default=40)
    p.add_argument("--val_n", type=int, default=10)
    p.add_argument("--q", type=float, default=0.99, help="Calibration quantile on baseline fused scores")
    p.add_argument("--device", default="cpu")

    # DAGMM hyperparameters
    p.add_argument("--z", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--ae_h1", type=int, default=128)
    p.add_argument("--ae_h2", type=int, default=64)
    p.add_argument("--est_h", type=int, default=32)
    p.add_argument("--drop", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lam_energy", type=float, default=0.1)
    p.add_argument("--lam_cov", type=float, default=0.005)

    return p

def main():
    args = build_arg_parser().parse_args()

    if args.pairs_dir:
        data_dir = Path(args.pairs_dir)
        files = sorted(data_dir.glob(args.pairs_pattern))
        if not files:
            raise FileNotFoundError(f"No CSVs in {data_dir} matching {args.pairs_pattern}")
        out_csv = Path(args.pairs_out or (data_dir / "dagmm_pairs.csv"))

        names = [f.stem for f in files]
        paths = {f.stem: f for f in files}
        if args.undirected:
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
        else:
            pairs = [(a, b) for a in names for b in names if a != b]

        total = len(pairs)
        for idx, (a, b) in enumerate(pairs, start=1):
            print(f"[{idx}/{total}] DAGMM {args.mode} :: {a} -> {b} (fusion={args.fusion})")
            u1 = paths[a].as_posix()
            u2 = paths[b].as_posix()
            try:
                if args.mode == "unimodal":
                    res = run_unimodal(
                        u1, u2,
                        n_per_user=args.n_per_user, keep_u2=args.keep_u2,
                        train_n=args.train_n, val_n=args.val_n, q=args.q, device=args.device,
                        z=args.z, K=args.K, ae_h1=args.ae_h1, ae_h2=args.ae_h2, est_h=args.est_h,
                        drop=args.drop, lr=args.lr, weight_decay=args.weight_decay,
                        epochs=args.epochs, patience=args.patience,
                        lam_energy=args.lam_energy, lam_cov=args.lam_cov
                    )
                else:
                    res = run_multimodal(
                        u1, u2,
                        n_per_user=args.n_per_user, keep_u2=args.keep_u2,
                        train_n=args.train_n, val_n=args.val_n, q=args.q,
                        fusion_choice=args.fusion, k_list=args.k_list, device=args.device,
                        z=args.z, K=args.K, ae_h1=args.ae_h1, ae_h2=args.ae_h2, est_h=args.est_h,
                        drop=args.drop, lr=args.lr, weight_decay=args.weight_decay,
                        epochs=args.epochs, patience=args.patience,
                        lam_energy=args.lam_energy, lam_cov=args.lam_cov
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
        res = run_unimodal(
            args.user1, args.user2,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, q=args.q, device=args.device,
            z=args.z, K=args.K, ae_h1=args.ae_h1, ae_h2=args.ae_h2, est_h=args.est_h,
            drop=args.drop, lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            lam_energy=args.lam_energy, lam_cov=args.lam_cov
        )
    else:
        res = run_multimodal(
            args.user1, args.user2,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, q=args.q,
            fusion_choice=args.fusion, k_list=args.k_list, device=args.device,
            z=args.z, K=args.K, ae_h1=args.ae_h1, ae_h2=args.ae_h2, est_h=args.est_h,
            drop=args.drop, lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            lam_energy=args.lam_energy, lam_cov=args.lam_cov
        )

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"[OUT] metrics.json -> {args.out_dir}")

if __name__ == "__main__":
    main()
