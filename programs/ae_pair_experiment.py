import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve

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

def load_csv_df(path: str) -> pd.DataFrame:
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

class AE(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int]):
        super().__init__()
        enc_layers: List[nn.Module] = []
        sizes = [input_dim] + hidden
        for i in range(len(sizes) - 1):
            enc_layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        ds = list(reversed(sizes))
        for i in range(len(ds) - 1):
            dec_layers.append(nn.Linear(ds[i], ds[i+1]))
            if i < len(ds) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z)
        return xr

def make_ae(input_dim: int) -> AE:
    if input_dim >= 256:
        hidden = [256, 128, 64, 32]
    elif input_dim >= 128:
        hidden = [128, 64, 32]
    elif input_dim >= 32:
        hidden = [32, 16]
    elif input_dim >= 8:
        hidden = [8, 4]
    else:
        hidden = [max(2, input_dim // 2) or 2]
    return AE(input_dim, hidden)

def train_ae(model: AE,
             X_train: np.ndarray,
             X_val: np.ndarray,
             device: str = "cpu",
             epochs: int = 60,
             batch_size: int = 16,
             lr: float = 1e-3) -> Dict[str, float]:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    n = Xtr.shape[0]

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xtr.index_select(0, idx)
            xr = model(xb)
            loss = loss_fn(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Xva).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_mse": float(best_val)}

@torch.no_grad()
def recon_errors(model: AE, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    Xb = torch.tensor(X, dtype=torch.float32, device=device)
    Xr = model(Xb).cpu().numpy()
    err = ((Xr - X)**2).mean(axis=1)
    return err

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def probs_from_errors(err: np.ndarray, cal_err: np.ndarray) -> np.ndarray:
    mu = cal_err.mean()
    sd = cal_err.std() + 1e-9
    z = (err - mu)/sd
    return sigmoid(z)

def frr_far_at_threshold(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Tuple[float, float]:
    y_pred = (scores >= tau).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    frr = fp / max(tn + fp, 1)
    far = fn / max(tp + fn, 1)
    return float(frr), float(far)

def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, th = roc_curve(y_true, scores)
    frr = fpr
    far = 1.0 - tpr
    i = int(np.argmin(np.abs(far - frr)))
    eer = 0.5 * (far[i] + frr[i])
    tau_eer = float(th[i]) if i < len(th) else float(th[-1])
    return float(eer), tau_eer

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Dict[str, float]:
    y_pred = (scores >= tau).astype(int)
    tn = int(((y_true==0) & (y_pred==0)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    tp = int(((y_true==1) & (y_pred==1)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2*prec*rec / max(prec + rec, 1e-9)
    frr, far = frr_far_at_threshold(y_true, scores, tau)
    eer, tau_eer = eer_from_scores(y_true, scores)
    return {
        "tn":tn,"fp":fp,"fn":fn,"tp":tp,
        "accuracy":acc,"precision":prec,"recall":rec,"f1":f1,
        "frr": frr, "far": far, "eer": eer, "tau_eer": tau_eer
    }

def fuse_scores(M: np.ndarray, fusion: str, k: Optional[int] = None) -> np.ndarray:
    f = fusion.lower()
    if f == "mean":
        return M.mean(axis=1)
    if f == "max":
        return M.max(axis=1)
    if f == "kofm":
        if k is None:
            raise ValueError("k must be provided for kofm fusion")
        return (M >= 0.5).sum(axis=1) / M.shape[1]
    raise ValueError(f"Unknown fusion: {fusion}")

def default_k_list(m: int) -> List[int]:
    start = int(np.ceil(m/2))
    return list(range(start, m+1))

def pick_best_fusion(M_tr: np.ndarray, M_va: np.ndarray, final_q: float,
                     fusion_choice: str, k_list: Optional[List[int]]) -> Tuple[str, Optional[int]]:
    candidates: List[Tuple[str, Optional[int]]] = []
    if fusion_choice == "auto":
        candidates += [("mean", None), ("max", None)]
        ks = k_list if k_list else default_k_list(M_tr.shape[1])
        for k in ks:
            candidates.append(("kofm", k))
    elif fusion_choice == "mean":
        candidates = [("mean", None)]
    elif fusion_choice == "max":
        candidates = [("max", None)]
    elif fusion_choice == "kofm":
        ks = k_list if k_list else default_k_list(M_tr.shape[1])
        candidates = [("kofm", k) for k in ks]
    else:
        raise ValueError(f"Unsupported --fusion '{fusion_choice}'")

    best = None
    best_frr = float("inf")

    for fu, kk in candidates:
        s_tr = fuse_scores(M_tr, fu, kk)
        tau_train = float(np.quantile(s_tr, final_q))
        s_va = fuse_scores(M_va, fu, kk)
        frr = float((s_va >= tau_train).mean())
        key = (fu, kk)
        if frr < best_frr or (abs(frr - best_frr) < 1e-12 and
                              ((best and best[0] != "mean" and fu == "mean") or
                               (best and best[0] not in ("mean",) and fu == "max") or
                               (best and best[0] == "kofm" and fu == "kofm" and (kk or 0) > (best[1] or 0)) )):
            best = key
            best_frr = frr

    return best[0], best[1]

def _prepare_data(user1_path: str, user2_path: str,
                  n_per_user: int, keep_u2: int,
                  train_n: int, val_n: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray],
                                                     np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df1 = load_csv_df(user1_path)
    df2 = load_csv_df(user2_path)

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

def _standardize_all(X_by: Dict[str, np.ndarray], tr: np.ndarray) -> Dict[str, np.ndarray]:
    Z = {}
    for k, X in X_by.items():
        Z_all, _ = standardize(X[tr], X)
        Z[k] = Z_all
    return Z

def pair_experiment_nn_multimodal(user1_path: str, user2_path: str,
                                  out_dir: Optional[str],
                                  write_artifacts: bool = True,
                                  n_per_user: int = 60, keep_u2: int = 30,
                                  train_n: int = 40, val_n: int = 10,
                                  final_q: float = 0.99,
                                  fusion_choice: str = "auto",
                                  k_list: Optional[List[int]] = None,
                                  device: str = "cpu",
                                  epochs_heatmap: int = 80,
                                  epochs_other: int = 60,
                                  batch_size: int = 16,
                                  lr: float = 1e-3) -> Dict[str, float]:

    if write_artifacts and out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data, X_by, tr, va, te_norm, te_anom = _prepare_data(user1_path, user2_path, n_per_user, keep_u2, train_n, val_n)
    cal_idx = np.concatenate([tr, va])
    test_idx = np.concatenate([te_norm, te_anom])
    y_true = np.concatenate([np.zeros(len(te_norm), dtype=int),
                             np.ones(len(te_anom), dtype=int)])

    Z = _standardize_all(X_by, tr)

    details = {}
    probs_tr, probs_va, probs_cal, probs_test = [], [], [], []

    for kmod in ["mouse_heatmap", "mouse_timing", "keystrokes", "gui"]:
        Xk = Z[kmod]
        in_dim = Xk.shape[1]
        model = make_ae(in_dim)
        e = epochs_heatmap if kmod == "mouse_heatmap" else epochs_other
        hist = train_ae(model, Xk[tr], Xk[va], device=device, epochs=e, batch_size=batch_size, lr=lr)

        err_tr   = recon_errors(model, Xk[tr],      device=device)
        err_va   = recon_errors(model, Xk[va],      device=device)
        err_cal  = recon_errors(model, Xk[cal_idx], device=device)
        err_test = recon_errors(model, Xk[test_idx],device=device)

        p_tr   = probs_from_errors(err_tr,   err_cal)
        p_va   = probs_from_errors(err_va,   err_cal)
        p_cal  = probs_from_errors(err_cal,  err_cal)
        p_test = probs_from_errors(err_test, err_cal)

        probs_tr.append(p_tr); probs_va.append(p_va)
        probs_cal.append(p_cal); probs_test.append(p_test)

        details[kmod] = {"input_dim": in_dim, "val_mse": hist["val_mse"]}

    M_tr   = np.vstack(probs_tr).T
    M_va   = np.vstack(probs_va).T
    M_cal  = np.vstack(probs_cal).T
    M_test = np.vstack(probs_test).T

    chosen_fusion, chosen_k = pick_best_fusion(M_tr, M_va, final_q, fusion_choice, k_list)

    s_cal  = fuse_scores(M_cal,  chosen_fusion, chosen_k)
    s_test = fuse_scores(M_test, chosen_fusion, chosen_k)
    tau = float(np.quantile(s_cal, final_q))

    metrics = compute_metrics(y_true, s_test, tau)

    if write_artifacts and out_dir:
        with open(os.path.join(out_dir, "nn_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "mode": "multimodal",
                "fusion": chosen_fusion,
                "k": chosen_k,
                "threshold": tau,
                "metrics": metrics,
                "details": details
            }, f, indent=2)

        out_rows = []
        for i, idx in enumerate(test_idx):
            row = {
                "window_start_s": float(data.loc[idx, "window_start_s"]),
                "window_end_s": float(data.loc[idx, "window_end_s"]),
                "label": int(y_true[i]),
                "score_fused": float(s_test[i]),
                "fusion": chosen_fusion,
                "k": int(chosen_k) if chosen_k is not None else ""
            }
            for j, mod in enumerate(["mouse_heatmap","mouse_timing","keystrokes","gui"]):
                row[f"{mod}_p"] = float(M_test[i, j])
            out_rows.append(row)
        pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, "nn_test_scores.csv"), index=False)

    return {
        "threshold": tau,
        **metrics,
        "mode": "multimodal",
        "fusion": chosen_fusion,
        "k": chosen_k
    }

def pair_experiment_nn_single_modality(user1_path: str, user2_path: str,
                                       out_dir: Optional[str],
                                       write_artifacts: bool = True,
                                       modality: str = "keystrokes",
                                       n_per_user: int = 60, keep_u2: int = 30,
                                       train_n: int = 40, val_n: int = 10,
                                       final_q: float = 0.99,
                                       device: str = "cpu",
                                       epochs_heatmap: int = 80,
                                       epochs_other: int = 60,
                                       batch_size: int = 16,
                                       lr: float = 1e-3) -> Dict[str, float]:

    if modality not in ALL_MODALITIES:
        raise ValueError(f"--single_modality must be one of {ALL_MODALITIES}")
    if write_artifacts and out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data, X_by, tr, va, te_norm, te_anom = _prepare_data(user1_path, user2_path, n_per_user, keep_u2, train_n, val_n)
    cal_idx = np.concatenate([tr, va])
    test_idx = np.concatenate([te_norm, te_anom])
    y_true = np.concatenate([np.zeros(len(te_norm), dtype=int),
                             np.ones(len(te_anom), dtype=int)])

    X = X_by[modality]
    Z_all, _ = standardize(X[tr], X)

    in_dim = Z_all.shape[1]
    model = make_ae(in_dim)
    ep = epochs_heatmap if modality == "mouse_heatmap" else epochs_other
    hist = train_ae(model, Z_all[tr], Z_all[va], device=device, epochs=ep, batch_size=batch_size, lr=lr)

    err_cal = recon_errors(model, Z_all[cal_idx], device=device)
    err_test = recon_errors(model, Z_all[test_idx], device=device)
    p_cal = probs_from_errors(err_cal, err_cal)
    p_test = probs_from_errors(err_test, err_cal)

    tau = float(np.quantile(p_cal, final_q))
    metrics = compute_metrics(y_true, p_test, tau)

    if write_artifacts and out_dir:
        with open(os.path.join(out_dir, "nn_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "mode": "single_modality",
                "modality": modality,
                "threshold": tau,
                "metrics": metrics,
                "details": {modality: {"input_dim": in_dim, "val_mse": hist["val_mse"]}}
            }, f, indent=2)

        out_rows = []
        for i, idx in enumerate(test_idx):
            out_rows.append({
                "window_start_s": float(data.loc[idx, "window_start_s"]),
                "window_end_s": float(data.loc[idx, "window_end_s"]),
                "label": int(y_true[i]),
                "score": float(p_test[i]),
            })
        pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, "nn_test_scores.csv"), index=False)

    return {"threshold": tau, **metrics, "mode": "single_modality", "modality": modality}

def pair_experiment_nn_unimodal_all(user1_path: str, user2_path: str,
                                    out_dir: Optional[str],
                                    write_artifacts: bool = True,
                                    n_per_user: int = 60, keep_u2: int = 30,
                                    train_n: int = 40, val_n: int = 10,
                                    final_q: float = 0.99,
                                    device: str = "cpu",
                                    epochs: int = 80,
                                    batch_size: int = 16,
                                    lr: float = 1e-3) -> Dict[str, float]:

    if write_artifacts and out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data, X_by, tr, va, te_norm, te_anom = _prepare_data(user1_path, user2_path, n_per_user, keep_u2, train_n, val_n)
    cal_idx = np.concatenate([tr, va])
    test_idx = np.concatenate([te_norm, te_anom])
    y_true = np.concatenate([np.zeros(len(te_norm), dtype=int),
                             np.ones(len(te_anom), dtype=int)])

    X_all = np.hstack([X_by["mouse_heatmap"], X_by["mouse_timing"], X_by["keystrokes"], X_by["gui"]]).astype(np.float32)
    Z_all, _ = standardize(X_all[tr], X_all)

    in_dim = Z_all.shape[1]
    model = make_ae(in_dim)
    hist = train_ae(model, Z_all[tr], Z_all[va], device=device, epochs=epochs, batch_size=batch_size, lr=lr)

    err_cal = recon_errors(model, Z_all[cal_idx], device=device)
    err_test = recon_errors(model, Z_all[test_idx], device=device)
    p_cal = probs_from_errors(err_cal, err_cal)
    p_test = probs_from_errors(err_test, err_cal)

    tau = float(np.quantile(p_cal, final_q))
    metrics = compute_metrics(y_true, p_test, tau)

    if write_artifacts and out_dir:
        with open(os.path.join(out_dir, "nn_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "mode": "unimodal_all",
                "input_dim": in_dim,
                "val_mse": hist["val_mse"],
                "threshold": tau,
                "metrics": metrics
            }, f, indent=2)

        out_rows = []
        for i, idx in enumerate(test_idx):
            out_rows.append({
                "window_start_s": float(data.loc[idx, "window_start_s"]),
                "window_end_s": float(data.loc[idx, "window_end_s"]),
                "label": int(y_true[i]),
                "score": float(p_test[i]),
            })
        pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, "nn_test_scores.csv"), index=False)

    return {"threshold": tau, **metrics, "mode": "unimodal_all", "input_dim": in_dim}

def run_one_pair(args,
                 u1_path: str,
                 u2_path: str,
                 write_artifacts: bool,
                 out_dir: Optional[str] = None) -> Dict[str, float]:
    if args.unimodal_all:
        return pair_experiment_nn_unimodal_all(
            user1_path=u1_path, user2_path=u2_path,
            out_dir=out_dir, write_artifacts=write_artifacts,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
            device=args.device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
    elif args.single_modality:
        return pair_experiment_nn_single_modality(
            user1_path=u1_path, user2_path=u2_path,
            out_dir=out_dir, write_artifacts=write_artifacts,
            modality=args.single_modality,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
            device=args.device, epochs_heatmap=args.epochs_heatmap, epochs_other=args.epochs_other,
            batch_size=args.batch_size, lr=args.lr
        )
    else:
        return pair_experiment_nn_multimodal(
            user1_path=u1_path, user2_path=u2_path,
            out_dir=out_dir, write_artifacts=write_artifacts,
            n_per_user=args.n_per_user, keep_u2=args.keep_u2,
            train_n=args.train_n, val_n=args.val_n, final_q=args.final_q,
            fusion_choice=args.fusion, k_list=args.k_list,
            device=args.device, epochs_heatmap=args.epochs_heatmap, epochs_other=args.epochs_other,
            batch_size=args.batch_size, lr=args.lr
        )

def append_row_realtime(csv_path: Path, row: Dict[str, object], header_order: Optional[List[str]] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        if header_order is None:
            header_order = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

def append_summary_rows(csv_path: Path, header_order: List[str]):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "threshold", "tau_eer",
        "tn","fp","fn","tp",
        "accuracy","precision","recall","f1",
        "frr","far","eer"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    mean_vals = df[numeric_cols].mean(numeric_only=True)
    std_vals  = df[numeric_cols].std(numeric_only=True)

    mean_row = {k: "" for k in header_order}
    std_row  = {k: "" for k in header_order}
    mean_row["base_user"] = "MEAN"
    std_row["base_user"]  = "STD"

    for c in numeric_cols:
        mean_row[c] = float(mean_vals[c])
        std_row[c]  = float(std_vals[c])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        writer.writerow(mean_row)
        writer.writerow(std_row)
        f.flush()

def build_arg_parser():
    p = argparse.ArgumentParser(description="Neural Baseline-Deviation Pair Experiment")

    # Single-pair args
    p.add_argument("--user1", help="CSV for baseline user")
    p.add_argument("--user2", help="CSV for impostor user")
    p.add_argument("--out_dir", help="Where to write results (single pair only)")

    # Shared knobs
    p.add_argument("--n_per_user", type=int, default=60)
    p.add_argument("--keep_u2", type=int, default=30)
    p.add_argument("--train_n", type=int, default=40)
    p.add_argument("--val_n", type=int, default=10)
    p.add_argument("--final_q", type=float, default=0.99, help="Calibration quantile on baseline probabilities")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs_heatmap", type=int, default=80)
    p.add_argument("--epochs_other", type=int, default=60)
    p.add_argument("--epochs", type=int, default=80, help="Epochs for --unimodal_all mode")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    # Fusion controls (multimodal only)
    p.add_argument("--fusion", choices=["auto","mean","max","kofm"], default="auto",
                   help="Fusion rule for multimodal: auto (validate FRR), mean, max, kofm")
    p.add_argument("--k_list", type=int, nargs="*", help="Candidate k values for F_k/M (defaults to ceil(M/2)...M)")

    # Mode selectors
    p.add_argument("--single_modality", choices=ALL_MODALITIES, help="Run ONLY this modality (no fusion).")
    p.add_argument("--unimodal_all", action="store_true", help="Train ONE AE on ALL features concatenated (no fusion).")

    # Folder mode
    p.add_argument("--pairs_dir", help="If set, run pair experiment for every CSV pair in this folder.")
    p.add_argument("--pairs_pattern", default="*.csv", help="Glob pattern inside --pairs_dir (default: *.csv)")
    p.add_argument("--pairs_out", help="CSV path where per-pair results are appended in real time.")
    p.add_argument("--undirected", action="store_true", help="If set, evaluate each unordered pair once (a as baseline). Otherwise run both directions.")
    return p

def main():
    args = build_arg_parser().parse_args()

    if args.pairs_dir:
        data_dir = Path(args.pairs_dir)
        files = sorted(data_dir.glob(args.pairs_pattern))
        if not files:
            raise FileNotFoundError(f"No CSV files in {data_dir} matching {args.pairs_pattern}")

        out_csv = Path(args.pairs_out or (data_dir / "pair_results.csv"))
        header = [
            "base_user", "imp_user", "direction", "mode", "modality",
            "fusion", "k",
            "threshold", "tau_eer",
            "tn","fp","fn","tp",
            "accuracy","precision","recall","f1",
            "frr","far","eer"
        ]

        names = [f.stem for f in files]
        paths = {f.stem: f for f in files}
        if args.undirected:
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
        else:
            pairs = [(a, b) for a in names for b in names if a != b]

        total = len(pairs)
        for kidx, (a, b) in enumerate(pairs, start=1):
            u1 = paths[a].as_posix()
            u2 = paths[b].as_posix()
            print(f"[{kidx}/{total}] running {a} -> {b} (fusion={args.fusion}) ...")
            try:
                res = run_one_pair(args, u1, u2, write_artifacts=False, out_dir=None)
                row = {
                    "base_user": a,
                    "imp_user": b,
                    "direction": f"{a}->{b}",
                    "mode": res.get("mode", ""),
                    "modality": res.get("modality", ""),
                    "fusion": res.get("fusion", ""),
                    "k": res.get("k", ""),
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
                append_row_realtime(out_csv, row, header_order=header)
            except Exception as e:
                print(f"[{kidx}/{total}] ERROR {a}->{b}: {e}")

        append_summary_rows(out_csv, header_order=header)
        print(f"[DONE] Wrote results and summary rows to {out_csv}")
        return

    if not (args.user1 and args.user2 and args.out_dir):
        raise SystemExit("For single-pair mode please provide --user1, --user2 and --out_dir")

    res = run_one_pair(args, args.user1, args.user2, write_artifacts=True, out_dir=args.out_dir)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
