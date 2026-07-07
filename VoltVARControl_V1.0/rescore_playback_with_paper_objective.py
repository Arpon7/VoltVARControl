# rescore_playback_with_paper_objective.py
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def _paper_objective(q: np.ndarray,
                     v: np.ndarray,
                     eta_over_sbar: np.ndarray,
                     X_inv: np.ndarray | None = None,
                     normalize_trace: bool = False) -> float:
    """
    F(q, v) = 0.5 * (v-1)^T X^{-1} (v-1) + 0.5 * sum_i (eta_i/sbar_i) * q_i^2
    If X_inv is None, we use the unweighted 0.5 * ||v-1||_2^2 instead.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    c = np.asarray(eta_over_sbar, dtype=float).reshape(-1)
    e = (v - 1.0).reshape(-1, 1)  # (n,1)

    if X_inv is None:
        term_v = 0.5 * float(np.dot(e.ravel(), e.ravel()))
    else:
        X_inv = np.asarray(X_inv, dtype=float)
        if normalize_trace:
            tr = float(np.trace(X_inv))
            if tr > 0:
                X_inv = X_inv / tr
        term_v = 0.5 * float((e.T @ X_inv @ e).item())

    term_q = 0.5 * float(np.sum(c * (q ** 2)))
    return term_v + term_q

def _load_xinv(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"X_inv not found: {p}")
    if p.suffix.lower() in (".npy",):
        return np.load(p)
    # assume CSV/TSV
    return np.loadtxt(p, delimiter=",")  # change if you use TSV

def rescore(csv_in: str,
            csv_out: str,
            eta_over_sbar: float = 0.01,
            xinv_path: str | None = None,
            normalize_trace: bool = False,
            v_prefix: str = "v_",
            q_prefix: str = "q_"):
    """
    Reads a DOE playback CSV with columns like: time, v_*, q_*
    Writes a copy with an added column:
      - F_paper_like   (if no X_inv provided)
      - F_paper_weighted (if X_inv provided)
    """
    df = pd.read_csv(csv_in)
    v_cols = [c for c in df.columns if c.lower().startswith(v_prefix)]
    q_cols = [c for c in df.columns if c.lower().startswith(q_prefix)]
    if not v_cols or not q_cols:
        raise ValueError(f"Could not find voltage ('{v_prefix}*') and reactive ('{q_prefix}*') columns in {csv_in}")

    v = df[v_cols].to_numpy(dtype=float)
    q = df[q_cols].to_numpy(dtype=float)
    n = q.shape[1]
    c = eta_over_sbar * np.ones(n, dtype=float)

    X_inv = _load_xinv(xinv_path)
    col_name = "F_paper_weighted" if X_inv is not None else "F_paper_like"

    F_vals = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        F_vals[i] = _paper_objective(q[i], v[i], c, X_inv=X_inv, normalize_trace=normalize_trace)

    df[col_name] = F_vals
    df.to_csv(csv_out, index=False)
    print(json.dumps({"rows": int(len(df)), "written": csv_out, "column": col_name}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input DOE CSV (trajectory_doe_ieee13_stable.csv)")
    ap.add_argument("--out_csv", required=True, help="Output rescored CSV")
    ap.add_argument("--eta_over_sbar", type=float, default=0.01, help="Per-bus weight factor")
    ap.add_argument("--xinv", type=str, default=None, help="Optional path to X_inv (.npy or .csv). If absent, uses unweighted ||v-1||^2/2.")
    ap.add_argument("--normalize_trace", action="store_true", help="Normalize X_inv by its trace before scoring (paper-style comparability)")
    ap.add_argument("--v_prefix", type=str, default="v_", help="Prefix of voltage columns (default: v_)")
    ap.add_argument("--q_prefix", type=str, default="q_", help="Prefix of reactive columns (default: q_)")
    args = ap.parse_args()

    rescore(args.in_csv, args.out_csv,
            eta_over_sbar=args.eta_over_sbar,
            xinv_path=args.xinv,
            normalize_trace=bool(args.normalize_trace),
            v_prefix=args.v_prefix,
            q_prefix=args.q_prefix)
