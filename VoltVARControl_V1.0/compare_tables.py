# compare_tables.py — robust CLI + tolerant to old/new metric keys
from __future__ import annotations
import argparse, json, pandas as pd, numpy as np
from pathlib import Path

# Paper defaults (their tables report negative numbers = sum of rewards)
PAPER_DEFAULT = {
    "13":  {"mean_recovery_time_s": 2.60, "mean_transient_reward": -6.76,  "mean_ss_objective": -0.11},
    "123": {"mean_recovery_time_s":12.08, "mean_transient_reward":-333.03, "mean_ss_objective": -5.95},
}

def read_text_any_encoding(path: str) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "mbcs", "cp1252"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")

def extract_last_json(text: str) -> str:
    # brace-balanced extraction of the LAST {...} blob
    stack = 0; start = -1; last = None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0: start = i
            stack += 1
        elif ch == "}":
            if stack:
                stack -= 1
                if stack == 0 and start != -1:
                    last = text[start:i+1]
    if last is None:
        raise ValueError("Could not find a complete JSON object.")
    return last

def load_json_lax(path: str):
    txt = read_text_any_encoding(path)
    blob = extract_last_json(txt)
    return json.loads(blob)

def fmt(x, n=4):
    try:
        return f"{float(x):.{n}g}"
    except Exception:
        return str(x)

def pick_transient(d: dict):
    # Prefer reward (paper sign), else convert cost -> reward, then legacy
    if d is None: return None
    if "mean_transient_reward" in d: return float(d["mean_transient_reward"])
    if "mean_transient_cost"   in d: return -float(d["mean_transient_cost"])
    if "mean_transient"        in d: return float(d["mean_transient"])  # legacy key
    return None

def pick_ss(d: dict):
    # Prefer our ΔF if present; else paper F(q)
    if d is None: return None
    if "mean_deltaF" in d:           return float(d["mean_deltaF"])
    if "mean_ss_objective" in d:     return float(d["mean_ss_objective"])
    return None

def summarize_doe(csv_path: str | None):
    if not csv_path: return None
    df = pd.read_csv(csv_path)
    out = {}
    # Try to find a paper-like F column
    fcol = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("paper_f","f_paper_like","f_paper","steady_state_f","f_q","fq"): fcol = c; break
        if ("paper" in lc) and ("f" in lc): fcol = c; break
    if fcol:
        out["mean_F"] = float(np.mean(df[fcol].values))
        out["median_F"] = float(np.median(df[fcol].values))
    vcols = [c for c in df.columns if c.lower().startswith("v_")]
    if vcols:
        v = df[vcols].to_numpy(dtype=float)
        inband = (v >= 0.95) & (v <= 1.05)
        out["pct_time_in_band"] = float(np.mean(np.all(inband, axis=1)) * 100.0)
    return out or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_13",  required=True, help="Your metrics JSON for IEEE-13.")
    ap.add_argument("--metrics_123", required=False, help="Your metrics JSON for IEEE-123 (optional).")
    ap.add_argument("--base",   default=None, help="Optional base.json (overrides --paper).")
    ap.add_argument("--paper",  action="store_true", help="Use hard-coded paper defaults for base.")
    ap.add_argument("--doe13",  default=None, help="Optional DOE/scored CSV for IEEE-13.")
    ap.add_argument("--doe123", default=None, help="Optional DOE/scored CSV for IEEE-123.")
    ap.add_argument("--out",    default="comparison_table.csv", help="Output CSV path.")
    args = ap.parse_args()

    # Load your metrics
    y13 = load_json_lax(args.metrics_13) if args.metrics_13 else None
    y123 = load_json_lax(args.metrics_123) if args.metrics_123 else None

    # Decide base (paper) side
    if args.base:
        b = load_json_lax(args.base)
        # normalize base keys to reward sign if needed
        for tag in ("13","123"):
            if tag in b and isinstance(b[tag], dict) and "mean_transient_cost" in b[tag]:
                b[tag]["mean_transient_reward"] = -float(b[tag]["mean_transient_cost"])
    else:
        b = PAPER_DEFAULT if args.paper or not args.base else PAPER_DEFAULT

    rows = []

    def add_case(tag: str, name: str, yours: dict | None):
        base = b.get(tag)
        if (yours is None) and (base is None):
            return
        rows.extend([
            {"Case": name, "Metric": "Mean recovery time (s)",
             "Base (paper/TASRL)": None if base is None else base.get("mean_recovery_time_s"),
             "Yours": None if yours is None else yours.get("mean_recovery_time_s")},
            {"Case": name, "Metric": "Mean transient (sum of rewards)",
             "Base (paper/TASRL)": None if base is None else base.get("mean_transient_reward"),
             "Yours": pick_transient(yours)},
            {"Case": name, "Metric": "Steady-state (ΔF if yours; paper F otherwise)",
             "Base (paper/TASRL)": None if base is None else base.get("mean_ss_objective"),
             "Yours": pick_ss(yours)},
        ])

    if y13 is not None or ("13" in b):
        add_case("13", "IEEE-13", y13)
    if y123 is not None or ("123" in b):
        add_case("123", "IEEE-123", y123)

    df = pd.DataFrame(rows)
    # Pretty print to stdout
    printed_cases = df["Case"].dropna().unique().tolist()
    if len(printed_cases):
        print("\n=== Apples-to-apples (paper protocol) ===")
        for name in printed_cases:
            sub = df[df["Case"] == name]
            print(f"\n{name}")
            for _, r in sub.iterrows():
                print(f"  {r['Metric']:<44}  Base: {fmt(r['Base (paper/TASRL)'])}   Yours: {fmt(r['Yours'])}")

    # Optional DOE summaries
    blocks = []
    s13 = summarize_doe(args.doe13) if args.doe13 else None
    if s13:  blocks.append(("IEEE-13 (DOE)", s13))
    s123 = summarize_doe(args.doe123) if args.doe123 else None
    if s123: blocks.append(("IEEE-123 (DOE)", s123))
    if blocks:
        print("\n=== Real-data DOE (complementary) ===")
        for name, s in blocks:
            print(f"\n{name}")
            for k, v in s.items():
                print(f"  {k.replace('_',' ').title():<22} {fmt(v)}")

    # Save
    out_path = args.out
    df.to_csv(out_path, index=False)
    print(f"\nSaved table to {out_path}")

if __name__ == "__main__":
    main()
