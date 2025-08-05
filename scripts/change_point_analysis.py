# scripts/change_point_analysis.py
"""
Run Bayesian single changepoint detection for Brent oil prices (Task 2).
Saves:
- reports/task2_summary.json
- reports/task2_nearby_events.csv (if key_events.csv provided)
- reports/task2_price_window_stats.json
- reports/fig_price_with_cp.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az
import pymc3 as pm

from change_point_model import fit_single_changepoint, summarize_change


def load_prices(path_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # adjust these names if needed
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Price' in the price CSV.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # log returns
    df["LogPrice"] = np.log(df["Price"])
    df["LogReturn"] = df["LogPrice"].diff()
    df = df.dropna().reset_index(drop=True)
    return df


def find_nearby_events(events_csv: Path, change_date: pd.Timestamp, window_days: int = 60) -> pd.DataFrame:
    if not events_csv.exists():
        return pd.DataFrame()
    ev = pd.read_csv(events_csv)
    # expected column name 'start_date' (adjust if needed)
    if "start_date" not in ev.columns:
        raise ValueError("Expected column 'start_date' in key_events.csv.")
    ev["start_date"] = pd.to_datetime(ev["start_date"])
    mask = (ev["start_date"] >= change_date - pd.Timedelta(days=window_days)) & (
        ev["start_date"] <= change_date + pd.Timedelta(days=window_days)
    )
    return ev.loc[mask].sort_values("start_date").reset_index(drop=True)


def price_window_stats(df: pd.DataFrame, change_date: pd.Timestamp, window_days: int = 90):
    before_mask = (df["Date"] >= change_date - pd.Timedelta(days=window_days)) & (df["Date"] < change_date)
    after_mask = (df["Date"] >= change_date) & (df["Date"] <= change_date + pd.Timedelta(days=window_days))
    mean_price_before = df.loc[before_mask, "Price"].mean()
    mean_price_after = df.loc[after_mask, "Price"].mean()
    pct_change = 100 * (mean_price_after - mean_price_before) / mean_price_before
    return {
        "window_days": window_days,
        "mean_price_before": None if pd.isna(mean_price_before) else float(mean_price_before),
        "mean_price_after": None if pd.isna(mean_price_after) else float(mean_price_after),
        "pct_change": None if pd.isna(pct_change) else float(pct_change),
    }


def plot_price_with_cp(df: pd.DataFrame, change_date: pd.Timestamp, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Price"])
    plt.axvline(pd.to_datetime(change_date), linestyle="--")
    plt.title("Brent Price with Detected Change Point")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task 2: Bayesian changepoint detection on Brent oil prices.")
    parser.add_argument("--prices", type=str, default="data/cleaned_oil_prices.csv", help="Path to prices CSV.")
    parser.add_argument("--events", type=str, default="data/key_events.csv", help="Path to key events CSV (optional).")
    parser.add_argument("--outdir", type=str, default="reports", help="Directory to write outputs.")
    parser.add_argument("--draws", type=int, default=2000, help="MCMC draws.")
    parser.add_argument("--tune", type=int, default=1000, help="MCMC tuning steps.")
    parser.add_argument("--target_accept", type=float, default=0.9, help="NUTS target_accept.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    prices_path = Path(args.prices)
    events_path = Path(args.events)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load & prepare data
    df = load_prices(prices_path)

    # 2) Fit model on log returns
    y = df["LogReturn"].values
    trace = fit_single_changepoint(
        y, draws=args.draws, tune=args.tune, target_accept=args.target_accept, seed=args.seed
    )

    # 3) Summaries
    summary_df = az.summary(trace)
    # extract change info + probs
    change_info = summarize_change(trace, df["Date"], df["LogReturn"])

    # 4) Save numeric summaries
    (outdir / "task2_summary.json").write_text(json.dumps(change_info, indent=2))

    # 5) Associate with events (optional)
    change_date = pd.to_datetime(change_info["change_date"])
    nearby = pd.DataFrame()
    if events_path.exists():
        nearby = find_nearby_events(events_path, change_date, window_days=60)
        if not nearby.empty:
            nearby.to_csv(outdir / "task2_nearby_events.csv", index=False)

    # 6) Price window stats around CP
    window_stats = price_window_stats(df, change_date, window_days=90)
    (outdir / "task2_price_window_stats.json").write_text(json.dumps(window_stats, indent=2))

    # 7) Plot price + change point
    plot_price_with_cp(df, change_date, out_path=outdir / "fig_price_with_cp.png")

    # 8) Print a short, kid-friendly summary to console
    print("\n=== Task 2 Results ===")
    print(f"Most likely change point date: {change_info['change_date']}")
    print(f"P(mu_after > mu_before): {change_info['P(mu_after>mu_before)']:.3f}")
    print(f"P(vol_after > vol_before): {change_info['P(vol_after>vol_before)']:.3f}")
    if window_stats["pct_change"] is not None:
        print(
            f"Average price +/-{window_stats['window_days']}d — Before: ${window_stats['mean_price_before']:.2f}, "
            f"After: ${window_stats['mean_price_after']:.2f}, Change: {window_stats['pct_change']:.1f}%"
        )
    if not nearby.empty:
        print(f"Found {len(nearby)} event(s) near change date. See: {outdir/'task2_nearby_events.csv'}")

    # 9) Save full ArviZ summary table
    summary_csv = outdir / "task2_posterior_summary.csv"
    summary_df.to_csv(summary_csv)
    print(f"Posterior summary saved to: {summary_csv}")
    print(f"Figure saved to: {outdir/'fig_price_with_cp.png'}")
    print(f"Core summary saved to: {outdir/'task2_summary.json'}")
    print(f"Window stats saved to: {outdir/'task2_price_window_stats.json'}")


if __name__ == "__main__":
    main()
