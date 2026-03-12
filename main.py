"""
=============================================================================
5004CMD – BTS Trips Analysis
main.py – Master runner

Runs all questions in three modes:
  1. Sequential   – no parallelism
  2. Parallel 10  – Dask distributed, 10 workers
  3. Parallel 20  – Dask distributed, 20 workers

Datasets:
  big   = Trips_By_Distance.csv  (~100k rows, daily data)
  small = Trips_Full_Data.csv    (32-week summary)

HOW TO RUN:
  python3 main.py
=============================================================================
"""

import time
import dask.dataframe as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dask.distributed import Client

from data_classification import data_classification
from question_a import question_a
from question_b import question_b
from question_c import question_c
from question_d import question_d

# LOAD DATASETS


def load_datasets():
    """Loads both CSVs as Dask DataFrames."""
    print("  Loading Trips_By_Distance.csv  (~100k rows) ...")
    big = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big["Date"] = dd.to_datetime(big["Date"])

    print("  Loading Trips_Full_Data.csv  (32-week summary) ...")
    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    return big, small


# RUN ALL QUESTIONS


def run_all(big, small, label):
    """Runs classification + Q-A through Q-D and returns elapsed time."""
    t_start = time.time()
    data_classification(big, small, label=label)
    question_a(big, small, label=label)
    question_b(big, small, label=label)
    question_c(big, small, label=label)
    question_d(big, small, label=label)
    elapsed = time.time() - t_start
    print(f"\n  Total time [{label}]: {elapsed:.2f} seconds")
    return elapsed

# TIMING COMPARISON PLOT


def plot_timing(timing_results):
    """Bar chart comparing execution times across all three modes."""
    labels = list(timing_results.keys())
    times  = list(timing_results.values())
    colors = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, times, color=colors[:len(labels)],
                  edgecolor="white", width=0.5)
    ax.set_ylabel("Total Execution Time (seconds)")
    ax.set_title("Sequential vs Parallel Processing – Total Execution Time")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{t:.1f}s",
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    seq_time = timing_results.get("Sequential", 1)
    for bar, lbl, t in zip(bars[1:], labels[1:], times[1:]):
        speedup = seq_time / t if t > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{speedup:.2f}x\nfaster",
                ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("timing_comparison.png", dpi=150)
    plt.close()
    print("\n  → Timing plot saved: timing_comparison.png")



# MAIN


def main():
    print("\n" + "=" * 70)
    print("  5004CMD – BTS Trips by Distance Analysis")
    print("  Sequential vs Parallel Processing Comparison")
    print("=" * 70)

    timing = {}

    # ── MODE 1: SEQUENTIAL 
    print("\n\n" + "-" * 70)
    print("  MODE 1: SEQUENTIAL PROCESSING")
    print("-" * 70)
    print("\n--- Loading datasets ---")
    big_seq, small_seq = load_datasets()

    t0 = time.time()
    data_classification(big_seq, small_seq, label="Sequential")
    question_a(big_seq, small_seq, label="Sequential")
    question_b(big_seq, small_seq, label="Sequential")
    question_c(big_seq, small_seq, label="Sequential")
    question_d(big_seq, small_seq, label="Sequential")
    timing["Sequential"] = time.time() - t0
    print(f"\n  Total time [Sequential]: {timing['Sequential']:.2f} seconds")

    # ── MODE 2: PARALLEL – 10 WORKERS
    print("\n\n" + "-" * 70)
    print("  MODE 2: PARALLEL – 10 WORKERS  (Dask Distributed)")
    print("-" * 70)
    client10 = Client(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    print("\n--- Loading datasets ---")
    big10, small10 = load_datasets()
    timing["Parallel (10 workers)"] = run_all(big10, small10,
                                              label="Parallel 10 workers (2 cores)")
    client10.close()

    # ── MODE 3: PARALLEL – 20 WORKERS ────────────────────────────────────────
    print("\n\n" + "-" * 70)
    print("  MODE 3: PARALLEL – 20 WORKERS  (Dask Distributed)")
    print("-" * 70)
    client20 = Client(n_workers=4, threads_per_worker=1, memory_limit="512MB")
    print("\n--- Loading datasets ---")
    big20, small20 = load_datasets()
    timing["Parallel (20 workers)"] = run_all(big20, small20,
                                              label="Parallel 20 workers (4 cores)")
    client20.close()

    # ── TIMING SUMMARY
    print("\n\n" + "=" * 70)
    print("  FINAL TIMING SUMMARY")
    print("=" * 70)
    print(f"  {'Mode':<30s} {'Time (s)':>10s}   {'Speedup':>10s}")
    print("  " + "-" * 55)
    for mode, t in timing.items():
        speedup = timing["Sequential"] / t if t > 0 else 0
        print(f"  {mode:<30s} {t:>10.2f}s  {speedup:>9.2f}x")

    plot_timing(timing)

    print("\n" + "=" * 70)
    print("  All done! Check this folder for all output PNG plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()