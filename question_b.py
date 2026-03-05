
import time
import dask.dataframe as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from dask.distributed import Client


def question_b(big_dataset, small_dataset, label="Sequential"):
    print(f"\n{'='*60}")
    print(f"  QUESTION B  [{label}]")
    print(f"{'='*60}")

    # ── Step 1: Filter to National level, keep only needed columns
    national = big_dataset[big_dataset["Level"] == "National"]
    national = national[["Date", "Number of Trips 10-25", "Number of Trips 50-100"]]
    national_pd = national.compute()
    national_pd["Date"] = pd.to_datetime(national_pd["Date"])

    # ── Step 2: Create the four filtered sets 
    set1 = national_pd[national_pd["Number of Trips 10-25"] > 10_000_000]
    set2 = national_pd[national_pd["Number of Trips 50-100"] > 10_000_000]
    set3 = national_pd[
        (national_pd["Number of Trips 10-25"] > 10_000_000) &
        (national_pd["Number of Trips 50-100"] > 10_000_000)
    ]
    set4 = national_pd[
        (national_pd["Number of Trips 10-25"] > 10_000_000) &
        (national_pd["Number of Trips 50-100"] < 10_000_000)
    ]

    print(f"\n  Set 1 – Dates with >10M people doing 10-25 trips : {len(set1):,} dates")
    print(f"  Set 2 – Dates with >10M people doing 50-100 trips: {len(set2):,} dates")
    print(f"  Set 3 – Both conditions met                       : {len(set3):,} dates")
    print(f"  Set 4 – Only 10-25 condition met                  : {len(set4):,} dates")

    # ── Plot 1: Side-by-side scatterplots 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    

    axes[0].scatter(set1["Date"], set1["Number of Trips 10-25"],
                    color="royalblue", alpha=0.6, s=20)
    axes[0].set_title("Dates: >10M People\nMaking 10-25 Trips", fontsize=11)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Number of Trips 10-25")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].scatter(set2["Date"], set2["Number of Trips 50-100"],
                    color="tomato", alpha=0.6, s=20)
    axes[1].set_title("Dates: >10M People\nMaking 50-100 Trips", fontsize=11)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Trips 50-100")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"Comparison of High-Volume Trip Days  [{label}]",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname1 = f"qb_scatterplot_sidebyside_{label.replace(' ', '_')}.png"
    plt.savefig(fname1, dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {fname1}")

    # ── Plot 2: Overlay scatterplot 
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(set1["Date"], set1["Number of Trips 10-25"],
               color="royalblue", alpha=0.5, s=20, label=">10M people – 10-25 trips")
    ax.scatter(set2["Date"], set2["Number of Trips 50-100"],
               color="tomato", alpha=0.5, s=20, label=">10M people – 50-100 trips")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Trips")
    ax.set_title(f"Q-B: >10M People – 10-25 Trips vs 50-100 Trips Overlay  [{label}]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    fname2 = f"qb_scatterplot_overlay_{label.replace(' ', '_')}.png"
    plt.savefig(fname2, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname2}")

    # ── Plot 3: Set 3 and Set 4
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    

    axes[0].scatter(set3["Date"], set3["Number of Trips 10-25"],
                    color="purple", alpha=0.6, s=20)
    axes[0].set_title("Set 3: Both Conditions Met", fontsize=11)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Number of Trips 10-25")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].scatter(set4["Date"], set4["Number of Trips 10-25"],
                    color="darkorange", alpha=0.6, s=20)
    axes[1].set_title("Set 4: Only 10-25 Condition Met", fontsize=11)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Trips 10-25")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"Q-B: Set 3 vs Set 4  [{label}]", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname3 = f"qb_set3_set4_{label.replace(' ', '_')}.png"
    plt.savefig(fname3, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname3}")

    return set1, set2, set3, set4

# STANDALONE RUN

if __name__ == "__main__":

    print("\n--- Loading datasets ---")
    big   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big["Date"] = dd.to_datetime(big["Date"])
    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    print("\n>>> Running SEQUENTIAL...")
    t0 = time.time()
    question_b(big, small, label="Sequential")
    seq_time = time.time() - t0
    print(f"\n  Sequential time: {seq_time:.2f}s")

    print("\n>>> Running PARALLEL – 10 workers...")
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    big10   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big10["Date"] = dd.to_datetime(big10["Date"])
    small10 = dd.read_csv("Trips_Full_Data.csv")
    small10["Date"] = dd.to_datetime(small10["Date"])
    t1 = time.time()
    question_b(big10, small10, label="Parallel 10 workers (2 cores)")
    par10_time = time.time() - t1
    print(f"\n  Parallel (10 workers) time: {par10_time:.2f}s")
    client.close()

    print("\n>>> Running PARALLEL – 20 workers...")
    client = Client(n_workers=4, threads_per_worker=1, memory_limit="512MB")
    big20   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big20["Date"] = dd.to_datetime(big20["Date"])
    small20 = dd.read_csv("Trips_Full_Data.csv")
    small20["Date"] = dd.to_datetime(small20["Date"])
    t2 = time.time()
    question_b(big20, small20, label="Parallel 20 workers (4 cores)")
    par20_time = time.time() - t2
    print(f"\n  Parallel (20 workers) time: {par20_time:.2f}s")
    client.close()

    print("\n--- Question B Timing Summary ---")
    print(f"  Sequential          : {seq_time:.2f}s")
    print(f"  Parallel 10 workers : {par10_time:.2f}s  (speedup: {seq_time/par10_time:.2f}x)")
    print(f"  Parallel 20 workers : {par20_time:.2f}s  (speedup: {seq_time/par20_time:.2f}x)")