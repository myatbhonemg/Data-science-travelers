import time
import dask.dataframe as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dask.distributed import Client


def question_a(big_dataset, small_dataset, label="Sequential"):

    print(f"\n{'='*60}")
    print(f"  QUESTION A  [{label}]")
    print(f"{'='*60}")

    # ── Step 1: Filter big dataset to National level only
    national = big_dataset[big_dataset["Level"] == "National"]
    national = national.drop(
        columns=["State FIPS", "State Postal Code", "County FIPS", "County Name"],
        errors="ignore"
    )

    # ── Step 2: Weekly average – population staying at home 
    weekly_home = (
        national
        .groupby("Week")["Population Staying at Home"]
        .mean()
        .reset_index()
        .compute()
    )
    weekly_home.columns = ["Week", "Avg Population Staying at Home"]

    print("\n  Average population staying at home per week:")
    print(weekly_home.to_string(index=False))

    # Plot 1: Bar chart – staying at home per week
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        weekly_home["Week"].astype(str),
        weekly_home["Avg Population Staying at Home"],
        color="steelblue", edgecolor="white"
    )
    ax.set_xlabel("Week Number")
    ax.set_ylabel("Average Population Staying at Home")
    ax.set_title(f"Q-A: Average Population Staying at Home per Week  [{label}]")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fname1 = f"qa_staying_home_{label.replace(' ', '_')}.png"
    plt.savefig(fname1, dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {fname1}")

    # ── Step 3: How far do people travel when NOT staying home? ───────────────
    distance_columns = [
        "Number of Trips <1", "Number of Trips 1-3", "Number of Trips 3-5",
        "Number of Trips 5-10", "Number of Trips 10-25", "Number of Trips 25-50",
        "Number of Trips 50-100", "Number of Trips 100-250",
        "Number of Trips 250-500", "Number of Trips >=500"
    ]
    available = [c for c in distance_columns if c in national.columns]
    mean_trips = national[available].mean().compute()
    short_labels = [c.replace("Number of Trips ", "") for c in mean_trips.index]

    # ── Plot 2: Bar chart – mean trips by distance range 
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(short_labels, mean_trips.values, color="coral", edgecolor="white")
    ax.set_xlabel("Distance Range (miles)")
    ax.set_ylabel("Mean Number of Trips")
    ax.set_title(f"Q-A: How Far People Travel When Not Staying Home  [{label}]")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fname2 = f"qa_distance_travelled_{label.replace(' ', '_')}.png"
    plt.savefig(fname2, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname2}")

    # ── Step 4: Small dataset – weekly distance breakdown 
    small_dist_cols = [
        "Trips <1 Mile", "Trips 1-3 Miles", "Trips 3-5 Miles",
        "Trips 5-10 Miles", "Trips 10-25 Miles", "Trips 25-50 Miles",
        "Trips 50-100 Miles", "Trips 100-250 Miles",
        "Trips 250-500 Miles", "Trips 500+ Miles"
    ]
    avail_small = [c for c in small_dist_cols if c in small_dataset.columns]
    mean_small  = small_dataset[avail_small].mean().compute()
    short_small = [c.replace(" Miles", "").replace("Trips ", "") for c in mean_small.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(short_small, mean_small.values, color="mediumseagreen", edgecolor="white")
    ax.set_xlabel("Distance Range")
    ax.set_ylabel("Mean Number of Trips")
    ax.set_title(f"Q-A: Mean Trips by Distance – 32-Week Summary  [{label}]")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fname3 = f"qa_small_dataset_distance_{label.replace(' ', '_')}.png"
    plt.savefig(fname3, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname3}")

    return weekly_home, mean_trips


# STANDALONE RUN

if __name__ == "__main__":

    print("\n--- Loading datasets ---")
    big   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big["Date"] = dd.to_datetime(big["Date"])

    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    # Sequential
    print("\n>>> Running SEQUENTIAL...")
    t0 = time.time()
    question_a(big, small, label="Sequential")
    seq_time = time.time() - t0
    print(f"\n  Sequential time: {seq_time:.2f}s")

    # Parallel – 10 workers
    print("\n>>> Running PARALLEL – 10 workers...")
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    big10   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big10["Date"] = dd.to_datetime(big10["Date"])
    small10 = dd.read_csv("Trips_Full_Data.csv")
    small10["Date"] = dd.to_datetime(small10["Date"])
    t1 = time.time()
    question_a(big10, small10, label="Parallel 10 workers (2 cores)")
    par10_time = time.time() - t1
    print(f"\n  Parallel (10 workers) time: {par10_time:.2f}s")
    client.close()

    # Parallel – 20 workers
    print("\n>>> Running PARALLEL – 20 workers...")
    client = Client(n_workers=4, threads_per_worker=1, memory_limit="512MB")
    big20   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big20["Date"] = dd.to_datetime(big20["Date"])
    small20 = dd.read_csv("Trips_Full_Data.csv")
    small20["Date"] = dd.to_datetime(small20["Date"])
    t2 = time.time()
    question_a(big20, small20, label="Parallel 20 workers (4 cores)")
    par20_time = time.time() - t2
    print(f"\n  Parallel (20 workers) time: {par20_time:.2f}s")
    client.close()

    print("\n--- Question A Timing Summary ---")
    print(f"  Sequential          : {seq_time:.2f}s")
    print(f"  Parallel 10 workers : {par10_time:.2f}s  (speedup: {seq_time/par10_time:.2f}x)")
    print(f"  Parallel 20 workers : {par20_time:.2f}s  (speedup: {seq_time/par20_time:.2f}x)")