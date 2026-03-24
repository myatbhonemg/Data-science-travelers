
import time
import dask.dataframe as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dask.distributed import Client


def question_d(big_dataset, small_dataset, label="Sequential"):
    """

    Parameters:
        big_dataset   – Dask DataFrame (Trips_By_Distance.csv, ~100k rows)
        small_dataset – Dask DataFrame (not heavily used here)
        label         – string label for plot titles and filenames
    
    """
    print(f"\n{'='*60}")
    print(f"  QUESTION D  [{label}]")
    print(f"{'='*60}")

    # ── Step 1: Filter to National level 
    national = big_dataset[big_dataset["Level"] == "National"]
    national = national.drop(
        columns=["State FIPS", "State Postal Code", "County FIPS", "County Name"],
        errors="ignore"
    )

    # ── Step 2: Distance-trip columns
    def classify_distance(band):
      short     = ['<1', '1-3', '3-5']
      medium    = ['5-10', '10-25']
      long_     = ['25-50', '50-100']
      very_long = ['100-250', '250-500', '>=500']
      if band in short:     return 'Short (<5 miles)'
      if band in medium:    return 'Medium (5-25 miles)'
      if band in long_:     return 'Long (25-100 miles)'
      return 'Very Long (>100 miles)'
 
    distance_columns = [
        "Number of Trips <1", "Number of Trips 1-3", "Number of Trips 3-5",
        "Number of Trips 5-10", "Number of Trips 10-25", "Number of Trips 25-50",
        "Number of Trips 50-100", "Number of Trips 100-250",
        "Number of Trips 250-500", "Number of Trips >=500"
    ]
    available    = [c for c in distance_columns if c in national.columns]
    mean_trips   = national[available].mean().compute()
    short_labels = [c.replace("Number of Trips ", "") for c in mean_trips.index]

    print("\n  Mean travellers per distance band:")
    for lbl, val in zip(short_labels, mean_trips.values):
        print(f"    {lbl:<12s}: {val:>15,.0f}")

    colors = plt.cm.tab10.colors[:len(mean_trips)]

    # ── Plot 1: Bar chart 
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(short_labels, mean_trips.values, color=colors, edgecolor="white")
    ax.set_xlabel("Distance / Trip Band (miles)")
    ax.set_ylabel("Mean Number of Travellers")
    ax.set_title(f"Q-D: Mean Number of Travellers by Distance Band  [{label}]")
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height() / 1e6:.1f}M",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fname1 = f"qd_bar_{label.replace(' ', '_')}.png"
    plt.savefig(fname1, dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {fname1}")

    # ── Plot 2: Pie chart 
    fig, ax = plt.subplots(figsize=(11, 9))
    wedges, texts, autotexts = ax.pie(
        mean_trips.values,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.legend(
        wedges, short_labels,
        title="Distance Band (miles)",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9
    )
    ax.set_title(f"Q-D: Proportional Share of Travellers by Distance Band  [{label}]")
    plt.tight_layout()
    fname2 = f"qd_pie_{label.replace(' ', '_')}.png"
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Plot saved: {fname2}")

    # ── Plot 3: Weekly trend 
    trend = (
        national.groupby("Week")[available]
        .mean()
        .compute()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    for col, color in zip(available[:5], colors[:5]):
        band_label = col.replace("Number of Trips ", "")
        ax.plot(trend["Week"], trend[col], marker="o", markersize=4,
                label=f"{band_label} miles", color=color)
    ax.set_xlabel("Week Number")
    ax.set_ylabel("Mean Number of Trips")
    ax.set_title(f"Q-D: Weekly Trend – Travellers by Distance Band  [{label}]")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fname3 = f"qd_trend_{label.replace(' ', '_')}.png"
    plt.savefig(fname3, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname3}")

    return mean_trips


# STANDALONE RUN
if __name__ == "__main__":

    print("\n--- Loading datasets ---")
    big   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big["Date"] = dd.to_datetime(big["Date"])
    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    print("\n>>> Running SEQUENTIAL...")
    t0 = time.time()
    question_d(big, small, label="Sequential")
    seq_time = time.time() - t0
    print(f"\n  Sequential time: {seq_time:.2f}s")

    print("\n>>> Running PARALLEL – 10 workers...")
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    big10   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big10["Date"] = dd.to_datetime(big10["Date"])
    small10 = dd.read_csv("Trips_Full_Data.csv")
    small10["Date"] = dd.to_datetime(small10["Date"])
    t1 = time.time()
    question_d(big10, small10, label="Parallel 10 workers (2 cores)")
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
    question_d(big20, small20, label="Parallel 20 workers (4 cores)")
    par20_time = time.time() - t2
    print(f"\n  Parallel (20 workers) time: {par20_time:.2f}s")
    client.close()

    print("\n--- Question D Timing Summary ---")
    print(f"  Sequential          : {seq_time:.2f}s")
    print(f"  Parallel 10 workers : {par10_time:.2f}s  (speedup: {seq_time/par10_time:.2f}x)")
    print(f"  Parallel 20 workers : {par20_time:.2f}s  (speedup: {seq_time/par20_time:.2f}x)")

    from dask.distributed import Client
    import time

    # Parallel 10 workers configuration
    client = Client(n_workers=2, threads_per_worker=1, memory_limit='512MB')
    t_start = time.time()
    
    elapsed = time.time() - t_start
    print(f'Parallel (10 workers) time: {elapsed:.2f}s')
    client.close()


