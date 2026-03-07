import time
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dask.distributed import Client
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def classify_distance(col_name):
    """Returns a distance category for a given column name."""
    if any(x in col_name for x in ["<1", "1-3", "3-5"]):
        return "Short (< 5 miles)"
    elif any(x in col_name for x in ["5-10", "10-25"]):
        return "Medium (5-25 miles)"
    elif any(x in col_name for x in ["25-50", "50-100"]):
        return "Long (25-100 miles)"
    elif any(x in col_name for x in ["100-250", "250-500", ">=500"]):
        return "Very Long (100+ miles)"
    return "Unknown"


def classify_activity_level(pop_not_home):
    """Classifies a day's travel activity level."""
    if pop_not_home < 50_000_000:
        return "Low Activity"
    elif pop_not_home < 100_000_000:
        return "Moderate Activity"
    elif pop_not_home < 150_000_000:
        return "High Activity"
    else:
        return "Very High Activity"


def data_classification(big_dataset, small_dataset, label="Sequential"):
    """
        big_dataset   – Dask DataFrame (Trips_By_Distance.csv, ~100k rows)
        small_dataset – Dask DataFrame (Trips_Full_Data.csv, 32-week summary)      
    """

    print(f"\n{'='*60}")
    print(f"  DATA CLASSIFICATION  [{label}]")
    print(f"{'='*60}")

    # ── Step 1: Filter to National level 
    national = big_dataset[big_dataset["Level"] == "National"]
    national = national.drop(
        columns=["State FIPS", "State Postal Code", "County FIPS", "County Name"],
        errors="ignore"
    )
    national_pd = national.compute()
    national_pd["Date"] = pd.to_datetime(national_pd["Date"])

    # ── Step 2: Rule-based distance classification 
    distance_columns = [
        "Number of Trips <1", "Number of Trips 1-3", "Number of Trips 3-5",
        "Number of Trips 5-10", "Number of Trips 10-25", "Number of Trips 25-50",
        "Number of Trips 50-100", "Number of Trips 100-250",
        "Number of Trips 250-500", "Number of Trips >=500"
    ]
    available = [c for c in distance_columns if c in national_pd.columns]

    classification_summary = pd.DataFrame({
        "Distance Band": available,
        "Category":      [classify_distance(c) for c in available],
        "Mean Trips":    [national_pd[c].mean() for c in available]
    })

    category_totals = (
        classification_summary
        .groupby("Category")["Mean Trips"]
        .sum()
        .reset_index()
        .sort_values("Mean Trips", ascending=False)
    )

    print("\n  Rule-Based Distance Classification:")
    print(classification_summary.to_string(index=False))
    print("\n  Totals by Category:")
    print(category_totals.to_string(index=False))

    # ── Plot 1: Bar chart – categories 
    category_order = ["Short (< 5 miles)", "Medium (5-25 miles)",
                      "Long (25-100 miles)", "Very Long (100+ miles)"]
    cat_plot = category_totals.set_index("Category").reindex(
        [c for c in category_order if c in category_totals["Category"].values]
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(cat_plot["Category"], cat_plot["Mean Trips"],
                  color=colors[:len(cat_plot)], edgecolor="white")
    ax.set_xlabel("Distance Category")
    ax.set_ylabel("Total Mean Trips")
    ax.set_title(f"Rule-Based Classification: Mean Trips by Distance Category  [{label}]")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height() / 1e6:.1f}M",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fname1 = f"classification_distance_categories_{label.replace(' ', '_')}.png"
    plt.savefig(fname1, dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {fname1}")

    # ── Step 3: Activity level classification
    activity_col = None
    for col in ["Population Not Staying at Home", "People Not Staying at Home"]:
        if col in national_pd.columns:
            activity_col = col
            break

    if activity_col:
        national_pd["Activity Level"] = national_pd[activity_col].apply(
            classify_activity_level
        )
        activity_counts = national_pd["Activity Level"].value_counts().reset_index()
        activity_counts.columns = ["Activity Level", "Number of Days"]
        print(f"\n  Daily Activity Level Classification:")
        print(activity_counts.to_string(index=False))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(activity_counts["Number of Days"],
               labels=activity_counts["Activity Level"],
               autopct="%1.1f%%", startangle=140,
               colors=colors[:len(activity_counts)])
        ax.set_title(f"Daily Activity Level Classification  [{label}]")
        plt.tight_layout()
        fname2 = f"classification_activity_levels_{label.replace(' ', '_')}.png"
        plt.savefig(fname2, dpi=150)
        plt.close()
        print(f"  → Plot saved: {fname2}")

    # ── Step 4: K-Means clustering 
    X_cluster = national_pd[available].dropna()
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_cluster)

    k      = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"\n  K-Means Clustering (k={k}):")
    print(f"    Silhouette Score: {sil_score:.4f}  (higher = better separated clusters)")

    national_pd_cluster = national_pd.loc[X_cluster.index].copy()
    national_pd_cluster["Cluster"] = cluster_labels

    cluster_counts = national_pd_cluster["Cluster"].value_counts().sort_index()
    print(f"\n  Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        print(f"    Cluster {cluster_id}: {count} days")

    # ── Plot 3: Cluster means 
    cluster_means = national_pd_cluster.groupby("Cluster")[available].mean()
    short_labels  = [c.replace("Number of Trips ", "") for c in available]
    cluster_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(13, 5))
    x         = np.arange(len(short_labels))
    bar_width = 0.2

    for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
        ax.bar(x + i * bar_width, row.values, bar_width,
               label=f"Cluster {cluster_id}",
               color=cluster_colors[i], edgecolor="white")

    ax.set_xlabel("Distance Band")
    ax.set_ylabel("Mean Number of Trips")
    ax.set_title(f"K-Means Clusters: Mean Trips per Distance Band  [{label}]")
    ax.set_xticks(x + bar_width * (k - 1) / 2)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fname3 = f"classification_kmeans_clusters_{label.replace(' ', '_')}.png"
    plt.savefig(fname3, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname3}")

    # ── Plot 4: Elbow method 
    inertia_values = []
    k_range = range(2, 8)
    for k_test in k_range:
        km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_values.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertia_values, marker="o",
            color="steelblue", linewidth=2, markersize=7)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title(f"Elbow Method – Optimal Number of Clusters  [{label}]")
    ax.axvline(x=4, color="red", linestyle="--", label="Chosen k=4")
    ax.legend()
    plt.tight_layout()
    fname4 = f"classification_elbow_{label.replace(' ', '_')}.png"
    plt.savefig(fname4, dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname4}")

    return classification_summary, sil_score



# STANDALONE RUN

if __name__ == "__main__":

    print("\n--- Loading datasets ---")
    big   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big["Date"] = dd.to_datetime(big["Date"])
    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    print("\n>>> Running SEQUENTIAL...")
    t0 = time.time()
    data_classification(big, small, label="Sequential")
    seq_time = time.time() - t0
    print(f"\n  Sequential time: {seq_time:.2f}s")

    print("\n>>> Running PARALLEL – 10 workers...")
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    big10   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big10["Date"] = dd.to_datetime(big10["Date"])
    small10 = dd.read_csv("Trips_Full_Data.csv")
    small10["Date"] = dd.to_datetime(small10["Date"])
    t1 = time.time()
    data_classification(big10, small10, label="Parallel 10 workers (2 cores)")
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
    data_classification(big20, small20, label="Parallel 20 workers (4 cores)")
    par20_time = time.time() - t2
    print(f"\n  Parallel (20 workers) time: {par20_time:.2f}s")
    client.close()

    print("\n--- Classification Timing Summary ---")
    print(f"  Sequential          : {seq_time:.2f}s")
    print(f"  Parallel 10 workers : {par10_time:.2f}s  (speedup: {seq_time/par10_time:.2f}x)")
    print(f"  Parallel 20 workers : {par20_time:.2f}s  (speedup: {seq_time/par20_time:.2f}x)")