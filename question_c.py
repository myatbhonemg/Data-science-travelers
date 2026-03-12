import time
import dask.dataframe as dd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dask.distributed import Client


def question_c(big_dataset, small_dataset, label="Sequential"):

    print(f"\n{'='*60}")
    print(f"  QUESTION C  [{label}]")                                                                                                             
    print(f"{'='*60}")

    # ── Step 1: Get national rows from big dataset 
    national = big_dataset[big_dataset["Level"] == "National"]
    national = national.drop(
        columns=["State FIPS", "State Postal Code", "County FIPS", "County Name"],
        errors="ignore"
    )

    # Get date range from small dataset to align the two datasets
    start_date = small_dataset["Date"].min().compute()
    end_date   = small_dataset["Date"].max().compute()

    big_filtered = national[
        national["Date"].between(str(start_date), str(end_date))
    ].compute()
    big_filtered["Date"] = pd.to_datetime(big_filtered["Date"])

    # ── Step 2: Prepare small dataset 
    small_pd = small_dataset.compute()
    small_pd["Date"] = pd.to_datetime(small_pd["Date"])

    # ── Step 3: Merge on Date so X and Y rows align 
    merged = pd.merge(
        big_filtered[["Date", "Number of Trips 5-10"]],
        small_pd[["Date", "Trips 1-25 Miles"]],
        on="Date",
        how="inner"
    ).dropna()

    print(f"\n  Rows available for model training: {len(merged)}")

    if len(merged) < 5:
        # Fallback if datasets don't overlap on dates
        print("  Using big dataset columns only as fallback.")
        avail = big_filtered[["Number of Trips 1-3", "Number of Trips 5-10"]].dropna()
        X = avail["Number of Trips 1-3"].values.reshape(-1, 1)
        y = avail["Number of Trips 5-10"].values
        x_label = "Number of Trips 1-3"
    else:
        X = merged["Trips 1-25 Miles"].values.reshape(-1, 1)
        y = merged["Number of Trips 5-10"].values
        x_label = "Trips 1-25 Miles"

    # ── Step 4: Train the Linear Regression model 
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # ── Step 5: Evaluate the model 
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\n  Model Results:")
    print(f"    Intercept   : {model.intercept_:.4f}")
    print(f"    Coefficient : {model.coef_[0]:.6f}")
    print(f"    R²          : {r2:.4f}   (closer to 1.0 = better fit)")
    print(f"    RMSE        : {rmse:,.2f}  (lower = fewer prediction errors)")

    # ── Step 6: Scatterplot + regression line
    sort_idx      = X[:, 0].argsort()
    X_sorted      = X[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color="royalblue", alpha=0.5, s=25, label="Actual data")
    ax.plot(X_sorted, y_pred_sorted, color="red", linewidth=2,
            label=f"Linear regression  (R²={r2:.3f})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of Trips 5-10")
    ax.set_title(f"Q-C: Predicting Trip Frequency from Trip Length  [{label}]")
    ax.legend()

    metrics_text = f"R² = {r2:.4f}\nRMSE = {rmse:,.0f}"
    ax.text(0.05, 0.88, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fname = f"qc_regression_{label.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {fname}")

    return r2, rmse


# STANDALONE RUN

if __name__ == "__main__":

    print("\n--- Loading datasets ---")
    big   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big["Date"] = dd.to_datetime(big["Date"])
    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    print("\n>>> Running SEQUENTIAL...")
    t0 = time.time()
    question_c(big, small, label="Sequential")
    seq_time = time.time() - t0
    print(f"\n  Sequential time: {seq_time:.2f}s")

    print("\n>>> Running PARALLEL – 10 workers...")
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    big10   = dd.read_csv("Trips_By_Distance.csv", assume_missing=True, dtype={'County Name': 'object'})
    big10["Date"] = dd.to_datetime(big10["Date"])
    small10 = dd.read_csv("Trips_Full_Data.csv")
    small10["Date"] = dd.to_datetime(small10["Date"])
    t1 = time.time()
    question_c(big10, small10, label="Parallel 10 workers (2 cores)")
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
    question_c(big20, small20, label="Parallel 20 workers (4 cores)")
    par20_time = time.time() - t2
    print(f"\n  Parallel (20 workers) time: {par20_time:.2f}s")
    client.close()

    print("\n--- Question C Timing Summary ---")
    print(f"  Sequential          : {seq_time:.2f}s")
    print(f"  Parallel 10 workers : {par10_time:.2f}s  (speedup: {seq_time/par10_time:.2f}x)")
    print(f"  Parallel 20 workers : {par20_time:.2f}s  (speedup: {seq_time/par20_time:.2f}x)")