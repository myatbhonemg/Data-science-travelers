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


# -----------------------------------------------------------------------------
# LOAD DATASETS
# -----------------------------------------------------------------------------

def load_datasets():
    """Loads both CSVs as Dask DataFrames."""
    print("  Loading Trips_By_Distance.csv  (~100k rows) ...")
    big = dd.read_csv("Trips_By_Distance.csv", assume_missing=True)
    big["Date"] = dd.to_datetime(big["Date"])

    print("  Loading Trips_Full_Data.csv  (32-week summary) ...")
    small = dd.read_csv("Trips_Full_Data.csv")
    small["Date"] = dd.to_datetime(small["Date"])

    return big, small


# -----------------------------------------------------------------------------
# RUN ALL QUESTIONS
# -----------------------------------------------------------------------------

def run_all(big, small, label):
    """Runs classification + Q-A through Q-D and returns elapsed time."""
    t_start = time.time()
    data_classification(big, small, label=label)
    question_a(big, small, label=label)
    #question_b(big, small, label=label)
    #question_c(big, small, label=label)
    #question_d(big, small, label=label)
    #elapsed = time.time() - t_start
    #print(f"\n  Total time [{label}]: {elapsed:.2f} seconds")
    #return elapsed