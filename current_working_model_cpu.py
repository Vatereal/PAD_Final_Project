import gc
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport

import mlflow
import mlflow.catboost
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

import optuna
import optuna.exceptions as optuna_w
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import HyperbandPruner

from catboost import CatBoostRegressor, Pool, cv

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DATA_DIR = Path("data_2024_sampled")
TARGET = "tip_amount"

warnings.filterwarnings("ignore", category=optuna_w.ExperimentalWarning)

EXPERIMENT = "YellowTaxi_10%_Sample"
TRIALS = 3
TIMEOUT_MIN = 1080
SPLITS = 3
MAX_ITERS = 10_000
EARLY_STOP = 350
TUNE_FRACTION = 0.1
TASK_TYPE, DEVICES = ("CPU", None)

mlflow.set_experiment(EXPERIMENT)

def prep(f):
    df = pl.read_parquet(f, low_memory=True)
    for c in ("tpep_pickup_datetime", "pickup_datetime", "tpep_dropoff_datetime", "dropoff_datetime"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Datetime("ns")))
    df = df.filter(pl.col(TARGET) >= 0)
    ns = 60000000000
    pick = next((c for c in ("tpep_pickup_datetime", "pickup_datetime") if c in df.columns), None)
    drop = next((c for c in ("tpep_dropoff_datetime", "dropoff_datetime") if c in df.columns), None)
    if pick and drop:
        df = (df.with_columns(((pl.col(drop).cast(pl.Int64) - pl.col(pick).cast(pl.Int64)) / ns).cast(pl.Float32).alias("trip_duration_min"))
                .with_columns([
                    pl.col(pick).dt.month().cast(pl.Int8).alias("pickup_month"),
                    pl.col(pick).dt.day().cast(pl.Int8).alias("pickup_day"),
                    pl.col(pick).dt.hour().cast(pl.Int8).alias("pickup_hour"),
                    pl.col(pick).dt.weekday().cast(pl.Int8).alias("pickup_dow")])
                .drop([pick, drop]))
    else:
        df = df.with_columns([
            pl.lit(0).cast(pl.Float32).alias("trip_duration_min"),
            pl.lit(0).cast(pl.Int8).alias("pickup_month"),
            pl.lit(0).cast(pl.Int8).alias("pickup_day"),
            pl.lit(0).cast(pl.Int8).alias("pickup_hour"),
            pl.lit(0).cast(pl.Int8).alias("pickup_dow")])
    for c, t in {"cbd_congestion_fee": pl.Float32, "airport_fee": pl.Float32, "congestion_surcharge": pl.Float32}.items():
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).cast(t).alias(c))
    int_cats = ["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "payment_type", "pickup_month", "pickup_day", "pickup_hour", "pickup_dow"]
    for c in int_cats:
        df = df.with_columns((pl.col(c).fill_null(-1) if c in df.columns else pl.lit(-1)).cast(pl.Int32).alias(c))
    if "store_and_fwd_flag" not in df.columns:
        df = df.with_columns(pl.lit("missing").cast(pl.Utf8).alias("store_and_fwd_flag"))
    df = df.with_columns(pl.col("store_and_fwd_flag").fill_null("missing").cast(pl.Categorical))
    return df

ddf = pl.concat([prep(f) for f in sorted(DATA_DIR.glob("*.parquet"))])
pdf = ddf.to_pandas(use_pyarrow_extension_array=True)
X = pdf.drop(columns=[TARGET])
y = pdf[TARGET]
cat_cols = ["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "payment_type", "pickup_month", "pickup_day", "pickup_hour", "pickup_dow", "store_and_fwd_flag"]
for c in cat_cols:
    X[c] = pd.Categorical(X[c].astype("string").fillna("missing")).codes.astype("int32")
num_cols = X.columns.difference(cat_cols)
X[num_cols] = X[num_cols].fillna(0).astype("float32")
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
full_pool = Pool(X, y, cat_features=cat_idx)
n_tune = int(TUNE_FRACTION * len(y))
idx = np.random.choice(len(y), n_tune, replace=False)
X_sub, y_sub = X.iloc[idx], y.iloc[idx]
tune_pool = Pool(X_sub, y_sub, cat_features=cat_idx)
input_example = X_sub.head(5)

del ddf, pdf
gc.collect()

pruner = optuna.pruners.HyperbandPruner(min_resource=500, max_resource=MAX_ITERS, reduction_factor=4)

def objective(trial):
    params = {
        "loss_function": "RMSE",
        "depth": trial.suggest_int("depth", 5, 7),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "bootstrap_type": "Bernoulli",
        "iterations": trial.suggest_int("iterations", 1000, MAX_ITERS, log=True),
        "early_stopping_rounds": EARLY_STOP,
        "task_type": TASK_TYPE,
        "devices": DEVICES,
        "thread_count": os.cpu_count(),
        "verbose": False
    }
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.set_tag("user_timezone", "Europe/Moscow")
        mlflow.set_tag("region", "Russia")
        mlflow.log_params(params)
        cvd = cv(pool=tune_pool, params=params, fold_count=SPLITS, partition_random_seed=SEED, early_stopping_rounds=EARLY_STOP, verbose=False)
        best_i = int(cvd["test-RMSE-mean"].idxmin())
        best_r = float(cvd["test-RMSE-mean"].min())
        mlflow.log_metric("val_rmse", best_r)
        mlflow.log_metric("best_iterations", best_i)
    trial.set_user_attr("best_iterations", best_i)
    return best_r

study = optuna.create_study(study_name="CatBoostOptunaStudy", direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=pruner)
start_tune = time.time()
study.optimize(objective, n_trials=TRIALS, timeout=TIMEOUT_MIN * 60, show_progress_bar=True)
tune_time = time.time() - start_tune
best = study.best_trial.params
best_iter = study.best_trial.user_attrs["best_iterations"]
final_params = {**best, "iterations": best_iter, "random_seed": SEED, "task_type": "CPU", "devices": None, "thread_count": os.cpu_count(), "verbose": False, "bootstrap_type": "Bernoulli"}

with mlflow.start_run(run_name="final_catboost_fit") as fin_run:
    mlflow.set_tag("user_timezone", "Europe/Moscow")
    mlflow.set_tag("region", "Russia")
    start_train = time.time()
    model = CatBoostRegressor(**final_params).fit(full_pool, verbose=False)
    train_time = time.time() - start_train
    sample_pred = model.predict(input_example)
    sig = infer_signature(input_example, sample_pred)
    mlflow.catboost.log_model(model, "model", signature=sig, input_example=input_example, registered_model_name="nyc_taxi_fare_catboost")
    mlflow.log_metric("best_rmse_cv", study.best_value)
    fi = model.get_feature_importance(type="FeatureImportance", prettified=False)
    cols = np.array(X.columns)
    rank = np.argsort(fi)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(cols)))
    ax.barh(range(len(rank)), fi[rank][::-1])
    ax.set_yticks(range(len(rank)))
    ax.set_yticklabels(cols[rank][::-1])
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("CatBoost feature importance")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

client = MlflowClient()
mv = client.get_latest_versions("nyc_taxi_fare_catboost", stages=["None"])[0]
client.transition_model_version_stage("nyc_taxi_fare_catboost", mv.version, stage="Staging")

print(f"Tuning time: {tune_time:.2f}s")
print(f"Training time: {train_time:.2f}s")
print(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} - Total time: {tune_time + train_time:.2f}s")
