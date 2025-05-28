import gc, os, random, warnings
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
import optuna, mlflow, mlflow.catboost
from catboost import CatBoostRegressor, Pool, cv
from mlflow.models import infer_signature
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import HyperbandPruner
import optuna.exceptions as optuna_w

warnings.filterwarnings("ignore", category=optuna_w.ExperimentalWarning)

SEED = 42
random.seed(SEED); np.random.seed(SEED)

DATA_DIR = Path("data_sampled")
TARGET = "tip_amount"
EXPERIMENT = "YellowTaxi_Optuna"
TRIALS = 1
TIMEOUT_MIN = 180
SPLITS = 2
MAX_ITERS = 5_000
EARLY_STOP = 50
TUNE_FRACTION = 0.01

TASK_TYPE, DEVICES = ("CPU", None)

def prep(f: Path) -> pl.DataFrame:
    df = pl.read_parquet(f, low_memory=True)
    for c in ("tpep_pickup_datetime","pickup_datetime","tpep_dropoff_datetime","dropoff_datetime"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Datetime("ns")))
    df = df.filter(pl.col(TARGET) >= 0)
    ns = 60_000_000_000
    pick = next((c for c in ("tpep_pickup_datetime","pickup_datetime") if c in df.columns), None)
    drop = next((c for c in ("tpep_dropoff_datetime","dropoff_datetime") if c in df.columns), None)
    if pick and drop:
        df = (df.with_columns(((pl.col(drop).cast(pl.Int64) - pl.col(pick).cast(pl.Int64)) / ns)
                              .cast(pl.Float32).alias("trip_duration_min"))
                .with_columns([
                    pl.col(pick).dt.month().cast(pl.Int8).alias("pickup_month"),
                    pl.col(pick).dt.day().cast(pl.Int8).alias("pickup_day"),
                    pl.col(pick).dt.hour().cast(pl.Int8).alias("pickup_hour"),
                    pl.col(pick).dt.weekday().cast(pl.Int8).alias("pickup_dow")
                ])
                .drop([pick, drop]))
    else:
        df = df.with_columns([
            pl.lit(0).cast(pl.Float32).alias("trip_duration_min"),
            pl.lit(0).cast(pl.Int8).alias("pickup_month"),
            pl.lit(0).cast(pl.Int8).alias("pickup_day"),
            pl.lit(0).cast(pl.Int8).alias("pickup_hour"),
            pl.lit(0).cast(pl.Int8).alias("pickup_dow")
        ])
    for c, t in {"cbd_congestion_fee": pl.Float32,
                 "airport_fee": pl.Float32,
                 "congestion_surcharge": pl.Float32}.items():
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).cast(t).alias(c))
    int_cats = ["VendorID","RatecodeID","PULocationID","DOLocationID","payment_type",
                "pickup_month","pickup_day","pickup_hour","pickup_dow"]
    for c in int_cats:
        df = df.with_columns((pl.col(c).fill_null(-1) if c in df.columns else pl.lit(-1))
                             .cast(pl.Int32).alias(c))
    if "store_and_fwd_flag" not in df.columns:
        df = df.with_columns(pl.lit("missing").cast(pl.Utf8).alias("store_and_fwd_flag"))
    df = df.with_columns(pl.col("store_and_fwd_flag").fill_null("missing").cast(pl.Categorical))
    return df

ddf = pl.concat([prep(f) for f in sorted(DATA_DIR.glob("*.parquet"))])
pdf = ddf.to_pandas(use_pyarrow_extension_array=True); del ddf; gc.collect()

y = pdf[TARGET]
X = pdf.drop(columns=[TARGET])
cat_cols = ["VendorID","RatecodeID","PULocationID","DOLocationID","payment_type",
            "pickup_month","pickup_day","pickup_hour","pickup_dow","store_and_fwd_flag"]
for c in cat_cols:
    X[c] = pd.Categorical(X[c].astype("string").fillna("missing")).codes.astype("int32")
num_cols = X.columns.difference(cat_cols)
X[num_cols] = X[num_cols].fillna(0).astype("float32")
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
full_pool = Pool(X, y, cat_features=cat_idx)

del pdf, X, y; gc.collect()

n_tune = int(TUNE_FRACTION * full_pool.shape[0])
idx_sub = np.random.choice(full_pool.shape[0], n_tune, replace=False)
tune_pool = full_pool.slice(idx_sub)

mlflow.set_experiment(EXPERIMENT)
if mlflow.active_run(): mlflow.end_run()
mlflow.start_run(run_name="optuna_catboost", log_system_metrics=True)

mlcb = MLflowCallback(metric_name="val_rmse", create_experiment=False, mlflow_kwargs={"nested": True})
pruner = HyperbandPruner()

def objective(trial):
    mlflow.set_tag("mlflow.runName", f"trial_{trial.number}")
    params = {
        "loss_function": "RMSE",
        "depth": trial.suggest_int("depth", 5, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "bootstrap_type": "Bernoulli",
        "iterations": MAX_ITERS,
        "early_stopping_rounds": EARLY_STOP,
        "task_type": TASK_TYPE,
        "thread_count": os.cpu_count(),
        "verbose": False
    }
    cv_data = cv(
        pool=tune_pool,
        params=params,
        fold_count=SPLITS,
        partition_random_seed=SEED,
        early_stopping_rounds=EARLY_STOP,
        verbose=False
    )
    best_iter = int(cv_data["test-RMSE-mean"].idxmin())
    best_rmse = float(cv_data["test-RMSE-mean"].min())
    mlflow.log_metric("rmse_cv", best_rmse)
    mlflow.log_metric("best_iterations", best_iter)
    trial.set_user_attr("best_iterations", best_iter)
    return best_rmse

study = optuna.create_study(study_name="CatBoostOptunaStudy", direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEED), pruner=pruner)
study.optimize(objective, n_trials=TRIALS, timeout=TIMEOUT_MIN * 60,
               callbacks=[mlcb], show_progress_bar=True)

best_params = study.best_trial.params
final_iter = study.best_trial.user_attrs["best_iterations"]
final_params = {**best_params, "iterations": final_iter, "random_seed": SEED,
                "task_type": TASK_TYPE,
                "thread_count": os.cpu_count(), "verbose": False,
                "bootstrap_type": "Bernoulli"}

model = CatBoostRegressor(**final_params).fit(full_pool, verbose=False)
signature = infer_signature(full_pool.get_features(), model.predict(full_pool.get_features()))
mlflow.catboost.log_model(model, "model", signature=signature, input_example=full_pool.get_features()[:5])
if Path("catboost_info").exists(): mlflow.log_artifacts("catboost_info", artifact_path="catboost_info")
mlflow.log_metric("best_rmse_cv", study.best_value)
mlflow.end_run()

print(f"Best CV RMSE (subsampled tuning): {study.best_value: .6f}")
print(f"Best params: {best_params}")
