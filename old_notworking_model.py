import gc, random, psutil, pynvml, polars as pl, numpy as np, pandas as pd, optuna, mlflow, mlflow.catboost
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from optuna.integration.mlflow import MLflowCallback

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data")
TARGET   = "tip_amount"
EXPERIMENT = "CatBoost_TimeSeries_Optuna"
TRIALS, SPLITS, MAX_ITERS, EARLY_STOP = 50, 5, 10_000, 100

def log_sys(prefix=""):
    mlflow.log_metric(f"{prefix}cpu_pct", psutil.cpu_percent())
    mlflow.log_metric(f"{prefix}mem_pct", psutil.virtual_memory().percent)
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        mlflow.log_metric(f"{prefix}gpu_util_pct", u.gpu)
        mlflow.log_metric(f"{prefix}gpu_mem_used_mb", m.used / 2**20)
        pynvml.nvmlShutdown()
    except Exception:
        pass

def gpu_info():
    try:
        pynvml.nvmlInit()
        h  = pynvml.nvmlDeviceGetHandleByIndex(0)
        nm = pynvml.nvmlDeviceGetName(h)
        name = nm.decode() if isinstance(nm, (bytes, bytearray)) else str(nm)
        mem = round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30, 2)
        pynvml.nvmlShutdown()
        return {"gpu_name": name, "gpu_mem_total_gb": mem}
    except Exception:
        return {"gpu_name": "NA", "gpu_mem_total_gb": 0}

def prep(f):
    df = pl.read_parquet(f, low_memory=True)
    for c in ("tpep_pickup_datetime", "pickup_datetime", "tpep_dropoff_datetime", "dropoff_datetime"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Datetime("ns")))
    df = df.filter(pl.col(TARGET) >= 0)
    ns = 60_000_000_000
    pick = next((c for c in ("tpep_pickup_datetime", "pickup_datetime") if c in df.columns), None)
    drop = next((c for c in ("tpep_dropoff_datetime", "dropoff_datetime") if c in df.columns), None)
    if pick and drop:
        df = (
            df.with_columns(((pl.col(drop).cast(pl.Int64) - pl.col(pick).cast(pl.Int64)) / ns).cast(pl.Float32).alias("trip_duration_min"))
            .with_columns([
                pl.col(pick).dt.month().cast(pl.Int8).alias("pickup_month"),
                pl.col(pick).dt.day().cast(pl.Int8).alias("pickup_day"),
                pl.col(pick).dt.hour().cast(pl.Int8).alias("pickup_hour"),
                pl.col(pick).dt.weekday().cast(pl.Int8).alias("pickup_dow"),
            ])
            .drop([pick, drop])
        )
    else:
        df = df.with_columns([
            pl.lit(0).cast(pl.Float32).alias("trip_duration_min"),
            pl.lit(0).cast(pl.Int8).alias("pickup_month"),
            pl.lit(0).cast(pl.Int8).alias("pickup_day"),
            pl.lit(0).cast(pl.Int8).alias("pickup_hour"),
            pl.lit(0).cast(pl.Int8).alias("pickup_dow"),
        ])
    for c, t in {"cbd_congestion_fee": pl.Float32, "airport_fee": pl.Float32, "congestion_surcharge": pl.Float32}.items():
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).cast(t).alias(c))
    int_cats = ["VendorID","RatecodeID","PULocationID","DOLocationID","payment_type",
                "pickup_month","pickup_day","pickup_hour","pickup_dow"]
    str_cats = ["store_and_fwd_flag"]
    for c in int_cats:
        if c not in df.columns:
            df = df.with_columns(pl.lit(-1).cast(pl.Int32).alias(c))
        else:
            df = df.with_columns(pl.col(c).cast(pl.Int32).fill_null(-1))
    for c in str_cats:
        if c not in df.columns:
            df = df.with_columns(pl.lit("missing").cast(pl.Utf8).alias(c))
        df = df.with_columns(pl.col(c).fill_null("missing").cast(pl.Categorical))
    return df

ddf = pl.concat([prep(f) for f in sorted(DATA_DIR.glob("*.parquet"))])
pdf = ddf.to_pandas(use_pyarrow_extension_array=True); del ddf; gc.collect()

y = pdf[TARGET]
X = pdf.drop(columns=[TARGET])

cat_cols = ["VendorID","RatecodeID","PULocationID","DOLocationID","payment_type",
            "pickup_month","pickup_day","pickup_hour","pickup_dow","store_and_fwd_flag"]

for c in cat_cols:
    X[c] = X[c].astype("string").fillna("missing")

num_cols = X.columns.difference(cat_cols)
X[num_cols] = X[num_cols].fillna(0).astype("float32")

tscv = TimeSeriesSplit(n_splits=SPLITS)
mlflow.set_experiment(EXPERIMENT)

system = {"cpu_cores": psutil.cpu_count(logical=True),
          "mem_total_gb": round(psutil.virtual_memory().total / 2**30, 2)}
system.update(gpu_info())

mlcb = MLflowCallback(metric_name="val_rmse", create_experiment=False, mlflow_kwargs={"nested": True})

@mlcb.track_in_mlflow()
def objective(trial):
    p = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "bootstrap_type":"Bernoulli",
        "iterations": MAX_ITERS,
        "early_stopping_rounds": EARLY_STOP,
        "eval_metric": "RMSE",
        "random_seed": SEED,
        "task_type": "GPU",
        "verbose": 0,
        "cat_features": cat_cols,
    }
    scores, iters = [], []
    for tr, vl in tscv.split(X):
        m = CatBoostRegressor(**p)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[vl], y.iloc[vl]), verbose=False)
        scores.append(mean_squared_error(y.iloc[vl], m.predict(X.iloc[vl])))
        iters.append(m.get_best_iteration())
        del m; gc.collect()
    rmse = float(np.mean(scores))
    trial.set_user_attr("best_iterations", int(np.mean(iters)))
    mlflow.log_metric("rmse_cv", rmse)
    mlflow.log_metric("best_iterations", trial.user_attrs["best_iterations"])
    log_sys()
    return rmse

with mlflow.start_run(run_name="optuna_catboost"):
    mlflow.log_params(system)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS, callbacks=[mlcb])
    best_p    = study.best_trial.params
    final_it  = study.best_trial.user_attrs["best_iterations"]
    final_p   = {**best_p, "iterations": final_it, "random_seed": SEED,
                 "task_type": "GPU", "verbose": 0, "cat_features": cat_cols}
    model = CatBoostRegressor(**final_p)
    model.fit(X, y, verbose=False)
    mlflow.log_params(final_p)
    mlflow.catboost.log_model(model, "model")
    log_sys("final_")

print(f"Best CV RMSE: {study.best_value:.6f}")
