import gc, random, psutil, pynvml, polars as pl, numpy as np, pandas as pd, optuna, mlflow, mlflow.catboost
from pathlib import Path
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from mlflow.models import infer_signature
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import HyperbandPruner
import warnings, optuna.exceptions as optuna_w

warnings.filterwarnings("ignore", category=optuna_w.ExperimentalWarning)

SEED = 42
random.seed(SEED); np.random.seed(SEED)

DATA_DIR = Path("data_sampled")
TARGET   = "tip_amount"
EXPERIMENT = "CatBoost_TimeSeries_Optuna"
TRIALS, SPLITS, MAX_ITERS, EARLY_STOP = 20, 5, 10000, 100

try: gpu_count = get_gpu_device_count()
except: gpu_count = 0
TASK_TYPE, DEVICES = ("GPU", "0") if gpu_count else ("CPU", None)

def log_sys(p=""):
    mlflow.log_metric(f"{p}cpu_pct", psutil.cpu_percent())
    mlflow.log_metric(f"{p}mem_pct", psutil.virtual_memory().percent)
    try:
        pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h); m = pynvml.nvmlDeviceGetMemoryInfo(h)
        mlflow.log_metric(f"{p}gpu_util_pct", u.gpu)
        mlflow.log_metric(f"{p}gpu_mem_used_mb", m.used/2**20); pynvml.nvmlShutdown()
    except: pass

def prep(f: Path) -> pl.DataFrame:
    df = pl.read_parquet(f, low_memory=True)
    for c in ("tpep_pickup_datetime","pickup_datetime","tpep_dropoff_datetime","dropoff_datetime"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Datetime("ns")))
    df = df.filter(pl.col(TARGET) >= 0)
    ns = 60000000000
    pick = next((c for c in ("tpep_pickup_datetime","pickup_datetime") if c in df.columns), None)
    drop = next((c for c in ("tpep_dropoff_datetime","dropoff_datetime") if c in df.columns), None)
    if pick and drop:
        df = (
            df.with_columns(((pl.col(drop).cast(pl.Int64)-pl.col(pick).cast(pl.Int64))/ns)
                            .cast(pl.Float32).alias("trip_duration_min"))
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
    for c,t in {"cbd_congestion_fee":pl.Float32,"airport_fee":pl.Float32,"congestion_surcharge":pl.Float32}.items():
        if c not in df.columns: df = df.with_columns(pl.lit(0).cast(t).alias(c))
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

y = pdf[TARGET]; X = pdf.drop(columns=[TARGET])
cat_cols = ["VendorID","RatecodeID","PULocationID","DOLocationID","payment_type",
            "pickup_month","pickup_day","pickup_hour","pickup_dow","store_and_fwd_flag"]
for c in cat_cols: X[c] = X[c].astype("string").fillna("missing")
num_cols = X.columns.difference(cat_cols); X[num_cols] = X[num_cols].fillna(0).astype("float32")

tscv = TimeSeriesSplit(n_splits=SPLITS)
mlflow.set_experiment(EXPERIMENT)

if mlflow.active_run(): mlflow.end_run()
root = mlflow.start_run(run_name="optuna_catboost", log_system_metrics=True)
log_sys("startup_")
mlflow.log_params({"cpu_cores": psutil.cpu_count(logical=True),
                   "mem_total_gb": round(psutil.virtual_memory().total/2**30,2),
                   "gpu_available": TASK_TYPE=="GPU",
                   "task_type": TASK_TYPE})

mlcb = MLflowCallback(metric_name="val_rmse", create_experiment=False, mlflow_kwargs={"nested": True})
pruner = HyperbandPruner()
@mlcb.track_in_mlflow()
def objective(t):
    p = {"depth": t.suggest_int("depth",4,10),
         "learning_rate": t.suggest_float("learning_rate",1e-3,0.3,log=True),
         "l2_leaf_reg": t.suggest_float("l2_leaf_reg",1e-3,10,log=True),
         "subsample": t.suggest_float("subsample",0.5,1.0),
         "min_data_in_leaf": t.suggest_int("min_data_in_leaf",1,100),
         "bootstrap_type":"Bernoulli",
         "iterations":MAX_ITERS,
         "early_stopping_rounds":EARLY_STOP,
         "eval_metric":"RMSE",
         "random_seed":SEED,
         "task_type":TASK_TYPE,
         "devices":DEVICES,
         "verbose":0,
         "cat_features":cat_cols}
    rms, its = [], []
    for tr, vl in tscv.split(X):
        m = CatBoostRegressor(**p)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[vl], y.iloc[vl]), verbose=False)
        rms.append(mean_squared_error(y.iloc[vl], m.predict(X.iloc[vl])))
        its.append(m.get_best_iteration()); del m; gc.collect()
    cv = float(np.mean(rms)); t.set_user_attr("best_iterations", int(np.mean(its)))
    mlflow.log_metric("rmse_cv", cv); mlflow.log_metric("best_iterations", t.user_attrs["best_iterations"]); log_sys("trial_")
    return cv

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=pruner)
study.optimize(objective, n_trials=TRIALS, callbacks=[mlcb])

best = study.best_trial.params
final_iter = study.best_trial.user_attrs["best_iterations"]
final = {**best, "iterations": final_iter, "random_seed": SEED,
         "task_type": TASK_TYPE, "devices": DEVICES, "verbose": 0,
         "bootstrap_type":"Bernoulli", "cat_features":cat_cols}
model = CatBoostRegressor(**final).fit(X, y, verbose=False)

sig = infer_signature(X.head(100), model.predict(X.head(100)))
mlflow.log_params(final)
mlflow.catboost.log_model(model, "model", signature=sig, input_example=X.head(5))
if Path("catboost_info").exists(): mlflow.log_artifacts("catboost_info", artifact_path="catboost_info")
mlflow.log_metric("best_rmse_cv", study.best_value)
log_sys("final_")
mlflow.end_run()

print(f"Best CV RMSE: {study.best_value:.6f}")
