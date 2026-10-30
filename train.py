#!/usr/bin/env python3
# Skycast - machine learning weather prediction
# Copyright (C) 2026 Skycast contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Training pipeline for the Skycast weather predictor.

Task
----
Given the previous 7-10 days of (temperature, humidity, wind speed, condition),
predict TODAY's
  * condition  (Sunny / Cloudy / Rainy)  -> classifier
  * mean temperature in degrees Celsius  -> regressor

Data sources (``--source``)
---------------------------
auto        Try Open-Meteo historical data for several cities; if the network is
            unavailable, fall back to the synthetic generator.
open-meteo  Real ERA5-based daily history from the free Open-Meteo archive API.
synthetic   Generator with a latent-moisture process that reproduces realistic
            covariance between humidity, wind, temperature and condition.
--csv FILE  Your own CSV with columns: date, temperature, humidity, wind_speed,
            condition (optional: city).

Usage
-----
    python train.py                      # full run (auto source, hyper-parameter search)
    python train.py --quick              # seconds-long run, fixed hyper-parameters
    python train.py --source synthetic   # fully offline
    python train.py --csv my_weather.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

log = logging.getLogger("skycast.train")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("WEATHER_MODEL_DIR", BASE_DIR)
CLASSIFIER_FILE = "weather_classifier.joblib"
REGRESSOR_FILE = "weather_regressor.joblib"
METRICS_FILE = "model_metrics.json"

CONDITIONS = ["Sunny", "Cloudy", "Rainy"]
COND_TO_ID = {name: i for i, name in enumerate(CONDITIONS)}
SUNNY, CLOUDY, RAINY = 0, 1, 2

MIN_DAYS, MAX_DAYS = 7, 10
FEATURE_WINDOW = 7  # lag / rolling features always use the most recent 7 days

# name, latitude, longitude
CITIES = [
    ("Dhaka", 23.81, 90.41),
    ("London", 51.51, -0.13),
    ("New York", 40.71, -74.01),
    ("Sydney", -33.87, 151.21),
    ("Mumbai", 19.08, 72.88),
    ("Berlin", 52.52, 13.40),
    ("Tokyo", 35.68, 139.69),
    ("Nairobi", -1.29, 36.82),
]

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
_X7 = np.arange(FEATURE_WINDOW, dtype=float) - (FEATURE_WINDOW - 1) / 2.0
_X7_DEN = float(np.dot(_X7, _X7))


def _slope(y: np.ndarray) -> float:
    """Least-squares slope of y against 0..6 (units per day)."""
    return float(np.dot(_X7, y - y.mean()) / _X7_DEN)


def _ewm_last(y: np.ndarray, alpha: float = 0.5) -> float:
    """Exponentially weighted mean that emphasises the most recent day."""
    w = (1.0 - alpha) ** np.arange(len(y) - 1, -1, -1)
    return float(np.dot(w, y) / w.sum())


def build_features(temp, hum, wind, cond) -> dict:
    """
    Turn a chronological history (oldest -> newest, 7-10 days) into one feature row.

    Parameters are sequences of equal length. ``cond`` holds condition ids
    (0 = Sunny, 1 = Cloudy, 2 = Rainy).

    Feature groups
    --------------
    * lags 1..7 of temperature, humidity and wind (lag1 = yesterday)
    * rain / sunny indicators for the last 3 days
    * rolling mean, std, min, max, range, slope, EWM and short-term deltas for
      temperature, humidity, wind and the approximate dew point
    * condition aggregates: fractions, days since rain/sun, current rain streak
    * whole-window means (uses days 8-10 when the caller supplies them)
    """
    temp = np.asarray(temp, dtype=float)
    hum = np.asarray(hum, dtype=float)
    wind = np.asarray(wind, dtype=float)
    cond = np.asarray(cond, dtype=int)
    n = len(temp)
    if not (MIN_DAYS <= n <= MAX_DAYS) or not (len(hum) == len(wind) == len(cond) == n):
        raise ValueError(f"Expected {MIN_DAYS}-{MAX_DAYS} aligned days, got {n}.")

    t, h, w, c = (a[-FEATURE_WINDOW:] for a in (temp, hum, wind, cond))
    dew = t - (100.0 - h) / 5.0  # Magnus-style approximation, good for RH > 50%

    f: dict[str, float] = {}

    for k in range(1, FEATURE_WINDOW + 1):
        f[f"temp_lag{k}"] = t[-k]
        f[f"hum_lag{k}"] = h[-k]
        f[f"wind_lag{k}"] = w[-k]

    for k in range(1, 4):
        f[f"rain_lag{k}"] = float(c[-k] == RAINY)
        f[f"sunny_lag{k}"] = float(c[-k] == SUNNY)
    f["cond_lag1"] = float(c[-1])

    for name, arr in (("temp", t), ("hum", h), ("wind", w), ("dew", dew)):
        f[f"{name}_mean3"] = arr[-3:].mean()
        f[f"{name}_mean7"] = arr.mean()
        f[f"{name}_std3"] = arr[-3:].std()
        f[f"{name}_std7"] = arr.std()
        f[f"{name}_min7"] = arr.min()
        f[f"{name}_max7"] = arr.max()
        f[f"{name}_range7"] = arr.max() - arr.min()
        f[f"{name}_slope7"] = _slope(arr)
        f[f"{name}_ewm"] = _ewm_last(arr)
        f[f"{name}_delta1"] = arr[-1] - arr[-2]
        f[f"{name}_delta3"] = arr[-1] - arr[-3]

    f["frac_rainy7"] = float(np.mean(c == RAINY))
    f["frac_sunny7"] = float(np.mean(c == SUNNY))
    f["frac_cloudy7"] = float(np.mean(c == CLOUDY))
    f["frac_rainy3"] = float(np.mean(c[-3:] == RAINY))

    rain_idx = np.flatnonzero(c == RAINY)
    sun_idx = np.flatnonzero(c == SUNNY)
    f["days_since_rain"] = float(FEATURE_WINDOW if rain_idx.size == 0 else FEATURE_WINDOW - 1 - rain_idx[-1])
    f["days_since_sun"] = float(FEATURE_WINDOW if sun_idx.size == 0 else FEATURE_WINDOW - 1 - sun_idx[-1])
    streak = 0
    for v in c[::-1]:
        if v != RAINY:
            break
        streak += 1
    f["rain_streak"] = float(streak)

    f["temp_mean_all"] = temp.mean()
    f["hum_mean_all"] = hum.mean()
    f["wind_mean_all"] = wind.mean()

    return {k: float(v) for k, v in f.items()}


FEATURE_NAMES: list[str] = list(
    build_features(np.full(7, 20.0), np.full(7, 60.0), np.full(7, 10.0), np.zeros(7, dtype=int)).keys()
)


def features_to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def label_condition(precip_mm: float, cloud_pct: float) -> int:
    """Daily label rule applied to real observations."""
    if precip_mm >= 1.0:
        return RAINY
    if cloud_pct < 45.0:
        return SUNNY
    return CLOUDY


def fetch_open_meteo(city: str, lat: float, lon: float, start: date, end: date, timeout: int = 45) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "wind_speed_10m_mean",
                "precipitation_sum",
                "cloud_cover_mean",
            ]
        ),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "skycast-train/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        payload = json.loads(resp.read().decode("utf-8"))
    d = payload["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "temperature": pd.to_numeric(pd.Series(d["temperature_2m_mean"]), errors="coerce"),
            "humidity": pd.to_numeric(pd.Series(d["relative_humidity_2m_mean"]), errors="coerce"),
            "wind_speed": pd.to_numeric(pd.Series(d["wind_speed_10m_mean"]), errors="coerce"),
            "precip": pd.to_numeric(pd.Series(d["precipitation_sum"]), errors="coerce"),
            "cloud": pd.to_numeric(pd.Series(d["cloud_cover_mean"]), errors="coerce"),
        }
    )
    num = ["temperature", "humidity", "wind_speed", "precip", "cloud"]
    df[num] = df[num].interpolate(limit=2)
    df = df.dropna(subset=num).reset_index(drop=True)
    df["condition_id"] = [label_condition(p, c) for p, c in zip(df["precip"], df["cloud"])]
    df["city"] = city
    return df[["city", "date", "temperature", "humidity", "wind_speed", "condition_id"]]


def load_open_meteo(years: int = 5) -> pd.DataFrame:
    end = date.today() - timedelta(days=7)  # the archive lags real time by a few days
    start = end - timedelta(days=365 * years)
    frames = []
    for city, lat, lon in CITIES:
        try:
            log.info("Fetching Open-Meteo history for %s ...", city)
            frames.append(fetch_open_meteo(city, lat, lon, start, end))
            time.sleep(0.4)  # be polite to the free API
        except Exception as exc:  # network, JSON or schema problems
            log.warning("  skipped %s (%s)", city, exc)
    if len(frames) < 2:
        raise RuntimeError("Could not download enough Open-Meteo data (need at least 2 cities).")
    return pd.concat(frames, ignore_index=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "temperature", "humidity", "wind_speed", "condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if "city" not in df.columns:
        df["city"] = "csv"
    df["date"] = pd.to_datetime(df["date"])
    lookup = {c.lower(): i for c, i in COND_TO_ID.items()}
    df["condition_id"] = df["condition"].astype(str).str.strip().str.lower().map(lookup)
    for col in ("temperature", "humidity", "wind_speed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["temperature", "humidity", "wind_speed", "condition_id"])
    df["condition_id"] = df["condition_id"].astype(int)
    return df[["city", "date", "temperature", "humidity", "wind_speed", "condition_id"]]


def generate_synthetic(n_cities: int = 6, days: int = 1800, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic climate generator with realistic covariance.

    A latent, persistent "moisture" process (AR(1) plus a seasonal term) drives
    humidity, wind, cloud cover and rain at the same time. Temperature follows a
    seasonal cycle plus an autocorrelated anomaly and is cooler on wet days.
    Because moisture persists, yesterday's humidity / condition carry genuine
    predictive information about today - exactly what the model must learn.
    """
    rng = np.random.default_rng(seed)
    frames = []
    start = pd.Timestamp("2018-01-01")
    for ci in range(n_cities):
        base_t = rng.uniform(8, 30)
        amp = rng.uniform(3, 12)
        hemi = rng.choice([-1.0, 1.0])
        h_off = rng.uniform(-10, 10)
        phi_m = rng.uniform(0.72, 0.88)
        wet_season_amp = rng.uniform(0.2, 0.8)
        doy = (start + pd.to_timedelta(np.arange(days), unit="D")).dayofyear.to_numpy()
        season = hemi * np.sin(2 * np.pi * (doy - 110) / 365.0)

        m = np.zeros(days)
        anomaly = np.zeros(days)
        for i in range(1, days):
            m[i] = phi_m * m[i - 1] + np.sqrt(1 - phi_m**2) * rng.normal()
            anomaly[i] = 0.7 * anomaly[i - 1] + rng.normal(0, 1.8)
        m_eff = m + wet_season_amp * np.cos(2 * np.pi * (doy - 200) / 365.0)
        m_obs = m_eff + rng.normal(0, 0.3, days)

        cond = np.where(m_obs > 0.85, RAINY, np.where(m_obs < -0.25, SUNNY, CLOUDY))
        humidity = np.clip(66 + 14 * m_obs + h_off + rng.normal(0, 5, days), 15, 100)
        wind = np.clip(rng.gamma(4.0, 2.4, days) * (1 + 0.25 * np.maximum(m_obs, 0)), 0, 90)
        temp = base_t + amp * season + anomaly + np.select([cond == RAINY, cond == CLOUDY], [-2.4, -0.8], 0.9)

        frames.append(
            pd.DataFrame(
                {
                    "city": f"synthetic-{ci + 1}",
                    "date": start + pd.to_timedelta(np.arange(days), unit="D"),
                    "temperature": np.round(temp, 1),
                    "humidity": np.round(humidity, 1),
                    "wind_speed": np.round(wind, 1),
                    "condition_id": cond.astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def get_daily_data(source: str, csv_path: str | None, quick: bool, years: int, seed: int) -> tuple[pd.DataFrame, str]:
    if csv_path:
        return load_csv(csv_path), f"csv:{os.path.basename(csv_path)}"
    if source == "synthetic" or (source == "auto" and quick):
        n_cities, days = (4, 1200) if quick else (6, 1800)
        return generate_synthetic(n_cities, days, seed), "synthetic"
    if source in ("open-meteo", "auto"):
        try:
            return load_open_meteo(years), "open-meteo"
        except Exception as exc:
            if source == "open-meteo":
                raise
            log.warning("Real data unavailable (%s). Falling back to the synthetic generator.", exc)
            return generate_synthetic(6, 1800, seed), "synthetic"
    raise ValueError(f"Unknown source: {source}")


# --------------------------------------------------------------------------- #
# Supervised dataset
# --------------------------------------------------------------------------- #
def make_samples(daily: pd.DataFrame, seed: int = 42):
    """Slide a variable 7-10 day window over every city and build (X, y) pairs."""
    rng = np.random.default_rng(seed)
    rows, y_cls, y_tmp, dates = [], [], [], []
    for _, g in daily.groupby("city", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        temp = g["temperature"].to_numpy(float)
        hum = g["humidity"].to_numpy(float)
        wind = g["wind_speed"].to_numpy(float)
        cond = g["condition_id"].to_numpy(int)
        day_no = g["date"].to_numpy("datetime64[D]").astype("int64")
        for i in range(MAX_DAYS, len(g)):
            n = int(rng.integers(MIN_DAYS, MAX_DAYS + 1))
            if day_no[i] - day_no[i - n] != n:  # gap in the record -> skip
                continue
            s = slice(i - n, i)
            rows.append(build_features(temp[s], hum[s], wind[s], cond[s]))
            y_cls.append(cond[i])
            y_tmp.append(temp[i])
            dates.append(day_no[i])
    if not rows:
        raise RuntimeError("No usable training windows were produced.")
    order = np.argsort(np.asarray(dates), kind="stable")  # chronological for honest validation
    X = features_to_frame(rows).iloc[order].reset_index(drop=True)
    return X, np.asarray(y_cls)[order], np.asarray(y_tmp)[order]


# --------------------------------------------------------------------------- #
# Model search
# --------------------------------------------------------------------------- #
def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _search(estimator, space, X, y, scoring, n_iter, cv, seed):
    s = RandomizedSearchCV(
        estimator,
        space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    s.fit(X, y)
    return s


def _classifier_candidates(seed):
    return {
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [8, 12, 16, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", 0.3, 0.5],
                "class_weight": [None, "balanced_subsample"],
            },
        ),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(random_state=seed),
            {
                "learning_rate": [0.03, 0.05, 0.08, 0.12],
                "max_depth": [3, 5, 7, None],
                "max_leaf_nodes": [15, 31, 63],
                "l2_regularization": [0.0, 0.1, 1.0],
                "min_samples_leaf": [20, 40, 80],
                "max_iter": [200, 350, 500],
            },
        ),
    }


def _regressor_candidates(seed):
    return {
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [8, 12, 16, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": [0.3, 0.5, 1.0],
            },
        ),
        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor(random_state=seed),
            {
                "learning_rate": [0.03, 0.05, 0.08, 0.12],
                "max_depth": [3, 5, 7, None],
                "max_leaf_nodes": [15, 31, 63],
                "l2_regularization": [0.0, 0.1, 1.0],
                "min_samples_leaf": [20, 40, 80],
                "max_iter": [200, 350, 500],
            },
        ),
    }


def _atomic_dump(obj, path: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, path)


def run_training(
    source: str = "auto",
    csv_path: str | None = None,
    quick: bool = False,
    model_dir: str | None = None,
    n_iter: int = 12,
    cv_splits: int = 3,
    years: int = 5,
    seed: int = 42,
) -> dict:
    """Train both models, evaluate on a chronological hold-out, save artefacts."""
    t0 = time.time()
    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    daily, source_used = get_daily_data(source, csv_path, quick, years, seed)
    log.info("Loaded %d daily records from '%s'.", len(daily), source_used)

    X, y_cls, y_tmp = make_samples(daily, seed)
    log.info("Built %d training windows with %d features.", len(X), X.shape[1])

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    yc_tr, yc_te = y_cls[:split], y_cls[split:]
    yt_tr, yt_te = y_tmp[:split], y_tmp[split:]
    cv = TimeSeriesSplit(n_splits=cv_splits)

    if quick:
        clf_name = "HistGradientBoostingClassifier"
        reg_name = "HistGradientBoostingRegressor"
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=5, max_iter=150, min_samples_leaf=40, random_state=seed
        ).fit(X_tr, yc_tr)
        reg = HistGradientBoostingRegressor(
            learning_rate=0.08, max_depth=5, max_iter=200, min_samples_leaf=40, random_state=seed
        ).fit(X_tr, yt_tr)
        cv_report = {"mode": "quick (fixed hyper-parameters, no search)"}
        clf_params, reg_params = clf.get_params(), reg.get_params()
    else:
        best_c = None
        for name, (est, space) in _classifier_candidates(seed).items():
            log.info("Searching %s ...", name)
            s = _search(est, space, X_tr, yc_tr, "f1_macro", n_iter, cv, seed)
            log.info("  CV macro-F1 = %.4f", s.best_score_)
            if best_c is None or s.best_score_ > best_c[1].best_score_:
                best_c = (name, s)
        best_r = None
        for name, (est, space) in _regressor_candidates(seed).items():
            log.info("Searching %s ...", name)
            s = _search(est, space, X_tr, yt_tr, "neg_root_mean_squared_error", n_iter, cv, seed)
            log.info("  CV RMSE = %.4f", -s.best_score_)
            if best_r is None or s.best_score_ > best_r[1].best_score_:
                best_r = (name, s)
        clf_name, clf = best_c[0], best_c[1].best_estimator_
        reg_name, reg = best_r[0], best_r[1].best_estimator_
        clf_params, reg_params = best_c[1].best_params_, best_r[1].best_params_
        cv_report = {
            "mode": f"RandomizedSearchCV, TimeSeriesSplit({cv_splits}), n_iter={n_iter}",
            "classifier_cv_macro_f1": round(float(best_c[1].best_score_), 4),
            "regressor_cv_rmse": round(float(-best_r[1].best_score_), 4),
        }

    # ---- honest hold-out evaluation (future data the models never saw) ----
    pred_c = clf.predict(X_te)
    pred_t = reg.predict(X_te)
    majority = int(np.bincount(yc_tr).argmax())
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source_used,
        "n_daily_records": int(len(daily)),
        "n_training_windows": int(len(X)),
        "n_features": int(X.shape[1]),
        "sklearn_version": sklearn.__version__,
        "classifier": clf_name,
        "regressor": reg_name,
        "search": cv_report,
        "holdout": {
            "n": int(len(X_te)),
            "accuracy": round(float(accuracy_score(yc_te, pred_c)), 4),
            "macro_f1": round(float(f1_score(yc_te, pred_c, average="macro")), 4),
            "baseline_majority_accuracy": round(float(np.mean(yc_te == majority)), 4),
            "baseline_persistence_accuracy": round(float(np.mean(X_te["cond_lag1"].to_numpy().astype(int) == yc_te)), 4),
            "confusion_matrix": confusion_matrix(yc_te, pred_c, labels=[0, 1, 2]).tolist(),
            "temperature_rmse": round(_rmse(yt_te, pred_t), 4),
            "temperature_mae": round(float(mean_absolute_error(yt_te, pred_t)), 4),
            "baseline_persistence_rmse": round(_rmse(yt_te, X_te["temp_lag1"].to_numpy()), 4),
        },
        "best_params": {
            "classifier": {k: (v if isinstance(v, (int, float, str, type(None), bool)) else str(v)) for k, v in clf_params.items() if k in _short_keys(clf_name)},
            "regressor": {k: (v if isinstance(v, (int, float, str, type(None), bool)) else str(v)) for k, v in reg_params.items() if k in _short_keys(reg_name)},
        },
        "classes": CONDITIONS,
    }

    # ---- refit on every window, then persist ----
    log.info("Refitting final models on all %d windows ...", len(X))
    clf_final = clone(clf).fit(X, y_cls)
    reg_final = clone(reg).fit(X, y_tmp)
    metrics["training_seconds"] = round(time.time() - t0, 1)

    _atomic_dump(clf_final, os.path.join(model_dir, CLASSIFIER_FILE))
    _atomic_dump(reg_final, os.path.join(model_dir, REGRESSOR_FILE))
    tmp_metrics = os.path.join(model_dir, f"{METRICS_FILE}.{os.getpid()}.tmp")
    with open(tmp_metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    os.replace(tmp_metrics, os.path.join(model_dir, METRICS_FILE))

    h = metrics["holdout"]
    log.info(
        "Hold-out accuracy %.1f%% (majority baseline %.1f%%, persistence %.1f%%) | temperature RMSE %.2f C (persistence %.2f C)",
        100 * h["accuracy"], 100 * h["baseline_majority_accuracy"], 100 * h["baseline_persistence_accuracy"],
        h["temperature_rmse"], h["baseline_persistence_rmse"],
    )
    log.info("Saved models to %s in %.1fs.", model_dir, metrics["training_seconds"])
    return metrics


def _short_keys(model_name: str) -> set[str]:
    if model_name.startswith("RandomForest"):
        return {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"}
    return {"learning_rate", "max_depth", "max_leaf_nodes", "l2_regularization", "min_samples_leaf", "max_iter"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the Skycast weather models.")
    p.add_argument("--source", choices=["auto", "open-meteo", "synthetic"], default="auto")
    p.add_argument("--csv", help="Train on your own CSV instead (date,temperature,humidity,wind_speed,condition).")
    p.add_argument("--quick", action="store_true", help="Fast run: synthetic data (when source=auto), no hyper-parameter search.")
    p.add_argument("--n-iter", type=int, default=12, help="Random-search candidates per model family.")
    p.add_argument("--cv", type=int, default=3, help="Time-series CV folds.")
    p.add_argument("--years", type=int, default=5, help="Years of Open-Meteo history per city.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", default=MODEL_DIR, help="Where to write the .joblib files.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        run_training(
            source=args.source,
            csv_path=args.csv,
            quick=args.quick,
            model_dir=args.model_dir,
            n_iter=args.n_iter,
            cv_splits=args.cv,
            years=args.years,
            seed=args.seed,
        )
    except Exception:
        log.exception("Training failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
