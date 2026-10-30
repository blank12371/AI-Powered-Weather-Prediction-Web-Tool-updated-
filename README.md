# Skycast: machine learning weather prediction

Skycast predicts **today's weather condition** (Sunny, Cloudy or Rainy) and **today's mean temperature (°C)** from the previous **7 to 10 days** of weather. It ships as a Flask web app with a dark glassmorphism dashboard, a JSON API, and a training pipeline that learns from real historical data (Open-Meteo) or a realistic synthetic generator when you are offline.

Licensed under the **GNU General Public License v3.0**. See [GPL-3.0 compliance](#gpl-30-compliance).

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project layout](#project-layout)
4. [Local setup](#local-setup)
5. [Training the models](#training-the-models)
6. [Running the app](#running-the-app)
7. [API reference](#api-reference)
8. [Deploying to Render](#deploying-to-render)
9. [How the model works and what accuracy to expect](#how-the-model-works-and-what-accuracy-to-expect)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)
12. [GPL-3.0 compliance](#gpl-30-compliance)

## Features

- **Two models**: a classifier for the condition (with per-class confidence) and a regressor for temperature (with an approximate 90% range).
- **Real data pipeline**: daily history for 8 cities from the free [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api), your own CSV, or a synthetic generator with realistic covariance between humidity, wind, temperature and rain.
- **82 engineered features**: lags 1 to 7, rolling mean/std/min/max/range, linear slopes, exponentially weighted means, short-term deltas, an approximate dew point, rain/sun streaks and days-since-rain, all computed from the most recent 7 days, plus whole-window means that use days 8 to 10.
- **Model selection**: randomized hyper-parameter search over Random Forest and Histogram Gradient Boosting, scored with chronological `TimeSeriesSplit` cross-validation, then evaluated on a held-out final 20% of time that the search never saw.
- **Honest baselines**: every training run reports accuracy against "always predict the most common class" and "same as yesterday", and temperature RMSE against "same as yesterday".
- **Startup safety**: if the `.joblib` files are missing or corrupt, the server starts immediately, trains a quick model in a background thread, and answers with a clearly labelled rule-based estimate until the model is ready. A file lock stops multiple gunicorn workers from training at once.
- **Input validation**: malformed JSON, wrong types, missing fields, bad conditions, wrong day counts and out-of-range values all return clear JSON errors with proper HTTP status codes.
- **Dashboard**: 7 to 10 day input grid with sensible defaults and example presets, client-side validation, loading state, animated confidence bars, and ☀️ / ☁️ / 🌧️ result graphics. Responsive and keyboard accessible.

## Architecture

```
            ┌──────────────────────────── train.py ────────────────────────────┐
            │                                                                  │
 Open-Meteo │  data loaders ──► daily table ──► sliding 7-10 day windows       │
 CSV        │  (or synthetic)   (city, date,     build_features() -> 82 cols   │
 synthetic  │                    temp, hum,                │                   │
            │                    wind, cond)               ▼                   │
            │                          chronological 80/20 split               │
            │                   RandomizedSearchCV + TimeSeriesSplit           │
            │                   (Random Forest vs Hist. Gradient Boosting)    │
            │                          │ hold-out metrics + baselines          │
            │                          ▼                                       │
            │      weather_classifier.joblib  weather_regressor.joblib         │
            │                   model_metrics.json                             │
            └───────────────────────────┬──────────────────────────────────────┘
                                        │ loaded at startup
            ┌───────────────────────────▼──────────────── app.py ──────────────┐
 Browser ──►│  GET /         dashboard (HTML/CSS/JS embedded in app.py)        │
            │  GET /health   status: missing | training | ready | failed       │
            │  POST /predict validate ► build_features() ► classifier + regressor
            │                         ► JSON (condition, temperature, confidence)
            │                                                                  │
            │  Background thread: models missing/corrupt? ► quick training     │
            │  (file lock across workers) ► rule-based fallback meanwhile      │
            └──────────────────────────────────────────────────────────────────┘
```

`app.py` imports `build_features` from `train.py`, so training and serving always use exactly the same feature code.

## Project layout

```
.
├── app.py              Flask app, API, startup safety, embedded dashboard
├── train.py            Data ingestion, feature engineering, search, persistence
├── requirements.txt    Python dependencies
├── render.yaml         Optional Render Blueprint
├── LICENSE             GNU GPL v3.0 full text
├── README.md
└── .gitignore
```

Generated at runtime (git-ignored): `weather_classifier.joblib`, `weather_regressor.joblib`, `model_metrics.json`.

## Local setup

Requires Python 3.10 or newer (3.11 recommended).

```bash
git clone <your-repo-url> skycast
cd skycast

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Training the models

```bash
python train.py
```

By default this downloads about 5 years of daily history for 8 cities from Open-Meteo, falls back to the synthetic generator if the network is unavailable, runs the model search, prints hold-out metrics, and writes the two `.joblib` files plus `model_metrics.json` next to `train.py`. A full run takes several minutes on a laptop.

Useful options:

| Command | What it does |
| --- | --- |
| `python train.py --quick` | Seconds-long run: synthetic data, fixed hyper-parameters, no search. This is what the app uses for emergency auto-training. |
| `python train.py --source open-meteo` | Require real data; fail instead of falling back. |
| `python train.py --source synthetic` | Fully offline. |
| `python train.py --csv my_data.csv` | Train on your own file with columns `date,temperature,humidity,wind_speed,condition` (optional `city`). Condition must be Sunny, Cloudy or Rainy. |
| `python train.py --n-iter 30 --cv 5` | Search harder (more candidates, more folds). |
| `python train.py --years 10` | More Open-Meteo history per city. |
| `python train.py --model-dir ./models` | Write artefacts somewhere else (set `WEATHER_MODEL_DIR` to match when serving). |

How real observations become labels: a day is **Rainy** if precipitation is at least 1 mm, otherwise **Sunny** if mean cloud cover is below 45%, otherwise **Cloudy**.

## Running the app

```bash
python app.py
```

Open <http://localhost:5000>. If you skipped training, the server still starts: the banner on the page says the models are training in the background, and a few seconds later predictions switch from the rule-based estimate to the machine learning model.

For a production-style local run (Linux/macOS):

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

## API reference

### `POST /predict`

Send the history **oldest day first**, 7 to 10 entries.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days": [
      {"temperature": 30, "humidity": 78, "wind_speed": 12, "condition": "Cloudy"},
      {"temperature": 29, "humidity": 82, "wind_speed": 15, "condition": "Rainy"},
      {"temperature": 28, "humidity": 86, "wind_speed": 18, "condition": "Rainy"},
      {"temperature": 28, "humidity": 88, "wind_speed": 20, "condition": "Rainy"},
      {"temperature": 27, "humidity": 90, "wind_speed": 22, "condition": "Cloudy"},
      {"temperature": 28, "humidity": 89, "wind_speed": 19, "condition": "Rainy"},
      {"temperature": 27, "humidity": 91, "wind_speed": 17, "condition": "Rainy"}
    ]
  }'
```

Successful response (`200`):

```json
{
  "condition": "Rainy",
  "temperature_c": 27.3,
  "temperature_range_c": [23.7, 30.9],
  "confidence": {"Sunny": 3.1, "Cloudy": 28.4, "Rainy": 68.5},
  "days_used": 7,
  "model": {
    "mode": "machine-learning",
    "status": "ready",
    "data_source": "open-meteo",
    "holdout_accuracy": 0.71,
    "holdout_temperature_rmse": 2.2
  }
}
```

`model.mode` is `rule-based-fallback` while models are still training. `temperature_range_c` is an approximate 90% interval derived from the hold-out RMSE. The `confidence` values are percentages that sum to 100.

Field rules:

| Field | Type | Allowed range |
| --- | --- | --- |
| `temperature` | number | -60 to 60 (°C) |
| `humidity` | number | 0 to 100 (%) |
| `wind_speed` | number | 0 to 250 (km/h) |
| `condition` | string | `Sunny`, `Cloudy` or `Rainy` (case-insensitive) |

Error responses are always JSON: `{"error": "...", "details": [{"day": 3, "field": "humidity", "message": "..."}]}`.

| Status | Meaning |
| --- | --- |
| 400 | Invalid JSON, body not an object, `days` not a list, missing field, wrong type, unknown condition |
| 415 | `Content-Type` is not `application/json` |
| 413 | Body larger than 64 KB |
| 422 | Not 7 to 10 days, or a number is outside its allowed range |
| 500 | Unexpected server error (details are logged, not returned) |

### `GET /health`

Returns `{"service": "skycast", "status": "ready", "models_ready": true, ...}`. `status` is one of `missing`, `training`, `ready` or `failed`. Always responds `200` so it is safe to use as a platform health check even while training.

## Deploying to Render

Skycast is a standard Python web service. Training happens during the build, so the first request is already served by a fully trained model.

### Option A: dashboard

1. Push the project to a GitHub or GitLab repository.
2. In Render choose **New > Web Service** and connect the repository.
3. Use these settings:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Root Directory | leave blank (repository root, where `app.py` lives) |
| Build Command | `pip install -r requirements.txt && python train.py` |
| Start Command | `gunicorn app:app` |
| Health Check Path | `/health` |

4. Add environment variables (Environment tab):
   - `PYTHON_VERSION` = `3.11.9`
   - `WEB_CONCURRENCY` = `1` (recommended on small instances, which keeps memory low)
5. Click **Create Web Service**. Render installs dependencies, trains the models, then starts gunicorn.

### Option B: Blueprint

Commit the included `render.yaml`, then choose **New > Blueprint** in Render and select the repository. It applies the same settings.

### Deployment notes

- **Build-time training**: Render's build step has internet access, so `python train.py` downloads real Open-Meteo data. If the download fails, it falls back to synthetic data and the build still succeeds. Check the build log for the line `data_source` to see which was used.
- **Free-tier build time**: a full search on a small instance can be slow. If builds time out, use `python train.py --n-iter 4` in the build command, or `python train.py --quick`.
- **Ephemeral disk**: Render's filesystem resets on every deploy and on free-tier restarts. That is why models are trained in the build step. If the files are ever missing at runtime, the app retrains a quick model in the background instead of crashing.
- **Do not use `--preload`** with gunicorn. The background model loader starts when each worker imports `app.py`.
- **Cold starts**: free instances spin down when idle; the first request after a pause can take about 30 to 60 seconds.

## How the model works and what accuracy to expect

**Target**: today's condition and mean temperature, predicted only from the previous 7 to 10 days of (temperature, humidity, wind, condition). There is no pressure, radar or satellite input, so this is a statistical pattern model, not a replacement for numerical weather forecasts.

**Evaluation**: data are ordered by time. The search uses `TimeSeriesSplit` inside the first 80%, and the reported hold-out metrics come from the final 20%, which the search never sees. The final models are then refit on all data before being saved.

**Calibrating expectations**: day-to-day weather is strongly persistent, so "same as yesterday" is a hard baseline to beat. In a test run on the built-in synthetic generator (about 10,700 windows), the models reached about 60% condition accuracy against a 36% majority-class baseline, but only matched the 61% "same as yesterday" baseline, and cut temperature RMSE from 2.25 °C (persistence) to about 2.15 °C. That is the realistic ceiling for this small feature set on that data. Results on real Open-Meteo data will differ by city and season; `model_metrics.json` records the baselines next to the model scores for every run, so you can see exactly what the model adds. To push accuracy further, add inputs that carry real forecasting signal (pressure and pressure trend, dew point, cloud cover, season, location) to the dashboard and `build_features`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `5000` | Port for `python app.py` (Render sets this for gunicorn). |
| `WEATHER_MODEL_DIR` | project folder | Where model files are read and written. Falls back to a temp directory if not writable. |
| `FLASK_DEBUG` | unset | Set to `1` for the Flask debugger with `python app.py`. |
| `WEB_CONCURRENCY` | gunicorn default | Number of gunicorn workers. |

## Troubleshooting

- **Banner says models are training**: wait a few seconds and predict again. Check `GET /health`.
- **Banner says training failed**: look at the server log. Run `python train.py --source synthetic --quick` to confirm the environment works, then restart.
- **`sklearn` version warnings or load failures after upgrading**: delete the `.joblib` files and retrain; the app also does this automatically if the saved files cannot be loaded.
- **Open-Meteo download fails**: the free API rate-limits heavy use. Retry later, use `--years 2`, or train on `--source synthetic` / your own `--csv`.

## GPL-3.0 compliance

Skycast is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version**.

- The full license text is in [`LICENSE`](LICENSE).
- Every source file carries the standard GPL notice header.
- Skycast is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of merchantability or fitness for a particular purpose.
- If you distribute copies or modified versions (including hosting that provides the source to recipients under the GPL), you must keep these notices, license the whole work under GPL-3.0, and make the corresponding source available.
- Third-party dependencies (Flask, scikit-learn, pandas, NumPy, joblib, gunicorn) are used under their own licenses, which are compatible with GPL-3.0. Open-Meteo data is provided under [CC BY 4.0](https://open-meteo.com/en/license); attribute it if you republish data or derived results.

Replace "Skycast contributors" and the year in the source headers with your own name before publishing.
