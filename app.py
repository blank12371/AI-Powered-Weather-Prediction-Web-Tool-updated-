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
Skycast web application.

    GET  /         dashboard
    GET  /health   liveness + model status (used by Render's health check)
    POST /predict  7-10 days of history in, today's condition + temperature out

Startup safety
--------------
If the .joblib models are missing or unreadable, a background thread trains a
quick model so the web server starts immediately and never crashes. Until the
models are ready, /predict answers with a transparent rule-based estimate and
labels it as such.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

import train as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("skycast.app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # a valid payload is < 2 KB
app.json.sort_keys = False

# --------------------------------------------------------------------------- #
# Model storage and lifecycle
# --------------------------------------------------------------------------- #
LOCK_STALE_SECONDS = 20 * 60
WAIT_FOR_OTHER_WORKER_SECONDS = 15 * 60


def _resolve_model_dir() -> str:
    """Use the configured directory if writable, otherwise a temp directory."""
    wanted = T.MODEL_DIR
    try:
        os.makedirs(wanted, exist_ok=True)
        if os.access(wanted, os.W_OK):
            return wanted
    except OSError:
        pass
    fallback = os.path.join(tempfile.gettempdir(), "skycast_models")
    os.makedirs(fallback, exist_ok=True)
    log.warning("Model directory %s is not writable; using %s", wanted, fallback)
    return fallback


MODEL_DIR = _resolve_model_dir()
CLF_PATH = os.path.join(MODEL_DIR, T.CLASSIFIER_FILE)
REG_PATH = os.path.join(MODEL_DIR, T.REGRESSOR_FILE)
METRICS_PATH = os.path.join(MODEL_DIR, T.METRICS_FILE)
LOCK_PATH = os.path.join(MODEL_DIR, ".training.lock")

_lock = threading.Lock()
_state = {"status": "missing", "classifier": None, "regressor": None, "metrics": {}, "error": None}
_last_reload_attempt = 0.0


def _try_load(quiet: bool = False) -> bool:
    """Load both models from disk and smoke-test them. Never raises."""
    if not (os.path.exists(CLF_PATH) and os.path.exists(REG_PATH)):
        return False
    try:
        clf = joblib.load(CLF_PATH)
        reg = joblib.load(REG_PATH)
        probe = T.features_to_frame([dict.fromkeys(T.FEATURE_NAMES, 0.0)])
        clf.predict_proba(probe)
        reg.predict(probe)
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        with _lock:
            _state.update(status="ready", classifier=clf, regressor=reg, metrics=metrics, error=None)
        log.info("Models loaded from %s", MODEL_DIR)
        return True
    except Exception as exc:  # corrupt file, version mismatch, schema drift ...
        if not quiet:
            log.warning("Could not load saved models (%s); they will be retrained.", exc)
        return False


def _acquire_training_lock() -> bool:
    """Cross-process lock so several gunicorn workers do not all train at once."""
    for _ in range(2):
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK_PATH) > LOCK_STALE_SECONDS:
                    os.remove(LOCK_PATH)
                    continue
            except OSError:
                pass
            return False
        except OSError:
            return False
    return False


def _release_training_lock() -> None:
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def _bootstrap_models() -> None:
    """Background routine: load models, or train them if absent."""
    try:
        if _try_load():
            return
        with _lock:
            _state["status"] = "training"
        for _attempt in range(2):
            if _acquire_training_lock():
                try:
                    log.info("No usable models found - training a quick model in the background ...")
                    T.run_training(source="auto", quick=True, model_dir=MODEL_DIR)
                finally:
                    _release_training_lock()
                break
            # Another worker holds the lock: wait until it finishes (lock released) or models load.
            log.info("Another worker is training; waiting for its models ...")
            deadline = time.time() + WAIT_FOR_OTHER_WORKER_SECONDS
            while time.time() < deadline:
                if _try_load(quiet=True):
                    return
                if not os.path.exists(LOCK_PATH):
                    break  # the other worker finished (or crashed); re-check / take over
                time.sleep(2)
        if not _try_load():
            raise RuntimeError("Models are still unavailable after training.")
    except Exception as exc:
        log.exception("Model bootstrap failed; serving rule-based estimates.")
        with _lock:
            _state.update(status="failed", error=str(exc))


def _maybe_reload() -> None:
    """If another worker finished training, pick its files up (rate limited)."""
    global _last_reload_attempt
    now = time.time()
    if _state["status"] == "ready" or now - _last_reload_attempt < 5:
        return
    _last_reload_attempt = now
    if os.path.exists(CLF_PATH) and os.path.exists(REG_PATH):
        _try_load(quiet=True)


def start_bootstrap() -> None:
    threading.Thread(target=_bootstrap_models, name="model-bootstrap", daemon=True).start()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
FIELD_RULES = {
    "temperature": (-60.0, 60.0, "degrees Celsius"),
    "humidity": (0.0, 100.0, "percent"),
    "wind_speed": (0.0, 250.0, "km/h"),
}
LOOKUP = {name.lower(): i for name, i in T.COND_TO_ID.items()}


def _to_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def validate_payload(payload):
    """
    Returns (days, errors, status). ``days`` is a list of dicts with numeric
    values and a condition id, oldest first. ``status`` is 400 for structural
    problems and 422 for out-of-range values.
    """
    if not isinstance(payload, dict):
        return None, [{"field": "body", "message": "Body must be a JSON object."}], 400
    raw_days = payload.get("days")
    if not isinstance(raw_days, list):
        return None, [{"field": "days", "message": "'days' must be a list of daily records (oldest first)."}], 400
    if not (T.MIN_DAYS <= len(raw_days) <= T.MAX_DAYS):
        return None, [{"field": "days", "message": f"Provide between {T.MIN_DAYS} and {T.MAX_DAYS} days; received {len(raw_days)}."}], 422

    errors, structural = [], False
    days = []
    for idx, item in enumerate(raw_days, start=1):
        if not isinstance(item, dict):
            errors.append({"day": idx, "field": "day", "message": "Each day must be a JSON object."})
            structural = True
            continue
        clean = {}
        for key, (lo, hi, unit) in FIELD_RULES.items():
            if key not in item:
                errors.append({"day": idx, "field": key, "message": f"Missing '{key}'."})
                structural = True
                continue
            num = _to_number(item[key])
            if num is None:
                errors.append({"day": idx, "field": key, "message": f"'{key}' must be a finite number."})
                structural = True
            elif not lo <= num <= hi:
                errors.append({"day": idx, "field": key, "message": f"'{key}' must be between {lo:g} and {hi:g} ({unit})."})
            else:
                clean[key] = num
        cond = item.get("condition")
        if cond is None:
            errors.append({"day": idx, "field": "condition", "message": "Missing 'condition'."})
            structural = True
        elif not isinstance(cond, str) or cond.strip().lower() not in LOOKUP:
            errors.append({"day": idx, "field": "condition", "message": f"'condition' must be one of {', '.join(T.CONDITIONS)}."})
            structural = True
        else:
            clean["condition"] = LOOKUP[cond.strip().lower()]
        days.append(clean)

    if errors:
        return None, errors, 400 if structural else 422
    return days, [], 200


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def heuristic_predict(f: dict):
    """Transparent rule-based estimate used only while the ML models are unavailable."""
    temp = 0.6 * f["temp_ewm"] + 0.4 * f["temp_lag1"] + 0.5 * f["temp_slope7"]
    rain = 0.07 * (f["hum_ewm"] - 72) + 0.8 * f["rain_lag1"] + 1.0 * f["frac_rainy7"] + 0.04 * f["hum_delta3"] - 1.0
    sunny = -0.07 * (f["hum_ewm"] - 55) + 0.8 * f["sunny_lag1"] + 1.0 * f["frac_sunny7"] - 0.5
    cloudy = 0.4 - 0.02 * abs(f["hum_ewm"] - 65)
    return _softmax([sunny, cloudy, rain]), float(temp)


def _percentages(probs) -> dict:
    """Round to one decimal while guaranteeing the values add up to 100.0."""
    vals = [round(float(p) * 100.0, 1) for p in probs]
    drift = round(100.0 - sum(vals), 1)
    vals[int(np.argmax(vals))] = round(vals[int(np.argmax(vals))] + drift, 1)
    return dict(zip(T.CONDITIONS, vals))


def run_inference(days: list) -> dict:
    feats = T.build_features(
        [d["temperature"] for d in days],
        [d["humidity"] for d in days],
        [d["wind_speed"] for d in days],
        [d["condition"] for d in days],
    )
    with _lock:
        clf, reg, status, metrics = _state["classifier"], _state["regressor"], _state["status"], dict(_state["metrics"])

    if clf is not None and reg is not None:
        row = T.features_to_frame([feats])
        raw = clf.predict_proba(row)[0]
        probs = np.zeros(len(T.CONDITIONS))
        for cls, p in zip(clf.classes_, raw):
            probs[int(cls)] = p
        temp = float(reg.predict(row)[0])
        mode = "machine-learning"
        rmse = float(metrics.get("holdout", {}).get("temperature_rmse", 2.5))
    else:
        probs, temp = heuristic_predict(feats)
        mode = "rule-based-fallback"
        rmse = 3.0

    confidence = _percentages(probs)
    condition = max(confidence, key=confidence.get)
    margin = 1.645 * rmse  # approximate 90% interval assuming roughly normal errors
    holdout = metrics.get("holdout", {})
    return {
        "condition": condition,
        "temperature_c": round(temp, 1),
        "temperature_range_c": [round(temp - margin, 1), round(temp + margin, 1)],
        "confidence": confidence,
        "days_used": len(days),
        "model": {
            "mode": mode,
            "status": status,
            "data_source": metrics.get("data_source"),
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_temperature_rmse": holdout.get("temperature_rmse"),
        },
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/health")
def health():
    _maybe_reload()
    with _lock:
        status, error, metrics = _state["status"], _state["error"], _state["metrics"]
    body = {"service": "skycast", "status": status, "models_ready": status == "ready"}
    if error:
        body["error"] = error
    if metrics:
        body["trained_at"] = metrics.get("trained_at")
        body["data_source"] = metrics.get("data_source")
    return jsonify(body), 200


@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify(error="Content-Type must be application/json."), 415
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request body is not valid JSON."), 400
    days, errors, status = validate_payload(payload)
    if errors:
        return jsonify(error="Invalid input.", details=errors), status
    _maybe_reload()
    try:
        return jsonify(run_inference(days)), 200
    except Exception:
        log.exception("Inference failed")
        return jsonify(error="Prediction failed on the server. Please try again."), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify(error="Method not allowed."), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Request body too large."), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Internal server error."), 500


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
    )
    if request.path == "/predict" or request.path == "/health":
        resp.headers["Cache-Control"] = "no-store"
    return resp


# --------------------------------------------------------------------------- #
# Frontend (single embedded page)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skycast - machine learning weather prediction</title>
<meta name="description" content="Enter the last 7 to 10 days of weather and get today's predicted condition and temperature.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%E2%9B%85%3C/text%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg0:#080d1c; --bg1:#0f1a33; --bg2:#16254a;
  --ink:#e9eefc; --muted:#94a3c6; --faint:#62718f;
  --line:rgba(255,255,255,.10); --glass:rgba(255,255,255,.055); --glass-strong:rgba(255,255,255,.09);
  --sunny:#ffc247; --cloudy:#9fb6e0; --rainy:#4cc3ff; --danger:#ff7a8a; --ok:#5ee3a1;
  --accent:var(--rainy);
  --r-lg:22px; --r-md:14px; --r-sm:10px;
  font-family:"Outfit",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{
  margin:0; min-height:100vh; color:var(--ink); line-height:1.5;
  background:
    radial-gradient(900px 520px at 12% -8%, rgba(76,195,255,.16), transparent 60%),
    radial-gradient(760px 480px at 100% 8%, rgba(255,194,71,.10), transparent 62%),
    linear-gradient(160deg,var(--bg0),var(--bg1) 55%,var(--bg2));
  background-attachment:fixed;
}
.shell{max-width:1240px; margin:0 auto; padding:40px 24px 56px}
.top{margin-bottom:28px}
h1{font-size:clamp(2rem,4vw,2.9rem); font-weight:600; letter-spacing:-.02em; margin:0 0 6px}
.top p{margin:0; color:var(--muted); font-size:1.05rem; max-width:60ch}
.banner{
  margin:0 0 20px; padding:12px 16px; border-radius:var(--r-md); border:1px solid var(--line);
  background:var(--glass); display:flex; gap:12px; align-items:center; font-size:.95rem;
}
.banner[hidden]{display:none}
.banner .dot{width:9px;height:9px;border-radius:50%;background:var(--sunny);flex:none;animation:pulse 1.4s ease-in-out infinite}
.banner.failed .dot{background:var(--danger);animation:none}
@keyframes pulse{50%{opacity:.35}}
.layout{display:grid; grid-template-columns:minmax(0,1.55fr) minmax(320px,1fr); gap:24px; align-items:start}
.panel{
  background:var(--glass); border:1px solid var(--line); border-radius:var(--r-lg);
  backdrop-filter:blur(18px) saturate(140%); -webkit-backdrop-filter:blur(18px) saturate(140%);
  box-shadow:0 24px 60px -28px rgba(0,0,0,.65); padding:24px;
}
.panel-head{display:flex; flex-wrap:wrap; gap:14px; align-items:center; justify-content:space-between; margin-bottom:18px}
h2{font-size:1.25rem; font-weight:600; margin:0; letter-spacing:-.01em}
.controls{display:flex; flex-wrap:wrap; gap:10px; align-items:center}
.stepper{display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; background:var(--glass-strong)}
.stepper button{width:36px;height:36px;border:0;background:transparent;color:var(--ink);font:inherit;font-size:1.2rem;cursor:pointer;border-radius:999px}
.stepper button:hover:not(:disabled){background:rgba(255,255,255,.08)}
.stepper button:disabled{color:var(--faint);cursor:not-allowed}
.stepper output{min-width:74px;text-align:center;font-weight:500;font-variant-numeric:tabular-nums}
select,input{
  font:inherit; color:var(--ink); background:rgba(8,13,28,.55); border:1px solid var(--line);
  border-radius:var(--r-sm); padding:9px 11px; width:100%;
}
.controls select{width:auto; padding:8px 12px; border-radius:999px; background:var(--glass-strong)}
select option{background:#0f1a33;color:var(--ink)}
input[type=number]{font-variant-numeric:tabular-nums}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
button.ghost{
  font:inherit; color:var(--muted); background:transparent; border:1px solid var(--line);
  border-radius:999px; padding:8px 14px; cursor:pointer;
}
button.ghost:hover{color:var(--ink); background:rgba(255,255,255,.06)}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px}
.day{
  --tone:var(--cloudy);
  border:1px solid var(--line); border-top:3px solid var(--tone); border-radius:var(--r-md);
  background:rgba(8,13,28,.38); padding:14px; transition:border-color .2s;
}
.day[data-cond="Sunny"]{--tone:var(--sunny)}
.day[data-cond="Cloudy"]{--tone:var(--cloudy)}
.day[data-cond="Rainy"]{--tone:var(--rainy)}
.day-head{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px}
.day-label{font-weight:500}
.day-icon{font-size:1.5rem; line-height:1}
.field{display:block; margin-top:9px}
.field>span{display:flex; justify-content:space-between; font-size:.82rem; color:var(--muted); margin-bottom:4px}
.field input.invalid{border-color:var(--danger)}
.err{display:block; min-height:0; color:var(--danger); font-size:.78rem; margin-top:3px}
.err:empty{display:none}
.form-error{
  margin-top:16px; padding:12px 14px; border-radius:var(--r-md);
  border:1px solid rgba(255,122,138,.45); background:rgba(255,122,138,.09); color:#ffc7ce; font-size:.95rem;
}
.form-error[hidden]{display:none}
.cta{
  margin-top:20px; width:100%; display:flex; align-items:center; justify-content:center; gap:12px;
  font:inherit; font-weight:600; font-size:1.05rem; color:#06101f; padding:15px 20px;
  border:0; border-radius:var(--r-md); cursor:pointer;
  background:linear-gradient(135deg,#7fd8ff,#4cc3ff 55%,#3aa0ff);
  box-shadow:0 14px 34px -14px rgba(76,195,255,.7);
}
.cta:hover:not(:disabled){filter:brightness(1.07)}
.cta:disabled{cursor:progress; filter:saturate(.7) brightness(.9)}
.spinner{display:none; width:18px; height:18px; border:2.5px solid rgba(6,16,31,.3); border-top-color:#06101f; border-radius:50%; animation:spin .7s linear infinite}
.cta.loading .spinner{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

.result{position:sticky; top:24px; min-height:420px; overflow:hidden; --accent:var(--rainy)}
.result[data-cond="Sunny"]{--accent:var(--sunny)}
.result[data-cond="Cloudy"]{--accent:var(--cloudy)}
.result[data-cond="Rainy"]{--accent:var(--rainy)}
.result::before{
  content:""; position:absolute; inset:-30% -20% auto auto; width:360px; height:360px; border-radius:50%;
  background:radial-gradient(circle,var(--accent),transparent 68%); opacity:.0; transition:opacity .6s; pointer-events:none;
}
.result.has-data::before{opacity:.22}
.empty{color:var(--muted); display:grid; gap:10px; place-items:start; padding-top:8px}
.empty .big{font-size:2.6rem}
.empty p{margin:0; max-width:34ch}
.out{position:relative}
.out[hidden],.empty[hidden],.skeleton[hidden]{display:none}
.hero-icon{font-size:5.2rem; line-height:1; filter:drop-shadow(0 10px 28px rgba(0,0,0,.35))}
.cond{font-size:2.1rem; font-weight:600; margin:6px 0 0; letter-spacing:-.02em}
.temp{display:flex; align-items:baseline; gap:10px; margin-top:2px}
.temp strong{font-size:3.3rem; font-weight:300; font-variant-numeric:tabular-nums; letter-spacing:-.03em}
.temp span{color:var(--muted); font-size:.95rem}
.bars{margin-top:22px; display:grid; gap:12px}
.bar-row{display:grid; grid-template-columns:78px 1fr 56px; gap:10px; align-items:center; font-size:.95rem}
.bar-row .name{color:var(--muted)}
.bar-row .pct{text-align:right; font-variant-numeric:tabular-nums; font-weight:500}
.track{height:10px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden}
.fill{height:100%; width:0; border-radius:999px; background:var(--c); transition:width .8s cubic-bezier(.2,.8,.2,1)}
.fill[data-c="Sunny"]{--c:var(--sunny)} .fill[data-c="Cloudy"]{--c:var(--cloudy)} .fill[data-c="Rainy"]{--c:var(--rainy)}
.meta{margin-top:22px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:.86rem}
.meta b{color:var(--ink); font-weight:500}
.tag{display:inline-block; padding:2px 9px; border-radius:999px; font-size:.78rem; margin-right:6px; border:1px solid var(--line)}
.tag.ml{color:var(--ok); border-color:rgba(94,227,161,.4)}
.tag.fb{color:var(--sunny); border-color:rgba(255,194,71,.45)}
.skeleton{display:grid; gap:14px}
.sk{height:16px; border-radius:8px; background:linear-gradient(90deg,rgba(255,255,255,.06),rgba(255,255,255,.14),rgba(255,255,255,.06)); background-size:200% 100%; animation:shine 1.2s linear infinite}
.sk.lg{height:84px;width:84px;border-radius:50%}
@keyframes shine{to{background-position:-200% 0}}
footer{margin-top:34px; color:var(--faint); font-size:.85rem}
footer a{color:var(--muted)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
@media (max-width:980px){.layout{grid-template-columns:1fr}.result{position:relative;top:0;min-height:0}}
@media (max-width:520px){.shell{padding:24px 14px 40px}.panel{padding:16px}.grid{grid-template-columns:1fr 1fr}}
@media (max-width:400px){.grid{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<main class="shell">
  <header class="top">
    <h1>Skycast</h1>
    <p>Enter the last 7 to 10 days of weather, oldest first. Skycast predicts today's condition and temperature.</p>
  </header>

  <div id="banner" class="banner" role="status" hidden><span class="dot"></span><span id="bannerText"></span></div>

  <div class="layout">
    <section class="panel" aria-labelledby="histTitle">
      <div class="panel-head">
        <h2 id="histTitle">Recent weather</h2>
        <div class="controls">
          <div class="stepper" role="group" aria-label="Number of days">
            <button type="button" id="dec" aria-label="Use one fewer day">&minus;</button>
            <output id="dayCount" aria-live="polite">10 days</output>
            <button type="button" id="inc" aria-label="Use one more day">+</button>
          </div>
          <select id="preset" aria-label="Load example data">
            <option value="">Load an example</option>
            <option value="default">Humid and unsettled</option>
            <option value="monsoon">Monsoon spell</option>
            <option value="heat">Dry heat</option>
            <option value="overcast">Cool and overcast</option>
          </select>
          <button type="button" id="reset" class="ghost">Reset</button>
        </div>
      </div>

      <form id="form" novalidate>
        <div id="grid" class="grid"></div>
        <div id="formError" class="form-error" role="alert" hidden></div>
        <button id="submit" class="cta" type="submit">
          <span id="ctaLabel">Predict today's weather</span><span class="spinner" aria-hidden="true"></span>
        </button>
      </form>
    </section>

    <aside id="result" class="panel result" aria-live="polite" aria-label="Prediction result">
      <div id="empty" class="empty">
        <div class="big" aria-hidden="true">&#9925;</div>
        <h2>Today's forecast</h2>
        <p>Fill in the recent days on the left, then choose Predict today's weather.</p>
      </div>
      <div id="loading" class="skeleton" hidden aria-hidden="true">
        <div class="sk lg"></div><div class="sk" style="width:60%"></div><div class="sk" style="width:85%"></div>
        <div class="sk" style="width:75%"></div><div class="sk" style="width:90%"></div>
      </div>
      <div id="out" class="out" hidden>
        <div class="hero-icon" id="heroIcon" aria-hidden="true"></div>
        <p class="cond" id="condText"></p>
        <div class="temp"><strong id="tempText"></strong><span id="rangeText"></span></div>
        <div class="bars" id="bars"></div>
        <div class="meta" id="meta"></div>
      </div>
    </aside>
  </div>

  <footer>Skycast is free software under the <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener">GNU GPL v3.0</a>. Predictions are statistical estimates, not official forecasts.</footer>
</main>

<script>
(function(){
  "use strict";
  var MIN_DAYS = 7, MAX_DAYS = 10;
  var ICON = {Sunny:"\u2600\uFE0F", Cloudy:"\u2601\uFE0F", Rainy:"\uD83C\uDF27\uFE0F"};
  var CONDS = ["Sunny","Cloudy","Rainy"];
  var FIELDS = [
    {key:"temperature", label:"Temperature", unit:"\u00B0C", min:-60, max:60, step:"0.1"},
    {key:"humidity",    label:"Humidity",    unit:"%",       min:0,   max:100, step:"1"},
    {key:"wind_speed",  label:"Wind speed",  unit:"km/h",    min:0,   max:250, step:"0.1"}
  ];

  // Index 0 = oldest day. Only the most recent `count` rows are shown.
  var PRESETS = {
    "default": {
      temperature:[29,30,31,30,29,28,28,27,28,29], humidity:[62,66,70,74,78,82,80,84,81,77],
      wind_speed:[9,10,12,14,16,18,15,17,13,12],
      condition:["Sunny","Cloudy","Cloudy","Cloudy","Rainy","Rainy","Cloudy","Rainy","Cloudy","Cloudy"]},
    "monsoon": {
      temperature:[30,29,28,28,27,28,27,27,28,28], humidity:[78,82,86,88,90,89,91,90,88,87],
      wind_speed:[12,15,18,20,22,19,17,21,18,16],
      condition:["Cloudy","Rainy","Rainy","Rainy","Rainy","Cloudy","Rainy","Rainy","Cloudy","Rainy"]},
    "heat": {
      temperature:[33,34,35,35,36,36,37,36,36,37], humidity:[38,35,33,32,30,29,28,30,29,27],
      wind_speed:[8,9,8,10,9,11,10,9,8,9],
      condition:["Sunny","Sunny","Sunny","Sunny","Sunny","Sunny","Sunny","Sunny","Sunny","Sunny"]},
    "overcast": {
      temperature:[16,15,15,14,14,13,14,13,13,14], humidity:[70,72,75,74,78,77,79,80,78,81],
      wind_speed:[15,17,16,19,18,20,17,18,19,16],
      condition:["Cloudy","Cloudy","Cloudy","Rainy","Cloudy","Cloudy","Cloudy","Cloudy","Rainy","Cloudy"]}
  };

  var count = 10;
  var rows = [];
  var $ = function(id){ return document.getElementById(id); };

  function esc(s){ return String(s).replace(/[&<>"']/g, function(c){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]; }); }

  function loadPreset(name){
    var p = PRESETS[name];
    rows = [];
    for (var i = 0; i < MAX_DAYS; i++){
      rows.push({temperature:String(p.temperature[i]), humidity:String(p.humidity[i]),
                 wind_speed:String(p.wind_speed[i]), condition:p.condition[i]});
    }
  }

  function dayLabel(offset){ return offset === 1 ? "Yesterday" : offset + " days ago"; }

  function render(){
    $("dayCount").textContent = count + " days";
    $("dec").disabled = count <= MIN_DAYS;
    $("inc").disabled = count >= MAX_DAYS;
    var html = "";
    for (var v = 0; v < count; v++){
      var idx = MAX_DAYS - count + v;      // position inside rows
      var offset = count - v;              // 1 = yesterday
      var r = rows[idx];
      html += '<div class="day" data-cond="'+esc(r.condition)+'" data-idx="'+idx+'">' +
        '<div class="day-head"><span class="day-label">'+dayLabel(offset)+'</span>' +
        '<span class="day-icon" aria-hidden="true">'+ICON[r.condition]+'</span></div>';
      FIELDS.forEach(function(f){
        var id = "f-"+idx+"-"+f.key;
        html += '<label class="field" for="'+id+'"><span>'+f.label+'<span>'+f.unit+'</span></span>' +
          '<input id="'+id+'" type="number" inputmode="decimal" step="'+f.step+'" min="'+f.min+'" max="'+f.max+'" ' +
          'data-idx="'+idx+'" data-key="'+f.key+'" value="'+esc(r[f.key])+'" aria-describedby="'+id+'-e" required>' +
          '<small class="err" id="'+id+'-e"></small></label>';
      });
      html += '<label class="field" for="c-'+idx+'"><span>Condition</span></label>' +
        '<select id="c-'+idx+'" data-idx="'+idx+'" data-key="condition" aria-label="Condition, '+dayLabel(offset)+'">';
      CONDS.forEach(function(c){
        html += '<option value="'+c+'"'+(c===r.condition?' selected':'')+'>'+ICON[c]+' '+c+'</option>';
      });
      html += '</select></div>';
    }
    $("grid").innerHTML = html;
  }

  $("grid").addEventListener("input", function(e){
    var t = e.target, idx = t.getAttribute("data-idx"), key = t.getAttribute("data-key");
    if (idx === null) return;
    rows[+idx][key] = t.value;
    if (t.tagName === "INPUT"){ t.classList.remove("invalid"); $(t.id+"-e").textContent = ""; }
  });
  $("grid").addEventListener("change", function(e){
    var t = e.target;
    if (t.tagName !== "SELECT") return;
    var idx = +t.getAttribute("data-idx");
    rows[idx].condition = t.value;
    var card = t.closest(".day");
    card.setAttribute("data-cond", t.value);
    card.querySelector(".day-icon").textContent = ICON[t.value];
  });

  $("inc").addEventListener("click", function(){ if (count < MAX_DAYS){ count++; render(); } });
  $("dec").addEventListener("click", function(){ if (count > MIN_DAYS){ count--; render(); } });
  $("reset").addEventListener("click", function(){ loadPreset("default"); count = 10; $("preset").value = ""; clearError(); render(); });
  $("preset").addEventListener("change", function(e){
    if (!e.target.value) return;
    loadPreset(e.target.value); clearError(); render();
  });

  function clearError(){ $("formError").hidden = true; $("formError").textContent = ""; }
  function showError(msg){ var el = $("formError"); el.textContent = msg; el.hidden = false; }

  function validate(){
    var firstBad = null, bad = 0;
    for (var idx = MAX_DAYS - count; idx < MAX_DAYS; idx++){
      FIELDS.forEach(function(f){
        var input = $("f-"+idx+"-"+f.key), err = $("f-"+idx+"-"+f.key+"-e");
        var raw = String(rows[idx][f.key]).trim(), num = Number(raw), msg = "";
        if (raw === "" || !isFinite(num)) msg = "Enter a number";
        else if (num < f.min || num > f.max) msg = "Use "+f.min+" to "+f.max;
        input.classList.toggle("invalid", !!msg);
        input.setAttribute("aria-invalid", msg ? "true" : "false");
        err.textContent = msg;
        if (msg){ bad++; if (!firstBad) firstBad = input; }
      });
    }
    if (firstBad){ firstBad.focus(); }
    return bad;
  }

  function setLoading(on){
    var btn = $("submit");
    btn.disabled = on; btn.classList.toggle("loading", on);
    $("ctaLabel").textContent = on ? "Predicting..." : "Predict today's weather";
    $("loading").hidden = !on;
    if (on){ $("empty").hidden = true; $("out").hidden = true; }
  }

  function showResult(d){
    var res = $("result");
    res.setAttribute("data-cond", d.condition); res.classList.add("has-data");
    $("heroIcon").textContent = ICON[d.condition];
    $("condText").textContent = d.condition;
    $("tempText").textContent = d.temperature_c.toFixed(1) + "\u00B0C";
    $("rangeText").textContent = "likely " + d.temperature_range_c[0].toFixed(1) + " to " + d.temperature_range_c[1].toFixed(1) + "\u00B0C";

    var order = CONDS.slice().sort(function(a,b){ return d.confidence[b] - d.confidence[a]; });
    $("bars").innerHTML = order.map(function(c){
      return '<div class="bar-row"><span class="name">'+ICON[c]+' '+c+'</span>' +
        '<div class="track"><div class="fill" data-c="'+c+'" data-w="'+d.confidence[c]+'"></div></div>' +
        '<span class="pct">'+d.confidence[c].toFixed(1)+'%</span></div>';
    }).join("");

    var m = d.model || {}, meta = "";
    if (m.mode === "machine-learning"){
      meta += '<span class="tag ml">Machine learning model</span>';
      if (typeof m.holdout_accuracy === "number")
        meta += 'Condition accuracy on unseen data: <b>' + (m.holdout_accuracy*100).toFixed(1) + '%</b>';
      if (typeof m.holdout_temperature_rmse === "number")
        meta += ' &middot; temperature error (RMSE): <b>' + m.holdout_temperature_rmse.toFixed(2) + '\u00B0C</b>';
      if (m.data_source) meta += '<br>Trained on ' + esc(m.data_source) + ' data, using ' + d.days_used + ' days of history.';
    } else {
      meta += '<span class="tag fb">Rule-based estimate</span>The machine learning models are still getting ready. Predict again in a minute for a model-based result.';
    }
    $("meta").innerHTML = meta;
    $("out").hidden = false;
    requestAnimationFrame(function(){ requestAnimationFrame(function(){
      document.querySelectorAll(".fill").forEach(function(el){ el.style.width = el.getAttribute("data-w") + "%"; });
    }); });
  }

  $("form").addEventListener("submit", function(e){
    e.preventDefault(); clearError();
    if (validate() > 0){ showError("Fix the highlighted fields and try again."); return; }
    var days = [];
    for (var idx = MAX_DAYS - count; idx < MAX_DAYS; idx++){
      var r = rows[idx];
      days.push({temperature:Number(r.temperature), humidity:Number(r.humidity),
                 wind_speed:Number(r.wind_speed), condition:r.condition});
    }
    setLoading(true);
    var ctrl = new AbortController(), timer = setTimeout(function(){ ctrl.abort(); }, 25000);
    fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"},
                       body:JSON.stringify({days:days}), signal:ctrl.signal})
      .then(function(resp){ return resp.json().catch(function(){ return {}; }).then(function(body){ return {ok:resp.ok, status:resp.status, body:body}; }); })
      .then(function(r){
        setLoading(false);
        if (!r.ok){
          var msg = (r.body && r.body.error) || ("Request failed (" + r.status + ").");
          if (r.body && r.body.details && r.body.details.length){
            msg += " " + r.body.details.slice(0,3).map(function(x){ return (x.day ? "Day "+x.day+": " : "") + x.message; }).join(" ");
          }
          if ($("out").hidden) $("empty").hidden = false;
          showError(msg); return;
        }
        showResult(r.body);
      })
      .catch(function(err){
        setLoading(false);
        if ($("out").hidden) $("empty").hidden = false;
        showError(err && err.name === "AbortError" ? "The server took too long to respond. Try again." : "Could not reach the server. Check your connection and try again.");
      })
      .then(function(){ clearTimeout(timer); });
  });

  // ---- model status banner ----
  function pollHealth(){
    fetch("/health", {cache:"no-store"}).then(function(r){ return r.json(); }).then(function(h){
      var b = $("banner"), t = $("bannerText");
      if (h.status === "training" || h.status === "missing"){
        b.className = "banner"; t.textContent = "The prediction models are training in the background. Predictions use a rule-based estimate until they are ready.";
        b.hidden = false; setTimeout(pollHealth, 3000);
      } else if (h.status === "failed"){
        b.className = "banner failed"; t.textContent = "The models could not be trained. Predictions use a rule-based estimate.";
        b.hidden = false;
      } else { b.hidden = true; }
    }).catch(function(){ /* ignore transient errors */ });
  }

  loadPreset("default"); render(); pollHealth();
})();
</script>
</body>
</html>
"""

# Start loading / training as soon as the module is imported (gunicorn app:app).
start_bootstrap()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
