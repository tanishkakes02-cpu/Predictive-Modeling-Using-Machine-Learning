from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import threading
import time

import numpy as np

from train_predictive_model import (
    DecisionTreeClassifier,
    LogisticRegressionGD,
    RandomForestClassifier,
    make_dataset,
    metrics,
    train_test_split,
)


APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
FEATURES = [
    "age",
    "monthly_income",
    "debt_ratio",
    "credit_score",
    "prior_defaults",
    "engagement_score",
]
MODEL_LOCK = threading.Lock()
STATE: dict = {}


def train_models() -> None:
    started = time.time()
    data = make_dataset()
    X = data[FEATURES].to_numpy(dtype=float)
    y = data["outcome_default_risk"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    models = {
        "Logistic Regression": LogisticRegressionGD().fit(X_train_std, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=22).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=6).fit(X_train, y_train),
    }

    model_metrics = []
    for name, model in models.items():
        test_input = X_test_std if name == "Logistic Regression" else X_test
        result = metrics(y_test, model.predict_proba(test_input))
        model_metrics.append(
            {
                "model": name,
                **{key: round(float(value), 4) for key, value in result.items()},
            }
        )
    model_metrics.sort(key=lambda item: item["roc_auc"], reverse=True)

    importance = models["Random Forest"].feature_importances_
    feature_importance = [
        {"feature": feature.replace("_", " ").title(), "value": round(float(value), 4)}
        for feature, value in sorted(zip(FEATURES, importance), key=lambda item: item[1], reverse=True)
    ]

    with MODEL_LOCK:
        STATE.clear()
        STATE.update(
            {
                "models": models,
                "mean": mean,
                "std": std,
                "metrics": model_metrics,
                "best_model": model_metrics[0]["model"],
                "feature_importance": feature_importance,
                "dataset_rows": len(data),
                "positive_rate": round(float(y.mean()), 4),
                "trained_in": round(time.time() - started, 2),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


def state_summary() -> dict:
    with MODEL_LOCK:
        return {
            "metrics": STATE["metrics"],
            "best_model": STATE["best_model"],
            "feature_importance": STATE["feature_importance"],
            "dataset_rows": STATE["dataset_rows"],
            "positive_rate": STATE["positive_rate"],
            "trained_in": STATE["trained_in"],
            "trained_at": STATE["trained_at"],
        }


def predict(payload: dict) -> dict:
    values = np.array([[float(payload[name]) for name in FEATURES]], dtype=float)
    with MODEL_LOCK:
        models = STATE["models"]
        standardized = (values - STATE["mean"]) / STATE["std"]
        probabilities = {
            name: float(model.predict_proba(standardized if name == "Logistic Regression" else values)[0])
            for name, model in models.items()
        }
        best_name = STATE["best_model"]

    probability = probabilities[best_name]
    risk_level = "High" if probability >= 0.65 else "Moderate" if probability >= 0.35 else "Low"
    decision = "Manual review recommended" if probability >= 0.5 else "Likely safe to approve"
    drivers = []
    if values[0, 2] > 0.45:
        drivers.append("High debt ratio")
    if values[0, 3] < 620:
        drivers.append("Low credit score")
    if values[0, 4] > 0:
        drivers.append("Previous defaults")
    if values[0, 5] < 40:
        drivers.append("Low engagement")
    if not drivers:
        drivers.append("No major negative signals")

    return {
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "decision": decision,
        "best_model": best_name,
        "model_probabilities": {name: round(value, 4) for name, value in probabilities.items()},
        "drivers": drivers,
    }


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/metrics":
            self.send_json(state_summary())
            return
        if self.path == "/health":
            self.send_json({"status": "ok"})
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/predict":
                self.send_json(predict(payload))
                return
            if self.path == "/api/retrain":
                train_models()
                self.send_json(state_summary())
                return
            self.send_json({"error": "Not found"}, 404)
        except (KeyError, ValueError, TypeError) as exc:
            self.send_json({"error": f"Invalid input: {exc}"}, 400)
        except Exception as exc:
            self.send_json({"error": f"Server error: {exc}"}, 500)


if __name__ == "__main__":
    print("Training models...")
    train_models()
    print(f"Dashboard ready at http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), AppHandler).serve_forever()
