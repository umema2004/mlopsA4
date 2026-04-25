from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import joblib, numpy as np, pandas as pd, time, os

app = FastAPI(title="Fraud Detection API")

# ── Prometheus metrics ────────────────────────────────────────────
REQUEST_COUNT   = Counter("fraud_api_requests_total",   "Total requests",          ["endpoint","status"])
REQUEST_LATENCY = Histogram("fraud_api_latency_seconds","Request latency",          ["endpoint"])
FRAUD_PREDICTED = Counter("fraud_predictions_total",    "Fraud predictions",        ["label"])
CONFIDENCE_HIST = Histogram("fraud_confidence_score",   "Prediction confidence",    buckets=[0.1*i for i in range(11)])
RECALL_GAUGE    = Gauge("fraud_model_recall",            "Rolling fraud recall")
FPR_GAUGE       = Gauge("fraud_model_fpr",               "False positive rate")

# Rolling window for recall/FPR
_tp=_fn=_fp=_tn=0

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/A4/models/hybrid_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️  Model not loaded: {e}")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: dict):
    global _tp, _fn, _fp, _tn
    start = time.time()
    try:
        features = np.array(data["features"]).reshape(1, -1)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        proba    = model.predict_proba(features)[0][1]
        label    = int(proba >= 0.45)               # optimal threshold from Task 4
        true_lbl = data.get("true_label", None)

        # Update rolling confusion matrix
        if true_lbl is not None:
            if true_lbl == 1 and label == 1: _tp += 1
            elif true_lbl == 1 and label == 0: _fn += 1
            elif true_lbl == 0 and label == 1: _fp += 1
            else: _tn += 1
            recall = _tp / (_tp + _fn + 1e-9)
            fpr    = _fp / (_fp + _tn + 1e-9)
            RECALL_GAUGE.set(recall)
            FPR_GAUGE.set(fpr)

        FRAUD_PREDICTED.labels(label=str(label)).inc()
        CONFIDENCE_HIST.observe(float(proba))
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time()-start)

        return {"fraud_probability": round(float(proba),4),
                "prediction": label,
                "threshold": 0.45}
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulate")
def simulate_traffic():
    """Simulate 50 predictions with random data (for demo/testing)"""
    import random
    results = []
    for _ in range(50):
        # Random feature vector (length 30 — adjust to your model's input size)
        feats   = [random.gauss(0,1) for _ in range(180)]
        true_lbl= random.choices([0,1], weights=[0.97,0.03])[0]
        r = predict({"features": feats, "true_label": true_lbl})
        results.append(r)
    return {"simulated": len(results), "sample": results[:3]}
