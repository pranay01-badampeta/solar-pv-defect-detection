"""Flask web interface for solar PV monitoring.

- Upload a panel image: the CNN returns the defect class with its confidence.
- Enter a sensor reading: the Random Forest / SVM model returns the fault type.

Train the models first (train_cnn.py and sensor_faults.py), then:
    python app.py        # http://127.0.0.1:5000
"""
import io
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image

from sensor_faults import FEATURES, add_features

MODELS = Path(__file__).parent / "models"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

ADVICE = {
    "Clean": "Panel looks clean. No action needed.",
    "Dusty": "Schedule cleaning: dust reduces output.",
    "Bird-drop": "Clean the affected area: droppings can cause hot spots.",
    "Snow-Covered": "Output blocked until snow melts or is cleared.",
    "Electrical-damage": "Inspect wiring and cells: electrical damage is a safety risk.",
    "Physical-Damage": "Inspect for cracks: the panel may need replacing.",
    "Normal": "Output matches the conditions.",
    "Partial shading": "Check for shade from nearby objects or debris.",
    "Soiling": "Output is uniformly low for the irradiance: schedule cleaning.",
    "Open circuit": "No current flowing: check connectors, fuses and cabling.",
    "Degradation": "Lower voltage and higher temperature: inspect the module.",
}


@lru_cache(maxsize=1)
def load_cnn():
    path = MODELS / "solar_cnn.pt"
    if not path.exists():
        return None
    import torch  # imported lazily so the sensor page works without PyTorch

    from model import build_model, eval_transform

    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], eval_transform, torch


@lru_cache(maxsize=1)
def load_sensor_model():
    path = MODELS / "sensor_model.joblib"
    return joblib.load(path) if path.exists() else None


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES,
                           cnn_ready=(MODELS / "solar_cnn.pt").exists(),
                           sensor_ready=(MODELS / "sensor_model.joblib").exists())


@app.post("/api/predict-image")
def predict_image():
    loaded = load_cnn()
    if loaded is None:
        return jsonify(error="No CNN model yet. Run train_cnn.py first."), 503
    file = request.files.get("image")
    if not file:
        return jsonify(error="Upload an image file."), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify(error="That file is not a readable image."), 400
    model, classes, transform, torch = loaded
    with torch.no_grad():
        probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
    ranked = sorted(zip(classes, probs.tolist()), key=lambda t: t[1], reverse=True)
    top = ranked[0][0]
    return jsonify(prediction=top, confidence=ranked[0][1], advice=ADVICE.get(top, ""),
                   probabilities=[{"label": c, "p": p} for c, p in ranked])


@app.post("/api/predict-sensor")
def predict_sensor():
    bundle = load_sensor_model()
    if bundle is None:
        return jsonify(error="No sensor model yet. Run sensor_faults.py first."), 503
    data = request.get_json(silent=True) or request.form
    try:
        row = {f: float(data[f]) for f in FEATURES if f != "power"}
    except (KeyError, ValueError):
        return jsonify(error="Enter a number for every reading."), 400
    if row["irradiance"] <= 0:
        return jsonify(error="Irradiance must be above 0 W/m²."), 400
    row["power"] = row["voltage"] * row["current"]
    X = add_features(pd.DataFrame([row]))
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda t: t[1], reverse=True)
    top = ranked[0][0]
    return jsonify(prediction=top, confidence=float(ranked[0][1]), model=bundle["name"],
                   advice=ADVICE.get(top, ""), power=round(row["power"], 1),
                   probabilities=[{"label": c, "p": float(p)} for c, p in ranked])


if __name__ == "__main__":
    app.run(debug=True)
