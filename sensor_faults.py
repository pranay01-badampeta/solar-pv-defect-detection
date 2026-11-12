"""Fault detection from PV sensor readings with Random Forest and SVM.

Classifies each reading (irradiance, module temperature, ambient temperature,
DC voltage, DC current, power) as Normal, Partial shading, Soiling,
Open circuit or Degradation.

Real plant data with labelled faults is rarely public, so `--simulate`
generates readings from a simple PV model with injected faults. Replace it
with your own labelled CSV (same columns plus a `fault` column) when you have one.

Usage:
    python sensor_faults.py --simulate
    python sensor_faults.py --csv data/sensors.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["irradiance", "module_temp", "ambient_temp", "voltage", "current", "power"]
FAULTS = ["Normal", "Partial shading", "Soiling", "Open circuit", "Degradation"]
SEED = 42


def simulate(n: int = 6000) -> pd.DataFrame:
    """Readings from a 300 W module model (Voc 40 V, Isc 9.5 A) with injected faults."""
    rng = np.random.default_rng(SEED)
    irr = rng.uniform(150, 1050, n)                                 # W/m²
    amb = rng.uniform(5, 38, n)                                     # °C
    mod = amb + irr / 800 * 25 + rng.normal(0, 1.5, n)              # NOCT-style heating
    fault = rng.choice(len(FAULTS), n, p=[0.6, 0.1, 0.1, 0.1, 0.1])

    current = 9.5 * irr / 1000 * (1 + 0.0005 * (mod - 25))
    voltage = 32.0 * (1 - 0.0035 * (mod - 25)) + 1.2 * np.log(irr / 1000 + 1e-3)
    current, voltage = current.copy(), voltage.copy()

    m = fault == 1  # partial shading: current drop, bypass diodes lower voltage
    current[m] *= rng.uniform(0.45, 0.75, m.sum()); voltage[m] *= rng.uniform(0.62, 0.8, m.sum())
    m = fault == 2  # soiling: uniform current loss
    current[m] *= rng.uniform(0.72, 0.9, m.sum())
    m = fault == 3  # open circuit: no current, voltage near Voc
    current[m] = rng.uniform(0, 0.15, m.sum()); voltage[m] = 40 * (1 - 0.0035 * (mod[m] - 25))
    m = fault == 4  # degradation: lower voltage and current, hotter module
    current[m] *= rng.uniform(0.85, 0.95, m.sum()); voltage[m] *= rng.uniform(0.82, 0.92, m.sum())
    mod[m] += rng.uniform(3, 8, m.sum())

    current += rng.normal(0, 0.08, n)
    voltage += rng.normal(0, 0.4, n)
    df = pd.DataFrame({
        "irradiance": irr.round(1), "module_temp": mod.round(2), "ambient_temp": amb.round(2),
        "voltage": voltage.round(3), "current": current.clip(0).round(3),
    })
    df["power"] = (df["voltage"] * df["current"]).round(2)
    df["fault"] = [FAULTS[i] for i in fault]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise by irradiance so the models see performance, not just weather."""
    X = df[FEATURES].copy()
    X["current_per_irr"] = X["current"] / (X["irradiance"] / 1000)
    X["power_per_irr"] = X["power"] / (X["irradiance"] / 1000)
    X["temp_rise"] = X["module_temp"] - X["ambient_temp"]
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.simulate:
        df = simulate()
    else:
        raise SystemExit("Pass --csv path/to/labelled_readings.csv or --simulate.")

    X, y = add_features(df), df["fault"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True,
                                                         random_state=SEED)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        summary[name] = {"accuracy": accuracy_score(y_te, pred),
                         "macro_f1": f1_score(y_te, pred, average="macro")}
        print(f"\n=== {name} ===")
        print(classification_report(y_te, pred, digits=3))

    best = max(summary, key=lambda k: summary[k]["macro_f1"])
    joblib.dump({"model": models[best], "name": best}, out / "sensor_model.joblib")
    (out / "sensor_metrics.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(summary).T.round(3).to_string())
    print(f"\nBest model ({best}) saved to {out}/sensor_model.joblib")


if __name__ == "__main__":
    main()
