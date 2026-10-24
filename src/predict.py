"""Load trained models and turn sensor readings into predictions."""
from pathlib import Path

import joblib
import pandas as pd

from .config import (CRITICAL_THRESHOLD, FEATURES, MODELS_DIR, SENSORS,
                     WARNING_THRESHOLD)
from .features import add_features


def status_from_probability(p):
    if p >= CRITICAL_THRESHOLD:
        return "Critical"
    if p >= WARNING_THRESHOLD:
        return "Warning"
    return "Healthy"


class Predictor:
    def __init__(self, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        self.classifier = joblib.load(models_dir / "failure_classifier.joblib")
        self.regressor = joblib.load(models_dir / "rul_regressor.joblib")

    def predict_features(self, feats):
        """feats: DataFrame that already contains the FEATURES columns."""
        x = feats[FEATURES]
        return self.classifier.predict_proba(x)[:, 1], self.regressor.predict(x)

    def predict_readings(self, readings):
        """readings: DataFrame with machine_id, step and raw sensor columns."""
        return self.predict_features(add_features(readings))

    def predict_latest(self, readings):
        """Predict for the most recent reading of a single machine's history."""
        proba, rul = self.predict_readings(readings)
        p, r = float(proba[-1]), float(max(rul[-1], 0.0))
        return {"failure_probability": round(p, 4),
                "status": status_from_probability(p),
                "estimated_rul": round(r, 1)}


def readings_from_records(records, machine_id="api"):
    """Build a readings DataFrame from a list of {sensor: value} dicts."""
    df = pd.DataFrame(records)
    missing = [s for s in SENSORS if s not in df.columns]
    if missing:
        raise ValueError(f"missing sensor fields: {missing}")
    df = df[SENSORS].astype(float).copy()
    df.insert(0, "step", range(len(df)))
    df.insert(0, "machine_id", machine_id)
    return df
