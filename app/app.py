"""Live dashboard: a simulated fleet of machines monitored by the trained models.

Run:  python app/app.py     then open http://127.0.0.1:5000
"""
import json
import sys
import threading
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.config import MODELS_DIR, RESULTS_DIR, SENSORS  # noqa: E402
from src.data_generator import Machine  # noqa: E402
from src.features import add_features  # noqa: E402
from src.predict import Predictor, readings_from_records, status_from_probability  # noqa: E402

NAMES = ["CNC Lathe 01", "Pump 02", "Compressor 03", "Conveyor 04", "Turbine 05", "Press 06"]
START_FRACTIONS = [0.15, 0.35, 0.55, 0.75, 0.85, 0.92]   # how worn each machine is at start-up
BUFFER = 40            # readings kept per machine for feature calculation and sparklines
DOWNTIME_TICKS = 4     # a failed machine is replaced after this many ticks


class Fleet:
    """Simulates several machines and scores them with the trained models."""

    def __init__(self, predictor, seed=7):
        self.predictor = predictor
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.clock = 0
        self.alerts = deque(maxlen=25)
        self.units = [self._new_unit(i, f) for i, f in enumerate(START_FRACTIONS)]
        for u in self.units:
            self._assess(u)

    def _new_unit(self, idx, wear_fraction=0.0):
        machine = Machine(f"M{idx + 1:02d}", self.rng)
        start = int(wear_fraction * machine.life)
        warm = min(start, BUFFER)
        machine.age = start - warm
        buffer = deque(maxlen=BUFFER)
        for _ in range(max(warm, 1)):
            buffer.append(machine.step())
        return {"idx": idx, "name": NAMES[idx], "machine": machine, "buffer": buffer,
                "status": "Healthy", "probability": 0.0, "rul": 0.0, "down": 0}

    def _alert(self, level, message):
        self.alerts.appendleft({"t": self.clock, "level": level, "message": message})

    def _assess(self, unit):
        feats = add_features(pd.DataFrame(list(unit["buffer"]))).tail(1)
        proba, rul = self.predictor.predict_features(feats)
        unit["probability"] = float(proba[0])
        unit["rul"] = float(max(rul[0], 0.0))
        new_status = status_from_probability(unit["probability"])
        old_status = unit["status"]
        if new_status != old_status and old_status != "Failed":
            if new_status == "Critical":
                self._alert("critical", f"{unit['name']}: CRITICAL, failure probability "
                                        f"{unit['probability']:.0%}. Schedule maintenance now.")
            elif new_status == "Warning" and old_status == "Healthy":
                self._alert("warning", f"{unit['name']}: warning, failure probability "
                                       f"{unit['probability']:.0%}.")
        unit["status"] = new_status

    def tick(self, steps=1):
        with self.lock:
            for _ in range(steps):
                self.clock += 1
                for u in self.units:
                    m = u["machine"]
                    if m.failed:
                        u["down"] += 1
                        if u["down"] >= DOWNTIME_TICKS:
                            self.units[u["idx"]] = self._new_unit(u["idx"])
                            self._alert("info", f"{u['name']}: replaced and back in service.")
                        continue
                    u["buffer"].append(m.step())
                    if m.failed:
                        warned = u["status"] in ("Warning", "Critical")
                        u["status"] = "Failed"
                        u["probability"], u["rul"] = 1.0, 0.0
                        self._alert("failure", f"{u['name']}: FAILED"
                                    + (" (was flagged in advance)." if warned else " (no prior warning)."))
            for u in self.units:
                if u["status"] != "Failed":
                    self._assess(u)

    def reset(self):
        with self.lock:
            self.clock = 0
            self.alerts.clear()
            self.units = [self._new_unit(i, f) for i, f in enumerate(START_FRACTIONS)]
            for u in self.units:
                self._assess(u)

    def snapshot(self):
        with self.lock:
            machines = []
            for u in self.units:
                buf = list(u["buffer"])
                machines.append({
                    "id": u["machine"].machine_id, "name": u["name"], "status": u["status"],
                    "probability": round(u["probability"], 3), "estimated_rul": round(u["rul"], 1),
                    "actual_rul": max(u["machine"].life - u["machine"].age, 0),
                    "latest": {s: round(buf[-1][s], 2) for s in SENSORS},
                    "spark": {s: [round(r[s], 3) for r in buf] for s in ("temperature", "vibration")},
                })
            return {"clock": self.clock, "machines": machines, "alerts": list(self.alerts)}


def create_app(models_dir=MODELS_DIR, results_dir=RESULTS_DIR):
    app = Flask(__name__)
    fleet = Fleet(Predictor(models_dir))
    predictor = fleet.predictor

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        return jsonify(fleet.snapshot())

    @app.post("/api/tick")
    def tick():
        steps = max(1, min(int(request.args.get("steps", 1)), 50))
        fleet.tick(steps)
        return jsonify(fleet.snapshot())

    @app.post("/api/reset")
    def reset():
        fleet.reset()
        return jsonify(fleet.snapshot())

    @app.get("/api/metrics")
    def metrics():
        path = Path(results_dir) / "metrics.json"
        return jsonify(json.loads(path.read_text()) if path.exists() else {})

    @app.post("/api/predict")
    def predict():
        """Score a machine from its recent sensor history.

        Body: {"readings": [{"temperature": .., "vibration": .., "pressure": ..,
                              "rotational_speed": .., "torque": ..}, ...]}   (oldest first)
        """
        body = request.get_json(silent=True) or {}
        records = body.get("readings")
        if not isinstance(records, list) or not records:
            return jsonify({"error": "send a non-empty 'readings' list"}), 400
        try:
            result = predictor.predict_latest(readings_from_records(records))
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


if __name__ == "__main__":
    if not (MODELS_DIR / "failure_classifier.joblib").exists():
        sys.exit("No trained models found. Run:  python -m src.train")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
