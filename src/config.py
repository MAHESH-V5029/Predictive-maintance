"""Central configuration shared by training, benchmarking and the dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "docs" / "figures"

# Raw sensor channels measured on every machine
SENSORS = ["temperature", "vibration", "pressure", "rotational_speed", "torque"]

# A reading is labelled "failure_soon" if the machine fails within HORIZON steps
HORIZON = 30

# Rolling-window sizes (in time steps) used for feature engineering
WINDOW = 5
LONG_WINDOW = 20

# Remaining-useful-life targets are capped; "more than 150 steps left" is simply "healthy"
RUL_CAP = 150

# Failure-probability thresholds that map to a health status on the dashboard
WARNING_THRESHOLD = 0.35
CRITICAL_THRESHOLD = 0.65

FEATURES = (
    SENSORS
    + [f"{s}_mean" for s in SENSORS]
    + [f"{s}_std" for s in SENSORS]
    + [f"{s}_trend" for s in SENSORS]
)
