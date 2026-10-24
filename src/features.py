"""Feature engineering: rolling statistics per machine."""
from .config import FEATURES, LONG_WINDOW, SENSORS, WINDOW


def add_features(df):
    """Add rolling mean / std / trend for every sensor, computed per machine.

    mean  : average of the last WINDOW readings (smooths noise)
    std   : spread of the last WINDOW readings (captures instability)
    trend : short-term mean minus long-term mean (captures drift)
    Only past and current readings are used, so the same function works for live data.
    """
    df = df.sort_values(["machine_id", "step"]).copy()
    grouped = df.groupby("machine_id", sort=False)
    for s in SENSORS:
        short = grouped[s].transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        long = grouped[s].transform(lambda x: x.rolling(LONG_WINDOW, min_periods=1).mean())
        df[f"{s}_mean"] = short
        df[f"{s}_std"] = grouped[s].transform(
            lambda x: x.rolling(WINDOW, min_periods=2).std()).fillna(0.0)
        df[f"{s}_trend"] = short - long
    return df


def feature_matrix(df):
    """Return the model input columns in the fixed training order."""
    return df[FEATURES]
