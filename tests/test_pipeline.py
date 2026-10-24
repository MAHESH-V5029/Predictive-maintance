"""Run with:  python -m unittest discover -s tests -v   (or: pytest)"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.app import create_app  # noqa: E402
from src import train  # noqa: E402
from src.config import FEATURES, HORIZON, SENSORS  # noqa: E402
from src.data_generator import generate_dataset  # noqa: E402
from src.features import add_features  # noqa: E402
from src.predict import Predictor, status_from_probability  # noqa: E402


class DataAndFeatureTests(unittest.TestCase):
    def test_dataset_shape_and_labels(self):
        df = generate_dataset(n_machines=5, seed=1)
        self.assertEqual(df["machine_id"].nunique(), 5)
        self.assertTrue(set(SENSORS).issubset(df.columns))
        # label must agree with the RUL definition
        self.assertTrue(((df["rul"] <= HORIZON).astype(int) == df["failure_soon"]).all())
        # every machine ends at RUL 1 (last reading before failure)
        self.assertTrue((df.groupby("machine_id")["rul"].min() == 1).all())

    def test_same_seed_is_reproducible(self):
        a, b = generate_dataset(3, seed=9), generate_dataset(3, seed=9)
        self.assertTrue(a.equals(b))

    def test_features_complete_and_causal(self):
        df = generate_dataset(n_machines=3, seed=2)
        feats = add_features(df)
        self.assertEqual(feats[FEATURES].isna().sum().sum(), 0)
        # Causality: features of the first 50 rows must not change if later rows are removed
        m = df[df["machine_id"] == "M001"]
        full = add_features(m).iloc[:50][FEATURES].to_numpy()
        part = add_features(m.iloc[:50])[FEATURES].to_numpy()
        self.assertTrue(np.allclose(full, part))

    def test_status_thresholds(self):
        self.assertEqual(status_from_probability(0.05), "Healthy")
        self.assertEqual(status_from_probability(0.50), "Warning")
        self.assertEqual(status_from_probability(0.90), "Critical")


class ModelAndAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.metrics = train.run(n_machines=40, seed=3, models_dir=root / "models",
                                fig_dir=root / "figs", results_dir=root / "results",
                                data_dir=None, verbose=False)
        cls.models_dir, cls.results_dir = root / "models", root / "results"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_beats_chance(self):
        best = self.metrics["selected_model"]
        test = self.metrics["classification"][best]["test"]
        self.assertGreater(test["roc_auc"], 0.9)
        self.assertGreater(test["recall"], 0.7)

    def test_predictor_outputs(self):
        predictor = Predictor(self.models_dir)
        df = generate_dataset(2, seed=5)
        proba, rul = predictor.predict_readings(df)
        self.assertEqual(len(proba), len(df))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        self.assertTrue((rul >= 0).all())

    def test_predictor_handles_missing_values(self):
        predictor = Predictor(self.models_dir)
        df = generate_dataset(1, seed=6)
        df.loc[df.index[::7], "vibration"] = np.nan
        proba, _ = predictor.predict_readings(df)
        self.assertFalse(np.isnan(proba).any())

    def test_api(self):
        client = create_app(self.models_dir, self.results_dir).test_client()
        self.assertEqual(client.get("/").status_code, 200)
        state = client.get("/api/state").get_json()
        self.assertEqual(len(state["machines"]), 6)
        state = client.post("/api/tick?steps=3").get_json()
        self.assertEqual(state["clock"], 3)
        ok = client.post("/api/predict", json={"readings": [
            {s: v for s, v in zip(SENSORS, [70, 0.5, 100, 1500, 40])}]})
        self.assertEqual(ok.status_code, 200)
        self.assertIn(ok.get_json()["status"], {"Healthy", "Warning", "Critical"})
        self.assertEqual(client.post("/api/predict", json={"readings": [{"temperature": 1}]}).status_code, 400)
        self.assertEqual(client.post("/api/predict", json={}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
