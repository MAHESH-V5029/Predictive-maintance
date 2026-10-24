"""Performance tests: latency, throughput, memory, model size and robustness.

Usage:
    python -m src.benchmark

Results are written to results/benchmark.json, results/performance_results.md and
docs/figures/robustness.png.
"""
import json
import time
import tracemalloc

import matplotlib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIG_DIR, MODELS_DIR, RESULTS_DIR, SENSORS  # noqa: E402
from .data_generator import NOISE, generate_dataset  # noqa: E402
from .features import add_features  # noqa: E402
from .predict import Predictor  # noqa: E402


def _scores(predictor, raw):
    feats = add_features(raw)
    proba, _ = predictor.predict_features(feats)
    pred = (proba >= 0.5).astype(int)
    y = feats["failure_soon"].to_numpy()
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def timing(predictor, raw, repeats=200):
    one_machine = raw[raw["machine_id"] == raw["machine_id"].iloc[0]]
    window = one_machine.iloc[60:100][["machine_id", "step"] + SENSORS]   # 40-reading buffer

    end_to_end = []
    for _ in range(repeats):
        t = time.perf_counter()
        predictor.predict_latest(window)
        end_to_end.append((time.perf_counter() - t) * 1000)

    feats = add_features(window).tail(1)
    model_only = []
    for _ in range(repeats):
        t = time.perf_counter()
        predictor.predict_features(feats)
        model_only.append((time.perf_counter() - t) * 1000)

    all_feats = add_features(raw)
    t = time.perf_counter()
    predictor.predict_features(all_feats)
    batch_seconds = time.perf_counter() - t

    tracemalloc.start()
    predictor.predict_features(all_feats)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "end_to_end_ms_mean": float(np.mean(end_to_end)),
        "end_to_end_ms_p95": float(np.percentile(end_to_end, 95)),
        "model_only_ms_mean": float(np.mean(model_only)),
        "model_only_ms_p95": float(np.percentile(model_only, 95)),
        "batch_rows": int(len(all_feats)),
        "batch_throughput_rows_per_s": float(len(all_feats) / batch_seconds),
        "peak_python_memory_mb_batch": float(peak / 1e6),
    }


def robustness(predictor, raw, seed=0):
    rng = np.random.default_rng(seed)
    noise_rows, missing_rows = [], []
    for k in [0, 0.5, 1, 2, 3]:
        noisy = raw.copy()
        for s in SENSORS:
            noisy[s] = noisy[s] + rng.normal(0, k * NOISE[s], len(noisy))
        noise_rows.append({"extra_noise_x_sigma": k, **_scores(predictor, noisy)})
    for p in [0, 0.05, 0.10, 0.20, 0.30]:
        gappy = raw.copy()
        for s in SENSORS:
            gappy.loc[rng.random(len(gappy)) < p, s] = np.nan
        missing_rows.append({"missing_fraction": p, **_scores(predictor, gappy)})
    return noise_rows, missing_rows


def make_figure(noise_rows, missing_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rows, key, title, xlabel in [
        (axes[0], noise_rows, "extra_noise_x_sigma", "Robustness to extra sensor noise",
         "Extra noise (multiples of normal sensor noise)"),
        (axes[1], missing_rows, "missing_fraction", "Robustness to missing readings",
         "Fraction of sensor readings missing")]:
        x = [r[key] for r in rows]
        for metric in ["precision", "recall", "f1"]:
            ax.plot(x, [r[metric] for r in rows], marker="o", label=metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_markdown(result):
    t, sizes = result["timing"], result["model_size_kb"]
    lines = ["# Performance test results", "",
             "## Speed, memory and size", "",
             "| Metric | Value |", "|---|---|",
             f"| Single prediction, model only (mean / p95) | {t['model_only_ms_mean']:.1f} ms / {t['model_only_ms_p95']:.1f} ms |",
             f"| Single prediction, end to end incl. features (mean / p95) | {t['end_to_end_ms_mean']:.1f} ms / {t['end_to_end_ms_p95']:.1f} ms |",
             f"| Batch throughput ({t['batch_rows']:,} rows) | {t['batch_throughput_rows_per_s']:,.0f} rows/s |",
             f"| Peak Python memory during batch prediction | {t['peak_python_memory_mb_batch']:.1f} MB |",
             f"| Classifier file size | {sizes['classifier']:.0f} KB |",
             f"| RUL regressor file size | {sizes['regressor']:.0f} KB |", "",
             "## Robustness to extra sensor noise", "",
             "| Extra noise (x sensor sigma) | Precision | Recall | F1 |", "|---|---|---|---|"]
    for r in result["noise"]:
        lines.append(f"| {r['extra_noise_x_sigma']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    lines += ["", "## Robustness to missing sensor readings", "",
              "| Missing fraction | Precision | Recall | F1 |", "|---|---|---|---|"]
    for r in result["missing"]:
        lines.append(f"| {r['missing_fraction']:.0%} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    predictor = Predictor(MODELS_DIR)
    raw = generate_dataset(n_machines=40, seed=2024)      # independent of training seed
    result = {"timing": timing(predictor, raw)}
    result["model_size_kb"] = {
        "classifier": (MODELS_DIR / "failure_classifier.joblib").stat().st_size / 1024,
        "regressor": (MODELS_DIR / "rul_regressor.joblib").stat().st_size / 1024}
    result["noise"], result["missing"] = robustness(predictor, raw)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(result, indent=2))
    (RESULTS_DIR / "performance_results.md").write_text(to_markdown(result))
    make_figure(result["noise"], result["missing"], FIG_DIR / "robustness.png")
    print(to_markdown(result))


if __name__ == "__main__":
    main()
