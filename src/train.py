"""Train and evaluate the failure classifier and the remaining-useful-life regressor.

Usage:
    python -m src.train                       # simulate data and train
    python -m src.train --machines 200        # more simulated machines
    python -m src.train --data my_data.csv    # use your own CSV (see README)
"""
import argparse
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import (DATA_DIR, FEATURES, FIG_DIR, HORIZON, MODELS_DIR,
                     RESULTS_DIR, RUL_CAP, SENSORS)
from .data_generator import generate_dataset
from .features import add_features


def load_data(path=None, n_machines=150, seed=42):
    if path is None:
        return generate_dataset(n_machines, seed)
    df = pd.read_csv(path)
    required = ["machine_id", "step", "rul"] + SENSORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df["failure_soon"] = (df["rul"] <= HORIZON).astype(int)
    return df


def split_by_machine(df, seed=42):
    """70/15/15 split by machine, so no machine appears in more than one split (no leakage)."""
    ids = np.array(df["machine_id"].unique(), dtype=object)
    np.random.default_rng(seed).shuffle(ids)
    n_train, n_val = int(0.70 * len(ids)), int(0.15 * len(ids))
    parts = (ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:])
    return [df[df["machine_id"].isin(p)] for p in parts]


def candidate_models(seed):
    imputer = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Logistic Regression": Pipeline([
            ("imputer", imputer()), ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("imputer", imputer()),
            ("model", RandomForestClassifier(
                n_estimators=150, max_depth=14, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=seed))]),
        "Gradient Boosting": Pipeline([
            ("imputer", imputer()),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_depth=6,
                class_weight="balanced", random_state=seed))]),
    }


def classification_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(fig_dir, results, models, best_name, x_test, y_test, rul_true, rul_pred):
    # 1. Model comparison
    names = list(results)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    for i, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(np.arange(len(names)) + (i - 1) * width,
               [results[n]["test"][metric] for n in names], width, label=metric.capitalize())
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on held-out test machines")
    ax.legend(loc="lower right")
    _save_fig(fig, fig_dir / "model_comparison.png")

    # 2. Confusion matrix of selected model
    cm = results[best_name]["test"]["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > matrix.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Failure soon"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Failure soon"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {best_name}")
    _save_fig(fig, fig_dir / "confusion_matrix.png")

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save_fig(fig, fig_dir / "roc_curves.png")

    # 4. Feature importance (model-agnostic permutation importance)
    imp = permutation_importance(models[best_name], x_test, y_test, scoring="f1",
                                 n_repeats=3, random_state=0, n_jobs=1)
    order = np.argsort(imp.importances_mean)[-12:]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([FEATURES[i] for i in order], imp.importances_mean[order], color="#c0392b")
    ax.set_xlabel("Drop in F1 when feature is shuffled")
    ax.set_title(f"Top features ({best_name})")
    _save_fig(fig, fig_dir / "feature_importance.png")

    # 5. RUL predicted vs actual
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(rul_true, rul_pred, s=3, alpha=0.3)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "k--", linewidth=0.8)
    ax.set_xlabel(f"Actual RUL (capped at {RUL_CAP})")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Remaining useful life: predicted vs actual")
    _save_fig(fig, fig_dir / "rul_prediction.png")


def run(n_machines=150, seed=42, data_path=None, models_dir=MODELS_DIR,
        fig_dir=FIG_DIR, results_dir=RESULTS_DIR, data_dir=DATA_DIR, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    raw = load_data(data_path, n_machines, seed)
    df = add_features(raw)
    train, val, test = split_by_machine(df, seed)
    log(f"Data: {len(df):,} rows, {df['machine_id'].nunique()} machines "
        f"(train {train['machine_id'].nunique()}, val {val['machine_id'].nunique()}, "
        f"test {test['machine_id'].nunique()}); failure_soon rate {df['failure_soon'].mean():.1%}")

    x_tr, y_tr = train[FEATURES], train["failure_soon"]
    x_va, y_va = val[FEATURES], val["failure_soon"]
    x_te, y_te = test[FEATURES], test["failure_soon"]

    # Train every candidate, choose by validation F1, report on the untouched test set
    models, results = {}, {}
    for name, pipe in candidate_models(seed).items():
        pipe.fit(x_tr, y_tr)
        models[name] = pipe
        results[name] = {
            "validation": classification_metrics(y_va, pipe.predict_proba(x_va)[:, 1]),
            "test": classification_metrics(y_te, pipe.predict_proba(x_te)[:, 1]),
        }
        log(f"  {name:20s} val F1 {results[name]['validation']['f1']:.3f} | "
            f"test F1 {results[name]['test']['f1']:.3f}  recall {results[name]['test']['recall']:.3f}")
    best_name = max(results, key=lambda n: results[n]["validation"]["f1"])
    log(f"Selected model: {best_name}")

    # Remaining-useful-life regressor
    regressor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                        n_jobs=-1, random_state=seed))])
    regressor.fit(x_tr, train["rul"].clip(upper=RUL_CAP))
    rul_true = test["rul"].clip(upper=RUL_CAP).to_numpy()
    rul_pred = regressor.predict(x_te)
    near = rul_true <= HORIZON
    rul_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(rul_true, rul_pred))),
        "mae": float(mean_absolute_error(rul_true, rul_pred)),
        "rmse_near_failure": float(np.sqrt(mean_squared_error(rul_true[near], rul_pred[near]))),
    }
    log(f"RUL regressor: RMSE {rul_metrics['rmse']:.1f}, MAE {rul_metrics['mae']:.1f}, "
        f"RMSE near failure {rul_metrics['rmse_near_failure']:.1f}")

    # Save artefacts
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best_name], models_dir / "failure_classifier.joblib", compress=3)
    joblib.dump(regressor, models_dir / "rul_regressor.joblib", compress=3)
    metrics = {
        "selected_model": best_name,
        "dataset": {"rows": int(len(df)), "machines": int(df["machine_id"].nunique()),
                    "failure_soon_rate": float(df["failure_soon"].mean()),
                    "horizon_steps": HORIZON, "seed": seed},
        "classification": results,
        "rul": rul_metrics,
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    make_figures(fig_dir, results, models, best_name, x_te, y_te, rul_true, rul_pred)

    if data_path is None and data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        keep = raw["machine_id"].unique()[:15]
        raw[raw["machine_id"].isin(keep)].round(3).to_csv(
            data_dir / "sample_sensor_data.csv", index=False)
    log(f"Done in {time.time() - t0:.0f}s. Models saved to {models_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machines", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None, help="optional CSV with real data")
    args = parser.parse_args()
    run(n_machines=args.machines, seed=args.seed, data_path=args.data)


if __name__ == "__main__":
    main()
