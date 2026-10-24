"""Draw the high-level architecture diagram used in the report.  Run: python docs/make_diagrams.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#1d2733", lw=1.2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.2, color="#333")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#1d2733"))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    w, h, y = 2.0, 1.3, 2.7
    stages = [
        ("Sensors", "temperature, vibration,\npressure, speed, torque", "#dbe9f6"),
        ("Data ingestion", "simulator / CSV / API\n(5 channels per step)", "#dbe9f6"),
        ("Feature engineering", "rolling mean, std, trend\nper machine", "#fdebc8"),
        ("ML models", "failure classifier\n+ RUL regressor", "#d8f0dc"),
        ("Health logic", "Healthy / Warning /\nCritical thresholds", "#d8f0dc"),
    ]
    xs = [0.2 + i * 2.35 for i in range(5)]
    for x, (t, s, c) in zip(xs, stages):
        box(ax, x, y, w, h, t, s, c)
    for i in range(4):
        arrow(ax, xs[i] + w + 0.02, y + h / 2, xs[i + 1] - 0.02, y + h / 2)
    box(ax, 3.3, 0.4, 2.6, 1.2, "Offline training", "src/train.py: 3 models compared,\nbest chosen on validation set", "#f3d9e3")
    box(ax, 6.7, 0.4, 2.6, 1.2, "Dashboard + REST API", "live cards, alerts, /api/predict", "#f3d9e3")
    arrow(ax, 4.6, 1.62, xs[3] + w / 2 - 0.3, y - 0.02)
    arrow(ax, xs[4] + w / 2 - 0.5, y - 0.02, 8.0, 1.62)
    ax.text(6, 4.3, "Predictive Maintenance: system architecture", ha="center", fontsize=13, weight="bold")
    fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    print("saved", OUT / "architecture.png")
