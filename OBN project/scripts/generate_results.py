from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
DAY1_CSV = BASE_DIR / "data" / "features" / "feature_dataset.csv"
DAY2_CSV = BASE_DIR / "data" / "features" / "feature_dataset_day2.csv"
OUT_DIR.mkdir(exist_ok=True)

FOUR_DISTS = [10, 15, 20, 25]
CLASS_LABELS = ['laptop', 'plastic box with foam', 'plastic box without foam', 'steel']
SHORT_LABELS = ['Laptop', 'PB w/foam', 'PB w/o foam', 'Steel']


# -----------------------------
# Helpers
# -----------------------------
def get_latest_model(prefix, exclude=None):
    files = sorted(MODEL_DIR.glob(f"{prefix}_*.pkl"))
    if exclude:
        files = [f for f in files if exclude not in f.name]
    if not files:
        raise FileNotFoundError(f"No model found for prefix: {prefix}")
    return joblib.load(files[-1]), files[-1].name


def load_data(csv_path, dist_filter=None):
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].str.lower().str.strip()
    if dist_filter is not None:
        df = df[df["distance_cm"].isin(dist_filter)]
    X = df.drop(columns=["label", "source_file"])
    y = df["label"]
    return df, X, y


def per_dist_acc(model, df, X, y, distances):
    accs = []
    for d in distances:
        mask = df["distance_cm"] == d
        if mask.sum() == 0:
            accs.append(np.nan)
        else:
            accs.append(accuracy_score(y[mask], model.predict(X[mask])))
    return accs


# -----------------------------
# Load data
# -----------------------------
df1_full, X1_full, y1_full = load_data(DAY1_CSV)
df2_full, X2_full, y2_full = load_data(DAY2_CSV)
df1_4d, X1_4d, y1_4d = load_data(DAY1_CSV, FOUR_DISTS)
df2_4d, X2_4d, y2_4d = load_data(DAY2_CSV, FOUR_DISTS)
all_distances = sorted(df2_full["distance_cm"].unique())

# -----------------------------
# Load models
# -----------------------------
models = {}
for name in ["Random_Forest", "SVM", "KNN"]:
    models[name], _ = get_latest_model(name, exclude="4dist")

for name in ["Random_Forest_4dist", "SVM_4dist", "KNN_4dist"]:
    models[name], _ = get_latest_model(name)

# -----------------------------
# Compute results
# -----------------------------
results = {}

# full-range models
for name in ["Random_Forest", "SVM", "KNN"]:
    m = models[name]
    results[name] = {
        "day2_full_acc": accuracy_score(y2_full, m.predict(X2_full)),
        "day2_4d_acc": accuracy_score(y2_4d, m.predict(X2_4d)),
        "day2_full_cm": confusion_matrix(y2_full, m.predict(X2_full), labels=CLASS_LABELS),
        "day2_4d_cm": confusion_matrix(y2_4d, m.predict(X2_4d), labels=CLASS_LABELS),
        "per_dist_full": per_dist_acc(m, df2_full, X2_full, y2_full, all_distances),
        "per_dist_4d": per_dist_acc(m, df2_4d, X2_4d, y2_4d, FOUR_DISTS),
    }

# 4-distance-trained models
for name in ["Random_Forest_4dist", "SVM_4dist", "KNN_4dist"]:
    m = models[name]
    results[name] = {
        "day2_4d_acc": accuracy_score(y2_4d, m.predict(X2_4d)),
        "day2_4d_cm": confusion_matrix(y2_4d, m.predict(X2_4d), labels=CLASS_LABELS),
        "per_dist_4d": per_dist_acc(m, df2_4d, X2_4d, y2_4d, FOUR_DISTS),
    }

# -----------------------------
# Figure 1: overall comparison
# Use official values from your report / training scripts
# -----------------------------
official_day1 = {
    "Random_Forest": 0.8241,
    "SVM": 0.6481,
    "KNN": 0.6574,
}

fig, ax = plt.subplots(figsize=(12, 6))
categories = ["Day 1\n(Train/Test)", "Day 2\nFull Range",
              "Day 2\n4 Distances\n(Full Model)", "Day 2\n4 Distances\n(4-dist Model)"]
x = np.arange(len(categories))
width = 0.25

for i, (name, label) in enumerate(zip(
    ["Random_Forest", "SVM", "KNN"],
    ["Random Forest", "SVM", "KNN"]
)):
    vals = [
        official_day1[name],
        results[name]["day2_full_acc"],
        results[name]["day2_4d_acc"],
        results[name + "_4dist"]["day2_4d_acc"],
    ]
    bars = ax.bar(x + i*width, [v*100 for v in vals], width, label=label)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random chance (25%)")
ax.set_xticks(x + width)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Overall Classification Accuracy — All Conditions", fontsize=14, fontweight="bold")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_overall_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 2: accuracy vs distance
# -----------------------------
fig, ax = plt.subplots(figsize=(13, 6))
for name, label in zip(["Random_Forest", "SVM", "KNN"], ["Random Forest", "SVM", "KNN"]):
    ax.plot(all_distances, [v*100 for v in results[name]["per_dist_full"]],
            marker="o", linewidth=2, label=label)

ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random chance")
ax.set_xlabel("Distance (cm)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy vs Distance — Full Range Models on Day 2 (5–30 cm)")
ax.set_xticks(all_distances)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_accuracy_vs_distance_fullrange.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 3: 4-distance comparison
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, algo, algo_label in zip(
    axes,
    ["Random_Forest", "SVM", "KNN"],
    ["Random Forest", "SVM", "KNN"]
):
    ax.plot(FOUR_DISTS, [v*100 for v in results[algo]["per_dist_4d"]],
            marker="o", linewidth=2, label="Full-range model")
    ax.plot(FOUR_DISTS, [v*100 for v in results[algo + "_4dist"]["per_dist_4d"]],
            marker="s", linestyle="--", linewidth=2, label="4-dist model")
    ax.axhline(y=25, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title(algo_label)
    ax.set_xlabel("Distance (cm)")
    ax.set_xticks(FOUR_DISTS)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Accuracy (%)")
fig.suptitle("Accuracy vs Distance — Full-Range vs 4-Distance Models on Day 2")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_accuracy_4dist_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 4: confusion matrices full-range Day 2
# -----------------------------
cmap_blue = LinearSegmentedColormap.from_list("blue", ["#FFFFFF", "#1A5276"])
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, name, label in zip(
    axes,
    ["Random_Forest", "SVM", "KNN"],
    ["Random Forest", "SVM", "KNN"]
):
    cm = results[name]["day2_full_cm"]
    cm_norm = cm.astype(float) / np.where(cm.sum(axis=1)[:, None] == 0, 1, cm.sum(axis=1)[:, None])
    ax.imshow(cm_norm, interpolation="nearest", cmap=cmap_blue, vmin=0, vmax=1)
    ax.set_title(f"{label} — Acc: {results[name]['day2_full_acc']*100:.1f}%")
    ax.set_xticks(range(len(SHORT_LABELS)))
    ax.set_yticks(range(len(SHORT_LABELS)))
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_confusion_fullrange_day2.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 5: confusion matrices 4-dist Day 2
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, name, label in zip(
    axes,
    ["Random_Forest_4dist", "SVM_4dist", "KNN_4dist"],
    ["Random Forest (4-dist)", "SVM (4-dist)", "KNN (4-dist)"]
):
    cm = results[name]["day2_4d_cm"]
    cm_norm = cm.astype(float) / np.where(cm.sum(axis=1)[:, None] == 0, 1, cm.sum(axis=1)[:, None])
    ax.imshow(cm_norm, interpolation="nearest", cmap=cmap_blue, vmin=0, vmax=1)
    ax.set_title(f"{label} — Acc: {results[name]['day2_4d_acc']*100:.1f}%")
    ax.set_xticks(range(len(SHORT_LABELS)))
    ax.set_yticks(range(len(SHORT_LABELS)))
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_confusion_4dist_day2.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 6: temporal drift
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.25
algo_labels = ["Random Forest", "SVM", "KNN"]

day1_accs = [official_day1[n]*100 for n in ["Random_Forest", "SVM", "KNN"]]
day2_accs = [results[n]["day2_full_acc"]*100 for n in ["Random_Forest", "SVM", "KNN"]]
day2_4d_accs = [results[n + "_4dist"]["day2_4d_acc"]*100 for n in ["Random_Forest", "SVM", "KNN"]]

ax.bar(x - width, day1_accs, width, label="Day 1 (same session)")
ax.bar(x, day2_accs, width, label="Day 2 - Full range model")
ax.bar(x + width, day2_4d_accs, width, label="Day 2 - 4-dist model")
ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random chance (25%)")
ax.set_xticks(x)
ax.set_xticklabels(algo_labels)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Temporal Drift: Day 1 vs Day 2 Accuracy")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_temporal_drift.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 7: F1 heatmap
# -----------------------------
model_display = ["RF Full", "SVM Full", "KNN Full", "RF 4-dist", "SVM 4-dist", "KNN 4-dist"]
model_keys = ["Random_Forest", "SVM", "KNN", "Random_Forest_4dist", "SVM_4dist", "KNN_4dist"]
test_X = [X2_full, X2_full, X2_full, X2_4d, X2_4d, X2_4d]
test_y = [y2_full, y2_full, y2_full, y2_4d, y2_4d, y2_4d]

f1_matrix = np.zeros((len(model_keys), len(CLASS_LABELS)))
for i, (key, X_t, y_t) in enumerate(zip(model_keys, test_X, test_y)):
    y_pred = models[key].predict(X_t)
    f1_matrix[i, :] = f1_score(y_t, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(f1_matrix, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="F1 Score")
ax.set_xticks(range(len(SHORT_LABELS)))
ax.set_yticks(range(len(model_display)))
ax.set_xticklabels(SHORT_LABELS, rotation=25, ha="right")
ax.set_yticklabels(model_display)
for i in range(len(model_display)):
    for j in range(len(CLASS_LABELS)):
        ax.text(j, i, f"{f1_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Per-Class F1 Score — All Models and Conditions (Day 2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig7_f1_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------
# Results summary text
# -----------------------------
lines = []
lines.append("=" * 70)
lines.append("STEP 9 — RESULTS SUMMARY")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 70)
lines.append(f"\n{'Model':<25} {'Day1':>8} {'Day2 Full':>12} {'Day2 4d (full)':>16} {'Day2 4d (4dist)':>17}")
lines.append("-" * 82)
for name, label in zip(["Random_Forest", "SVM", "KNN"], ["Random Forest", "SVM", "KNN"]):
    lines.append(
        f"{label:<25} {official_day1[name]*100:>7.1f}%"
        f" {results[name]['day2_full_acc']*100:>11.1f}%"
        f" {results[name]['day2_4d_acc']*100:>15.1f}%"
        f" {results[name + '_4dist']['day2_4d_acc']*100:>16.1f}%"
    )

with open(OUT_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ Step 9 complete. All figures and summary saved in:", OUT_DIR)