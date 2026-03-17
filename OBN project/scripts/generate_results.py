from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Paths ── CHANGE THESE TWO LINES TO MATCH YOUR COMPUTER
BASE_DIR = Path("C:/Machine Learning/Machine-Learning---OBN-1/OBN project")
OUT_DIR  = Path("C:/Machine Learning/Machine-Learning---OBN-1/OBN project/results")

MODEL_DIR = BASE_DIR / "models"
DAY1_CSV  = BASE_DIR / "data" / "features" / "feature_dataset.csv"
DAY2_CSV  = BASE_DIR / "data" / "features" / "feature_dataset_day2.csv"
OUT_DIR.mkdir(exist_ok=True)

FOUR_DISTS   = [10, 15, 20, 25]
CLASS_LABELS = ['laptop', 'plastic box with foam', 'plastic box without foam', 'steel']
SHORT_LABELS = ['Laptop', 'PB w/foam', 'PB w/o foam', 'Steel']

# ── Helpers ────────────────────────────────────────────────────────────────
def get_latest_model(prefix):
    files = sorted(MODEL_DIR.glob(f"{prefix}_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No model found for prefix: {prefix}")
    return joblib.load(files[-1]), str(files[-1].name)

def load_data(csv_path, dist_filter=None):
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].str.lower().str.strip()
    if dist_filter:
        df = df[df['distance_cm'].isin(dist_filter)]
    X = df.drop(columns=['label', 'source_file'])
    y = df['label']
    return df, X, y

def per_dist_acc(model, df, X, y, distances):
    accs = []
    for d in distances:
        mask = df['distance_cm'] == d
        if mask.sum() == 0:
            accs.append(np.nan)
            continue
        accs.append(accuracy_score(y[mask], model.predict(X[mask])))
    return accs

# ── Load all data ──────────────────────────────────────────────────────────
print("Loading datasets...")
df1_full, X1_full, y1_full = load_data(DAY1_CSV)
df2_full, X2_full, y2_full = load_data(DAY2_CSV)
df2_4d,   X2_4d,   y2_4d   = load_data(DAY2_CSV, FOUR_DISTS)
df1_4d,   X1_4d,   y1_4d   = load_data(DAY1_CSV, FOUR_DISTS)
all_distances = sorted(df2_full['distance_cm'].unique())

# ── Load models ────────────────────────────────────────────────────────────
print("Loading models...")
models = {}
model_names_full  = ['Random_Forest', 'SVM', 'KNN']
model_names_4dist = ['Random_Forest_4dist', 'SVM_4dist', 'KNN_4dist']

for name in model_names_full + model_names_4dist:
    try:
        models[name], fname = get_latest_model(name)
        print(f"  Loaded {name}: {fname}")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

# ── Compute all results ────────────────────────────────────────────────────
print("\nComputing results...")
results = {}

for name in model_names_full:
    if name not in models:
        continue
    m = models[name]
    results[name] = {
        'day1_acc':      accuracy_score(y1_full, m.predict(X1_full)),
        'day2_full_acc': accuracy_score(y2_full, m.predict(X2_full)),
        'day2_4d_acc':   accuracy_score(y2_4d,   m.predict(X2_4d)),
        'day2_full_cm':  confusion_matrix(y2_full, m.predict(X2_full), labels=CLASS_LABELS),
        'day2_4d_cm':    confusion_matrix(y2_4d,   m.predict(X2_4d),   labels=CLASS_LABELS),
        'per_dist_full': per_dist_acc(m, df2_full, X2_full, y2_full, all_distances),
        'per_dist_4d':   per_dist_acc(m, df2_4d,   X2_4d,   y2_4d,   FOUR_DISTS),
    }

for name in model_names_4dist:
    if name not in models:
        continue
    m = models[name]
    results[name] = {
        'day1_acc':    accuracy_score(y1_4d, m.predict(X1_4d)),
        'day2_4d_acc': accuracy_score(y2_4d, m.predict(X2_4d)),
        'day2_4d_cm':  confusion_matrix(y2_4d, m.predict(X2_4d), labels=CLASS_LABELS),
        'per_dist_4d': per_dist_acc(m, df2_4d, X2_4d, y2_4d, FOUR_DISTS),
    }

cmap_blue = LinearSegmentedColormap.from_list('blue', ['#FFFFFF', '#1A5276'])

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Overall Accuracy Comparison
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1...")
fig, ax = plt.subplots(figsize=(12, 6))
categories = ['Day 1\n(Train/Test)', 'Day 2\nFull Range', 'Day 2\n4 Distances\n(Full Model)', 'Day 2\n4 Distances\n(4-dist Model)']
x = np.arange(len(categories))
width = 0.25

for i, (name, label, color) in enumerate(zip(
    ['Random_Forest', 'SVM', 'KNN'],
    ['Random Forest', 'SVM', 'KNN'],
    ['#2E86AB', '#E84855', '#3BB273']
)):
    if name not in results:
        continue
    r  = results[name]
    r4 = results.get(name + '_4dist', {})
    vals = [r['day1_acc'], r['day2_full_acc'], r['day2_4d_acc'],
            r4.get('day2_4d_acc', np.nan)]
    bars = ax.bar(x + i * width, [v * 100 if not np.isnan(v) else 0 for v in vals],
                  width, label=label, color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(y=25, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Random chance (25%)')
ax.set_xticks(x + width)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Overall Classification Accuracy — All Conditions', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_overall_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_overall_accuracy.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Accuracy vs Distance Full Range Day 2
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2...")
fig, ax = plt.subplots(figsize=(13, 6))
for name, label, color in zip(
    ['Random_Forest', 'SVM', 'KNN'],
    ['Random Forest', 'SVM', 'KNN'],
    ['#2E86AB', '#E84855', '#3BB273']
):
    if name not in results:
        continue
    ax.plot(all_distances, [v * 100 for v in results[name]['per_dist_full']],
            marker='o', markersize=4, label=label, color=color, linewidth=2, alpha=0.85)

ax.axhline(y=25, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Random chance')
ax.set_xlabel('Distance (cm)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy vs Distance — Full Range Models on Day 2 (5–30 cm)', fontsize=13, fontweight='bold')
ax.set_xticks(all_distances)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_accuracy_vs_distance_fullrange.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_accuracy_vs_distance_fullrange.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — 4 Distances: Full Model vs 4-dist Model
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, algo, algo_label in zip(axes,
    ['Random_Forest', 'SVM', 'KNN'],
    ['Random Forest', 'SVM', 'KNN']
):
    if algo in results:
        ax.plot(FOUR_DISTS, [v*100 for v in results[algo]['per_dist_4d']],
                marker='o', markersize=7, label='Full-range model',
                color='#2E86AB', linewidth=2)
    if algo + '_4dist' in results:
        ax.plot(FOUR_DISTS, [v*100 for v in results[algo + '_4dist']['per_dist_4d']],
                marker='s', markersize=7, label='4-dist model',
                color='#E84855', linewidth=2, linestyle='--')
    ax.axhline(y=25, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_title(algo_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Distance (cm)', fontsize=10)
    ax.set_xticks(FOUR_DISTS)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9)
axes[0].set_ylabel('Accuracy (%)', fontsize=11)
fig.suptitle('Accuracy vs Distance — Full-Range vs 4-Distance Models on Day 2',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_accuracy_4dist_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_accuracy_4dist_comparison.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Confusion Matrices Full Range Day 2
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 4...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, name, label in zip(axes,
    ['Random_Forest', 'SVM', 'KNN'],
    ['Random Forest', 'SVM', 'KNN']
):
    if name not in results:
        continue
    cm = results[name]['day2_full_cm']
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / np.where(row_sums == 0, 1, row_sums)
    ax.imshow(cm_norm, interpolation='nearest', cmap=cmap_blue, vmin=0, vmax=1)
    ax.set_title(f'{label}  —  Acc: {results[name]["day2_full_acc"]*100:.1f}%',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(SHORT_LABELS)))
    ax.set_yticks(range(len(SHORT_LABELS)))
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9)
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)',
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
fig.suptitle('Confusion Matrices — Full-Range Models Tested on Day 2',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_confusion_fullrange_day2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_confusion_fullrange_day2.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Confusion Matrices 4-dist Models Day 2
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 5...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, name, label in zip(axes,
    ['Random_Forest_4dist', 'SVM_4dist', 'KNN_4dist'],
    ['Random Forest (4-dist)', 'SVM (4-dist)', 'KNN (4-dist)']
):
    if name not in results:
        continue
    cm = results[name]['day2_4d_cm']
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / np.where(row_sums == 0, 1, row_sums)
    ax.imshow(cm_norm, interpolation='nearest', cmap=cmap_blue, vmin=0, vmax=1)
    ax.set_title(f'{label}  —  Acc: {results[name]["day2_4d_acc"]*100:.1f}%',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(SHORT_LABELS)))
    ax.set_yticks(range(len(SHORT_LABELS)))
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9)
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)',
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
fig.suptitle('Confusion Matrices — 4-Distance Models Tested on Day 2',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_confusion_4dist_day2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_confusion_4dist_day2.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Temporal Drift Summary
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 6...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.25
algo_labels = ['Random Forest', 'SVM', 'KNN']

day1_accs    = [results[n]['day1_acc']*100        for n in model_names_full if n in results]
day2_accs    = [results[n]['day2_full_acc']*100    for n in model_names_full if n in results]
day2_4d_accs = [results.get(n+'_4dist', {}).get('day2_4d_acc', np.nan)*100
                for n in model_names_full]

b1 = ax.bar(x - width, day1_accs,    width, label='Day 1 (same session)',      color='#2E86AB', alpha=0.9)
b2 = ax.bar(x,         day2_accs,    width, label='Day 2 - Full range model',  color='#E84855', alpha=0.9)
b3 = ax.bar(x + width, day2_4d_accs, width, label='Day 2 - 4-dist model',      color='#3BB273', alpha=0.9)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=25, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random chance (25%)')
ax.set_xticks(x)
ax.set_xticklabels(algo_labels, fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Temporal Drift: Day 1 vs Day 2 Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_temporal_drift.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_temporal_drift.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Per-Class F1 Score Heatmap
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 7...")
model_display = ['RF Full', 'SVM Full', 'KNN Full', 'RF 4-dist', 'SVM 4-dist', 'KNN 4-dist']
model_keys    = ['Random_Forest', 'SVM', 'KNN', 'Random_Forest_4dist', 'SVM_4dist', 'KNN_4dist']
test_X        = [X2_full, X2_full, X2_full, X2_4d, X2_4d, X2_4d]
test_y        = [y2_full, y2_full, y2_full, y2_4d, y2_4d, y2_4d]

f1_matrix = np.zeros((len(model_keys), len(CLASS_LABELS)))
for i, (key, X_t, y_t) in enumerate(zip(model_keys, test_X, test_y)):
    if key not in models:
        f1_matrix[i, :] = np.nan
        continue
    y_pred = models[key].predict(X_t)
    f1_matrix[i, :] = f1_score(y_t, y_pred, labels=CLASS_LABELS,
                                average=None, zero_division=0)

cmap_f1 = LinearSegmentedColormap.from_list('f1', ['#FADBD8', '#FFFFFF', '#D5F5E3', '#1E8449'])
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(f1_matrix, cmap=cmap_f1, vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='F1 Score')
ax.set_xticks(range(len(SHORT_LABELS)))
ax.set_yticks(range(len(model_display)))
ax.set_xticklabels(SHORT_LABELS, rotation=25, ha='right', fontsize=10)
ax.set_yticklabels(model_display, fontsize=10)
for i in range(len(model_display)):
    for j in range(len(CLASS_LABELS)):
        val = f1_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10,
                    color='white' if val > 0.7 else 'black', fontweight='bold')
ax.set_title('Per-Class F1 Score — All Models and Conditions (Day 2)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Object Class', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig7_f1_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_f1_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY
# ══════════════════════════════════════════════════════════════════════════
lines = []
lines.append("=" * 70)
lines.append("STEP 9 — RESULTS SUMMARY")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 70)
lines.append(f"\n{'Model':<25} {'Day1':>8} {'Day2 Full':>12} {'Day2 4d (full)':>16} {'Day2 4d (4dist)':>17}")
lines.append("-" * 82)
for name, label in zip(model_names_full, ['Random Forest', 'SVM', 'KNN']):
    if name not in results:
        continue
    r  = results[name]
    r4 = results.get(name + '_4dist', {})
    d4 = r4.get('day2_4d_acc', float('nan'))
    lines.append(f"{label:<25} {r['day1_acc']*100:>7.1f}%"
                 f" {r['day2_full_acc']*100:>11.1f}%"
                 f" {r['day2_4d_acc']*100:>15.1f}%"
                 f" {d4*100 if not np.isnan(d4) else float('nan'):>16.1f}%")

summary = "\n".join(lines)
print("\n" + summary)
with open(OUT_DIR / 'results_summary.txt', 'w') as f:
    f.write(summary)

print(f"\n✅ Done! All files saved to: {OUT_DIR}")