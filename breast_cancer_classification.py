# =============================================================================
# Breast Cancer Classification using scikit-learn
# Dataset: Breast Cancer Wisconsin (Diagnostic) — built into scikit-learn
# Task: Binary Classification — Malignant vs. Benign
# Model: Logistic Regression (Beginner-friendly)
# Group 2: AI and ML | Current Trends and Topics in Computing | March 2026
# =============================================================================

# --- 1. IMPORTS ---
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  BREAST CANCER CLASSIFICATION — MACHINE LEARNING DEMO")
print("  Group 2 | Current Trends and Topics in Computing")
print("=" * 60)


# --- 2. LOAD DATASET ---
data = load_breast_cancer()
X = data.data           # Features (30 numeric measurements)
y = data.target         # Labels: 0 = Malignant, 1 = Benign
feature_names = data.feature_names
target_names  = data.target_names  # ['malignant', 'benign']

print(f"\n[DATASET] Breast Cancer Wisconsin (Diagnostic)")
print(f"  Total samples   : {X.shape[0]}")
print(f"  Total features  : {X.shape[1]}")
print(f"  Class labels    : {list(target_names)}")
print(f"  Malignant (0)   : {sum(y == 0)}")
print(f"  Benign    (1)   : {sum(y == 1)}")


# --- 3. TRAIN / TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n[SPLIT] 80% Train / 20% Test")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")


# --- 4. FEATURE SCALING ---
# Logistic Regression is sensitive to feature magnitude — StandardScaler
# transforms all features to have mean=0 and std=1.
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)  # Fit on train ONLY
X_test   = scaler.transform(X_test)       # Apply same scale to test


# --- 5. TRAIN THE MODEL ---
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print(f"\n[MODEL] Logistic Regression — training complete.")


# --- 6. MAKE PREDICTIONS ---
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of Benign


# --- 7. EVALUATE ---
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[RESULTS]")
print(f"  Accuracy  : {accuracy * 100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))


# --- 8. VISUALIZATIONS ---
output_dir = "/mnt/user-data/outputs/"

# --- 8a. Confusion Matrix ---
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Logistic Regression\nBreast Cancer Classification",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(output_dir + "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: confusion_matrix.png")


# --- 8b. Top 10 Most Important Features (by coefficient magnitude) ---
coefficients = np.abs(model.coef_[0])
top10_idx    = np.argsort(coefficients)[-10:][::-1]
top10_names  = [feature_names[i] for i in top10_idx]
top10_coefs  = coefficients[top10_idx]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2196F3" if c > 0 else "#F44336" for c in model.coef_[0][top10_idx]]
bars = ax.barh(top10_names[::-1], top10_coefs[::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Absolute Coefficient (Feature Importance)", fontsize=11)
ax.set_title("Top 10 Most Influential Features\nLogistic Regression — Breast Cancer Dataset",
             fontsize=12, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.8)
for bar, val in zip(bars, top10_coefs[::-1]):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(output_dir + "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: feature_importance.png")


# --- 8c. Class Distribution Bar Chart ---
fig, ax = plt.subplots(figsize=(5, 4))
classes = ["Malignant\n(Class 0)", "Benign\n(Class 1)"]
counts  = [sum(y == 0), sum(y == 1)]
colors_bar = ["#EF5350", "#42A5F5"]
bars = ax.bar(classes, counts, color=colors_bar, edgecolor="white", width=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha="center", fontsize=13, fontweight="bold")
ax.set_title("Class Distribution\nBreast Cancer Wisconsin Dataset",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Samples")
ax.set_ylim(0, max(counts) + 60)
plt.tight_layout()
plt.savefig(output_dir + "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: class_distribution.png")


# --- 9. SAMPLE PREDICTION DEMO ---
print("\n[DEMO] Predicting on 5 unseen test samples:")
print(f"  {'Sample':>8}  {'Actual':>12}  {'Predicted':>12}  {'Correct?':>10}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
for i in range(5):
    actual    = target_names[y_test[i]]
    predicted = target_names[y_pred[i]]
    correct   = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"  {i+1:>8}  {actual:>12}  {predicted:>12}  {correct:>10}")

print("\n" + "=" * 60)
print(f"  Final Accuracy: {accuracy * 100:.2f}%")
print("  All plots saved to /mnt/user-data/outputs/")
print("=" * 60)
