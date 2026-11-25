import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)
import pickle
import os

# ─────────────────────────────────────────────
# FIND CURRENT PROJECT DIRECTORY
# (works on both Windows and Linux - uses the script's own location)
# ─────────────────────────────────────────────
current_folder = os.path.dirname(os.path.abspath(__file__))

# Full path of CSV file
csv_path = os.path.join(current_folder, "student_data.csv")

print("Looking for dataset at:")
print(csv_path)

# ─────────────────────────────────────────────
# CHECK IF FILE EXISTS
# ─────────────────────────────────────────────
if not os.path.exists(csv_path):
    print("\n❌ ERROR: student_data.csv NOT FOUND!")
    print("Make sure student_data.csv is inside the SAME folder as train_model.py")
    print("Expected location:", csv_path)
    exit()

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv(csv_path)

print("\n✅ Dataset Loaded Successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# ─────────────────────────────────────────────
# FEATURES  (updated to match new dataset schema)
# ─────────────────────────────────────────────
# Old features (removed):
#   study_hours, attendance, assignment_marks, internal_marks
#
# New features (Assignment=10, Attendance=10, Mid term=20,
#               Class test=10, Semester final=50  → total 100)
features = [
    "assignment",
    "attendance",
    "mid_term",
    "class_test",
    "semester_final"
]

# ─────────────────────────────────────────────
# SAFETY CHECK: verify CSV has the new-schema columns
# ─────────────────────────────────────────────
missing = [c for c in features + ["final_marks", "result"] if c not in df.columns]
if missing:
    print("\n❌ ERROR: Your student_data.csv is using the OLD schema.")
    print("Missing columns:", missing)
    print("\nColumns found in your CSV:", list(df.columns))
    print("\n👉 Fix: run generate_student_data.py FIRST to regenerate the CSV")
    print("   with the new schema (assignment, attendance, mid_term,")
    print("   class_test, semester_final), then run train_model.py again.")
    exit()

X = df[features]

# ═════════════════════════════════════════════
# MODEL 1: PASS / FAIL CLASSIFICATION
# ═════════════════════════════════════════════
y_class = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n──────── PASS / FAIL MODEL ────────")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Pass", "Fail"]
)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pass", "Fail"],
    yticklabels=["Pass", "Fail"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

confusion_path = os.path.join(current_folder, "confusion_matrix.png")
plt.savefig(confusion_path)
print("\n✅ Saved:", confusion_path)

plt.show()  # ← Graph will pop up here

# ═════════════════════════════════════════════
# MODEL 2: FINAL MARKS PREDICTION
# ═════════════════════════════════════════════
y_reg = df["final_marks"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

reg.fit(X_train2, y_train2)

y_pred2 = reg.predict(X_test2)

print("\n──────── FINAL MARKS MODEL ────────")
print(f"Mean Absolute Error : {mean_absolute_error(y_test2, y_pred2):.2f}")
print(f"R² Score            : {r2_score(y_test2, y_pred2):.4f}")

# ─────────────────────────────────────────────
# ACTUAL VS PREDICTED GRAPH
# ─────────────────────────────────────────────
plt.figure(figsize=(6, 4))

plt.scatter(
    y_test2,
    y_pred2,
    alpha=0.5,
    color="teal"
)

plt.plot(
    [0, 100],
    [0, 100],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted")
plt.legend()

plt.tight_layout()

prediction_graph = os.path.join(current_folder, "actual_vs_predicted.png")
plt.savefig(prediction_graph)
print("✅ Saved:", prediction_graph)

plt.show()  # ← Graph will pop up here

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE GRAPH
# ─────────────────────────────────────────────
importances = pd.Series(
    clf.feature_importances_,
    index=features
)

plt.figure(figsize=(6, 4))

importances.sort_values().plot(
    kind="barh",
    color="steelblue"
)

plt.title("Feature Importance")

plt.tight_layout()

importance_graph = os.path.join(current_folder, "feature_importance.png")
plt.savefig(importance_graph)
print("✅ Saved:", importance_graph)

plt.show()  # ← Graph will pop up here

# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────
classifier_path = os.path.join(current_folder, "classifier_model.pkl")
regressor_path = os.path.join(current_folder, "regressor_model.pkl")

with open(classifier_path, "wb") as f:
    pickle.dump(clf, f)

with open(regressor_path, "wb") as f:
    pickle.dump(reg, f)

print("\n✅ Models Saved Successfully!")
print("Classifier:", classifier_path)
print("Regressor :", regressor_path)
