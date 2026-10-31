import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

df = pd.read_csv("dataset.csv")

print("\nDataset Loaded:")
print(df.head())

# ─────────────────────────────────────────────
# FEATURES + LABELS
# ─────────────────────────────────────────────

X = df.drop("label", axis=1)

y = df["label"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nLabels:")
print(y.unique())

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ─────────────────────────────────────────────
# CREATE MODEL
# ─────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")

# ─────────────────────────────────────────────
# TEST MODEL
# ─────────────────────────────────────────────

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

print("\nFeature Importance:")

importance = model.feature_importances_

for feature, score in zip(X.columns, importance):
    print(f"{feature:<20} {score:.3f}")

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────

joblib.dump(model, "exercise_model.pkl")

print("\nModel saved as exercise_model.pkl")