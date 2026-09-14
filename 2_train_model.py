"""
=================================================================
Step 2: Model Training
=================================================================
Loads the collected dataset, trains a Machine Learning classifier,
evaluates it, and saves the trained model.

Two model options:
  - Random Forest (default, fast, good accuracy)
  - Neural Network (MLP, slightly better but slower)

Output: models/gesture_model.pkl, models/scaler.pkl
=================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import joblib
import os

from config import (
    DATASET_FILE, TEST_SIZE, RANDOM_STATE,
    MODEL_TYPE, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    LABEL_ENCODER_PATH, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    NN_HIDDEN_LAYERS, NN_MAX_ITER, GESTURE_DISPLAY,
)


def train_model():
    # ---- Load data ----
    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: Dataset not found at {DATASET_FILE}")
        print("Run 1_collect_data.py first!")
        return

    df = pd.read_csv(DATASET_FILE)
    print(f"  Loaded {len(df)} samples from {DATASET_FILE}")
    print(f"  Classes: {df['gesture'].unique()}")

    # ---- Prepare features & labels ----
    X = df.drop(columns=['gesture']).values
    y = df['gesture'].values

    # Encode labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n  Label encoding:")
    for i, cls in enumerate(le.classes_):
        print(f"    {cls} -> {i}")

    # ---- Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"\n  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")

    # ---- Scale features ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Train model ----
    print(f"\n  Training model: {MODEL_TYPE}...")

    if MODEL_TYPE == "random_forest":
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif MODEL_TYPE == "neural_network":
        model = MLPClassifier(
            hidden_layer_sizes=NN_HIDDEN_LAYERS,
            max_iter=NN_MAX_ITER,
            random_state=RANDOM_STATE,
            early_stopping=True,
            verbose=True,
        )
    else:
        print(f"Unknown model type: {MODEL_TYPE}")
        return

    model.fit(X_train_scaled, y_train)
    print("  ✓ Training complete!")

    # ---- Evaluate ----
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Classification report
    target_names = [GESTURE_DISPLAY.get(cls, cls) for cls in le.classes_]
    print("\n  Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=3,
    )
    print(report)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    print("  Confusion matrix saved: models/confusion_matrix.png")

    # ---- Feature importance (Random Forest only) ----
    if MODEL_TYPE == "random_forest":
        importances = model.feature_importances_
        top_n = 20
        indices = np.argsort(importances)[-top_n:]
        
        feature_names = df.drop(columns=['gesture']).columns
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=150)
        print("  Feature importance saved: models/feature_importance.png")

    # ---- Save model, scaler, label encoder ----
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"\n  Model saved:     {MODEL_SAVE_PATH}")
    print(f"  Scaler saved:    {SCALER_SAVE_PATH}")
    print(f"  Labels saved:    {LABEL_ENCODER_PATH}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Model type:     {MODEL_TYPE}")
    print(f"  Accuracy:       {accuracy*100:.1f}%")
    print(f"  Dataset size:   {len(df)}")
    print(f"  Num classes:    {len(le.classes_)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_model()
