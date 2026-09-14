"""
=================================================================
Step 1: Data Collection
=================================================================
This script opens your webcam and captures hand gesture samples
for each gesture class. The data is saved as a CSV file that
will be used for training the ML model.

How to use:
  1. Run this script
  2. Press SPACE to start collecting for the current gesture
  3. Show the gesture clearly in front of the camera
  4. Wait until all samples are collected
  5. Repeat for each gesture
  6. Press Q to quit early

Output: data/gesture_dataset.csv
=================================================================
"""
import cv2
import numpy as np
import pandas as pd
import os
import time

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    GESTURES, NUM_SAMPLES_PER_GESTURE,
    CAPTURE_DELAY_MS, DATA_DIR, DATASET_FILE,
)
from utils.hand_tracker import HandTracker


def collect_data():
    # ---- Setup ----
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    tracker = HandTracker()

    all_features = []
    all_labels = []

    print("\n" + "=" * 60)
    print("  GESTURE DATA COLLECTION")
    print("=" * 60)
    print(f"  Gestures to collect: {len(GESTURES)}")
    print(f"  Samples per gesture: {NUM_SAMPLES_PER_GESTURE}")
    print(f"  Output file: {DATASET_FILE}")
    print("=" * 60)

    for gesture_id, gesture_name in GESTURES.items():
        print(f"\n--- Gesture: {gesture_name} (class {gesture_id}) ---")
        print(f"Show the '{gesture_name}' gesture to the camera.")
        print("Press SPACE to start collecting, Q to quit.")

        collected = 0
        collecting = False

        while collected < NUM_SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                print("Camera error!")
                break

            frame = cv2.flip(frame, 1)  # mirror
            landmarks, annotated = tracker.find_hand(frame)

            # ---- UI overlay ----
            color = (0, 255, 0) if collecting else (0, 200, 255)
            status = f"COLLECTING {collected}/{NUM_SAMPLES_PER_GESTURE}" if collecting \
                else f"Press SPACE to start: {gesture_name}"

            cv2.putText(annotated, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated, f"Gesture: {gesture_name}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Progress bar
            bar_width = int(400 * collected / NUM_SAMPLES_PER_GESTURE)
            cv2.rectangle(annotated, (20, 440), (420, 460), (50, 50, 50), -1)
            cv2.rectangle(annotated, (20, 440), (20 + bar_width, 460), (0, 255, 0), -1)

            cv2.imshow("Data Collection", annotated)

            key = cv2.waitKey(CAPTURE_DELAY_MS) & 0xFF

            if key == ord(' '):  # SPACE = start collecting
                collecting = True
            elif key == ord('q'):  # Q = quit
                print("Quit requested.")
                cap.release()
                cv2.destroyAllWindows()
                tracker.release()
                return

            # ---- Collect sample if hand detected ----
            if collecting and landmarks is not None:
                features = tracker.extract_features(landmarks)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(gesture_name)
                    collected += 1

        print(f"  ✓ Collected {collected} samples for '{gesture_name}'")

    # ---- Save dataset ----
    feature_columns = [f"lm_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(all_features, columns=feature_columns)
    df['gesture'] = all_labels
    df.to_csv(DATASET_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Dataset saved: {DATASET_FILE}")
    print(f"  Total samples: {len(df)}")
    print(f"  Classes: {df['gesture'].nunique()}")
    print(f"  Samples per class:")
    for g, cnt in df['gesture'].value_counts().items():
        print(f"    {g}: {cnt}")
    print(f"{'=' * 60}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


if __name__ == "__main__":
    collect_data()
