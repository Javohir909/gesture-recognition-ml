"""
=================================================================
Step 3: Real-time Gesture Recognition
=================================================================
Opens webcam, detects hand, extracts features, and predicts
the gesture using the trained ML model in real-time.

Optionally sends commands to Arduino robot arm.

How to use:
  1. Make sure you have trained the model (run 2_train_model.py)
  2. Run this script
  3. Show hand gestures to the camera
  4. Press Q to quit
=================================================================
"""
import cv2
import numpy as np
import joblib
import time

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH, LABEL_ENCODER_PATH,
    ARDUINO_ENABLED, DISPLAY_WINDOW,
    GESTURE_DISPLAY, GESTURE_SERVO_MAP,
)
from utils.hand_tracker import HandTracker
from utils.arduino_comm import ArduinoComm


def recognize_realtime():
    # ---- Load trained model ----
    print("\n" + "=" * 60)
    print("  REAL-TIME GESTURE RECOGNITION")
    print("=" * 60)

    try:
        model = joblib.load(MODEL_SAVE_PATH)
        scaler = joblib.load(SCALER_SAVE_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        print(f"  Model loaded: {MODEL_SAVE_PATH}")
        print(f"  Classes: {list(le.classes_)}")
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found! ({e})")
        print("Run 2_train_model.py first!")
        return

    # ---- Setup camera ----
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    tracker = HandTracker()

    # ---- Setup Arduino (optional) ----
    arduino = None
    if ARDUINO_ENABLED:
        arduino = ArduinoComm()
        if arduino.connect():
            print("  Arduino connected!")
        else:
            print("  Arduino connection failed — continuing without Arduino")
            arduino = None

    # ---- Gesture smoothing ----
    # To avoid flickering, we use a voting buffer:
    # keep last N predictions and show the most common one
    BUFFER_SIZE = 5
    prediction_buffer = []

    current_gesture = "none"
    last_arduino_cmd = ""
    fps_time = time.time()

    print("\n  Show your hand to the camera!")
    print("  Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        landmarks, annotated = tracker.find_hand(frame)

        # ---- Predict gesture ----
        if landmarks is not None:
            features = tracker.extract_features(landmarks)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
                gesture_name = le.inverse_transform([prediction])[0]
                confidence = np.max(model.predict_proba(features_scaled))

                # Add to buffer for smoothing
                prediction_buffer.append(gesture_name)
                if len(prediction_buffer) > BUFFER_SIZE:
                    prediction_buffer.pop(0)

                # Most common prediction in buffer
                from collections import Counter
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                current_gesture = most_common

                # ---- Display on frame ----
                display_name = GESTURE_DISPLAY.get(gesture_name, gesture_name)
                conf_text = f"{confidence*100:.0f}%"

                # Big gesture label
                cv2.putText(annotated, display_name, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                cv2.putText(annotated, f"Confidence: {conf_text}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # Finger count (from MediaPipe directly)
                finger_count = tracker.count_fingers_mediapipe(landmarks)
                cv2.putText(annotated, f"Fingers: {finger_count}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # ---- Send to Arduino ----
                if arduino and current_gesture != last_arduino_cmd:
                    arduino.send_gesture(current_gesture)
                    last_arduino_cmd = current_gesture

        else:
            current_gesture = "none"
            cv2.putText(annotated, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ---- FPS counter ----
        now = time.time()
        fps = 1.0 / (now - fps_time) if (now - fps_time) > 0 else 0
        fps_time = now
        cv2.putText(annotated, f"FPS: {fps:.0f}", (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(DISPLAY_WINDOW, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---- Cleanup ----
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    if arduino:
        arduino.disconnect()
    print("\nGoodbye!")


if __name__ == "__main__":
    recognize_realtime()
