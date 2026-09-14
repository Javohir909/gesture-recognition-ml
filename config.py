"""
Configuration file for Gesture Recognition ML Project.
All settings in one place — easy to tweak.
"""

# ============================================================
# CAMERA SETTINGS
# ============================================================
CAMERA_INDEX = 0           # 0 = default webcam, 1 = external cam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ============================================================
# GESTURE DEFINITIONS
# Change these to match YOUR gestures!
# ============================================================
GESTURES = {
    0: "fist",
    1: "one_finger",
    2: "two_fingers",
    3: "three_fingers",
    4: "four_fingers",
    5: "open_hand",
}

# Display names for each gesture (friendly labels)
GESTURE_DISPLAY = {
    "fist":         "✊ Кулак",
    "one_finger":   "☝️ Один палец",
    "two_fingers": "✌️ Два пальца",
    "three_fingers":"🤟 Три пальца",
    "four_fingers": "🖖 Четыре",
    "open_hand":    "🖐️ Открытая рука",
}

# ============================================================
# DATA COLLECTION
# ============================================================
NUM_SAMPLES_PER_GESTURE = 200     # images per gesture class
CAPTURE_DELAY_MS = 50            # delay between captures (ms)
DATA_DIR = "data"                 # folder to save collected data
DATASET_FILE = "data/gesture_dataset.csv"

# ============================================================
# MODEL TRAINING
# ============================================================
TEST_SIZE = 0.2          # 20% of data for testing
RANDOM_STATE = 42        # reproducibility seed
MODEL_TYPE = "random_forest"  # "random_forest" or "neural_network"
MODEL_SAVE_PATH = "models/gesture_model.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# Random Forest hyperparameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20

# Neural Network hyperparameters
NN_HIDDEN_LAYERS = (128, 64, 32)
NN_MAX_ITER = 500

# ============================================================
# ARDUINO SERIAL COMMUNICATION
# ============================================================
ARDUINO_PORT = "/dev/ttyUSB0"    # Change for your system
                                # Windows: "COM3", Mac: "/dev/tty.usbmodem..."
ARDUINO_BAUD_RATE = 9600
ARDUINO_ENABLED = False          # Set True if Arduino is connected

# Gesture-to-servo mapping (0-180 degrees for each servo finger)
GESTURE_SERVO_MAP = {
    "fist":          [0, 0, 0, 0, 0],      # all fingers closed
    "one_finger":    [180, 0, 0, 0, 0],    # thumb up
    "two_fingers":  [0, 180, 180, 0, 0],  # index + middle
    "three_fingers":[0, 180, 180, 180, 0],
    "four_fingers": [0, 180, 180, 180, 180],
    "open_hand":     [180, 180, 180, 180, 180],  # all open
}

# ============================================================
# MEDIAPIPE HAND SETTINGS
# ============================================================
MP_MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.7
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================
# DISPLAY
# ============================================================
DISPLAY_WINDOW = "Gesture Recognition ML"
FONT = None  # will use cv2.FONT_HERSHEY_SIMPLEX
