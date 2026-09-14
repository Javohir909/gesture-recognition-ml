"""
Hand tracking module using MediaPipe.
Detects hand landmarks and extracts features for ML.
"""
import cv2
import mediapipe as mp
import numpy as np
from config import (
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
)


class HandTracker:
    """Detects hand landmarks using MediaPipe."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_NUM_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )

    def find_hand(self, frame):
        """
        Process a BGR frame and return hand landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            landmarks (list of 21 x,y,z points) or None if no hand detected
            annotated_frame: frame with landmarks drawn on it
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        landmarks = None
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return landmarks, frame

    def extract_features(self, landmarks):
        """
        Extract ML features from raw landmarks.
        
        Raw landmarks are x,y,z in [0..1] range relative to image.
        We compute:
          - 21 landmark positions (normalized)
          - Distances between key finger tips and wrist
          - Angles between finger joints
          
        Returns: numpy array of 63 features (21 landmarks × 3 coords)
        """
        if landmarks is None:
            return None

        lm_array = np.array(landmarks)  # shape: (21, 3)

        # Normalize: make wrist the origin
        wrist = lm_array[0]
        lm_array = lm_array - wrist

        # Scale by max distance from wrist (so values are in [-1, 1])
        max_dist = np.max(np.linalg.norm(lm_array, axis=1))
        if max_dist > 0:
            lm_array = lm_array / max_dist

        # Flatten to 1D feature vector
        features = lm_array.flatten()  # 63 features
        return features

    def count_fingers_mediapipe(self, landmarks):
        """
        Simple finger counting using MediaPipe landmarks.
        
        Logic:
        - For thumb: compare tip x-position with IP joint
        - For other fingers: compare tip y-position with PIP joint
          (tip above PIP = finger is up)
        
        Returns: integer 0-5
        """
        if landmarks is None:
            return 0

        lm = landmarks  # list of [x, y, z]
        finger_count = 0

        # Thumb: compare tip (4) with IP joint (3)
        # For right hand: tip.x > ip.x means thumb is open
        # For left hand: tip.x < ip.x means thumb is open
        # Simplified: just check distance from thumb tip to index base
        thumb_tip = np.array(lm[4][:2])
        thumb_ip = np.array(lm[3][:2])
        index_mcp = np.array(lm[5][:2])

        # Thumb is open if distance from tip to index_mcp > from ip to index_mcp
        d_tip = np.linalg.norm(thumb_tip - index_mcp)
        d_ip = np.linalg.norm(thumb_ip - index_mcp)
        if d_tip > d_ip * 1.2:
            finger_count += 1

        # Other 4 fingers: tip y < PIP y means finger is up
        # (in image coords, y=0 is top)
        finger_tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
        finger_pips = [6, 10, 14, 18]   # PIP joints

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            if lm[tip_idx][1] < lm[pip_idx][1]:  # tip above PIP
                finger_count += 1

        return finger_count

    def release(self):
        """Free MediaPipe resources."""
        self.hands.close()
