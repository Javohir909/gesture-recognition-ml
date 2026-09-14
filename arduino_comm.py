"""
Arduino serial communication module.
Sends gesture commands to Arduino via Serial (USB).
"""
import serial
import time
from config import ARDUINO_PORT, ARDUINO_BAUD_RATE, GESTURE_SERVO_MAP


class ArduinoComm:
    """Communicates with Arduino over Serial USB."""

    def __init__(self, port=ARDUINO_PORT, baud_rate=ARDUINO_BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None

    def connect(self):
        """
        Open serial connection to Arduino.
        Waits 2 seconds for Arduino to reset after connection.
        """
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Arduino resets on serial connect
            print(f"[Arduino] Connected on {self.port} @ {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"[Arduino] Connection failed: {e}")
            return False

    def send_gesture(self, gesture_name):
        """
        Send a gesture command to Arduino.
        
        The Arduino receives a comma-separated string of 5 servo angles.
        Example: "180,0,180,0,0\n"
        
        Args:
            gesture_name: string like 'fist', 'open_hand', etc.
        """
        if self.serial_conn is None or not self.serial_conn.is_open:
            return False

        servo_values = GESTURE_SERVO_MAP.get(gesture_name, [0, 0, 0, 0, 0])
        message = ",".join(str(v) for v in servo_values) + "\n"

        try:
            self.serial_conn.write(message.encode('utf-8'))
            return True
        except serial.SerialException:
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[Arduino] Disconnected")
