import sys
import cv2
import mediapipe as mp
import time

from collections import deque

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout
)

from PyQt6.QtGui import QImage, QPixmap
from PyQt.QtCore import QTimer

from angle_helper import calculate_angle


# ── CONFIG ─────────────────────────────────────────────

SMOOTHING_WINDOW = 6

UP_ANGLE = 160
DOWN_ANGLE = 90


# ── REP TRACKER ────────────────────────────────────────

class RepTracker:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.angle_buffer = deque(maxlen=SMOOTHING_WINDOW)

    def update(self, angle):
        self.angle_buffer.append(angle)

        smooth_angle = sum(self.angle_buffer) / len(self.angle_buffer)

        if smooth_angle > UP_ANGLE:
            self.stage = "up"

        if smooth_angle < DOWN_ANGLE and self.stage == "up":
            self.stage = "down"
            self.counter += 1

        return smooth_angle


# ── MAIN WINDOW ────────────────────────────────────────

class GymTrackerApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gym AI Tracker")

        # Camera label
        self.image_label = QLabel()

        # Buttons
        self.next_set_button = QPushButton("Next Set")
        self.reset_button = QPushButton("Reset Workout")
        self.quit_button = QPushButton("Quit")

        # Layouts
        button_layout = QHBoxLayout()

        button_layout.addWidget(self.next_set_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.quit_button)

        main_layout = QVBoxLayout()

        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Button actions
        self.next_set_button.clicked.connect(self.next_set)
        self.reset_button.clicked.connect(self.reset_workout)
        self.quit_button.clicked.connect(self.close)

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose()

        # Webcam
        self.cap = cv2.VideoCapture(0)

        # Rep tracking
        self.tracker = RepTracker()

        # Set tracking
        self.current_set = 1

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def next_set(self):
        print(f"Set {self.current_set} completed")

        self.current_set += 1
        self.tracker.counter = 0

    def reset_workout(self):
        self.current_set = 1
        self.tracker.counter = 0

        print("Workout reset")

    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        # Convert image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose processing
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Get points
            hip = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]

            knee = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ]

            ankle = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            smooth_angle = self.tracker.update(angle)

            # Draw pose
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # Overlay text
            cv2.putText(
                frame,
                f"Reps: {self.tracker.counter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Angle: {int(smooth_angle)}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Set: {self.current_set}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 200, 100),
                2
            )

        # Convert frame to Qt image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape

        bytes_per_line = ch * w

        qt_image = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


# ── APP START ──────────────────────────────────────────

app = QApplication(sys.argv)

window = GymTrackerApp()
window.show()

sys.exit(app.exec_())