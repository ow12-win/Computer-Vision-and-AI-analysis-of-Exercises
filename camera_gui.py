import sys
import cv2

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout
)

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer


class CameraWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gym AI Tracker")

        self.image_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.RGB888
        )

        self.image_label.setPixmap(QPixmap.fromImage(qt_image))


app = QApplication(sys.argv)

window = CameraWindow()
window.show()

sys.exit(app.exec())