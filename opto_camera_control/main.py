"""Entry point for the opto-camera control GUI.

Run from the dedicated Python 3.10 venv (the one with PySpin + nidaqmx +
PySide6 installed):

    .venv310\\Scripts\\python opto_camera_control\\main.py

The window opens even without the camera/DAQ SDKs present (PySide6 only); it
reports a clear error when you try to Connect or Start without them, so layout
can be checked on any interpreter that has PySide6.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 560)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
