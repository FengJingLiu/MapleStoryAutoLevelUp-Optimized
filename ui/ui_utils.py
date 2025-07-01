

# PySide 6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QCheckBox, QListWidget, QScrollArea, QFileDialog, QHBoxLayout, QLineEdit,
    QKeySequenceEdit, QPlainTextEdit
)
from PySide6.QtGui import QPixmap, QKeySequence, QKeyEvent
from PySide6.QtCore import Qt, QObject, Signal


# Local import
from logger import logger

def validate_numerical_input(input_str: str, error_label: QLabel,
                              val_lowest: float, val_highest: float):
    try:
        val = float(input_str)
        if not (val_lowest <= val <= val_highest):
            raise ValueError
        error_label.setVisible(False)
        return True
    except ValueError:
        error_label.setText(f"Please enter number between {val_lowest} and {val_highest}.")
        error_label.setVisible(True)
        logger.debug(f"Invalid numerical input: {input_str}")
        return False

def validate_key_input(key_edit: QKeySequenceEdit):
    sequence = key_edit.keySequence()

    if sequence.count() == 0:
        return  # Nothing pressed yet

    if sequence.count() > 1 or '+' in sequence.toString():
        # Take only the last valid single key
        last_seq = sequence[sequence.count() - 1]
        key_edit.setKeySequence(QKeySequence(last_seq))
        logger.info("Multiple keys/combo detected. Using only the last key.")
        key_edit.setToolTip("Only one key allowed. Replaced with the last one.")
    else:
        key_edit.setToolTip("")
