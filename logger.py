import logging
import datetime
import os

# Create today's date string
now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Ensure 'log/' directory exists
os.makedirs("log", exist_ok=True)

# File handler
file_handler = logging.FileHandler(f'log/MSBot_{now_str}.log', mode='w', encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def set_level(level):
    """Change log level at runtime."""
    logger.setLevel(level)
    file_handler.setLevel(level)
    console_handler.setLevel(level)
