import logging
import datetime

# Create today's date string
today_str = datetime.datetime.now().strftime("%Y-%m-%d")

# Create logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# File handler
file_handler = logging.FileHandler(f'maplestory_{today_str}.log', mode='w', encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
