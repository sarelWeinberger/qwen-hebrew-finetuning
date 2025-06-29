# text_cleaning/logger.py
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Create and return a simple logger
logger = logging.getLogger('TextCleaning')