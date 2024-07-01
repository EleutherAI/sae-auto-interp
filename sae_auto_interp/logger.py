import logging
from . import log_path

logger = logging.getLogger("sae_auto_interp")
logger.setLevel(logging.INFO)  

file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO) 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)