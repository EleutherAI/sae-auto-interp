import logging

def get_logger(name, path):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger