import os
import logging
from colorlog import ColoredFormatter

def model_select(path):
    """
    PTM - SIB 에서 학습된 모델 중 가장 loss가 작은 것을 가져옵니다.
    PTM - KMEANS 에서 학습된 모델 중 가장 loss가 작은 것을 가져옵니다.

    path : 모델이 담겨져 있는 디렉토리
    """
    
    model_list = os.listdir(path)

    valid_loss_list = list()
    for model in model_list:
        valid_loss = model.split('valid_loss')[1].split('_model.pt')[0]
        valid_loss_list.append(float(valid_loss))

    selected_model = f"valid_loss{min(valid_loss_list)}_model.pt"

    return path + selected_model

def set_logger(log_dir):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        ('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        log_colors={
        'DEBUG':    'cyan',
        'INFO':     'yellow,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(f"{log_dir}IDoFew.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
            
