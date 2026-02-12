import logging
import os
from datetime import datetime
import pytz

def setup_logger(dataset_name='all', logs_dir='logs'):
    """
    设置logger，使用中国时区，日志文件以时间戳命名，并为每个数据集创建独立目录
    
    Args:
        dataset_name (str): 数据集名称，默认为'all'
        logs_dir (str): 日志文件根目录
        
    Returns:
        logging.Logger: 配置好的logger实例
        timestamp (str): 时间戳字符串
    """
    # 为特定数据集创建子目录
    if dataset_name != 'all':
        dataset_logs_dir = os.path.join(logs_dir, dataset_name)
    else:
        dataset_logs_dir = logs_dir
        
    # 创建目录（如果不存在）
    os.makedirs(dataset_logs_dir, exist_ok=True)
    
    # 获取中国时区的时间戳作为文件名
    china_tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.now(china_tz).strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(dataset_logs_dir, f"{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger(f"HEMM_{dataset_name}")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter并添加到处理器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    if not logger.handlers:  # 避免重复添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger, timestamp