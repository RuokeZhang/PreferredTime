import logging
import sys
from typing import Optional

def setup_logger(name: str, level: Optional[str] = "INFO") -> logging.Logger:
    """设置并返回logger实例"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


