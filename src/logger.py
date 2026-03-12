"""
Logging configuration for FX prediction pipeline.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import keras


def setup_logger(
    name: str = "fx_prediction",
    log_dir: str = "../logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log"
    )
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger