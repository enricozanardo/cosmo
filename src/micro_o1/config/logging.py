import sys
from pathlib import Path
from loguru import logger

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
(logs_dir / "tests").mkdir(exist_ok=True)

def setup_logger(name: str, filename: str):
    """Setup logger configuration
    
    Args:
        name: Name of the logger (e.g., 'tokenizer', 'embeddings')
        filename: Log file name
    """
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        colorize=True
    )
    
    # Add file handler
    logger.add(
        f"logs/{filename}",
        rotation="500 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    return logger.bind(name=name) 