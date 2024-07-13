import sys

from loguru import logger

def set_log_level(level: str):
    logger.remove()
    
    format1="<green>{time:YY-MM-DD HH:mm:ss.SSS}</green> ""<level>{level}</level> ""<cyan>{file}:{line}</cyan> ""<yellow>{thread.name}</yellow> ""- {message}"
    format2 = "<level>{level}</level> ""- {message}"
    
    logger.add(
        sys.stdout,
        colorize = True,
        format = format2,
        level = level
    )


logger.remove()
set_log_level('DEBUG')


def get_logger() -> logger:
    return logger


def main():
    logger.debug("这是一个调试信息")
    logger.info("这是一个信息")
    logger.warning("这是一个警告")
    logger.error("这是一个错误")
    logger.critical("这是一个严重错误")
    pass
 
 
if __name__ == '__main__':
    main()
    pass
