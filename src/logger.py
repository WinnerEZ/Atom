import logging
import time
import os
from datetime import timedelta


_logger = None


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)#消耗的时间

        prefix = "%s [%d] - %s - %s" % (
            record.levelname,
            os.getpid(),
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def get_logger(filepath=None):
    """
    Create a logger.
    创造一个logger
    """
    global _logger
    if _logger is not None:
        assert _logger is not None
        return _logger
    assert filepath is not None
    # create log formatter 创造log格式化
    log_formatter = LogFormatter()

    # create file handler and set level to debug 创建文本输出并将模式设为debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info 创建控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug 创建logger，并将模式设为debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time 重置logger消耗时间
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    _logger = logger
    return logger
