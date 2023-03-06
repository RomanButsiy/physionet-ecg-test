import logging
import sys
import os
from pythonjsonlogger.jsonlogger import JsonFormatter


FORMAT = "%(asctime)s <%(name)s> [%(levelname)s] - u%(message)s"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)

    if os.getenv('LOGGING_CONSOLE_FORMATTER', 'json') == 'json':
        console_handler.setFormatter(JsonFormatter(FORMAT))
    else:
        console_handler.setFormatter(logging.Formatter(FORMAT))
    return console_handler


def add_global_logger_extra(**extra):
    current_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        record.__dict__.update(extra)
        return record

    logging.setLogRecordFactory(record_factory)


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        # better to have too much log than not enough
        logger.setLevel(logging.DEBUG)

        logger.addHandler(get_console_handler())

        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False

    return logger
