import os
import logging


def get_logger(name, level="warn"):
    """Get the logger for printing informations.
    Used for layout the information of various stages while executing the program.
    Set the environment variable ``LOG_LEVEL`` to change the default level.
    Parameters
    ----------
    name: str
        Name of the logger.
    level: {'debug', 'info', 'warn', 'warning', 'error', 'critical'}
        Level of the logger. The level 'warn' and 'warning' are different. The former
        is the default level and the actual level is set to logging.INFO, and for
        'warning' which will be set to true logging.WARN level. The purpose behind this
        design is to categorize the message layout into several different formats.
    """
    logger = logging.getLogger(name)
    level = os.environ.get("LOG_LEVEL", level)

    msg_formats = {
        "debug": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "info": "%(asctime)s %(message)s  [at %(filename)s:%(lineno)d]",
        "warn": "%(asctime)s %(message)s",
        "warning": "%(asctime)s %(message)s",
        "error": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "critical": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
    }
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=msg_formats[level.lower()], datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if len(logger.handlers) > 0:
        rm_idx = [idx for idx, handler in enumerate(logger.handlers) if isinstance(handler, logging.StreamHandler)]
        for idx in rm_idx:
            del logger.handlers[idx]
    logger.addHandler(handler)
    logger.setLevel(level_mapping[level.lower()])
    return logger