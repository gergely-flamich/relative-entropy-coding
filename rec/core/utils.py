import logging


class CoreError(Exception):
    """
    Base error thrown by modules in the core
    """


def setup_logger(name, level, log_file=None, to_console=False, format="%(levelname)s:%(name)s:%(message)s"):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(format)

    if log_file is not None:

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    if to_console:

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger

