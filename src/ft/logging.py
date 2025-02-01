import logging
import os


def init_logger(name: str | None = None) -> logging.Logger:
    if name is not None:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    os.environ["KINETO_LOG_LEVEL"] = "5"
    return logger
