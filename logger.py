import logging
import sys
from pathlib import Path


def get_logger(name: str = "airflow_ai", log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_metrics(logger, metrics: dict, prefix: str = "") -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("%s%s: %.4f", prefix, key, value)
        else:
            logger.info("%s%s: %s", prefix, key, value)


def log_section(logger, title: str, width: int = 60, char: str = "=") -> None:
    bar = char * width
    logger.info(bar)
    logger.info(title.center(width))
    logger.info(bar)
