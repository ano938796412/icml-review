import logging
import os
import sys


def get_logger(log_dir, log_name, log_level):
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(
        f"{os.path.join(log_dir, log_name)}.log", mode="w", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def chain_remove_none(xs):
    """like itertools.chain but remove `None`"""
    for s in xs:
        if s is not None:
            for x in s:
                yield x
