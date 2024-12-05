"""Logging utilities for fitscube."""

from __future__ import annotations

import io
import logging

logging.captureWarnings(True)

# Following guide from gwerbin/multiprocessing_logging.py
# https://gist.github.com/gwerbin/e9ab7a88fef03771ab0bf3a11cf921bc


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger: logging.Logger, level: int | None = None):
        super().__init__()
        self.logger = logger
        self.level = level if level else logging.INFO

    def write(self, buf: str) -> int:
        self.buf = buf.strip("\r\n\t ")
        return len(buf)

    def flush(self) -> None:
        if not self.logger:
            return
        if not self.level:
            level = logging.INFO
        self.logger.log(level, self.buf)


# pylint: disable=W0621
def set_verbosity(logger: logging.Logger, verbosity: int) -> None:
    """Set the logger verbosity.

    Args:
        logger (logging.Logger): The logger
        verbosity (int): Verbosity level
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    else:
        level = logging.CRITICAL

    logger.setLevel(level)


logger = logging.getLogger("fitscube")
formatter = logging.Formatter(
    fmt="[%(threadName)s] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)
logger.addHandler(ch)
