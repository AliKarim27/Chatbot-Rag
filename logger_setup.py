import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure the root logger (idempotent).

    If logging is already configured (handlers exist) this is a no-op.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def get_logger(name: str):
    return logging.getLogger(name)
