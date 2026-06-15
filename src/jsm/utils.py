"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import logging
import math
import re
import fsspec
import lightning
import torch


def parse_sequence(sequence):
    try:
        utr_3_match = re.search(r'<utr_3_bos>(.*?)<utr_3_eos>', sequence).group(1)
    except AttributeError:
        utr_3_match = re.search(r'<utr_3_bos>(.*?)', sequence).group(1)
    utr_5_match = re.search(r'<utr_5_bos>(.*?)<utr_5_eos>', sequence).group(1)
    cds_match = re.search(r'<cds_bos>(.*?)<cds_eos>', sequence).group(1)
    return utr_5_match, cds_match, utr_3_match


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(name, tensor)
        raise ValueError(
          f"Tensor {name} contains NaN values. Please check your data or model.")


class LoggingContext:
    """Context manager for selective logging."""
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ('debug', 'info', 'warning', 'error',
                  'exception', 'fatal', 'critical'):
        setattr(logger,
                level,
                lightning.pytorch.utilities.rank_zero_only(
                  getattr(logger, level)))

    return logger

