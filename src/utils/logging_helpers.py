# utils/logging_helpers.py  – new tiny helper
import logging
def enable_full_debug(logger):
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s – %(message)s"))
