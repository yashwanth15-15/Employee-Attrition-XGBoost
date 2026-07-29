import logging
import sys


class FlushHandler(logging.StreamHandler):
    """A StreamHandler that flushes after every emit.
    Ensures logs are visible immediately in containerised environments (e.g. Render).
    """

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging():
    logger = logging.getLogger("api")
    logger.setLevel(logging.INFO)

    handler = FlushHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logging()

