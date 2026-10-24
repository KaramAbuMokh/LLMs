import logging
import os

try:
    import psutil
except Exception:
    psutil = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.debug("Entering setup_logging")
    try:
        os.makedirs("logs", exist_ok=True)
        logger.debug("Ensured logs directory exists")
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        logger.info("Logging has been configured")
    except Exception:
        logger.error("Failed to set up logging", exc_info=True)
        raise


def log_memory_usage(stage: str):
    logger = logging.getLogger(__name__)
    logger.debug("Entering log_memory_usage for stage=%s", stage)
    try:
        if psutil is None:
            logger.warning("psutil is not available; skipping memory usage logging")
            return
        process = psutil.Process(os.getpid())
        logger.debug("Retrieved process information")
        mem = process.memory_info()
        logger.info(
            "Memory usage at %s - RSS: %.2f MB, VMS: %.2f MB",
            stage,
            mem.rss / (1024 * 1024),
            mem.vms / (1024 * 1024),
        )
    except Exception:
        logger.error("Failed to log memory usage for stage=%s", stage, exc_info=True)
