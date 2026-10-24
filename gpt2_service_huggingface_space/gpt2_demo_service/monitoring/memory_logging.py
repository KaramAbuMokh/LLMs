# Reads and logs how much memory the service uses, so the Space logs show real peak numbers
import logging  # writes to the Space log
import resource  # peak memory counter
import sys  # tells Linux from macOS
import psutil  # current memory counter

logger = logging.getLogger(__name__)  # logger for this module


def read_memory_megabytes():  # returns (current, peak) memory in megabytes
    current_bytes = psutil.Process().memory_info().rss  # memory held right now
    peak_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # highest memory so far
    peak_bytes = peak_units if sys.platform == "darwin" else peak_units * 1024  # macOS counts bytes, Linux kilobytes
    return round(current_bytes / 1048576), round(peak_bytes / 1048576)  # convert bytes to megabytes


def log_memory_usage(stage):  # writes one memory line, stage says where we are
    current_megabytes, peak_megabytes = read_memory_megabytes()  # current and peak numbers
    logger.info("memory at %s: now %s MB, peak %s MB", stage, current_megabytes, peak_megabytes)  # log line
