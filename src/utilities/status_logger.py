import os
import sys
import time
from typing import Optional


class StatusLogger:
    def __init__(self, start_msg: str, finish_msg: Optional[str] = None):
        self.start_msg = start_msg
        self.finish_msg = finish_msg

    def __enter__(self):
        sys.stdout.write("\r" + self.start_msg)
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            if self.finish_msg is None:
                sys.stdout.write("\r\033[K\r")
            else:
                sys.stdout.write("\r" + self.finish_msg + os.linesep)
            sys.stdout.flush()


class TimedStatusLogger:
    def __init__(self, start_msg: str, finish_msg: Optional[str] = None):
        self.start_msg = start_msg
        self.start_time = time.perf_counter_ns()
        self.finish_msg = finish_msg

    def __enter__(self):
        sys.stdout.write("\r" + self.start_msg)
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter_ns()
        elapsed_secs = (end_time - self.start_time) / 1e9
        if exc_type is None:
            if self.finish_msg is None:
                sys.stdout.write("\r\033[K\r")
            else:
                sys.stdout.write("\r" + f"Took {elapsed_secs :.1}s: " + self.finish_msg + os.linesep)
            sys.stdout.flush()