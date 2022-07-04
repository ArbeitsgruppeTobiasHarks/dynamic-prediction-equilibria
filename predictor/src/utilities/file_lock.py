
import os
import time
from typing import IO, Callable, List, Optional


def NoOp(_):
    pass


def with_file_lock(file_path: str, handle: Callable[[Callable[[str], IO]], None] = NoOp):
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    lock_path = os.path.join(directory, f".lock.{filename}")
    if os.path.exists(lock_path):
        print(f"Detected lock file for {filename}. Skipping...")
        return
    elif os.path.exists(file_path):
        print(f"{filename} already exists. Skipping...")
        return

    try:
        with open(lock_path, "w") as lock_file:
            lock_file.write("")
        handle(lambda mode: open(file_path, mode))
    finally:
        try:
            os.remove(lock_path)
        except OSError as e:
            print(
                f"An error occurred while removing lock file {lock_path}.", e)


def wait_for_locks(dir: str):
    locks: Optional[List[str]] = None
    curr_backoff_secs = 1
    while locks is None or len(locks) > 0:
        locks = [file for file in os.listdir(dir) if file.startswith(".lock.")]
        if len(locks) > 0:
            print(f"Found locks in {dir}. Waiting for {curr_backoff_secs} seconds...")
            time.sleep(curr_backoff_secs)
            curr_backoff_secs = min(curr_backoff_secs*2, 30)
        elif len(locks) == 0 and curr_backoff_secs > 1:
            print(f"Locks have been removed. Proceeding.")
