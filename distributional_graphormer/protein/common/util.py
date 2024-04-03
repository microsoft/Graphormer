import functools
import os


def ensure_dir(func):
    def func_wrapper(*args, **kwargs):
        path = func(*args, **kwargs).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    return func_wrapper


@functools.lru_cache(maxsize=1)
def is_remote_running():
    return "AMLT_OUTPUT_DIR" in os.environ


def no_remote(func):
    def func_wrapper(*args, **kwargs):
        if is_remote_running():
            raise Exception("RemoteNotSupport")
        return func(*args, **kwargs)

    return func_wrapper
