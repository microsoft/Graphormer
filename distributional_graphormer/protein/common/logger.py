from datetime import datetime


class Logger:
    def __init__(self, path, rotate_size=2e6, buffer_size=4e3):
        self._path = path
        self._rotate_size = rotate_size
        self._buffer_size = buffer_size

        self._file = None
        self._buffer = ""

    def info(self, msg):
        self._buffer += msg + "\n"
        if len(self._buffer) > self._buffer_size:
            self.flush()

    def flush(self):
        if len(self._buffer) > 0:
            self._write_fn(self._buffer)
            self._buffer = ""

    def _write_fn(self, msg):
        if self._file is None or self._rotate_fn(msg):
            self._init_file()

        with open(self._file, "a") as fp:
            fp.write(msg)

    def _prepare_new_path(self):
        if self._rotate_size is None:
            return self._path
        time = datetime.now().__format__("%Y-%m-%d_%H-%M-%S_%f")
        suffix = "." + time + self._path.suffix
        return self._path.with_suffix(suffix)

    def _init_file(self):
        self._file = self._prepare_new_path()

    def _rotate_fn(self, msg):
        return (
            self._rotate_size is not None
            and self._file.stat().st_size + len(msg) > self._rotate_size
        )
