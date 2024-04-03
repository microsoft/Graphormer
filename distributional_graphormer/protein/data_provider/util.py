import functools
import pickle
import zlib

import lmdb


class LMDBReader:
    def __init__(self, db_path):
        env_arg = {
            "max_readers": 1,
            "readonly": True,
            "lock": False,
            "readahead": False,
            "meminit": False,
        }

        self._env_clu = lmdb.open(str(db_path / "cluster"), **env_arg)
        self._env_str = lmdb.open(str(db_path / "structure"), **env_arg)

    @functools.lru_cache(maxsize=1)
    def get_num_cluster(self):
        with self._env_clu.begin(write=False) as txn:
            v = txn.get(b"n_clu")
            return pickle.loads(v)

    def get_clu(self, tp, index):
        with self._env_clu.begin(write=False) as txn:
            v = txn.get(f"{tp}:{index}".encode("ascii"))
            return pickle.loads(v)

    def get_structure(self, index):
        with self._env_str.begin(write=False) as txn:
            v = txn.get(f"{index}".encode("ascii"))
            return pickle.loads(v)
