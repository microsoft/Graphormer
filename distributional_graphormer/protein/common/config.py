import functools
import hashlib
from pathlib import Path

import yaml

from .logger import Logger
from .util import ensure_dir, is_remote_running, no_remote

__all__ = ["config"]


class Config:
    def __init__(self):
        # for data
        self.max_tokens_per_batch = 256 * 128
        self.max_tokens_per_gpu = 256 * 16
        self.max_tokens_per_sample = 256

        # for training
        self.mixed_precision = True
        self.nr_step = 500000
        self.warmup_step = 30000
        try:
            import torch

            self.n_gpu = torch.distributed.get_world_size()
        except:
            self.n_gpu = 1
        self.n_accumulate = (
            self.max_tokens_per_batch // self.n_gpu // self.max_tokens_per_gpu
        )
        self.lr = 0.001

        # for checkpoint
        self.chk_time_interval = 3600
        self.chk_step_interval = [10000, 50000]

        # for job
        self._job = self._load_job_config()

    @property
    @functools.lru_cache(maxsize=1)
    def experiment_root(self):
        return Path(__file__).resolve().parent.parent

    @property
    @no_remote
    @functools.lru_cache(maxsize=1)
    def project_root(self):
        cur = Path(__file__).resolve().parent

        while True:
            if (cur / ".projectile").exists():
                return cur
            if cur == "/":
                raise Exception("ProjectRootNotFound")
            cur = cur.parent

    def _create_logger(self, path, **kwargs):
        return Logger(path, **kwargs)

    @property
    @functools.lru_cache(maxsize=1)
    def train_logger(self):
        return self._create_logger(self.log_dir / "train_log.txt")

    @property
    @functools.lru_cache(maxsize=1)
    def dataset_dir(self):
        path = [self.experiment_root / "dataset/Mars2"]
        for p in path:
            if p.exists():
                return p
        raise Exception("DatasetNotFoundError")

    @property
    @no_remote
    @functools.lru_cache(maxsize=1)
    def unique_id(self):
        relpath = self.experiment_root.relative_to(self.project_root)
        return hashlib.md5(str(relpath).encode()).hexdigest()

    @property
    def _job_config_path(self):
        return self.experiment_root / ".job.yaml"

    @no_remote
    def _generate_job_config(self):
        print("Generating new config...")
        project_saved_dir = Path("/blob/dig_protein")
        relpath = self.experiment_root.relative_to(self.project_root)
        saved_dir = str(project_saved_dir / relpath)
        job = {
            "unique_id": self.unique_id,
            "saved_dir": saved_dir,
        }
        with open(self._job_config_path, "w") as fp:
            yaml.dump(job, fp)
        return job

    def _load_job_config(self):
        job = None
        try:
            with open(self._job_config_path, "r") as fp:
                job = yaml.safe_load(fp.read())
        except:
            job = self._generate_job_config()
        if not is_remote_running() and job.get("unique_id", "") != self.unique_id:
            return self._generate_job_config()
        return job

    @property
    @ensure_dir
    def saved_dir(self):
        return Path(self._job["saved_dir"])

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def log_dir(self):
        return self.saved_dir / "log"

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def checkpoint_dir(self):
        return Path("/tmp/checkpoint")

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def model_dir(self):
        return self.saved_dir / "model"

    @property
    @functools.lru_cache(maxsize=1)
    def finetune_model_path(self):
        return None


config = Config()
