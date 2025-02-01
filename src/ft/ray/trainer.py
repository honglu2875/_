import ray
import os

from ft.job_config import JobConfig
from ft.ray.base_actor import BaseActor
from ft.ray.dist_info import DistInfo
from ft.trainer import Trainer


class RayTrainer(Trainer, BaseActor):
    def __init__(self, job_config: JobConfig, dist_info: DistInfo):
        dist_info.set_env_var()
        self.dist_info = dist_info
        Trainer.__init__(self, job_config)
        BaseActor.__init__(self)
