import os

import torch

from torchtitan import utils
from torchtitan.parallelisms import ParallelDims
from torchtitan.utils import device_module, device_type


class MeshHandler:
    """Handles distributed training mesh and model parallelization"""

    def __init__(self, job_config):
        self.job_config = job_config
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        self.parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=self.world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )

        utils.init_distributed(job_config)
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)

        if self.parallel_dims.dp_enabled:
            self.dp_mesh = self.world_mesh["dp"]
            self.dp_degree = self.dp_mesh.size()
            self.dp_rank = self.dp_mesh.get_local_rank()
        else:
            self.dp_degree = 1
            self.dp_rank = 0

        if self.parallel_dims.pp_enabled:
            self.pp_mesh = self.world_mesh["pp"]
        else:
            self.pp_mesh = None

    def setup_deterministic(self, seed: int, deterministic: bool):
        """Set random seed and deterministic mode"""
        utils.set_determinism(self.world_mesh, self.device, seed, deterministic)
