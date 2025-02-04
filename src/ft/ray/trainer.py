
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.utils import get_ip, get_open_port

from ft.job_config import JobConfig
from ft.ray.base_actor import BaseActor
from ft.ray.dist_info import DistInfo
from ft.ray.mesh import ClusterMesh
from ft.trainer import Trainer


class RayTrainer(Trainer, BaseActor):
    def __init__(self, job_config: JobConfig, dist_info: DistInfo):
        dist_info.set_env_var()
        self.dist_info = dist_info
        Trainer.__init__(self, job_config)
        BaseActor.__init__(self)

    def update_weights(self):
        """Simulate weight updates with Gaussian noise."""
        param_info = []

        with torch.no_grad():
            for name, dtype, shape in self.weight_info:
                param = self.model.state_dict()[name]
                noise = torch.randn_like(param) * 0.01
                param.add_(noise)
                param_info.append((name, dtype, shape))

                self._sync_comm.broadcast(param, src=0, stream=torch.cuda.current_stream())


class RayTrainerGroup:
    def __init__(self, job_config: JobConfig, cluster_mesh: ClusterMesh):
        """Set up the ray trainer group."""
        self.config = job_config
        self.mesh = cluster_mesh

    def init_all(self):
        n_gpu = self.mesh.num_gpus_per_node
        n_cpu = self.mesh.num_cpu_per_node
        world_size = n_gpu * self.mesh.num_nodes
        #self.pg = placement_group([{"GPU": n_gpu, "CPU": n_cpu}] * self.mesh.num_nodes)
        self.pg = [placement_group([{"GPU": 1, "CPU": n_cpu}] * n_gpu) for _ in range(self.mesh.num_nodes)]
        for p in self.pg:
            ray.get(p.ready())

        dist_info = DistInfo(
            world_size=world_size,
            rank=0,
            master_addr=get_ip(),
            master_port=get_open_port(),
        )
        self.trainers = [
            ray.remote(
                num_gpus=1,
                num_cpus=max(n_cpu // n_gpu, 1),
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg[i // n_gpu],
                    placement_group_bundle_index=i % n_gpu,
                ),
            )(RayTrainer).remote(self.config, dist_info.replace("rank", i))
            for i in range(world_size)
        ]

    def train(self):
        ray.get([t.train.remote() for t in self.trainers])
