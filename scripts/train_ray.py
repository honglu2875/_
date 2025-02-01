from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ft.ray.dist_info import DistInfo
from ft.ray.trainer import RayTrainer
from ft.job_config import JobConfig
import ray
import torch
from vllm.utils import get_ip, get_open_port


def main():
    config = JobConfig()
    config.parse_args()
    pg_train = placement_group([{"GPU": 4, "CPU": 40}] * 1)
    ray.get(pg_train.ready())
    base_dist_info = DistInfo(
        world_size=4,
        rank=0,
        master_addr=get_ip(),
        master_port=get_open_port(),
    )
    trainers = [
        ray.remote(
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg_train,
                placement_group_bundle_index=0,
            ),
        )(RayTrainer).remote(config, base_dist_info.replace("rank", i)) for i in range(4)
    ]
    refs = [t.train.remote() for t in trainers]
    ray.get(refs)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
