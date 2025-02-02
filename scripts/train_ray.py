from ft.job_config import JobConfig
from ft.ray.mesh import ClusterMesh
from ft.ray.trainer import RayTrainerGroup


def main():
    config = JobConfig()
    config.parse_args()

    cluster_mesh = ClusterMesh(
        num_gpus_per_node=4,
        num_cpu_per_node=32,
        num_nodes=1,
    )
    trainer_group = RayTrainerGroup(config, cluster_mesh)
    trainer_group.init_all()
    trainer_group.train()


if __name__ == "__main__":
    main()
