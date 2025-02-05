from ft.job_config import JobConfig
from ft.ray.grpo_trainer import GRPORayTrainerGroup
from ft.ray.mesh import ClusterMesh


def main():
    config = JobConfig()
    config.parse_args()

    cluster_mesh = ClusterMesh(
        num_gpus_per_node=4,
        num_cpu_per_node=32,
        num_nodes=1,
    )
    trainer_group = GRPORayTrainerGroup(config, cluster_mesh)
    trainer_group.init_all()
    trainer_group.train()


if __name__ == "__main__":
    main()
