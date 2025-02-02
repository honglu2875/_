import dataclasses


@dataclasses.dataclass
class ClusterMesh:
    num_gpus_per_node: int
    num_nodes: int
    num_cpu_per_node: int = 8
