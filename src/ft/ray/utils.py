import dataclasses
from functools import cached_property

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from ft.ray.mock_learner import MockLearner


@dataclasses.dataclass
class ResourceConfig:
    gpu: int  # number of gpus
    cpu: int  # number of cpus
    n_worker: int  # number of subprocesses

    def __post_init__(self):
        assert self.cpu % self.gpu == 0

    def get_dict(self):
        return {k.name: getattr(self, k.name) for k in dataclasses.fields(self)}


@dataclasses.dataclass
class ResourceAssignment:
    vllm: ResourceConfig
    learner: ResourceConfig

    def get_dict(self):
        return {k.name: getattr(self, k.name).get_dict() for k in dataclasses.fields(self)}

    @cached_property
    def pg(self) -> dict:
        pg_dict = {
            k: [
                [
                    {
                        "GPU": 1,
                        "CPU": v["cpu"] // v["gpu"],
                    }
                ]
                * v["gpu"]
            ]
            * v["n_worker"]
            for k, v in self.get_dict().items()
        }
        pg = {k: [placement_group(d) for d in v] for k, v in pg_dict.items()}
        ray.get([w.ready() for v in pg.values() for w in v])
        return pg


def create_llm_workers(
    model_name: str,
    resources: ResourceAssignment,
) -> list[ray.actor.ActorHandle]:
    return [
        ray.remote(
            num_gpus=resources.vllm.gpu,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=resources.pg["vllm"][i],
            )
        )(LLM).remote(
            model=model_name,
            tensor_parallel_size=resources.vllm.gpu,  # assume tp = gpus each worker
            worker_cls="ft.ray.vllm_worker.EnhancedVLLMWorker",
        )
        for i in range(resources.vllm.n_worker)
    ]


def create_mock_learner(
    model_name: str,
    resources: ResourceAssignment,
) -> list[ray.actor.ActorHandle]:
    ray_args = [
        {
            "num_cpus": resources.learner.cpu,
            "num_gpus": resources.learner.gpu,
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=resources.pg["learner"][i],
            ),
        }
        for i in range(resources.learner.n_worker)
    ]
    return [
        ray.remote(**args)(MockLearner).remote(
            model=model_name,
        )
        for i, args in zip(range(resources.learner.n_worker), ray_args, strict=False)
    ]
