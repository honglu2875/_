import torch
from ray.train.torch import get_device
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class BaseActor:
    device: torch.device

    def __init__(self):
        self.device = get_device()

    def init_sync_group(self, master_address: str, master_port: int, world_size: int) -> None:
        self._sync_group = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=0, world_size=world_size
        )
        self._sync_comm = PyNcclCommunicator(self._sync_group, device=self.device)
