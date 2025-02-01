# This is an example of a mock actor that groups with vllm engines and communicates
import time
import torch
import ray
from ray.train.torch import get_device
from transformers import AutoModelForCausalLM
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from typing import List, Tuple, Dict


class MockLearner:
    """Learner for simulating weight updates with Gaussian noise."""
    
    def __init__(
        self,
        model: str,
        update_interval: float = 5.0
    ):
        self.device = get_device()
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.to(self.device)
        self.weight_info = [(name, param.dtype, list(param.shape)) for name, param in self.model.named_parameters()]
        self.update_interval = update_interval
        self._weight_version = 0
        self._sync_group = None

    def get_weight_info(self):
        return self.weight_info

    def init_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int
    ) -> None:
        self._sync_group = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=0,
            world_size=world_size
        )
        self._sync_comm = PyNcclCommunicator(self._sync_group, device=self.device)

    def update_weights(self) -> List[Tuple[str, torch.dtype, List[int]]]:
        """Simulate weight updates with Gaussian noise."""
        time.sleep(self.update_interval)
        param_info = []
        
        with torch.no_grad():
            for name, dtype, shape in self.weight_info:
                param = self.model.state_dict()[name]
                noise = torch.randn_like(param) * 0.01
                param.add_(noise)
                param_info.append((name, dtype, shape))
                
                self._sync_comm.broadcast(
                    param,
                    src=0,
                    stream=torch.cuda.current_stream()
                )
        
        self._weight_version += 1

    def get_weight_version(self) -> int:
        return self._weight_version

