import torch
from vllm.distributed.parallel_state import get_world_group
from vllm.worker.worker import Worker

from ft.logging import init_logger

logger = init_logger(__name__)


class EnhancedVLLMWorker(Worker):
    """Enhanced VLLM worker with weight update capabilities."""

    def init_weight_sync_group(self, master_address: str, master_port: int, rank_offset: int, world_size: int) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        rank = get_world_group().rank + rank_offset
        self.weight_sync_comm = PyNcclCommunicator(
            StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size),
            device=self.device,
        )

    def update_weights(self, weight_info: list[tuple]) -> None:
        logger.info("Updating %d weights.", len(weight_info))
        for name, dtype, shape in weight_info:
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            self.weight_sync_comm.broadcast(weight, src=0, stream=torch.cuda.current_stream())
            self.model_runner.model.load_weights(weights=[(name, weight)])
            del weight
        logger.info("Weight update is finished.")
