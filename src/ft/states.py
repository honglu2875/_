from io import BytesIO
from typing import Any
import torch
from torchtitan.checkpoint import CheckpointManager, TrainState
import dataclasses

@dataclasses.dataclass
class MTPredTrainState(TrainState):
    global_avg_fut_losses: list[float] = dataclasses.field(default_factory=list)

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        global_avg_fut_losses = BytesIO()
        torch.save(self.global_avg_fut_losses, global_avg_fut_losses)
        sd["global_avg_fut_losses"] = global_avg_fut_losses

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        state_dict["global_avg_fut_losses"].seek(0)
        self.global_avg_fut_losses = torch.load(state_dict["global_avg_fut_losses"])

