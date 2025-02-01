import dataclasses
from io import BytesIO
from typing import Any

import torch
from torchtitan.checkpoint import TrainState


@dataclasses.dataclass(frozen=True, slots=True)
class Metadata:
    num_tokens: torch.Tensor  # only masked token count in the current process
    num_tokens_full: int  # both masked and unmasked in the current process

    def __post_init__(self):
        assert isinstance(self.num_tokens, torch.Tensor)
        assert self.num_tokens.shape == (1,)
        assert self.num_tokens.dtype == torch.int32


@dataclasses.dataclass
class MTPredTrainState(TrainState):
    global_avg_fut_losses: list[float] = dataclasses.field(default_factory=list)

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        global_avg_fut_losses = BytesIO()
        torch.save(self.global_avg_fut_losses, global_avg_fut_losses)
        sd["global_avg_fut_losses"] = global_avg_fut_losses
        return sd

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        state_dict["global_avg_fut_losses"].seek(0)
        self.global_avg_fut_losses = torch.load(state_dict["global_avg_fut_losses"])
