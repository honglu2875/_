import dataclasses
from collections.abc import Callable
from dataclasses import field
from enum import Enum


class DatasetType(Enum):
    PRETRAIN = 0
    SFT = 1
    REASONING = 2


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable
    extra_processor: Callable | None = None
    type: DatasetType = DatasetType.PRETRAIN


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetMixConfig:
    datasets: list[DatasetConfig] = field(default_factory=list)

    def __post_init__(self):
        assert all(self.datasets[0].type == d.type for d in self.datasets)

    @property
    def type(self):
        return self.datasets[0].type
