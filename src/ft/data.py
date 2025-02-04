from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import datasets
import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader

from ft.dataset_config import DatasetConfig, DatasetMixConfig, DatasetType
from ft.logging import init_logger
from torchtitan.datasets.hf_datasets import DPAwareDataLoader, HuggingFaceDataset

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    extra: Any


class Dataset(HuggingFaceDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: Any,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        shuffle: bool = False,
        seed: int = 42,
        padding: bool = False,
        bos: bool = True,
        eos: bool = True,
    ) -> None:
        self.config = dataset_config
        ds = self._get_ds(self.config, shuffle=shuffle, seed=seed)
        self.dataset_name = self.name.split("/")[-1]
        self.seed = seed
        self.seq_len = seq_len
        self.infinite = infinite
        if dataset_config.type == DatasetType.PRETRAIN and padding:
            logger.warning(
                "The dataset type is set to be PRETRAIN. "
                "But I do not see any reason to do padding in pretraining. "
                "Emitting this warning for awareness."
            )
        if dataset_config.type != DatasetType.PRETRAIN and not padding:
            logger.error(
                "The dataset type is not PRETRAIN but padding=False. "
                "It is very rare to do packing on SFT or RL. "
                "I will raise an Exception here to ensure there is no mistake. "
                "If an actual need arises, I will change the behavior."
            )
            raise ValueError()
        self.padding = padding
        self.bos = bos
        self.eos = eos
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self._text_processor = self.config.text_processor
        self._extra_processor = getattr(self.config, "extra_processor", lambda *args: None)

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: list[int] = []

    def _get_ds(self, config: DatasetConfig, shuffle: bool, seed: int = 42) -> datasets.Dataset:
        assert isinstance(config, DatasetConfig)
        path = config.path
        dataset_loader = config.loader
        ds = dataset_loader(path)
        if hasattr(ds, "__len__"):
            logger.info("Loaded dataset: %s, Number of samples: %d.", ds.info.dataset_name, len(ds))
        else:
            logger.info("Loaded dataset: %s.", ds.info.dataset_name)
        if shuffle:
            if config.type == DatasetType.PRETRAIN:
                logger.warning(
                    "Dataset is marked as PRETRAIN. Despite"
                    "shuffle=True, shuffling will be ignored (due to"
                    "overhead of shuffling a large pretraining"
                    "dataset)."
                )
            else:
                ds = ds.shuffle(seed=seed)
        return ds

    @property
    def name(self):
        assert isinstance(self.config, DatasetConfig)
        return self.config.path.split("/")[-1]

    def __iter__(self):
        if not self.padding:
            yield from map(lambda ret: Batch(
                input_ids=ret[0],
                labels=ret[1],
                extra=None,  # extra_processor is never invoked on packing
            ), super().__iter__())
        else:
            while True:
                for sample in self._get_data_iter():
                    sample_text = self._text_processor(sample)
                    sample_tokens = self._tokenizer.encode(sample_text, bos=self.bos, eos=self.eos)
                    # always filter for longer sequences
                    if len(sample_tokens) > self.seq_len + 1:
                        continue

                    extra = self._extra_processor(sample)

                    with torch.no_grad():
                        input = torch.tensor(sample_tokens, dtype=torch.int64)
                        # Allocate+copy is much faster than torch.concat ;)
                        input_buffer = torch.zeros((self.seq_len,), dtype=torch.int64)
                        input_buffer[:input.shape[0] - 1].copy_(input[:-1])
                        label_buffer = torch.full_like(input_buffer, -100)
                        label_buffer[:input.shape[0] - 1].copy_(input[1:])
                    yield Batch(
                        input_ids=input_buffer,
                        labels=label_buffer,
                        extra=extra,
                    )

                if not self.infinite:
                    logger.warning(f"Dataset {self.dataset_name} has run out of data")
                    break
                else:
                    # Reset offset for the next iteration
                    self._sample_idx = 0
                    logger.warning(f"Dataset {self.dataset_name} is being re-looped")


class MixDataset(HuggingFaceDataset):
    def __init__(
        self,
        dataset_config: DatasetMixConfig,
        tokenizer: Any,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        shuffle: bool = False,
        seed: int = 42,
        padding: bool = False,
        bos: bool = True,
        eos: bool = True,
        weights: list | None = None,
    ) -> None:
        assert isinstance(dataset_config, DatasetMixConfig)
        self._datasets = [
            Dataset(
                cfg,
                tokenizer=tokenizer,
                seq_len=seq_len,
                world_size=world_size,
                rank=rank,
                infinite=infinite,
                shuffle=shuffle,
                seed=seed,
                padding=padding,
                bos=bos,
                eos=eos,
            ) for cfg in dataset_config.datasets
        ]
        self._weights = weights
        if self._weights is not None and any(x < 0 for x in self._weights):
            raise ValueError(f"Weights must all be non-negative. Got {self._weights}.")
        self._counter = 0
        self._rng = np.random.default_rng(seed=seed)
        # In case version problem changes default random generator
        assert isinstance(self._rng.bit_generator, np.random.PCG64)
        self._seed = seed

    def __iter__(self):
        if self._weights is not None:
            p = np.array(self._weights, dtype=np.float32)
            p = p / p.sum()
        else:
            p = None

        iters = [iter(d) for d in self._datasets]
        while True:
            idx = self._rng.choice(range(len(self._datasets)), 1, p=p).item()
            yield next(iters[idx])
            self._counter += 1


    def state_dict(self) -> dict:
        return {
            "weights": self._weights,
            "counter": self._counter,
            "seed": self._seed,
        } | {
            i: self._datasets[i].state_dict()
            for i in range(len(self._datasets))
        }

    def load_state_dict(self, state_dict):
        for i, ds in enumerate(self._datasets):
            ds.load_state_dict(state_dict[i])
        self._weights = state_dict["weights"]
        self._counter = state_dict["counter"]
        self._seed = state_dict["seed"]
        self._rng = np.random.default_rng(seed=self._seed)
        assert isinstance(self._rng.bit_generator, np.random.PCG64)
        self._rng.bit_generator.advance(self._counter)


class DataLoader(DPAwareDataLoader):
    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        StatefulDataLoader.__init__(self, hf_ds, batch_size, collate_fn=self._collate_fn)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def _collate_fn(self, batches: list[Batch]) -> Batch:
        return Batch(
            input_ids=torch.stack([b.input_ids for b in batches], dim=0),
            labels=torch.stack([b.labels for b in batches], dim=0),
            extra=[b.extra for b in batches],  # extra can be anything
        )



      
def build_hf_data_loader(
    dataset_config: DatasetConfig | DatasetMixConfig,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
    padding: bool = True,
):
    """Build a data loader for HuggingFace datasets."""
    cls = Dataset if isinstance(dataset_config, DatasetConfig) else MixDataset
    hf_ds = cls(dataset_config, tokenizer, seq_len, world_size, rank, infinite, padding=padding)
    return DataLoader(rank, hf_ds, batch_size=batch_size)
