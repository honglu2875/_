from typing import Any

import datasets
import torch
from datasets.distributed import split_dataset_by_node

from ft.dataset_config import DatasetConfig, DatasetMixConfig, DatasetType
from ft.logging import init_logger
from torchtitan.datasets.hf_datasets import DPAwareDataLoader, HuggingFaceDataset

logger = init_logger(__name__)


DConfigs = DatasetConfig | DatasetMixConfig


class Dataset(HuggingFaceDataset):
    def __init__(
        self,
        dataset_config: DConfigs,
        tokenizer: Any,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        shuffle: bool = False,
        seed: int = 42,
        padding: bool = False,
        bos=True,
        eos=True,
    ) -> None:
        self.config = dataset_config
        ds = self._get_ds(self.config, shuffle=shuffle, seed=seed)
        text_processor = self.config.text_processor
        self.dataset_name = self.name.split("/")[-1]
        self.seed = seed
        self.seq_len = seq_len
        self.infinite = infinite
        self.padding = padding
        self.bos = bos
        self.eos = eos
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: list[int] = []

    def _get_ds(self, config: DConfigs, shuffle: bool, seed: int = 42) -> datasets.Dataset:
        assert isinstance(config, DatasetConfig)
        path = config.path
        dataset_loader = config.loader
        ds = dataset_loader(path)
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
            yield from super().__iter__()
        else:
            while True:
                for sample in self._get_data_iter():
                    sample_text = self._text_processor(sample)
                    sample_tokens = self._tokenizer.encode(sample_text, bos=self.bos, eos=self.eos)
                    # always filter for longer sequences
                    if len(sample_tokens > self.seq_len + 1):
                        continue
                    input = torch.LongTensor(sample_tokens)
                    label = input[1:]
                    yield input, label

                if not self.infinite:
                    logger.warning(f"Dataset {self.dataset_name} has run out of data")
                    break
                else:
                    # Reset offset for the next iteration
                    self._sample_idx = 0
                    logger.warning(f"Dataset {self.dataset_name} is being re-looped")


class MixDataset(Dataset):
    def _get_ds(self, config: DConfigs, shuffle: bool, seed: int = 42) -> datasets.Dataset:
        assert isinstance(config, DatasetMixConfig)
        ds = []
        for cfg in config.datasets:
            ds.append(super()._get_ds(cfg, shuffle=shuffle, seed=seed))
        return datasets.concatenate_datasets(ds).shuffle(seed=seed)

    @property
    def name(self):
        assert isinstance(self.config, DatasetMixConfig)
        return "_".join(c.path.split("/")[-1] for c in self.config.datasets)


def build_hf_data_loader(
    dataset_config: DatasetConfig | DatasetMixConfig,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
    padding: bool = False,
):
    """Build a data loader for HuggingFace datasets."""
    cls = Dataset if isinstance(dataset_config, DatasetConfig) else MixDataset
    hf_ds = cls(dataset_config, tokenizer, seq_len, world_size, rank, infinite, padding=padding)
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
