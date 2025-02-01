from typing import Any

import torch
from datasets.distributed import split_dataset_by_node
from torchtitan.datasets.hf_datasets import DatasetConfig, DPAwareDataLoader, HuggingFaceDataset
from torchtitan.utils import logger


class Dataset(HuggingFaceDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: Any,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        path = dataset_config.path
        dataset_loader = dataset_config.loader
        text_processor = dataset_config.text_processor
        ds = dataset_loader(path)

        self.dataset_name = path.split("/")[-1]
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: list[int] = []


class PaddingDataset(Dataset):
    def __init__(
        self,
        *args,
        bos=True,
        eos=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bos = bos
        self.eos = eos

    def __iter__(self):
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


def build_hf_data_loader(
    dataset_config: DatasetConfig,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
    padding: bool = False,
):
    """Build a data loader for HuggingFace datasets."""
    cls = PaddingDataset if padding else Dataset
    hf_ds = cls(dataset_config, tokenizer, seq_len, world_size, rank, infinite)
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
