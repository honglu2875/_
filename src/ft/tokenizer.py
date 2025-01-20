from torchtitan.datasets.tokenizer import Tokenizer
from transformers import AutoTokenizer


class WrappedHFTokenizer(Tokenizer):
    def __init__(self, tokenizer_path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._n_words = 8

    def encode(self, *args, **kwargs) -> list[int]:
        bos = kwargs.pop("bos")
        eos = kwargs.pop("eos")

        tokens = self._tokenizer.encode(*args, **kwargs)
        if bos and (bos_id := self._tokenizer.bos_token_id) is not None and tokens[0] != bos_id:
            tokens = [bos_id] + tokens

        if eos and (eos_id := self._tokenizer.eos_token_id) is not None and tokens[-1] != eos_id:
            tokens = tokens + [eos_id]

        return tokens

    def decode(self, *args, **kwargs) -> str:
        return self._tokenizer.decode(*args, **kwargs)

    @property
    def n_words(self) -> int:
        return self._n_words
