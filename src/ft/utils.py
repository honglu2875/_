import contextlib
from torchtitan.models.llama.model import ModelArgs
from torchtitan.utils import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from ft.tokenizer import WrappedHFTokenizer
import torch


@contextlib.contextmanager
def maybe_wait(wait: bool):
    rank = torch.distributed.get_rank()
    if wait:
        logger.info("Rank %d is waiting.", rank)
        torch.distributed.barrier()
        logger.info("Rank %d finished waiting.", rank)
    yield
    if not wait:
        torch.distributed.barrier()


def get_model_and_tokenizer(model_name):
    with maybe_wait(wait=torch.distributed.get_node_local_rank() != 0):
        tokenizer = WrappedHFTokenizer(model_name) 
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True)
        model.config.use_cache = False
        model.eval()

    return model, tokenizer

def hf_to_titan_config(hf_config: PretrainedConfig) -> ModelArgs:
    """from HF config to torchtitan config dict.
    Note: This is entirely ad-hoc because Huggingface configs are different for different models.
        Will do the best-effort here but please update when expanding the use of open-source models.
    Note: This conversion will ignore fields in HF but not present in Titan config.
        The converted config is for information only. Please do not use
        it to initialize a titan model (unless it is Llama).
    """

    def _find_key(config, keys):
        for k in keys:
            if hasattr(config, k):
                return getattr(config, k)
        raise ValueError(f"Cannot find the keys {keys} in the config {config}.")

    n_kv_heads = _find_key(hf_config, ("num_key_value_heads", "num_attention_heads"))
    norm_eps = _find_key(hf_config, ("rms_norm_eps", "layer_norm_eps"))

    titan_config = ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=hf_config.vocab_size,
        multiple_of=256,
        ffn_dim_multiplier=hf_config.intermediate_size / hf_config.hidden_size,
        norm_eps=norm_eps,  # need to adapt for other models using layernorm
        norm_type="rmsnorm",
        rope_theta=hf_config.rope_theta,
        max_seq_len=hf_config.max_position_embeddings,
    )
    return titan_config

