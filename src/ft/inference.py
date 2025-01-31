from contextlib import contextmanager
import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import StaticCache

@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    do_sample: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2

@dataclass 
class TimingStats:
    prefill_time: float = 0.0
    total_decode_time: float = 0.0
    decode_times: list = None
    tokens_generated: int = 0
    
    def __post_init__(self):
        self.decode_times = []
    
    def get_summary(self) -> Dict[str, float]:
        avg_decode_time = sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0
        return {
            "prefill_time": self.prefill_time,
            "total_decode_time": self.total_decode_time,
            "tokens_per_sec": self.tokens_generated / self.total_decode_time if self.total_decode_time else 0,
            "avg_decode_time": avg_decode_time,
            "total_time": self.compile_prefill_time + self.compile_decode_time + self.prefill_time + self.total_decode_time
        }

@contextmanager
def timeit(stat: TimingStats, name: str, append=False):
    start = time.perf_counter()
    assert hasattr(stat, name)
    yield
    if append:
        getattr(stat, name).append(time.perf_counter() - start)
    else:
        setattr(stat, name, time.perf_counter() - start)
    

@torch.jit.script
def top_k_top_p_min_p_filtering(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    min_p: float,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Apply min_p: zero out tokens with probability < min_p * max_prob
    if min_p > 0.0:
        max_probs, _ = probs.max(dim=-1, keepdim=True)
        min_p_threshold = max_probs * min_p
        logits[probs < min_p_threshold] = float('-inf')
    
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    return logits

class PrefillModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        past_key_values: Optional[StaticCache] = None,
    ) -> Tuple[torch.Tensor, StaticCache]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        return outputs.logits, outputs.past_key_values

def next_buffer(buffer: torch.Tensor, block_size: int, fill=None):
    if fill:
        new_buffer = torch.full((buffer.shape[0], buffer.shape[1] + block_size), fill, dtype=buffer.dtype, device=buffer.device)
    else:
        new_buffer = torch.zeros((buffer.shape[0], buffer.shape[1] + block_size), dtype=buffer.dtype, device=buffer.device)
    new_buffer[:, :buffer.shape[1]].copy_(buffer)
    return new_buffer

def scrub_cache(static_cache: StaticCache, fr: int):
    for t in static_cache.key_cache + static_cache.value_cache:
        t[:, :, fr:].zero_()

class DecodeOneTokenModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(
        self,
        input_ids: torch.Tensor,  # shape: (batch_size, 1)
        attention_mask: torch.Tensor | None = None,  # shape: (batch_size, block_size)
        cache_position: torch.Tensor | None = None,  # shape: (1,)
        past_key_values: Optional[StaticCache] = None,
    ) -> Tuple[torch.Tensor, StaticCache]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        return outputs.logits, outputs.past_key_values

def generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    generation_config: Optional[GenerationConfig] = None,
    static_cache: Optional[StaticCache] = None,
    block_size: int = 1024,
) -> Tuple[torch.Tensor, TimingStats]:
    """
    Performs autoregressive generation with static cache and block-wise processing.
    
    Args:
        model: HuggingFace model
        input_ids: Tensor of shape (batch_size, seq_len) containing input token ids 
        generation_config: Configuration for generation
        static_cache: Optional pre-initialized static cache
        block_size: Size of fixed blocks for processing
    """
    torch.set_float32_matmul_precision('high')
    timing_stats = TimingStats()
    
    if generation_config is None:
        generation_config = GenerationConfig()

    device = input_ids.device
    batch_size = input_ids.shape[0]
    input_length = input_ids.shape[1]
    max_new_tokens = generation_config.max_new_tokens
    initial_buffer_len = ((input_length + block_size - 1) // block_size) * block_size

    if static_cache is None:
        static_cache = StaticCache(
            model.config,
            max_batch_size=batch_size,
            max_cache_len=max(max_new_tokens + input_length, initial_buffer_len),
            device=device,
            dtype=model.dtype
        )

    # Compile models
    prefill_model = torch.compile(
        PrefillModel(model),
        mode="reduce-overhead",
        fullgraph=True
    )

    decode_model = torch.compile(
        DecodeOneTokenModel(model),
        mode="reduce-overhead",
        fullgraph=True
    )
    
    # Initialize outputs
    token_buffer = torch.zeros((batch_size, initial_buffer_len), dtype=input_ids.dtype, device=device)
    token_buffer[:, :input_length].copy_(input_ids)
    #buffer_mask = torch.zeros((batch_size, initial_buffer_len), dtype=torch.bool, device=device)
    #buffer_mask[:, :input_length].copy_(attention_mask)
    cache_position = torch.arange(initial_buffer_len, dtype=torch.long, device=device)
    
    with torch.inference_mode():
        # Prefill phase
        #scrub_cache(static_cache, fr=input_length)
        with timeit(timing_stats, "prefill_time"):
            next_token_logits, static_cache = prefill_model(
                input_ids=token_buffer,
                past_key_values=static_cache,
                cache_position=cache_position,
            )
        next_token_logits = next_token_logits[:, input_length - 1, :]
        
        # Decode phase - process in fixed-size blocks
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        pos = input_length
        
        with timeit(timing_stats, "total_decode_time"):
            for block_idx in range(num_blocks):
                for pos_in_block in range(pos % block_size, min(block_size, max_new_tokens - block_idx * block_size)):
                    with timeit(timing_stats, "decode_times", append=True):
                        # Sample next token
                        if generation_config.temperature != 1.0:
                            next_token_logits = next_token_logits / generation_config.temperature
                        
                        if generation_config.do_sample:
                            filtered_logits = top_k_top_p_min_p_filtering(
                                next_token_logits,
                                generation_config.top_k,
                                generation_config.top_p,
                                generation_config.min_p
                            )
                            probs = F.softmax(filtered_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        
                        # FIXME: Early stopping
                        if (next_token == generation_config.eos_token_id).any():
                            return token_buffer[:, :pos], timing_stats
                        
                        # Update sequences
                        token_buffer[:, pos: pos + 1].copy_(next_token)
                        #buffer_mask[:, pos] = True
                        
                        # Get next token logits
                        cache_position = torch.tensor([pos], dtype=torch.long, device=device)
                        scrub_cache(static_cache, fr=pos)
                        next_token_logits, static_cache = decode_model(
                            input_ids=next_token,
                            past_key_values=static_cache,
                            cache_position=cache_position,
                        )
                        next_token_logits = next_token_logits.squeeze(1)
                        
                        timing_stats.tokens_generated += 1
                        pos += 1

                token_buffer = next_buffer(token_buffer, block_size)
                #buffer_mask = next_buffer(buffer_mask, block_size, fill=0)

        
    
    return token_buffer[:, :pos], timing_stats


if __name__ == "__main__":
    # test generation
    name = "NousResearch/Meta-Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(name)
    model = model.cuda()
    tok = AutoTokenizer.from_pretrained(name)
    p = tok(["Hi, how are you?"] * 20, return_tensors="pt")
    output, timing_stats = generate(model, p.input_ids.cuda())
    #print(tok.batch_decode(output))
    print(timing_stats)
