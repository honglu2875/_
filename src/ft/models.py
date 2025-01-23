import dataclasses
from typing import Any, Union
from transformers import Cache, LlamaForCausalLM
import torch
from transformers.generation.utils import CausalLMOutputWithPast


AnyCache = Union[Cache, list[torch.Tensor]]


@dataclasses.dataclass
class CausalLMOutputWithPastAndFuture(CausalLMOutputWithPast):
    future_logits: torch.Tensor | None = None


class LlamaMTPred(torch.nn.Module):
    def __init__(self, llama: LlamaForCausalLM, stride: int = 100, num_tokens: int = 5):
        super().__init__()
        self.llama = llama
        self.config = llama.config
        self.stride = stride
        self.num_tokens = num_tokens
        self.model = llama.model

        hd = self.config.hidden_size
        padded_hd = self.config.num_attention_heads * self.config.hidden_size // self.config.num_attention_heads
        self.side_dense = torch.nn.Linear(
            hd,
            padded_hd * num_tokens,
            bias=self.config.attention_bias
        ).to(llama.dtype)


    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: AnyCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        num_logits_to_keep: int = 0,
        **kwargs: Any,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.llama.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        bs, seq_len, _ = hidden_states.shape
        future_hidden = self.side_dense(hidden_states).reshape(bs, seq_len, self.num_tokens, -1)
        future_logits = self.llama.lm_head(future_hidden)

        loss = None
        if labels is not None:
            loss = self.llama.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndFuture(
            loss=loss,
            logits=logits,
            future_logits=future_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
