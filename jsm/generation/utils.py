import torch
from mamba_ssm.utils.generation import (
    InferenceParams,
    modify_logits_for_min_p_filtering,
    modify_logits_for_top_p_filtering,
)
from dataclasses import dataclass, field
from transformers.utils import ModelOutput
from typing import Optional, Callable
from torch import Tensor


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None
    steering: Optional[Tensor] = None
    steering_vector: Optional[Tensor] = None
    steering_layer_index: Optional[int] = None
    steering_weight: float = 1.0
    debug_counter: torch.Tensor = field(default_factory=lambda: torch.zeros((), dtype=torch.int32))

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@dataclass
class GenerationOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None
    metrics: Optional[dict] = None

class GenerationMetrics:
    """Track confidence and quality metrics during autoregressive generation."""
    def __init__(self, pad_token_id=None, device=None, number_sequences=1):
        self.pad_token_id = pad_token_id
        self.device = device if device is not None else "cpu"
        self.number_sequences = number_sequences
        self.reset()

    def reset(self):
        self.sum_logp = torch.zeros((self.number_sequences,), device=self.device)
        self.sum_entropy = torch.zeros((self.number_sequences,), device=self.device)
        self.count_tokens = torch.zeros((self.number_sequences,), device=self.device, dtype=torch.long)
    
    def to_dict(self):
        """Convert metrics to a dictionary."""
        return {
            "sum_logp": self.sum_logp.cpu().numpy().tolist(),
            "sum_entropy": self.sum_entropy.cpu().numpy().tolist(),
            "count_tokens": self.count_tokens.cpu().numpy().tolist(),
        }

    @torch.no_grad()
    def update(self, logits, tokens):
        """
        Update metrics from one generation step.

        Args:
            logits: (B, V) raw model logits BEFORE any sampling tricks (temperature, top-k/p, etc.)
            tokens: (B,) chosen tokens for this step (sampled or teacher-forced)
        """
        logp = torch.log_softmax(logits, dim=-1)  # (B, V)
        step_logp = logp.gather(1, tokens.unsqueeze(1)).squeeze(1)  # (B,)
        
        logp = logp[torch.isfinite(logp)]
        p = torch.exp(logp)                       # (B, V)
        # entropy
        step_entropy = -(p * logp).sum(dim=-1)  # (B,)
        # valid mask (ignore pads)
        if self.pad_token_id is not None:
            valid = (tokens != self.pad_token_id)
        else:
            valid = torch.ones_like(tokens, dtype=torch.bool)
        self.sum_logp += step_logp * valid
        self.sum_entropy += step_entropy * valid
        self.count_tokens += valid.long()
    
    @torch.no_grad()
    def summarize(self):
        """Summarize metrics after generation is done."""
        avg_logp = self.sum_logp / self.count_tokens.clamp(min=1)
        avg_entropy = self.sum_entropy / self.count_tokens.clamp(min=1)
        return {
            "avg_logp": avg_logp.cpu().numpy().tolist(),
            "avg_entropy": avg_entropy.cpu().numpy().tolist(),
            "count_tokens": self.count_tokens.cpu().numpy().tolist(),
        }



def sample(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            softmaxed = torch.softmax(logits_top, dim=-1)
            if softmaxed.isnan().any().item():
                print("got nan in softmaxed!")
                return None
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(softmaxed, num_samples=1).squeeze(dim=-1),
            ]
        else:
            if min_p > 0.0:
                logits_top = logits.clone()
                max_prob = logits_top[..., 0].item()
                min_prob = max_prob * min_p
                modify_logits_for_min_p_filtering(logits_top, min_prob)
                if temperature != 1.0:
                    logits_top /= temperature
                return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            softmaxed = torch.softmax(logits_top, dim=-1)
            return torch.multinomial(softmaxed, num_samples=1).squeeze(
                dim=-1
            )