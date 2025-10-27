# Copyright (c) 2023, Albert Gu, Tri Dao.
import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Optional
import tqdm
import torch
from transformers.generation import TextStreamer
from mamba_ssm.utils.generation import (
    InferenceParams,
)
from jsm.data.utils import DecodingController
from jsm.generation.utils import GenerationMetrics, sample, GenerationOutput, DecodingCGCache


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    eos_token_id,
    pad_token_id,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    output_scores=False,
    streamer: Optional[TextStreamer] = None,
    protein_embeddings=None,
    protein_sequences=None,
    progress_bar=True,
    cuda_monitor=False,
    expected_utr_5_length=None,
    expected_utr_3_length=None
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape
    seqlen_og = seqlen_og + protein_embeddings.shape[1] if protein_embeddings is not None else seqlen_og
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if not decoding:
            assert protein_embeddings is not None, "protein_embeddings must be provided for prompting"
            logits = model.generating_forward(
                input_ids,
                protein_embeddings=protein_embeddings,
                return_hidden_states=False,
                inference_params=inference_params,
                num_last_tokens=1,
            ).squeeze(dim=1)
        else:
            if cg:
                logits = model._decoding_cache.run(
                    input_ids, 
                    inference_params.seqlen_offset
                ).squeeze(dim=1)
            else:
                logits = model.generating_forward(
                    input_ids,
                    return_hidden_states=False,
                    inference_params=inference_params,
                    num_last_tokens=1,
                ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)
    
    if cuda_monitor:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    if enable_timing:
        start.record()
    
    if progress_bar:
        pbar = tqdm.tqdm(desc="Generating RNA sequence")
    else:
        pbar = None
    
    metrics = GenerationMetrics(
        pad_token_id=pad_token_id, 
        device=input_ids.device,
        number_sequences=batch_size,
    )
    controller = DecodingController(
            batch_size, 
            max_length, 
            device=input_ids.device, 
            protein_sequences=protein_sequences,
            expected_utr_5_length=expected_utr_5_length,
            expected_utr_3_length=expected_utr_3_length
    )
    controller.update(input_ids)

    while not controller.should_stop():
        logits = get_logits(
            controller.sequences[-1], 
            inference_params
        )
        inference_params.seqlen_offset += controller.sequences[-1].shape[1]
        logits = controller.modify_logits(logits)
        sampled_tokens = sample_tokens(logits, inference_params)
        # sampled_tokens = controller.pad_tokens(sampled_tokens) 
        controller.update(sampled_tokens)
        # print(logits[0])
        metrics.update(logits, sampled_tokens.squeeze(1))
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
        pbar.update(controller.sequences[-1].shape[1]) if pbar is not None else None
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)/1000):.0f} s")
    if cuda_monitor:
        torch.cuda.synchronize()
        print("Peak allocated (MB):", torch.cuda.max_memory_allocated()/1024**2)
        print("Peak reserved  (MB):", torch.cuda.max_memory_reserved()/1024**2)

    return GenerationOutput(
            sequences=torch.cat(controller.sequences, dim=1), 
            metrics=metrics.summarize()
        )


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids, self, max_length, top_k=top_k, top_p=top_p, min_p = min_p, temperature=temperature, output_scores=output_scores, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    dtype=None,
    n_warmups=2,
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(model, "allocate_inference_cache"), "CUDA graph decoding requires that the model has a method allocate_inference_cache"
        inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,  # original sequence length
            key_value_memory_dict=inf_cache,  # pre-allocated cache
            lengths_per_sample=lengths_per_sample,
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(input_ids, seqlen):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
    model, 
    inference_params, 
    batch_size, 
    max_seqlen, 
    decoding_seqlen=1, 
    mempool=None, 
    n_warmups=2,
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model.generating_forward(
                input_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
                return_hidden_states=False,
            )
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model.generating_forward(
            input_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
            return_hidden_states=False
        )

    def run(new_input_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og
    return run