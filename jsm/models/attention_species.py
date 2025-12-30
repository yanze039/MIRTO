# Copyright (c) 2023, Albert Gu, Tri Dao.
import math
from functools import partial
import copy
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from einops import rearrange
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/orcd/home/002/yanze039/orcd/pool/RNA_design/test/software/mamba")
from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.modules.mha import MHA
from flash_attn.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
# from mamba_ssm.modules.block import Block


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.is_attention = True if isinstance(self.mixer, MHA) else False
        
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
    
    def set_precision_for_alibi_slopes(self, dtype):
        if self.is_attention and hasattr(self.mixer.inner_attn, "alibi_slopes"):
            self.mixer.inner_attn.alibi_slopes = self.mixer.inner_attn.alibi_slopes.to(dtype)
        if self.is_attention and hasattr(self.mixer.inner_cross_attn, "alibi_slopes"):
            self.mixer.inner_cross_attn.alibi_slopes = self.mixer.inner_cross_attn.alibi_slopes.to(dtype)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        # import pdb; pdb.set_trace()
        if self.is_attention:
            mixer_kwargs.pop('seq_idx', None)
            if mixer_kwargs.get('cu_seqlens', None) is not None:
                hidden_states = hidden_states.squeeze(0)
        else:
            mixer_kwargs.pop('max_seqlen', None)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        if self.is_attention:
            if mixer_kwargs.get('cu_seqlens', None) is not None:
                hidden_states = hidden_states.unsqueeze(0)
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
    
    @torch.no_grad()
    def calculate_attention_weights(self, 
                                    hidden_states: Tensor, 
                                    residual: Optional[Tensor] = None, 
                                    inference_params=None, 
                                    **mixer_kwargs):
        assert self.is_attention, "calculate_attention_weights is only available for attention layers"
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        if self.is_attention:
            mixer_kwargs.pop('seq_idx', None)
            if mixer_kwargs.get('cu_seqlens', None) is not None:
                hidden_states = hidden_states.squeeze(0)
        else:
            mixer_kwargs.pop('max_seqlen', None)
        # hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        qkv = self.mixer.Wqkv(hidden_states)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", 
                       three=3, h=self.mixer.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scale = 1 / math.sqrt(self.mixer.head_dim)
        scores = torch.einsum("b i h d, b j h d -> b h i j", q * scale, k * scale)
        
        # if self.mixer.inner_attn.alibi_slopes is not None:
        #     # print(self.mixer.inner_attn.alibi_slopes)
        #     alibi_bias = self._get_alibi_bias(hidden_states.shape[1], self.mixer.inner_attn.alibi_slopes, device=hidden_states.device)
        #     scores = scores + alibi_bias
        
        # if self.mixer.causal:
        #     causal_mask = torch.tril(torch.ones(scores.shape[-2:], device=scores.device)).unsqueeze(0).unsqueeze(0)
        #     scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        
        # attn_weights = F.softmax(scores, dim=-1)
        return scores

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def _get_alibi_bias(self, seqlen, alibi_slopes, device):
        """
        Compute ALiBi bias matrix
        alibi_slopes: (num_heads,) tensor with slopes for each head
        """
        # Create relative position matrix: [i - j] for all i, j
        # Shape: (seqlen, seqlen)
        position = torch.arange(seqlen, device=device).unsqueeze(0)
        relative_position = position - position.transpose(0, 1)
        
        # ALiBi adds a bias of: slope * (i - j) for query position i and key position j
        # For causal attention, we typically only care about i >= j
        # Shape after broadcasting: (num_heads, seqlen, seqlen)
        alibi_bias = alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_position.unsqueeze(0)
        
        # Add batch dimension: (1, num_heads, seqlen, seqlen)
        alibi_bias = alibi_bias.unsqueeze(0)
        
        return alibi_bias


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        mixer_cls = partial(
            Mamba2,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
        if module.weight is not None:
            # for transformer
            nn.init.normal_(module.weight, std=initializer_range)
            # for SSM, we use the default initialization
    elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    
    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )
    
    def set_precision_for_alibi_slopes(self, dtype):
        for layer in self.layers:
            layer.set_precision_for_alibi_slopes(dtype)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, hidden_layer_idx=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            if hidden_layer_idx is not None and layer.layer_idx == hidden_layer_idx:
                return hidden_states + residual if residual is not None else hidden_states
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
    
    def calculate_attention_weights(self, 
                                    hidden_states, 
                                    inference_params=None, 
                                    hidden_layer_idx=None, 
                                    **mixer_kwargs):
        residual = None
        output_attention_weights = {}
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            attention_weights = layer.calculate_attention_weights(hidden_states, residual)
            # if hidden_layer_idx is not None and layer.layer_idx == hidden_layer_idx:
            #     return hidden_states + residual if residual is not None else hidden_states
            output_attention_weights[layer.layer_idx] = attention_weights
        return output_attention_weights


class SpeciesSpecificJointSequenceAttentionModel(nn.Module):

    def __init__(
        self,
        config,
        rna_vocab_size,
        protein_vocab_size,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}
        
        vocab_size = rna_vocab_size
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
            
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.rna_embeddings = nn.Embedding(rna_vocab_size, d_model)
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.criterion = None
        
        self.rna_lm_head = nn.Linear(d_model, self.rna_vocab_size, bias=True)
        self.protein_align_head = nn.Linear(config.protein_hidden_size, config.d_model, bias=True)
        self.protein_embedding_norm = RMSNorm(config.protein_hidden_size, eps=1e-6)
        
        self.species_embedding = nn.Embedding(config.num_species, d_model)
        self.modality_embedding = nn.Embedding(8, d_model)  # 3 modalities: 5'UTR, CDS, 3'UTR

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        nn.init.zeros_(self.protein_align_head.bias)
        nn.init.normal_(self.protein_align_head.weight, std=0.01)
        nn.init.zeros_(self.rna_lm_head.bias)
        nn.init.normal_(self.rna_lm_head.weight, std=0.01)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def set_precision_for_alibi_slopes(self, dtype):
        self.backbone.set_precision_for_alibi_slopes(dtype)

    def forward(
            self, 
            input_ids,
            species_ids,                 
            protein_embeddings,
            row_wise_col_perms,
            inverse_row_wise_col_perms,
            attention_mask,
            modality_type_ids=None,
            modality_mask=None,
            seq_idx=None,
            inference_params=None, 
            **mixer_kwargs,
        ):
        B, Lr = input_ids.shape
        Lp = protein_embeddings.shape[1]
        Lct = species_ids.shape[1]
        L = Lr + Lp + Lct
        inputs_embeds = self.rna_embeddings(input_ids)
        
        # if modality_type_ids is not None:
        modality_embeds = self.modality_embedding(modality_type_ids) * modality_mask.unsqueeze(-1)
        inputs_embeds = inputs_embeds + modality_embeds
        
        
        species_embeds = self.species_embedding(species_ids)
        protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))
        inputs_embeds = torch.cat([species_embeds, protein_embeddings, inputs_embeds], dim=1)
        inputs_embeds = torch.gather(inputs_embeds, dim=1, index=row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))  
        inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)
        
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        inputs_embeds = index_first_axis(rearrange(inputs_embeds, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        outputs = self.backbone(
            hidden_states=inputs_embeds,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            seq_idx=seq_idx,
        )
        hidden_states = pad_input(outputs.squeeze(0), indices, B, L)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lct+Lp:,:]
        rna_logits = self.rna_lm_head(hidden_states)
        
        return rna_logits


    def _forward(self, 
                input_ids,                 
                protein_embeddings,
                row_wise_col_perms,
                inverse_row_wise_col_perms,
                attention_mask,
                seq_idx=None,
                inference_params=None, 
                **mixer_kwargs,
        ):
        B, Lr = input_ids.shape
        Lp = protein_embeddings.shape[1]
        L = Lr + Lp
        inputs_embeds = self.rna_embeddings(input_ids)
        codon_protein_translation_logits = self.translation_lm_head(inputs_embeds)  # 60, 1180, 33
        
        protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))
        inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
        inputs_embeds = torch.gather(inputs_embeds, dim=1, index=row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))  
        indices, cu_seqlens, max_sequence_len = get_unpad_data(attention_mask)
        inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)
        outputs = self.backbone(
            hidden_states=inputs_embeds[:, :max_sequence_len, :],
        )
        outputs = F.pad(outputs, (0, 0, 0, L - max_sequence_len))  # Pad to original length
        hidden_states = outputs * attention_mask.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lp:,:]
        
        rna_logits = self.rna_lm_head(hidden_states)
        modality_logits = self.modality_prediction_head(hidden_states)

        return rna_logits, codon_protein_translation_logits, modality_logits
    
    @torch.no_grad()
    def generating_forward(
            self,
            input_ids,                 
            protein_embeddings=None,
            species_ids=None,
            return_hidden_states=True,
            inference_params=None, 
            num_last_tokens=0,
            hidden_layer_idx=None
        ):
        B, Lr = input_ids.shape
        inputs_embeds = self.rna_embeddings(input_ids)  
        if protein_embeddings is not None:
            protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))   
            inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
        if species_ids is not None:
            species_embeds = self.species_embedding(species_ids)
            inputs_embeds = torch.cat([species_embeds, inputs_embeds], dim=1)
        
        outputs = self.backbone(
            hidden_states=inputs_embeds,
            inference_params=inference_params,
            hidden_layer_idx=hidden_layer_idx
        )
        if return_hidden_states:
            return outputs
        
        if num_last_tokens > 0:
            outputs = outputs[:, -num_last_tokens:, :]
        rna_logits = self.rna_lm_head(outputs)
        return rna_logits
    
    def calculate_attention_weights(
            self, 
            input_ids,                 
            protein_embeddings=None,
            inference_params=None, 
            hidden_layer_idx=None
        ):
        inputs_embeds = self.rna_embeddings(input_ids)  
        if protein_embeddings is not None:
            protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))   
            inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
        attention_weights = self.backbone.calculate_attention_weights(
            hidden_states=inputs_embeds,
            inference_params=inference_params,
            hidden_layer_idx=hidden_layer_idx
        )
        return attention_weights
        


    