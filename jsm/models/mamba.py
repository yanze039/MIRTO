# Copyright (c) 2023, Albert Gu, Tri Dao.
import math
from functools import partial
import copy
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from einops import rearrange
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/orcd/home/002/yanze039/orcd/pool/RNA_design/test/software/mamba")
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
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

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
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


class JointSequenceMambaModel(nn.Module):

    def __init__(
        self,
        config,
        rna_vocab_size,
        protein_vocab_size,
        num_modalities=5,
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
            
        # we predict modality, so we don't use any 
            
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
        # self.modality_embeddings = nn.Embedding(, config.d_model)
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.criterion = None
        
        self.rna_lm_head = nn.Linear(d_model, self.rna_vocab_size, bias=True)
        self.translation_lm_head = nn.Linear(d_model, self.protein_vocab_size, bias=True)
        self.protein_align_head = nn.Linear(config.protein_hidden_size, config.d_model, bias=True)
        self.modality_prediction_head = nn.Linear(d_model, num_modalities, bias=True)  # Assuming 2 modalities: RNA and Protein
        self.protein_embedding_norm = RMSNorm(config.protein_hidden_size, eps=1e-6)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, 
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
        # codon_protein_translation_logits = self.translation_lm_head(inputs_embeds)  # 60, 1180, 33
        # import pdb; pdb.set_trace(header="Translation inputs_embeds")
        protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))
        inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
        inputs_embeds = torch.gather(inputs_embeds, dim=1, index=row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))  
        inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)
        
        indices, cu_seqlens, _ = get_unpad_data(attention_mask)
        inputs_embeds = index_first_axis(rearrange(inputs_embeds, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        outputs = self.backbone(
            hidden_states=inputs_embeds,
            seqlen=None,
            cu_seqlens=cu_seqlens,
            seq_idx=seq_idx
        )
        hidden_states = pad_input(outputs.squeeze(0), indices, B, L)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lp:,:]
        rna_logits = self.rna_lm_head(hidden_states)
        # modality_logits = self.modality_prediction_head(hidden_states)
        return rna_logits, None, None
        # return rna_logits, codon_protein_translation_logits, modality_logits


    def forward_padded(self, 
                input_ids,                 
                protein_embeddings,
                row_wise_col_perms,
                inverse_row_wise_col_perms,
                attention_mask,
                joint_mask,
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
        inputs_embeds = inputs_embeds * joint_mask.unsqueeze(-1)  # Apply joint masking
        inputs_embeds = torch.gather(inputs_embeds, dim=1, index=row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))  
        
        outputs = self.backbone(
            hidden_states=inputs_embeds,
        )
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
            return_hidden_states=True,
            inference_params=None, 
            num_last_tokens=0,
        ):
        B, Lr = input_ids.shape
        inputs_embeds = self.rna_embeddings(input_ids)  
        if protein_embeddings is not None:
            protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))   
            # print(protein_embeddings.shape)
            # print(inputs_embeds.shape)
            inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
        
        outputs = self.backbone(
            hidden_states=inputs_embeds,
            inference_params=inference_params,
        )
        if num_last_tokens > 0:
            outputs = outputs[:, -num_last_tokens:, :]
        rna_logits = self.rna_lm_head(outputs)
        # modality_logits = self.modality_prediction_head(outputs)
        if return_hidden_states:
            return rna_logits, outputs
        else:
            return rna_logits
        


    