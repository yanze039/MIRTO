import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from einops import rearrange
from flash_attn.ops.triton.layer_norm import RMSNorm
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
from flash_attn.layers.rotary import apply_rotary_emb


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000, max_len=8192*8):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # emb = torch.cat((freqs, freqs), dim=-1) # Shape: [max_len, dim]
        emb = freqs # Shape: [max_len, dim/2]
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, max_seqlen_in_batch):
        # Return the cache sliced to the longest sequence in the CURRENT batch
        # to save a bit of memory bandwidth, or just return the whole thing.
        return self.cos_cached[:max_seqlen_in_batch], self.sin_cached[:max_seqlen_in_batch]
		


#################################################################################
#                                 Core Model                                    #
#################################################################################

class EmbeddingLayer(nn.Module):
	def __init__(self, dim, vocab_dim):
		super().__init__()
		self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
		torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

	def forward(self, x):
		return self.embedding[x]


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads,mlp_ratio=4, residual_in_fp32=True, layer_index=None):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.residual_in_fp32=residual_in_fp32
        self.layer_index = layer_index

    def forward(self, 
                x, 
                rotary_cos_sin,
                cu_seqlens=None,
                max_seqlen=None,
                residual=None,
                return_qkv=False
     ): # Added residual to signature
        batch_size, _ = x.shape[0], x.shape[1]

		# --- ATTENTION BLOCK ---
        norm_x, x = layer_norm_fn(
			x, self.norm1.weight, self.norm1.bias,
			residual=residual,
			eps=self.norm1.eps,
			residual_in_fp32=self.residual_in_fp32,
			prenorm=True,
			is_rms_norm=isinstance(self.norm1, RMSNorm) # Detect if it's RMS
		)

        qkv = self.attn_qkv(norm_x)
        
        cos, sin = rotary_cos_sin
        if cu_seqlens is None:
			# qkv: (batch_size, seqlen, 3, nheads, headdim) if cu_seqlens is None
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
            raise NotImplementedError("cu_seqlens must be provided for variable-length sequences.")
			# qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
			# qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        else:
            qkv = rearrange(qkv, 'ts (three h d) -> ts three h d', three=3, h=self.n_heads)
			# we need to split qkv to qk, because v should not be rotated
            qkv[:, 0] = apply_rotary_emb(
				qkv[:, 0],
				cos, 
				sin, 
				interleaved=False, 
				cu_seqlens=cu_seqlens, 
				max_seqlen=max_seqlen
			)   
            qkv[:, 1] = apply_rotary_emb(
				qkv[:, 1],
				cos, 
				sin, 
				interleaved=False, 
				cu_seqlens=cu_seqlens, 
				max_seqlen=max_seqlen
			)
		
        attn_out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
			qkv, cu_seqlens, max_seqlen, 0., causal=False)
		
        if cu_seqlens is None:
            attn_out = rearrange(attn_out, '(b s) h d -> b s (h d)', b=batch_size)
            raise NotImplementedError("cu_seqlens must be provided for variable-length sequences.")
        else:
            attn_out = rearrange(attn_out, 'ts h d -> ts (h d)')
        attn_out = self.attn_out(attn_out)

		# --- MLP BLOCK ---
		# 2. Second Fused Norm + Residual
		# We pass 'x' (which is the output residual from the first block) 
		# and 'attn_out' as the input to the next norm.
        norm_x_mlp, x = layer_norm_fn(
			attn_out, self.norm2.weight, self.norm2.bias,
			residual=x,
			eps=self.norm2.eps,
			prenorm=True,
			residual_in_fp32=self.residual_in_fp32,
			is_rms_norm=isinstance(self.norm2, RMSNorm)
		)
        x_mlp = self.mlp(norm_x_mlp)
        if return_qkv:
            return x_mlp, x, qkv
        return x_mlp, x


class TransformerBackbone(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
            self, 
            config,
        ):
        super().__init__()
        self.config = config
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)
        self.residual_in_fp32 = config.get('residual_in_fp32', True)
        blocks = []
        for layer_index in range(config.n_blocks):
            blocks.append(TransformerBlock(config.hidden_size,
									config.n_heads,
									residual_in_fp32=self.residual_in_fp32,
									layer_index=layer_index))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = LayerNorm(config.hidden_size)
        self.D = config.hidden_size // config.n_heads
        
    def forward(
            self, 
            hidden_states, 
            cu_seqlens=None,
            max_seqlen=None
        ):
        # x is the embedding
        rotary_cos_sin = self.rotary_emb(max_seqlen)
        residual = None
        for i in range(len(self.blocks)):
            hidden_states, residual = self.blocks[i](hidden_states, 
								rotary_cos_sin, 
								cu_seqlens=cu_seqlens,
								max_seqlen=max_seqlen, residual=residual)
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
    
    @torch.inference_mode()
    def get_attention_weights(
            self, 
            hidden_states, 
            cu_seqlens=None,
            max_seqlen=None,
            indices=None,
            batch_size=None,
            padding_length=None,
            attention_mask=None
        ):
        # x is the embedding
        rotary_cos_sin = self.rotary_emb(max_seqlen)
        residual = None
        scale = 1.0 / math.sqrt(self.D)
        attention_scores_per_layer = {}
        for i in range(len(self.blocks)):
            hidden_states, residual, qkv = self.blocks[i](
                                hidden_states, 
								rotary_cos_sin, 
								cu_seqlens=cu_seqlens,
								max_seqlen=max_seqlen, 
                                residual=residual,
                                return_qkv=True
            )
            qkv = pad_input(qkv.squeeze(0), indices, batch_size, padding_length)
            qkv = qkv * attention_mask[:,:,None,None, None]
            
            
            # Unpack: [B,S,H,D]
            q, k, v = qkv.unbind(dim=2)

            # Move to [B,H,S,D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute scores: [B,H,S,S]
            # Use float32 for stability, then cast back if you want
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            attention_scores_per_layer[i] = scores.cpu()
            
        return attention_scores_per_layer
    
    def get_embeddings(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        layer_indices=[]
        ):
        # x is the embedding
        rotary_cos_sin = self.rotary_emb(max_seqlen)
        residual = None
        output_embeddings = {}
        for i in range(len(self.blocks)):
            hidden_states, residual = self.blocks[i](hidden_states, 
								rotary_cos_sin, 
								cu_seqlens=cu_seqlens,
								max_seqlen=max_seqlen, residual=residual)
            if i in layer_indices:
                output_embeddings[i] = hidden_states + residual
        return output_embeddings


class SpeciesSpecificJointSequenceTransformer(nn.Module):
    def __init__(
        self,
        config,
        rna_vocab_size,
        protein_vocab_size,
    ) -> None:
        self.config = config
        vocab_size = rna_vocab_size
        pad_vocab_size_multiple = 8
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
            
        self.backbone = TransformerBackbone(config)
        self.rna_embeddings = nn.Embedding(rna_vocab_size, config.hidden_size)
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        
        self.rna_lm_head = nn.Linear(config.hidden_size, self.rna_vocab_size, bias=True)
        self.protein_align_head = nn.Linear(config.protein_hidden_size, config.hidden_size, bias=True)
        self.protein_embedding_norm = RMSNorm(config.protein_hidden_size, eps=1e-6)
        
        self.species_embedding = nn.Embedding(config.num_species, config.hidden_size)
        self.modality_embedding = nn.Embedding(8, config.hidden_size)  # 3 modalities: 5'UTR, CDS, 3'UTR
        
        nn.init.zeros_(self.protein_align_head.bias)
        nn.init.normal_(self.protein_align_head.weight, std=0.01)
        nn.init.zeros_(self.rna_lm_head.bias)
        nn.init.normal_(self.rna_lm_head.weight, std=0.01)
        self.initialize_weights()

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
        inputs_embeds = index_first_axis(rearrange(inputs_embeds, "b s ... -> (b s) ..."), indices)
        outputs = self.backbone(
            hidden_states=inputs_embeds,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        
        hidden_states = pad_input(outputs.squeeze(0), indices, B, L)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lct+Lp:,:]
        rna_logits = self.rna_lm_head(hidden_states)
        return rna_logits
    
    def get_attention_weights(
            self, 
            input_ids,
            species_ids,                 
            protein_embeddings,
            row_wise_col_perms,
            inverse_row_wise_col_perms,
            attention_mask,
            modality_type_ids=None,
            modality_mask=None,
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
        inputs_embeds = index_first_axis(rearrange(inputs_embeds, "b s ... -> (b s) ..."), indices)
        scores = self.backbone.get_attention_weights(
            hidden_states=inputs_embeds,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            indices=indices,
            batch_size=B,
            padding_length=L,
            attention_mask=attention_mask
        )
        return scores
        
        
    
    def get_embeddings(
            self, 
            input_ids,
            species_ids,                 
            protein_embeddings,
            row_wise_col_perms,
            inverse_row_wise_col_perms,
            attention_mask,
            modality_type_ids=None,
            modality_mask=None,
            layer_indices=[],
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
        inputs_embeds = index_first_axis(rearrange(inputs_embeds, "b s ... -> (b s) ..."), indices)
        outputs = self.backbone.get_embeddings(
            hidden_states=inputs_embeds,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            layer_indices=layer_indices
        )
        
        for layer_index in layer_indices:
             outputs[layer_index] = pad_input(outputs[layer_index].squeeze(0), indices, B, L)
             outputs[layer_index] = outputs[layer_index] * attention_mask.unsqueeze(-1)
             outputs[layer_index] = torch.gather(outputs[layer_index], dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lct+Lp:,:]
        
        return outputs


    def initialize_weights(self):
        nn.init.normal_(self.rna_embeddings.weight, std=0.02)
        nn.init.normal_(self.species_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.rna_lm_head.weight, std=0.01)
        if self.rna_lm_head.bias is not None:
            nn.init.zeros_(self.rna_lm_head.bias)
        
        nn.init.normal_(self.protein_align_head.weight, std=0.01)
        if self.protein_align_head.bias is not None:
            nn.init.zeros_(self.protein_align_head.bias)
        
        for block in self.backbone.blocks:

			# Attention QKV projection
            nn.init.xavier_uniform_(block.attn_qkv.weight)
            std = 0.02 / math.sqrt(2 * self.config.n_blocks)
            nn.init.normal_(block.attn_out.weight, std=std)
            nn.init.xavier_uniform_(block.mlp[0].weight)
            nn.init.normal_(block.mlp[2].weight, std=std)
            if block.mlp[0].bias is not None:
                nn.init.zeros_(block.mlp[0].bias)
            if block.mlp[2].bias is not None:
                nn.init.zeros_(block.mlp[2].bias)
        
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, RMSNorm)):
                nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)