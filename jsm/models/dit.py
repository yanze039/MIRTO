import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.ops.triton.layer_norm import RMSNorm
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
		x: torch.Tensor,
		bias: typing.Optional[torch.Tensor],
		scale: torch.Tensor,
		residual: typing.Optional[torch.Tensor],
		prob: float,
		training: bool) -> torch.Tensor:
	if bias is not None:
		out = scale * F.dropout(x + bias, p=prob, training=training)
	else:
		out = scale * F.dropout(x, p=prob, training=training)

	if residual is not None:
		out = residual + out
	return out


def get_bias_dropout_add_scale(training):
	def _bias_dropout_add(x, bias, scale, residual, prob):
		return bias_dropout_add_scale(
			x, bias, scale, residual, prob, training)

	return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
						 shift: torch.Tensor,
						 scale: torch.Tensor) -> torch.Tensor:
	return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
		x: torch.Tensor,
		bias: typing.Optional[torch.Tensor],
		scale: torch.Tensor,
		residual: typing.Optional[torch.Tensor],
		prob: float) -> torch.Tensor:
	return bias_dropout_add_scale(
		x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
		x: torch.Tensor,
		bias: typing.Optional[torch.Tensor],
		scale: torch.Tensor,
		residual: typing.Optional[torch.Tensor],
		prob: float) -> torch.Tensor:
	return bias_dropout_add_scale(
		x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
									 shift: torch.Tensor,
									 scale: torch.Tensor) -> torch.Tensor:
	return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
	def __init__(self, dim, base=10_000):
		super().__init__()
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer('inv_freq', inv_freq)
		self.seq_len_cached = None
		self.cos_cached = None
		self.sin_cached = None

	def forward(self, x, seq_dim=1):
		seq_len = x.shape[seq_dim]
		if seq_len != self.seq_len_cached:
			self.seq_len_cached = seq_len
			t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
			freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
			emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
			# dims are: batch, seq_len, qkv, head, dim
			self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
			self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
			# This makes the transformation on v an identity.
			self.cos_cached[:,:,2,:,:].fill_(1.)
			self.sin_cached[:,:,2,:,:].fill_(0.)

		return self.cos_cached, self.sin_cached


def rotate_half(x):
	x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
	cos = cos[0,:,0,0,:cos.shape[-1]//2]
	sin = sin[0,:,0,0,:sin.shape[-1]//2]
	return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# function overload
def modulate(x, shift, scale):
	return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.weight = nn.Parameter(torch.ones([dim]))
		self.dim = dim
	def forward(self, x):
		with torch.cuda.amp.autocast(enabled=False):
			x = F.layer_norm(x.float(), [self.dim])
		return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
	"""x_skip + residual_scale * W @ x"""
	dim_out, dim_in = W.shape[0], W.shape[1]
	return torch.addmm(
		x_skip.view(-1, dim_out),
		x.view(-1, dim_in),
		W.T,
		alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
	"""
	Embeds scalar timesteps into vector representations.
	"""
	def __init__(self, hidden_size, frequency_embedding_size=256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True))
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t, dim, max_period=10000):
		"""
		Create sinusoidal timestep embeddings.
		:param t: a 1-D Tensor of N indices, one per batch element.
											These may be fractional.
		:param dim: the dimension of the output.
		:param max_period: controls the minimum frequency of the embeddings.
		:return: an (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(
			- math.log(max_period)
			* torch.arange(start=0, end=half, dtype=torch.float32)
			/ half).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat(
				[embedding,
				 torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t):
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb


class LabelEmbedder(nn.Module):
	"""Embeds class labels into vector representations.
	
	Also handles label dropout for classifier-free guidance.
	"""
	def __init__(self, num_classes, cond_size):
		super().__init__()
		self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
		self.num_classes = num_classes

		# TODO think of initializing with 0.02 std deviation like in original DiT paper

	def forward(self, labels):
		embeddings = self.embedding_table(labels)
		return embeddings
		

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
	def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
		super().__init__()
		self.n_heads = n_heads

		self.norm1 = LayerNorm(dim)
		self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
		self.attn_out = nn.Linear(dim, dim, bias=False)
		self.dropout1 = nn.Dropout(dropout)

		self.norm2 = LayerNorm(dim)
		self.mlp = nn.Sequential(
			nn.Linear(dim, mlp_ratio * dim, bias=True),
			nn.GELU(approximate='tanh'),
			nn.Linear(mlp_ratio * dim, dim, bias=True))
		self.dropout2 = nn.Dropout(dropout)
		self.dropout = dropout

		self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
		self.adaLN_modulation.weight.data.zero_()
		self.adaLN_modulation.bias.data.zero_()


	def _get_bias_dropout_scale(self):
		if self.training:
			return bias_dropout_add_scale_fused_train
		else:
			return bias_dropout_add_scale_fused_inference


	def forward(self, x, rotary_cos_sin, c, 
                cu_seqlens=None,
                max_seqlen=None):
		batch_size, seq_len = x.shape[0], x.shape[1]

		bias_dropout_scale_fn = self._get_bias_dropout_scale()

		(shift_msa, scale_msa, gate_msa, shift_mlp,
		 scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

		# attention operation
		x_skip = x
		x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

		qkv = self.attn_qkv(x)
		qkv = rearrange(qkv,
						'b s (three h d) -> b s three h d',
						three=3,
						h=self.n_heads)
		with torch.cuda.amp.autocast(enabled=False):
			cos, sin = rotary_cos_sin
			qkv = apply_rotary_pos_emb(
				qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
		qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        # qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        # cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        #    of the sequences in the batch, used to index into qkv.
        # max_seqlen: int. Maximum sequence length in the batch.
        # flash_attn_varlen_qkvpacked_func(
		# 	qkv,
		# 	cu_seqlens,
		# 	max_seqlen,
		# 	dropout_p=0.0,
		# 	softmax_scale=None,
		# 	causal=False,
		# 	window_size=(-1, -1),  # -1 means infinite context window
		# 	softcap=0.0, # 0.0 means deactivated
		# 	alibi_slopes=None,
		# 	deterministic=False,
		# 	return_attn_probs=False,
		# )
		x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
			qkv, cu_seqlens, max_seqlen, 0., causal=False)
		
		x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

		x = bias_dropout_scale_fn(self.attn_out(x),
								None,
								gate_msa,
								x_skip,
								self.dropout)

		# mlp operation
		x = bias_dropout_scale_fn(
			self.mlp(modulate_fused(
				self.norm2(x), shift_mlp, scale_mlp)),
			None, gate_mlp, x, self.dropout)
		return x



class EmbeddingLayer(nn.Module):
	def __init__(self, dim, vocab_dim):
		super().__init__()
		self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
		torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

	def forward(self, x):
		return self.embedding[x]


class DDitFinalLayer(nn.Module):
	def __init__(self, hidden_size, out_channels, cond_dim):
		super().__init__()
		self.norm_final = LayerNorm(hidden_size)
		self.linear = nn.Linear(hidden_size, out_channels)
		self.linear.weight.data.zero_()
		self.linear.bias.data.zero_()

		self.adaLN_modulation = nn.Linear(cond_dim,
										2 * hidden_size,
										bias=True)
		self.adaLN_modulation.weight.data.zero_()
		self.adaLN_modulation.bias.data.zero_()

	def forward(self, x, c):
		shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
		x = modulate_fused(self.norm_final(x), shift, scale)
		x = self.linear(x)
		return x


class DITBackbone(nn.Module, huggingface_hub.PyTorchModelHubMixin):
	def __init__(
            self, 
            config,
        ):
		super().__init__()
		self.config = config
	
		# self.vocab_embed = EmbeddingLayer(config.model.hidden_size, self.vocab_size)
		self.sigma_map = TimestepEmbedder(config.cond_dim)
		self.rotary_emb = Rotary(config.hidden_size // config.n_heads)

		blocks = []
		for _ in range(config.n_blocks):
			blocks.append(DDiTBlock(config.hidden_size,
									config.n_heads,
									config.cond_dim,
									dropout=config.dropout))
		self.blocks = nn.ModuleList(blocks)
		self.scale_by_sigma = config.scale_by_sigma

	def _get_bias_dropout_scale(self):
		if self.training:
			return bias_dropout_add_scale_fused_train
		else:
			return  bias_dropout_add_scale_fused_inference

	def forward(self, 
                hidden_states, 
                sigma,
                cu_seqlens=None,
                max_seqlen=None
                ):     
        # x is the embedding
		c = F.silu(self.sigma_map(sigma))
		rotary_cos_sin = self.rotary_emb(hidden_states)
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			for i in range(len(self.blocks)):
				hidden_states = self.blocks[i](hidden_states, 
                                   rotary_cos_sin, 
                                   c, 
                                   cu_seqlens=cu_seqlens,
                                   max_seqlen=max_seqlen)
		return hidden_states


class SpeciesSpecificJointSequenceDIT(nn.Module):
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
            
        self.backbone = DITBackbone(config)
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

    def forward(
            self, 
            input_ids,
            sigma,
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
        inputs_embeds = index_first_axis(rearrange(inputs_embeds, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        outputs = self.backbone(
            hidden_states=inputs_embeds,
            sigma=sigma,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        
        hidden_states = pad_input(outputs.squeeze(0), indices, B, L)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lct+Lp:,:]
        rna_logits = self.rna_lm_head(hidden_states)
        return rna_logits


    # def _forward(self, 
    #             input_ids,                 
    #             protein_embeddings,
    #             row_wise_col_perms,
    #             inverse_row_wise_col_perms,
    #             attention_mask,
    #             seq_idx=None,
    #             inference_params=None, 
    #             **mixer_kwargs,
    #     ):
    #     B, Lr = input_ids.shape
    #     Lp = protein_embeddings.shape[1]
    #     L = Lr + Lp
    #     inputs_embeds = self.rna_embeddings(input_ids)
    #     codon_protein_translation_logits = self.translation_lm_head(inputs_embeds)  # 60, 1180, 33
        
    #     protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))
    #     inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
    #     inputs_embeds = torch.gather(inputs_embeds, dim=1, index=row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))  
    #     indices, cu_seqlens, max_sequence_len = get_unpad_data(attention_mask)
    #     inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)
    #     outputs = self.backbone(
    #         hidden_states=inputs_embeds[:, :max_sequence_len, :],
    #     )
    #     outputs = F.pad(outputs, (0, 0, 0, L - max_sequence_len))  # Pad to original length
    #     hidden_states = outputs * attention_mask.unsqueeze(-1)
    #     hidden_states = torch.gather(hidden_states, dim=1, index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))[:,Lp:,:]
        
    #     rna_logits = self.rna_lm_head(hidden_states)
    #     modality_logits = self.modality_prediction_head(hidden_states)

    #     return rna_logits, codon_protein_translation_logits, modality_logits
    
    # @torch.no_grad()
    # def generating_forward(
    #         self,
    #         input_ids,                 
    #         protein_embeddings=None,
    #         species_ids=None,
    #         return_hidden_states=True,
    #         inference_params=None, 
    #         num_last_tokens=0,
    #         hidden_layer_indices=None
    #     ):
    #     B, Lr = input_ids.shape
    #     inputs_embeds = self.rna_embeddings(input_ids)  
    #     if protein_embeddings is not None:
    #         protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))   
    #         inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
    #     if species_ids is not None:
    #         species_embeds = self.species_embedding(species_ids)
    #         inputs_embeds = torch.cat([species_embeds, inputs_embeds], dim=1)
        
    #     outputs = self.backbone(
    #         hidden_states=inputs_embeds,
    #         inference_params=inference_params,
    #         hidden_layer_indices=hidden_layer_indices
    #     )
    #     if return_hidden_states:
    #         return outputs
        
    #     if num_last_tokens > 0:
    #         outputs = outputs[:, -num_last_tokens:, :]
    #     rna_logits = self.rna_lm_head(outputs)
    #     return rna_logits
    
    # def calculate_attention_weights(
    #         self, 
    #         input_ids,                 
    #         protein_embeddings=None,
    #         inference_params=None, 
    #         hidden_layer_idx=None,
    #         species_ids=None,
    #     ):
    #     inputs_embeds = self.rna_embeddings(input_ids)  
    #     if protein_embeddings is not None:
    #         protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))   
    #         inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
    #     if species_ids is not None:
    #         species_embeds = self.species_embedding(species_ids)
    #         inputs_embeds = torch.cat([species_embeds, inputs_embeds], dim=1)
    #     attention_weights = self.backbone.calculate_attention_weights(
    #         hidden_states=inputs_embeds,
    #         inference_params=inference_params,
    #         hidden_layer_idx=hidden_layer_idx
    #     )
    #     return attention_weights
        


    


