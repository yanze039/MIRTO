import sys, os, json
import math
import torch
import torch.nn as nn
from vortex.logging import activations_logger
from jsm.models.vortex_model import StripedHyena  # suspect import
from flash_attn.ops.triton.layer_norm import RMSNorm as FlashRMSNorm
import torch.nn.functional as F
import sys
sys.path.append('/home/yanze039/orcd/scratch/software/flash-linear-attention')
from fla.layers.utils import get_unpad_data
import transformer_engine.pytorch as te


class JointSequenceStripedHyenaBackbone(StripedHyena):
    def forward(
            self, 
            hidden_states,
            inference_params_dict=None, 
            padding_mask=None,
    ):
        if inference_params_dict is not None:
            hidden_states, inference_params_dict_out = self.stateful_forward(
                hidden_states,
                inference_params_dict=inference_params_dict,
            )
        else:
            hidden_states, inference_params_dict_out = self.stateless_forward(hidden_states, padding_mask=padding_mask)
        if self.print_activations:
            activations_logger.info(f"post norm: {hidden_states}, {hidden_states.min()}, {hidden_states.max()}, {self.norm.scale}")
        return hidden_states, inference_params_dict_out


class JointSequenceStripedHyenaModel(nn.Module):

    def __init__(
        self,
        config,
        rna_vocab_size,
        protein_vocab_size,
        num_modalities=5,
    ) -> None:
        self.config = config

        super().__init__()
        pad_vocab_size_multiple = self.config.pad_vocab_size_multiple
        if rna_vocab_size % pad_vocab_size_multiple != 0:
            rna_vocab_size += pad_vocab_size_multiple - (rna_vocab_size % pad_vocab_size_multiple)
        
        self.backbone = JointSequenceStripedHyenaBackbone(config.hyena_config)
        
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.criterion = None
        
        self.rna_embeddings = nn.Embedding(rna_vocab_size, config.hyena_config.hidden_size)
        self.translation_lm_head = nn.Linear(config.hyena_config.hidden_size, self.protein_vocab_size, bias=True)
        self.protein_align_head = nn.Linear(config.protein_hidden_size, config.hyena_config.hidden_size, bias=True)
        # self.modality_prediction_head = nn.Linear(config.hyena_config.vocab_size, num_modalities, bias=True)  # Assuming 2 modalities: RNA and Protein
        self.protein_embedding_norm = FlashRMSNorm(config.protein_hidden_size, dtype=torch.bfloat16, eps=1e-6)
        self.final_norm = FlashRMSNorm(config.hyena_config.hidden_size, dtype=torch.bfloat16, eps=1e-6)
        self.init_weight()
    
    def init_weight(self):
        # Initialize weights
        nn.init.normal_(self.rna_embeddings.weight, mean=0.0, std=0.02)
        std = 0.02
        nn.init.normal_(self.translation_lm_head.weight, mean=0.0, std=std)
        nn.init.zeros_(self.translation_lm_head.bias)
        nn.init.normal_(self.protein_align_head.weight, mean=0.0, std=std)
        nn.init.zeros_(self.protein_align_head.bias)
        # nn.init.normal_(self.modality_prediction_head.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.modality_prediction_head.bias)
        self.protein_embedding_norm.weight.data.fill_(1.0)
        self.final_norm.weight.data.fill_(1.0)

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
        
        Lp = protein_embeddings.shape[1]
        inputs_embeds = self.rna_embeddings(input_ids)
        codon_protein_translation_logits = self.translation_lm_head(inputs_embeds) * 0.2
        protein_embeddings = self.protein_align_head(self.protein_embedding_norm(protein_embeddings))
        inputs_embeds = torch.cat([protein_embeddings, inputs_embeds], dim=1)
        inputs_embeds = torch.gather(inputs_embeds, dim=1, index=row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]))  
        inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)
        _, L, _ = inputs_embeds.shape
        _, _, max_seqlen_in_batch = get_unpad_data(attention_mask)
        # remove padding zeros
        inputs_embeds = inputs_embeds[:, :max_seqlen_in_batch, :].contiguous()
        hidden_states, inference_params_dict_out = self.backbone(
            hidden_states=inputs_embeds,
            inference_params_dict=inference_params,
            padding_mask=attention_mask[:, :max_seqlen_in_batch] if attention_mask is not None else None,
        )
        # put back to padded, hidden_states has shape (B, max_seqlen_in_batch, D))
        hidden_states = F.pad(hidden_states, (0, 0, 0, L - max_seqlen_in_batch))
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, 
                                     index=inverse_row_wise_col_perms.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
                                     )[:,Lp:,:]
        # rna_logits = self.rna_lm_head(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        rna_logits = F.linear(hidden_states, self.rna_embeddings.weight)
            
        # 60, 1180, 33
        # modality_logits = self.modality_prediction_head(hidden_states)
        modality_logits = None
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
        


    