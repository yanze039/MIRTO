import torch
from torch.optim.lr_scheduler import LambdaLR
import pickle
import lightning as L
from typing import Optional
from dataclasses import dataclass
import time
import numpy as np
import torch.nn as nn
import jsm.utils as utils
from jsm.data.utils import codon_table, ConcatenatedAlphabet
from jsm.data.joint_sequence import prepare_inputs_for_model, tokenize_inputs
try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None
import torch.nn.functional as F

def hamming_distance_numpy(s1, s2):
    a = np.frombuffer(s1.encode(), dtype=np.uint8)
    b = np.frombuffer(s2.encode(), dtype=np.uint8)
    return np.count_nonzero(a != b)


BASE_ORDER = ["A", "C", "G", "U"]

@torch.no_grad()
def _kmer_index_table(k: int, device: torch.device) -> torch.Tensor:
    grids = torch.meshgrid(*[torch.arange(4, device=device) for _ in range(k)], indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, k)  # [M, k], M=4**k


def expected_kmer_dist_from_logits_masked(
    logits: torch.Tensor,          # [B, L, V]
    tok_to_idx: dict,
    utr5_mask: torch.Tensor,     # [B, L] bool
    utr3_mask: torch.Tensor,     # [B, L] bool
    k: int,
    eps: float = 1e-8,
    normalize: bool = True,
    idx_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Uses logits directly; internally uses log_softmax over (A,C,G,U).
    Returns q: [B, 4**k] expected k-mer distribution restricted to region_mask.
    """
    B, L, V = logits.shape

    # restrict to A,C,G,U logits: [B, L, 4]
    idx4 = torch.tensor([tok_to_idx[b] for b in BASE_ORDER], device=logits.device)
    sub_logits = logits.index_select(dim=-1, index=idx4)

    # log probabilities over A,C,G,U: [B, L, 4]
    logP = F.log_softmax(sub_logits, dim=-1)

    T = L - k + 1
    M = 4**k
    if T <= 0:
        return torch.full((B, M), 1.0 / M, device=logits.device, dtype=sub_logits.dtype)

    # compute expected k-mer probs per start
    log_terms = []
    for i in range(k):
        logP_slice = logP[:, i:i+T, :]  # [B, T, 4]
        bases = idx_table[:, i].view(1, 1, -1).expand(B, T, -1)  # [B, T, M]
        log_terms.append(torch.gather(logP_slice, dim=2, index=bases))

    log_prod = torch.stack(log_terms, dim=0).sum(dim=0)  # [B, T, M]
    p_kmer_at_t = log_prod.exp()                         # [B, T, M]

    # mask invalid windows
    # valid window starts inside region
    win_ok_utr5 = utr5_mask.unfold(1, k, 1).all(dim=-1)  # [B, T]
    any_win_utr5 = win_ok_utr5.any(dim=1)
    p_kmer_at_t_utr5 = p_kmer_at_t * win_ok_utr5.to(p_kmer_at_t.dtype).unsqueeze(-1)
    counts_utr5 = p_kmer_at_t_utr5.sum(dim=1)  # [B, M]
    q_utr5 = counts_utr5 / counts_utr5.sum(dim=1, keepdim=True).clamp_min(eps)
    if (~any_win_utr5).any():
        q_utr5[~any_win_utr5] = 1.0 / M
    
    win_ok_utr3 = utr3_mask.unfold(1, k, 1).all(dim=-1)  # [B, T]
    any_win_utr3 = win_ok_utr3.any(dim=1)
    p_kmer_at_t_utr3 = p_kmer_at_t * win_ok_utr3.to(p_kmer_at_t.dtype).unsqueeze(-1)
    counts_utr3 = p_kmer_at_t_utr3.sum(dim=1)  # [B, M]
    q_utr3 = counts_utr3 / counts_utr3.sum(dim=1, keepdim=True).clamp_min(eps)
    if (~any_win_utr3).any():
        q_utr3[~any_win_utr3] = 1.0 / M

    return q_utr5, q_utr3


def js_divergence(q_model: torch.Tensor, q_ref: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    q_model: [B, M]
    q_ref:   [M] or [B, M]
    returns: [B]
    """

    q_m = q_model.clamp_min(eps)
    q_r = q_ref.clamp_min(eps)
    m = 0.5 * (q_m + q_r)

    kl_m = (q_m * (q_m.log() - m.log())).sum(dim=-1)
    kl_r = (q_r * (q_r.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_m + kl_r)


@dataclass
class Loss:
    loss: torch.FloatTensor
    rna_lm_loss: Optional[torch.FloatTensor] = None
    translation_loss: Optional[torch.FloatTensor] = None
    num_codon_aa_errors: int = 0
    total_aa_length: int = 0
    modality_prediction_loss: Optional[torch.FloatTensor] = None
    
    rna_lm_loss_codon: Optional[torch.FloatTensor] = None
    rna_lm_loss_utr5: Optional[torch.FloatTensor] = None
    rna_lm_loss_utr3: Optional[torch.FloatTensor] = None
    perplexity: Optional[torch.FloatTensor] = None
    perplexity_utr5: Optional[torch.FloatTensor] = None
    perplexity_utr3: Optional[torch.FloatTensor] = None
    perplexity_codon: Optional[torch.FloatTensor] = None
    divergence: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None


def info_nce_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Computes the InfoNCE loss between two sets of features.
    Args:
        features_a: Tensor of shape (batch_size, feature_dim)
        features_b: Tensor of shape (batch_size, feature_dim)
        temperature: Scaling factor for the logits
    Returns:
        Scalar tensor representing the InfoNCE loss
    """
    batch_size = features_a.shape[0]

    logits = torch.matmul(features_a, features_b.T) / temperature
    labels = torch.arange(batch_size, device=features_a.device)

    loss_a_to_b = F.cross_entropy(logits, labels)
    loss_b_to_a = F.cross_entropy(logits.T, labels)

    loss = (loss_a_to_b + loss_b_to_a) * 0.5
    return loss


def grad_norm_autograd(loss, params):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True
    )
    total_norm = 0.0
    for g in grads:
        if g is not None:
            total_norm += g.norm(2).item() ** 2
    return total_norm ** 0.5


class JointSequenceModeling(L.LightningModule):
    def __init__(
        self,
        config,
        global_tokenizer,
        protein_tokenizer,
        rna_vocab_size,
        protein_vocab_size,
        protein_encoder,
        codon_tokenizer = None,
        utr_5_tokenizer = None,
        utr_3_tokenizer = None,
    ):
        
        L.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.tokenizer = None
        self.config = config
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.protein_tokenizer = protein_tokenizer
        self.global_tokenizer = global_tokenizer
        self.padding_index = self.global_tokenizer.padding_idx
        self.mask_index = self.global_tokenizer.mask_idx
        self.cls_index = self.global_tokenizer.cls_idx
        self.eos_index = self.global_tokenizer.eos_idx
        self.unknown_index = self.global_tokenizer.unk_idx
        self.N_index = self.global_tokenizer.tok_to_idx.get('N', self.global_tokenizer.unk_idx)
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.training_style = config.get("training_style", "standard")
        if self.config.backbone == 'mamba2':
            from jsm.models.mamba import JointSequenceMambaModel
            assert self.training_style == 'standard', "Mamba backbone only supports standard training style"
            self.backbone = JointSequenceMambaModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
                num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
            )
        elif self.config.backbone == 'hyena':
            from jsm.models.vortex_striped_hyena import JointSequenceStripedHyenaModel
            assert self.training_style == 'standard', "Hyena backbone only supports standard training style"
            self.backbone = JointSequenceStripedHyenaModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
                num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
            )
        elif self.config.backbone == 'hyena_nemo':
            assert self.training_style == 'standard', "Hyena Nemo backbone only supports standard training style"
            # We init model in the setup stage, because nemo needs the megatron initialization 
            # to be done after the trainer is setup
            self.backbone = None
        elif self.config.backbone == 'hybrid_mamba':
            assert self.training_style == 'standard', "Mamba/Attention hybrid backbone only supports standard training style"
            from jsm.models.hybrid_mamba import JointSequenceMambaModel
            self.backbone = JointSequenceMambaModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
                num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
            )
        elif self.config.backbone == 'attention_cell':
            assert self.training_style == 'cell_type_specific', "cell_type_finetune must be True when using attention_cell backbone"
            from jsm.models.attention_cell import CellTypeSpecificJointSequenceAttentionModel
            self.backbone = CellTypeSpecificJointSequenceAttentionModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
            )
        elif self.config.backbone == 'attention_species':
            assert self.training_style == 'species_specific', "species_specific must be True when using attention_species backbone"
            from jsm.models.attention_species import SpeciesSpecificJointSequenceAttentionModel
            self.backbone = SpeciesSpecificJointSequenceAttentionModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
            )
        else:
            raise ValueError(f'Unknown backbone: {self.config.backbone}')
        # metrics are automatically reset at end of epoch
        self.lr = self.config.optim.lr
       
        self.codon_aa_errors = []
        self.total_length = []
    
        # freeze protein encoder
        self.protein_encoder = protein_encoder
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
        self.rna_lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
            reduction="none"
        )
        self.launch_timestamp = time.time()
        self.codon_tokenizer = codon_tokenizer
        self.utr_5_tokenizer = utr_5_tokenizer
        self.utr_3_tokenizer = utr_3_tokenizer
        self.strict_loading = False
        self.resumed_dataloader_state_from_ckpt = None
        self.kmer_idx_table_cache = None
        self.k = 5  # k-mer size for k-mer distribution calculation
        if self.config.training.contrastive_training:
            self.contrastive_protein_proj = nn.Linear(self.config.model.protein_hidden_size, 256)
            self.contrastive_utr5_proj     = nn.Linear(self.config.model.d_model, 256)
            self.contrastive_cds_proj      = nn.Linear(self.config.model.d_model, 256)
            self.contrastive_utr3_proj     = nn.Linear(self.config.model.d_model, 256)
        else:
            self.contrastive_protein_proj = None
            self.contrastive_utr5_proj     = None
            self.contrastive_cds_proj      = None
            self.contrastive_utr3_proj     = None
    
    def setup(self, stage=None):
        if self.config.backbone == 'hyena_nemo':
            from jsm.models.nemo_model import safe_init_megatron
            safe_init_megatron()
            if self.config.backbone == 'hyena_nemo':
                from jsm.models.nemo_hyena import JointSequenceStripedHyenaModel
                self.backbone = JointSequenceStripedHyenaModel(
                    self.config.model,
                    rna_vocab_size=self.rna_vocab_size,
                    protein_vocab_size=self.protein_vocab_size,
                    num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
                )
        
    def register_tokenizer(self, tokenizer_name, tokenizer):
        """Register a tokenizer to the model."""
        if tokenizer_name == "global":
            self.global_tokenizer = tokenizer
        elif tokenizer_name == "protein":
            self.protein_tokenizer = tokenizer
        elif tokenizer_name == "codon":
            self.codon_tokenizer = tokenizer
        elif tokenizer_name == "utr_5":
            self.utr_5_tokenizer = tokenizer
        elif tokenizer_name == "utr_3":
            self.utr_3_tokenizer = tokenizer
        else:
            raise ValueError(f"Unknown tokenizer name: {tokenizer_name}")
    
    def forward(
            self, 
            batch,
        ):
        with torch.no_grad():
            protein_output = self.protein_encoder(batch["protein_input_ids"])
            protein_embeddings = protein_output.embeddings
        
        if self.training_style == "cell_type_specific":
            rna_logits, codon_protein_translation_logits, modality_logits = self.backbone(
                input_ids=batch["rna_input_ids"],    
                cell_type_ids=batch["cell_type_ids"],             
                protein_embeddings=protein_embeddings,
                row_wise_col_perms=batch["row_wise_col_perms"],
                inverse_row_wise_col_perms=batch["inverse_row_wise_col_perms"],
                attention_mask=batch["attention_mask"],
                seq_idx=batch["seq_idx"],
                inference_params=None,
            )
        elif self.training_style == "species_specific":
            rna_logits = self.backbone(
                input_ids=batch["rna_input_ids"],    
                species_ids=batch["species_ids"],             
                protein_embeddings=protein_embeddings,
                row_wise_col_perms=batch["row_wise_col_perms"],
                inverse_row_wise_col_perms=batch["inverse_row_wise_col_perms"],
                attention_mask=batch["attention_mask"],
                seq_idx=batch["seq_idx"],
                inference_params=None,
                modality_type_ids=batch['modality_type_ids'],
                modality_mask=batch['modality_mask'],
                return_middle_hidden_states=self.config.training.contrastive_training,
            )
            if self.config.training.contrastive_training:
                rna_logits = (*rna_logits, protein_embeddings)
        elif self.training_style == "standard":
            rna_logits = self.backbone(
                input_ids=batch["rna_input_ids"],                 
                protein_embeddings=protein_embeddings,
                row_wise_col_perms=batch["row_wise_col_perms"],
                inverse_row_wise_col_perms=batch["inverse_row_wise_col_perms"],
                attention_mask=batch["attention_mask"],
                seq_idx=batch["seq_idx"],
                inference_params=None, 
            )
        else:
            raise ValueError(f"Unknown training style: {self.training_style}")
        return rna_logits
    
    def _loss(self, batch, prefix='train'):
        rna_outputs = self.forward(batch)
        if self.config.training.contrastive_training:
            rna_logits, middle_hidden_states, protein_embeddings = rna_outputs
        else:
            rna_logits = rna_outputs
            middle_hidden_states = None
            protein_embeddings = None
        
        utils.print_nans(rna_logits, 'rna_logits')
        
        if not self.trainer.training:
            try:
                codon_logits = torch.argmax(rna_logits, dim=-1)[batch["translation_rna_mask"].bool()]
                codon_tokenizer = self.trainer.datamodule.codon_alphabet
                aa_list = []
                for clogit in codon_logits:
                    codon = codon_tokenizer.get_tok(clogit.item())
                    aa = codon_table.get(codon, "-")
                    aa_list.append(aa)
                aa_list = "".join(aa_list)
                full_protein_sequence = "".join([x[1:]+"*" for x in batch["protein_sequence"]])
                assert len(aa_list) == len(full_protein_sequence)
                
                n_errors = hamming_distance_numpy(aa_list, full_protein_sequence)
                total_length = len(full_protein_sequence)
            except Exception as e:
                print(f"Error calculating amino acid errors: {e}")
                n_errors = 1
                total_length = 1
        else:
            n_errors = None
            total_length = None
        
        labels = batch["rna_input_ids"]
        labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], self.rna_lm_loss_fn.ignore_index)), 1)
        rna_lm_loss_per_token = self.rna_lm_loss_fn(rna_logits.view(labels.numel(), -1), labels.view(-1)).view(labels.shape)
        
        utr5_mask = batch["utr5_mask"]
        cds_mask = batch["cds_mask"]
        utr3_mask = batch["utr3_mask"]
        
        rna_lm_loss_utr5 = (rna_lm_loss_per_token * utr5_mask).sum() / torch.clamp(utr5_mask.sum(), min=1.0)
        rna_lm_loss_codon = (rna_lm_loss_per_token * cds_mask).sum() / torch.clamp(cds_mask.sum(), min=1.0)
        rna_lm_loss_utr3 = (rna_lm_loss_per_token * utr3_mask).sum() / torch.clamp(utr3_mask.sum(), min=1.0)
        rna_lm_loss = (rna_lm_loss_utr5 + rna_lm_loss_codon + rna_lm_loss_utr3) / 3.0
        
        # kmer alignment loss
        divergence = None
        if self.config.training.kmer_alignment:
            if self.kmer_idx_table_cache is None:
                self.kmer_idx_table_cache = _kmer_index_table(k=self.k, device=rna_logits.device)
            kmer_utr_5, kmer_utr_3 = expected_kmer_dist_from_logits_masked(
                rna_logits,
                tok_to_idx=self.trainer.datamodule.utr_alphabet.tok_to_idx,
                utr5_mask=utr5_mask.bool(),
                utr3_mask=utr3_mask.bool(),
                k=self.k,
                idx_table=self.kmer_idx_table_cache,
            )
            
            divergence_utr5 = js_divergence(kmer_utr_5, batch['batch_kmer_utr5']).mean()
            divergence_utr3 = js_divergence(kmer_utr_3, batch['batch_kmer_utr3']).mean()
            
            divergence = 0.9 * divergence_utr5 + 0.1 * divergence_utr3
            if self.config.training.kmer_alignment_warmup_steps > 0 and self.trainer.training:
                if self.trainer.global_step < self.config.training.kmer_alignment_warmup_start:
                    kmer_alignment_weight = 0.0
                else:
                    kmer_alignment_weight = (
                        min(1.0, (self.trainer.global_step - self.config.training.kmer_alignment_warmup_start) / self.config.training.kmer_alignment_warmup_steps)
                        * self.config.training.kmer_alignment_weight
                    )
            else:
                kmer_alignment_weight = self.config.training.kmer_alignment_weight
            rna_lm_loss = rna_lm_loss + kmer_alignment_weight * divergence
        
        if self.config.training.contrastive_training:
            pooled_protein_embeddings = (protein_embeddings * batch['protein_padding_mask'].unsqueeze(-1)).sum(dim=1) / torch.clamp(batch['protein_padding_mask'].sum(dim=1, keepdim=True), min=1.0)
            pooled_utr_5_embeddings = (middle_hidden_states * utr5_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(utr5_mask.sum(dim=1, keepdim=True), min=1.0)
            pooled_cds_embeddings = (middle_hidden_states * cds_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(cds_mask.sum(dim=1, keepdim=True), min=1.0)
            pooled_utr_3_embeddings = (middle_hidden_states * utr3_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(utr3_mask.sum(dim=1, keepdim=True), min=1.0)
            protein_info = F.normalize(self.contrastive_protein_proj(pooled_protein_embeddings), dim=-1)
            utr5_info = F.normalize(self.contrastive_utr5_proj(pooled_utr_5_embeddings), dim=-1)
            cds_info = F.normalize(self.contrastive_cds_proj(pooled_cds_embeddings), dim=-1)
            utr3_info = F.normalize(self.contrastive_utr3_proj(pooled_utr_3_embeddings), dim=-1)
            info_nce_loss_utr5 = info_nce_loss(protein_info, utr5_info, temperature=self.config.training.contrastive_temperature)
            info_nce_loss_cds = info_nce_loss(protein_info, cds_info, temperature=self.config.training.contrastive_temperature)
            info_nce_loss_utr3 = info_nce_loss(protein_info, utr3_info, temperature=self.config.training.contrastive_temperature)
            contrastive_loss = (info_nce_loss_utr5 + info_nce_loss_cds + info_nce_loss_utr3) / 3.0
            # warm up the contrastive_weight
            if self.config.training.contrastive_warmup_steps > 0 and self.trainer.training:
                contrastive_weight = (
                    min(1.0, self.trainer.global_step / self.config.training.contrastive_warmup_steps)
                    * self.config.training.contrastive_weight
                )
            else:
                contrastive_weight = self.config.training.contrastive_weight
            rna_lm_loss = rna_lm_loss + contrastive_weight * contrastive_loss
        else:
            contrastive_loss = None
            
        with torch.no_grad():
            perplexity_utr5 = torch.exp(rna_lm_loss_utr5)
            perplexity_codon = torch.exp(rna_lm_loss_codon)
            perplexity_utr3 = torch.exp(rna_lm_loss_utr3)
        
        return Loss(
            loss=rna_lm_loss,
            num_codon_aa_errors=n_errors,
            total_aa_length=total_length,
            rna_lm_loss_codon=rna_lm_loss_codon,
            rna_lm_loss_utr5=rna_lm_loss_utr5,
            rna_lm_loss_utr3=rna_lm_loss_utr3,
            perplexity_utr5=perplexity_utr5,
            perplexity_codon=perplexity_codon,
            perplexity_utr3=perplexity_utr3,
            divergence=divergence,
            contrastive_loss=contrastive_loss,
        )

    def _compute_loss(self, batch, prefix='train'):
        try:
            losses = self._loss(batch, prefix=prefix)
        except ValueError as e:
            print(e)
            print(batch['rna_input_ids'].shape)
            print(batch['protein_input_ids'].shape)
            # save the batch data
            with open("error_batch.pkl", "wb") as fp:
                pickle.dump(batch, fp)
            raise ValueError from e
        
        if prefix == 'train':
            on_step = True
            on_epoch = False
            batch_size = batch['rna_input_ids'].shape[0]
        else:
            on_step = False
            on_epoch = True
            batch_size = batch['rna_input_ids'].shape[0]
            
        
        self.log_dict(
            {
                f"{prefix}/loss": losses.loss,
                f"{prefix}/rna_lm_loss_codon": losses.rna_lm_loss_codon,
                f"{prefix}/rna_lm_loss_utr5": losses.rna_lm_loss_utr5,
                f"{prefix}/rna_lm_loss_utr3": losses.rna_lm_loss_utr3,
                f"{prefix}/perplexity_utr5": losses.perplexity_utr5,
                f"{prefix}/perplexity_codon": losses.perplexity_codon,
                f"{prefix}/perplexity_utr3": losses.perplexity_utr3,  
            },
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        
        if prefix == 'train':
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log(name=f'{prefix}/lr',
                    value=lr,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True,
                 )
        if prefix == 'val':
            if hasattr(losses, "num_codon_aa_errors"):
                self.codon_aa_errors.append(losses.num_codon_aa_errors)
            if hasattr(losses, "total_aa_length"):
                self.total_length.append(losses.total_aa_length)
        if self.config.training.kmer_alignment:
            self.log(
                name=f"{prefix}/kmer_js_divergence",
                value=losses.divergence,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size
            )
        if self.config.training.contrastive_training:
            self.log(
                name=f"{prefix}/contrastive_loss",
                value=losses.contrastive_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size
            )
        return losses
        
    def on_fit_start(self):
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()

    def on_train_epoch_start(self):
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
        self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
    
    def on_train_epoch_end(self):
        self.trainer.train_dataloader.batch_sampler.current_batch_idx = 0
    
    def training_step(self, batch, batch_idx):
        if self.config.backbone == 'hyena_nemo':
            torch.compiler.cudagraph_mark_step_begin()
        losses = self._compute_loss(batch, prefix='train')        
        return losses.loss

    def on_validation_epoch_start(self):
        self.backbone.eval()
        self.codon_aa_errors = []
        self.total_length = []

    def validation_step(self, batch, batch_idx):
        if self.config.backbone == 'hyena_nemo':
            torch.compiler.cudagraph_mark_step_begin()
        losses = self._compute_loss(batch, prefix='val')
        return losses.loss
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        self.trainer.fit_loop.setup_data()
        if self.trainer.train_dataloader is None or self.trainer.max_epochs is None:
            raise ValueError("Trainer must have a train_dataloader.")
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches)
        # Check if num_steps is zero
        if num_steps == 0:
            raise ValueError("num_steps is zero. Please check your training configuration.")
        return num_steps
    
    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        current_time = time.time()
        elapsed_time = current_time - self.launch_timestamp
        if len(self.total_length) >0 and len(self.codon_aa_errors) >0:
            try:
                total_errors = np.sum(self.codon_aa_errors)
                total_length = np.sum(self.total_length)
                aa_error_rate = total_errors / total_length
                self.log("validation/aa_error_rate", aa_error_rate, prog_bar=True, sync_dist=True)
            except:
                print("Error calculating aa_error_rate. Skipping logging.")
                pass
        if elapsed_time // 3600 > 1:
            self.trainer.should_stop = True
        self.backbone.train()
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()        
    
    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        # trainable_parameters = [p for p in  if p.requires_grad]
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
        trainable_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        number_of_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        print(f">>> Number of trainable parameters: {number_of_trainable_parameters}")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)        
        
        if self.trainer.max_epochs is not None:
            def warmup_lr_lambda(current_step):
                return min(1.0, current_step / max(1.0, self.config.optim.num_warmup_steps))
            
            # Here we iterate scheduler by each step, so we need to set last_epoch to global_step - 1
            # to make sure the first step is warmup
            global_step = self.trainer.global_step
            last_epoch = global_step - 1   
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda, last_epoch=last_epoch)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=self.num_training_steps(), 
                                                            eta_min=self.config.optim.min_learning_rate, 
                                                            last_epoch=last_epoch, 
                                                            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                            schedulers=[warmup_scheduler, 
                                                                            cosine_scheduler], 
                                                            last_epoch=last_epoch,
                                                            milestones=[self.config.optim.num_warmup_steps])
        else:
            raise ValueError(
                'Lightning currently does not support warmup with `max_epochs=None`.'
                'Please set `max_epochs` to a positive integer.')

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    def on_save_checkpoint(self, checkpoint):
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['_total'] = self.trainer.num_training_batches
        checkpoint['loops']['fit_loop'][
        'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total'][
                'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
        'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['current'][
                'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
        'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        # the batch index in data loader is ahead of the current batch idx because of the prefetch
        # we use the optimizer step count to correct for that
        dataloader_state = checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]
        dataloader_state['current_batch_idx'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0] = dataloader_state
    
    def on_load_checkpoint(self, checkpoint):
        print("loading checkpoint")
        self.resumed_dataloader_state_from_ckpt = checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]

    @torch.no_grad()
    def generate(
            self, 
            protein_sequence,
            prompt=None,
            max_generating_length=8000,
            global_tokenizer=None,
            codon_tokenizer=None,
            utr_tokenizer=None,
            temperature=1.0,
            progress_bar=True,
            cg=False,
            batch_size=1,
            cuda_monitor=False,
            expected_utr_5_length=None,
            expected_utr_3_length=None,
            species_id=None,
        ):
        
        """Generate RNA sequence from protein sequence."""
        self.eval()
        global_tokenizer = self.global_tokenizer if global_tokenizer is None else global_tokenizer
        codon_tokenizer = self.trainer.datamodule.codon_alphabet if codon_tokenizer is None else codon_tokenizer
        utr_tokenizer = self.trainer.datamodule.utr_alphabet if utr_tokenizer is None else utr_tokenizer
        protein_embeddings, prompt_input_ids = prepare_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
            self.protein_encoder,
            self.global_tokenizer,
            batch_size=batch_size,
            prompt=prompt,
            utr_5_tokenizer=utr_tokenizer,
            codon_tokenizer=codon_tokenizer,
        )
        input_ids = prompt_input_ids
        if species_id is not None:
            species_input_ids = torch.tensor([species_id], device=protein_embeddings.device, dtype=torch.int64).view(1, 1)
            species_input_ids = species_input_ids.repeat(batch_size, 1)
        else:
            species_input_ids = None
        ctokenizer = ConcatenatedAlphabet(
            [
                global_tokenizer, codon_tokenizer, utr_tokenizer
            ]
        )
        if self.config.backbone == 'mamba2':
            from jsm.generation.mamba_generation import decode
        else:
            from jsm.generation.flash_attention_generation import decode
        output = decode(
            input_ids,
            self.backbone,
            max_length=max_generating_length,
            eos_token_id=global_tokenizer.eos_idx,
            pad_token_id=global_tokenizer.padding_idx,
            top_k=4,
            top_p=0.0,
            min_p=0.0,
            temperature=temperature,
            # repetition_penalty=1.0,
            cg=cg,
            protein_embeddings=protein_embeddings,
            protein_sequences=[protein_sequence],
            species_ids=species_input_ids,
            progress_bar=progress_bar,
            cuda_monitor=cuda_monitor,
            expected_utr_5_length=expected_utr_5_length,
            expected_utr_3_length=expected_utr_3_length
        )
        batch_size = output.sequences.shape[0]
        sequence_list = []
        for seq_i in range(batch_size):
            sequence = []
            for token_id in output.sequences[seq_i].cpu().numpy().flatten():
                sequence.append(ctokenizer.decode(token_id.item()))
                if token_id == global_tokenizer.eos_idx:
                    break
            sequence_list.append("".join(sequence))
        results = {"sequence": sequence_list}
        results.update(output.metrics)
        return expand_list_in_dict(results)

    @torch.no_grad()
    def encode(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            sentence_level=False,
            style="average",
            hidden_layer_idx=None,
            species_id=None,
        ):
        tokens, protein_embeddings = tokenize_inputs(
                protein_sequence, 
                utr5_sequence,
                cds_sequence,
                utr3_sequence,
                self.protein_tokenizer,
                self.protein_encoder,
                self.global_tokenizer,
                self.codon_tokenizer,
                self.utr_5_tokenizer,
                self.utr_3_tokenizer
        )
        if species_id is not None:
            species_input_ids = torch.tensor([species_id], device=protein_embeddings.device, dtype=torch.int64).view(1, 1)
            species_input_ids = species_input_ids.repeat(1, 1)
            hidden_states = self.backbone.generating_forward(
                tokens,
                protein_embeddings,
                return_hidden_states=True,
                hidden_layer_idx=hidden_layer_idx,
                species_ids=species_input_ids,
            )
        else:
            species_input_ids = None
            hidden_states = self.backbone.generating_forward(
                    tokens,
                    protein_embeddings,
                    return_hidden_states=True,
                    hidden_layer_idx=hidden_layer_idx
            )
        
        seq_length = tokens.shape[1]
        hidden_states = hidden_states[:, -seq_length:, :]
        if sentence_level:
            if style == "average":
                # return the last hidden state
                return hidden_states.mean(dim=1)
            elif style == "concatenate":
                return format_embeddings(hidden_states, style="concatenate", seq_lens=[len(utr5_sequence), len(cds_sequence), len(utr3_sequence)])
            else:
                raise ValueError(f"Unknown style: {style}")
        else:
            return hidden_states
    
    @torch.no_grad()
    def get_logits(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=None,
        ):
        tokens, protein_embeddings = tokenize_inputs(
                protein_sequence, 
                utr5_sequence,
                cds_sequence,
                utr3_sequence,
                self.protein_tokenizer,
                self.protein_encoder,
                self.global_tokenizer,
                self.codon_tokenizer,
                self.utr_5_tokenizer,
                self.utr_3_tokenizer
        )
        if species_id is not None:
            species_input_ids = torch.tensor([species_id], device=protein_embeddings.device, dtype=torch.int64).view(1, 1)
            species_input_ids = species_input_ids.repeat(1, 1)
            logits = self.backbone.generating_forward(
                tokens,
                protein_embeddings,
                return_hidden_states=False,
                species_ids=species_input_ids,
            )
        else:
            species_input_ids = None
            logits = self.backbone.generating_forward(
                    tokens,
                    protein_embeddings,
                    return_hidden_states=False,
            )
        return logits
    
    def get_attention_weights(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            hidden_layer_idx=None
        ):
        tokens, protein_embeddings = tokenize_inputs(
                protein_sequence, 
                utr5_sequence,
                cds_sequence,
                utr3_sequence,
                self.protein_tokenizer,
                self.protein_encoder,
                self.global_tokenizer,
                self.codon_tokenizer,
                self.utr_5_tokenizer,
                self.utr_3_tokenizer
        )
        attention_weights = self.backbone.calculate_attention_weights(
                tokens,
                protein_embeddings,
        )
        return attention_weights
        
    
    @torch.no_grad()
    def score(
        self,
        protein_sequence, 
        utr5_sequence, 
        cds_sequence, 
        utr3_sequence,
    ):
        tokens, protein_embeddings = tokenize_inputs(
                protein_sequence, 
                utr5_sequence,
                cds_sequence,
                utr3_sequence,
                self.protein_tokenizer,
                self.protein_encoder,
                self.global_tokenizer,
                self.codon_tokenizer,
                self.utr_5_tokenizer,
                self.utr_3_tokenizer
        )
        
        logits = self.backbone.generating_forward(
                tokens,
                protein_embeddings,
                return_hidden_states=False,
                num_last_tokens=tokens.shape[1]
                
        )
        # import pdb; pdb.set_trace()
        tokens = tokens.squeeze(0)[1:]
        logits = logits.squeeze(0)[:-1]
        logp = torch.log_softmax(logits, dim=-1)  # (B, V)
        logp = logp.gather(1, tokens.unsqueeze(1)).squeeze(1)  # (B,)
        ppl = torch.exp(-logp.mean())  # (B,)
        return ppl
        
def expand_list_in_dict(d):
    all_keys = list(d.keys())
    length = len(d[all_keys[0]])
    rlist = []
    for i in range(length):
        rlist.append({k: d[k][i] for k in all_keys})
    return rlist
    

def format_embeddings(embeddings, style="average", seq_lens=None):
    """
    Format embedding of sequence to a fixed size embeddings on sequence level.
      - embeddings: [1, seq_len, dim]
    """
    if style == "average":
        return torch.mean(embeddings[:, 1:-1, :], dim=1)
    elif style == "concatenate":
        utr_5_len, cds_length, utr_3_len = seq_lens
        cds_length = cds_length // 3
        utr_5 = embeddings[:, 1:utr_5_len+3, :]
        cds = embeddings[:, utr_5_len+3:utr_5_len+cds_length+5, :]
        utr_3 = embeddings[:, -utr_3_len-3:-1, :]
        utr_5 = torch.mean(utr_5, dim=1)
        cds = torch.mean(cds, dim=1)
        utr_3 = torch.mean(utr_3, dim=1)
        return torch.cat([utr_5, cds, utr_3], dim=1)
    else:
        raise ValueError("Unknown aggregation style: {}".format(style))