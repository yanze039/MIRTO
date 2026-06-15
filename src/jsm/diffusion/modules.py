import torch
import torch.nn.functional as F
import jsm.diffusion.noise_schedule as noise_schedule
from jsm.models.transformer import SpeciesSpecificJointSequenceTransformer
from jsm.data.utils import codon_table
import jsm.utils as utils
import torch.nn as nn
import pickle
from jsm.diffusion.core import Diffusion, _sample_categorical
import lightning as L
from typing import Optional
from dataclasses import dataclass
import time
import numpy as np
import tqdm
import math
from jsm.data.utils import (esm_tokenize, modality_map, tokenize_cds_sequence, 
                            tokenize_utr_sequence, aminoacid_list, aminoacid_to_highly_used_codons)
from jsm.data.species_specific import tokenize_inputs
from jsm.diffusion.decoding_utils import DiffusionDecodingConstraint

LOG2 = math.log(2)
logger = utils.get_logger(__name__)

def hamming_distance_numpy(s1, s2):
    a = np.frombuffer(s1.encode(), dtype=np.uint8)
    b = np.frombuffer(s2.encode(), dtype=np.uint8)
    return np.count_nonzero(a != b)


def sequence_ablation(sequence, 
                      start=0, 
                      end=-1,
                      ablation_window=10,
                      ablation_token=''):
    if end == -1:
        end = len(sequence)
    ablated_sequences = {}
    for i in range(start, end, ablation_window):
        ablated_seq = list(sequence)
        for j in range(i, min(i+ablation_window, end)):
            ablated_seq[j] = ablation_token
        ablated_seq = "".join(ablated_seq)
        ablated_sequences[f"{i:04d}_{min(i+ablation_window, end):04d}"] = ablated_seq
    return ablated_sequences


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

class JointSequenceDiffusion(Diffusion):
    def __init__(
        self,
        config,
        global_tokenizer,
        protein_tokenizer,
        rna_vocab_size,
        protein_vocab_size,
        protein_encoder
    ):
        L.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.tokenizer = None
        self.config = config
        self.rna_vocab_size = rna_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.protein_tokenizer = protein_tokenizer
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        self.global_tokenizer = global_tokenizer
        self.padding_index = self.global_tokenizer.padding_idx
        self.mask_index = self.global_tokenizer.mask_idx
        self.cls_index = self.global_tokenizer.cls_idx
        self.eos_index = self.global_tokenizer.eos_idx
        self.unknown_index = self.global_tokenizer.unk_idx
        self.N_index = self.global_tokenizer.tok_to_idx.get('N', self.global_tokenizer.unk_idx)
        self.parameterization = self.config.parameterization
        self.backbone = SpeciesSpecificJointSequenceTransformer(
            self.config.model,
            rna_vocab_size,
            protein_vocab_size,
        )
        self.T = 0

        self.subs_masking = self.config.subs_masking
        self.subs_masking = False

        self.softplus = torch.nn.Softplus()
        self.eval_model_tokenizer = self.tokenizer
        self.noise = noise_schedule.get_noise(self.config,
                                            dtype=self.dtype)
        self.ema = False
    
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self.consumed_batch_keys = []
        self._validate_configuration()
        
        self.protein_encoder = protein_encoder
        self.protein_encoder.eval()
        self.codon_aa_errors = []
        self.total_length = []
    
        # freeze protein encoder
        if self.protein_encoder is not None:
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
        
        self.translation_lm_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.launch_timestamp = time.time()
        self.resumed_dataloader_state_from_ckpt = None
        self.score_cg = None
    
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
    
    def forward(self, 
                x, 
                batch,
        ):
        """Returns log score."""
        with torch.no_grad():
            protein_output = self.protein_encoder(batch["protein_input_ids"])
            protein_embeddings = protein_output.embeddings
        rna_lm_logits = self.backbone(
            input_ids=x,
            species_ids=batch["species_ids"],                 
            protein_embeddings=protein_embeddings,
            row_wise_col_perms=batch["row_wise_col_perms"],
            inverse_row_wise_col_perms=batch["inverse_row_wise_col_perms"],
            attention_mask=batch["attention_mask"],
            modality_type_ids=batch['modality_type_ids'],
            modality_mask=batch['modality_mask'],
        )
        rna_lm_logits = self.process_parameterization(
            logits=rna_lm_logits,
            xt=x,
        )
        return rna_lm_logits        
    
    
    def _forward_pass_diffusion(self, x0, t, batch):
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])   
        # respect the special tokens
        move_chance = move_chance.repeat(1, x0.shape[1])
        move_chance[batch["special_token_mask"].bool()] = 0.0
        xt = self.q_xt(x0, move_chance)
        rna_lm_logits = self.forward(
                        xt, 
                        batch, 
        )
        
        if not self.trainer.training:
            codon_logits = torch.argmax(rna_lm_logits, dim=-1)[batch["translation_rna_mask"].bool()]
            codon_tokenizer = self.trainer.datamodule.codon_alphabet
            aa_list = []
            for clogit in codon_logits:
                codon = codon_tokenizer.get_tok(clogit.item())
                aa = codon_table.get(codon, "-")
                aa_list.append(aa)
            aa_list = "".join(aa_list)
            full_protein_sequence = "".join(batch["protein_sequence"])
            assert len(aa_list) == len(full_protein_sequence)
            n_errors = hamming_distance_numpy(aa_list, full_protein_sequence)
            total_length = len(full_protein_sequence)

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=rna_lm_logits,
            dim=-1,
            index=x0[:, :, None]
        ).squeeze(-1)        
        rna_lm_loss =  - log_p_theta * (
            dsigma / torch.expm1(sigma))[:, None]

        return {
            "rna_lm_loss": rna_lm_loss,
            "n_errors": n_errors if not self.trainer.training else 0,
            "total_length": total_length if not self.trainer.training else 0
        }
    
    
    def _loss(self, batch):
        rna_padding_mask = batch['rna_padding_mask']
        
        input_tokens = batch['rna_input_ids']
        t = self._sample_t(input_tokens.shape[0], input_tokens.device)
        outputs = self._forward_pass_diffusion(input_tokens, t, batch)        

        nlls = outputs["rna_lm_loss"] * rna_padding_mask
        # count = rna_padding_mask.sum()
        # batch_nll = nlls.sum()
        # token_nll = batch_nll / count
        
        # utr5 nlls
        utr5_nlls = (nlls * batch['utr5_mask']).sum()
        utr5_count = batch['utr5_mask'].sum()
        utr5_token_nll = utr5_nlls / utr5_count
        
        # codon nlls
        cds_nlls = (nlls * batch['cds_mask']).sum()
        cds_count = batch['cds_mask'].sum()
        cds_token_nll = cds_nlls / cds_count
        
        # utr3 nlls
        utr3_nlls = (nlls * batch['utr3_mask']).sum()
        utr3_count = batch['utr3_mask'].sum()
        utr3_token_nll = utr3_nlls / utr3_count
        
        mean_token_nll = (utr5_token_nll + cds_token_nll + utr3_token_nll) / 3.0
        
        with torch.no_grad():
            perplexity_utr5 = torch.exp(utr5_token_nll)
            perplexity_codon = torch.exp(cds_token_nll)
            perplexity_utr3 = torch.exp(utr3_token_nll)
            perplexity = torch.exp(mean_token_nll)
        
        return Loss(loss=mean_token_nll,
                    rna_lm_loss=mean_token_nll.detach(),
                    rna_lm_loss_codon=cds_token_nll.detach(),
                    rna_lm_loss_utr5=utr5_token_nll.detach(),
                    rna_lm_loss_utr3=utr3_token_nll.detach(),
                    perplexity=perplexity,
                    perplexity_utr5=perplexity_utr5,
                    perplexity_utr3=perplexity_utr3,
                    perplexity_codon=perplexity_codon,
                    num_codon_aa_errors=outputs["n_errors"],
                    total_aa_length=outputs["total_length"]
                    )

    def _compute_loss(self, batch, prefix):
        try:
            losses = self._loss(batch)
        except ValueError as e:
            print(batch['rna_input_ids'].shape)
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
    
        return losses
    
    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='train')
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(name='trainer/lr',
                 value=lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return losses.loss

    def on_validation_epoch_start(self):
        self.backbone.eval()
        self.noise.eval()
        self.codon_aa_errors = []
        self.total_length = []

    def validation_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='val')
        if hasattr(losses, "num_codon_aa_errors"):
            self.codon_aa_errors.append(losses.num_codon_aa_errors)
        if hasattr(losses, "total_aa_length"):
            self.total_length.append(losses.total_aa_length)
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
        if elapsed_time // 3600 > 2.5:
            self.trainer.should_stop = True
        self.backbone.train()
        if self.protein_encoder is not None:
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
            self.protein_encoder.eval()        
    
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
        if "combined_loader" in checkpoint['loops']['fit_loop']['state_dict']:
            self.resumed_dataloader_state_from_ckpt = checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]
        else:
            self.resumed_dataloader_state_from_ckpt = None
            print("Warning: combined_loader not found in checkpoint. Dataloader state will not be restored.")
    
    
    def _generating_forward(
            self,
            x,
            input_batch,
            cg=None,
            allowed_token_mask=None,
        ):
        if cg is not None:
            rna_lm_logits = cg(input_ids=x)
        else:
            rna_lm_logits = self.backbone(
                    input_ids=x, 
                    protein_embeddings=input_batch["protein_embeddings"],
                    row_wise_col_perms=input_batch["row_wise_col_perms"],
                    inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"],
                    attention_mask=input_batch["attention_mask"],
                    species_ids=input_batch["species_ids"],
                    modality_type_ids=input_batch["modality_type_ids"],
                    modality_mask=input_batch["modality_mask"]
                )
        rna_lm_logits = self.process_parameterization(
            logits=rna_lm_logits,
            xt=x,
            allowed_token_mask=allowed_token_mask
        )
        return rna_lm_logits
    
    def process_parameterization(self, logits, xt, allowed_token_mask=None):
        neg_infinity = -torch.finfo(logits.dtype).max
        logits[:, :, self.mask_index] = neg_infinity
        if allowed_token_mask is not None:
            logits[:, ~allowed_token_mask] = neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1,
                                            keepdim=True)

        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    
    
    def _sample_prior(self, batch_size, seq_length, utr_5_length=0, utr_3_length=0, cds_length=0):
        raw_prior = self.mask_index * torch.ones(batch_size, seq_length, dtype=torch.int64)
        raw_prior[:, 0] = self.global_tokenizer.cls_idx
        raw_prior[:, 1] = self.global_tokenizer.get_idx("<utr_5_bos>")
        raw_prior[:, 2+utr_5_length] = self.global_tokenizer.get_idx("<utr_5_eos>")
        raw_prior[:, 3+utr_5_length] = self.global_tokenizer.get_idx("<cds_bos>")
        raw_prior[:, 4+utr_5_length+cds_length] = self.global_tokenizer.get_idx("<cds_eos>")
        raw_prior[:, 5+utr_5_length+cds_length] = self.global_tokenizer.get_idx("<utr_3_bos>")
        raw_prior[:, 6+utr_5_length+cds_length+utr_3_length] = self.global_tokenizer.get_idx("<utr_3_eos>")
        raw_prior[:, 7+utr_5_length+cds_length+utr_3_length] = self.global_tokenizer.eos_idx
        return raw_prior
    
    @torch.no_grad()
    def sample(self, 
                protein_sequence, 
                utr5_length, 
                utr3_length, 
                species_id=85, 
                num_steps=None, 
                eps=1e-5,
                use_cg=False,
                prompt=None,
                codon_tokenizer=None,
                utr_5_tokenizer=None,
                utr_3_tokenizer=None,
        ):
        
        self.backbone.eval()
        batch_size_per_gpu = self.config.sampling.batch_size
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        cds_length = len(protein_sequence)+1
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        sequence_length = (8 + utr5_length + utr3_length + len(protein_sequence)+1)
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )
            
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64).to(self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        x = self._sample_prior(
            batch_size = batch_size_per_gpu,
            seq_length = sequence_length,
            utr_5_length=utr5_length,
            utr_3_length=utr3_length,
            cds_length=len(protein_sequence)+1,
            ).to(self.device)
        
        if prompt is not None:
            if "utr5_sequence" in prompt:
                utr_5_sequence = prompt["utr5_sequence"]
                assert len(utr_5_sequence) <= utr5_length, f"Provided utr5_sequence length {len(utr_5_sequence)} exceeds specified utr5_length {utr5_length}."
                utr_5_input_ids = tokenize_utr_sequence(utr_5_sequence, utr_5_tokenizer, mask_token_id=self.global_tokenizer.get_idx("<mask>")).to(self.device)
                x[:, 2:2+len(utr_5_input_ids)] = utr_5_input_ids
            if "utr3_sequence" in prompt:
                utr_3_sequence = prompt["utr3_sequence"]
                assert len(utr_3_sequence) <= utr3_length, f"Provided utr3_sequence length {len(utr_3_sequence)} exceeds specified utr3_length {utr3_length}."
                utr_3_input_ids = tokenize_utr_sequence(utr_3_sequence, utr_3_tokenizer, mask_token_id=self.global_tokenizer.get_idx("<mask>")).to(self.device)
                x[:, 6+utr5_length+cds_length:6+utr5_length+cds_length+len(utr_3_input_ids)] = utr_3_input_ids
            if "cds_sequence" in prompt:
                cds_sequence = prompt["cds_sequence"]
                assert len(cds_sequence)//3 <= cds_length, f"Provided cds_sequence length {len(cds_sequence)} exceeds specified cds_length {cds_length}."
                cds_input_ids = tokenize_cds_sequence(cds_sequence, codon_tokenizer).to(self.device)
                x[:, 4+utr5_length:4+utr5_length+len(cds_input_ids)] = cds_input_ids
        
        input_batch = self.preprocess(
            rna_input_ids=x,
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        input_batch['modality_type_ids'] = modality_type_ids
        input_batch['species_ids'] = species_ids
        if use_cg:
            cg = CUDAGraphForward(self.backbone, input_batch)
            cg.capture(example_input_ids=x)
        else:
            cg = None
        
        constraint = DiffusionDecodingConstraint()
        allowed_token_mask = constraint.get_allowed_token_mask(
            protein_sequence=protein_sequence,
            utr5_length=utr5_length,
            utr3_length=utr3_length
        )
        
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        logger.info(f"starting sampling using sampler [{self.sampler}] | CudaGraph: {use_cg} | num_steps: {num_steps}")
        assert self.sampler == 'ddpm_cache', "Currently, only ddpm_cache sampler is implemented. Please set sampler to 'ddpm_cache' in the config or implement the ddpm sampler."
        for i in tqdm.tqdm(range(num_steps), desc=f"[{self.sampler}] Sampling", unit="step", leave=False):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                raise NotImplementedError("ddpm sampler is not implemented yet. Please use ddpm_cache or analytic sampler.")
                x = self._ddpm_update(x, t, dt,
                                      protein_input_ids=protein_input_ids,
                    species_ids=species_ids,
                    modality_input_ids=modality_type_ids)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache, 
                    input_batch=input_batch,
                    cg=cg,
                    allowed_token_mask=allowed_token_mask
                    )
                if (not torch.allclose(x_next, x)
                    or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                            device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                x = self._generating_forward(x, input_batch, cg=cg).argmax(dim=-1)
        return x
    
    
    
    def _ddpm_caching_update(
                self, 
                x, 
                t, 
                dt, 
                p_x0=None,
                input_batch=None,   
                cg=None,
                allowed_token_mask=None
        ):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self._generating_forward(x, input_batch, cg=cg, allowed_token_mask=allowed_token_mask).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt, 
                     input_batch=None,
                     cg=None,
                     allowed_token_mask=None
                     ):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        log_p_x0 = self._generating_forward(x,
                                           input_batch=input_batch,
                                             cg=cg,
                                             allowed_token_mask=allowed_token_mask
                                        )
        # mask out <unknown>, <bos> and <cls> after the first token
        log_p_x0[:, 1:, self.cls_index] = -torch.finfo(log_p_x0.dtype).max
        log_p_x0[:, :, self.unknown_index] = -torch.finfo(log_p_x0.dtype).max
        log_p_x0[:, :, self.N_index] = -torch.finfo(log_p_x0.dtype).max
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x
    
    
    
    def preprocess(
            self,
            rna_input_ids, 
            protein_input_ids,
            species_ids,
            modality_input_ids
        ):
        batch_size = rna_input_ids.shape[0]
        rna_padding_mask = rna_input_ids.ne(self.global_tokenizer.padding_idx).to(torch.long).to(self.device)
        protein_padding_mask = protein_input_ids.ne(self.protein_tokenizer.vocab["<pad>"]).to(torch.long).to(self.device).repeat(batch_size, 1)
        species_padding_mask = torch.ones((batch_size, 1), dtype=torch.long).to(self.device)
        L = rna_padding_mask.shape[1] + protein_padding_mask.shape[1] + species_ids.shape[1]
        joint_masking = torch.cat([species_padding_mask, protein_padding_mask, rna_padding_mask], dim=1)        
        arange_tensor = torch.arange(L).unsqueeze(0).expand(batch_size, L).to(self.device)
        product = joint_masking * (arange_tensor + 1)
        product[product == 0] = L + 1
        row_wise_col_perms = torch.argsort(product, dim=1, descending=False, stable=True).to(self.device)
        inverse_indices = torch.empty_like(row_wise_col_perms).to(self.device)
        inverse_indices.scatter_(1, row_wise_col_perms, arange_tensor)
        attention_mask = torch.gather(joint_masking, dim=1, index=row_wise_col_perms).to(torch.int64)
        utr5_mask = (modality_input_ids == modality_map["utr_5"]).to(torch.long)
        utr3_mask = (modality_input_ids == modality_map["utr_3"]).to(torch.long)
        cds_mask = (modality_input_ids == modality_map["cds"]).to(torch.long)
        modality_mask = utr5_mask + utr3_mask + cds_mask
        
        protein_embeddings = self.protein_encoder(protein_input_ids).embeddings
        protein_embeddings = protein_embeddings.repeat(batch_size, 1, 1).to(self.device)
        return {
            "row_wise_col_perms": row_wise_col_perms,
            "inverse_row_wise_col_perms": inverse_indices,
            "attention_mask": attention_mask,
            "modality_mask": modality_mask,
            "protein_embeddings": protein_embeddings
        }
    
    
    def create_modality_type_tensor(
            self,
            batch_size,
            utr5_length,
            utr3_length,
            cds_length
        ):
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        modality_type_ids = torch.ones(batch_size, sequence_length, dtype=torch.int64).to(self.device)
        modality_type_ids[:, 0] = modality_map["global_special_tokens"]
        modality_type_ids[:, 1:3+utr5_length] = modality_map["utr_5"]
        modality_type_ids[:, 3+utr5_length:5+utr5_length+cds_length] = modality_map["cds"]
        modality_type_ids[:, 5+utr5_length+cds_length:7+utr5_length+cds_length+utr3_length] = modality_map["utr_3"]
        modality_type_ids[:, 7+utr5_length+cds_length+utr3_length:] = modality_map["global_special_tokens"]
        return modality_type_ids
    
    
    @torch.no_grad()
    def score_per_position(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
            score_batch_size=4,
            use_cg=True,
            retain_cg=False,
            regions_to_score=["utr5", "cds", "utr3"],
            update_protein=False
        ):
        """Compute the log-probability of a given RNA sequence conditioned on the protein sequence."""
        
        def _get_block_indices(block_idx, max_position, offset=0):
            start_idx = block_idx * score_batch_size
            end_idx = min((block_idx + 1) * score_batch_size, max_position)
            return start_idx+offset, end_idx+offset
        rna_input_ids, _ = tokenize_inputs(
            protein_sequence, 
            utr5_sequence=utr5_sequence,
            cds_sequence=cds_sequence,
            utr3_sequence=utr3_sequence,
            protein_tokenizer=self.protein_tokenizer,
            protein_encoder=self.protein_encoder,
            global_tokenizer=self.global_tokenizer,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            complete_sequence=True
        )
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        
        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        
        rna_input_ids = rna_input_ids.repeat(score_batch_size, 1).to(self.device)
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = score_batch_size
        
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64, device=self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        input_batch['modality_type_ids'] = modality_type_ids
        input_batch['species_ids'] = species_ids
        if use_cg:
            if retain_cg and self.score_cg is not None:
                cg = self.score_cg
            else:
                cg = CUDAGraphForward(self.backbone, input_batch, update_protein=update_protein)
                cg.capture(example_input_ids=rna_input_ids)
                if retain_cg:
                    self.score_cg = cg
        else:
            cg = None
        
        constraint = DiffusionDecodingConstraint()
        allowed_token_mask = constraint.get_allowed_token_mask(
            protein_sequence=protein_sequence,
            utr5_length=utr5_length,
            utr3_length=utr3_length
        )
        
        def _calculate_region_log_probs(region_length, region_offset, region_name):
            region_scores = torch.zeros(region_length, dtype=torch.float32).to(self.device)
            region_entropy = torch.zeros(region_length, dtype=torch.float32).to(self.device)
            n_blocks = (region_length + score_batch_size -1) // score_batch_size
            if region_length < score_batch_size:
                print(f"Warning: {region_name} sequence length {region_length} is smaller than score_batch_size {score_batch_size}. Consider reducing score_batch_size for faster scoring.")
            for block_idx in range(n_blocks):
                start_pos, end_pos = _get_block_indices(block_idx, region_length, offset=region_offset)
                current_batch_size = end_pos - start_pos
                corrupted_rna_input_ids = rna_input_ids[:current_batch_size].clone()
                column_indices = torch.arange(start_pos, end_pos, device=self.device)
                row_indices = torch.arange(current_batch_size).to(self.device)
                corrupted_rna_input_ids[row_indices, column_indices] = self.mask_index
                if use_cg:
                    size_match_cg = (corrupted_rna_input_ids.shape == cg.input_size)
                    # if not size_match_cg:
                    #     print(f"Warning: Corrupted input size {corrupted_rna_input_ids.shape} does not match CUDAGraph input size {cg.input_size}. Disabling CUDAGraph for this scoring.")
                if use_cg and score_batch_size == current_batch_size and size_match_cg:
                    rna_lm_logits = cg(corrupted_rna_input_ids, protein_embeddings=input_batch["protein_embeddings"])
                else:
                    rna_lm_logits = self.backbone(
                                    input_ids=corrupted_rna_input_ids, 
                                    protein_embeddings=input_batch["protein_embeddings"][:current_batch_size],
                                    row_wise_col_perms=input_batch["row_wise_col_perms"][:current_batch_size],
                                    inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"][:current_batch_size],
                                    attention_mask=input_batch["attention_mask"][:current_batch_size],
                                    species_ids=species_ids[:current_batch_size],
                                    modality_type_ids=modality_type_ids[:current_batch_size],
                                    modality_mask=input_batch["modality_mask"][:current_batch_size]
                                )
                rna_lm_logits = self.process_parameterization(
                        logits=rna_lm_logits,
                        xt=corrupted_rna_input_ids,
                        allowed_token_mask=allowed_token_mask
                    )
                logits_of_interest = rna_lm_logits[row_indices, column_indices]
                log_probs = logits_of_interest.log_softmax(dim=-1)
                
                # score
                original_tokens = rna_input_ids[0, column_indices]
                logp = log_probs[row_indices, original_tokens]
                region_scores[start_pos - region_offset:end_pos - region_offset] = logp
                
                # entropy
                probs = logits_of_interest.softmax(dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                region_entropy[start_pos - region_offset:end_pos - region_offset] = entropy
            return {
                "score": region_scores,
                "entropy": region_entropy
            }
        
        final_scores = {}
        if "utr5" in regions_to_score:
            utr5_perposition_scores = _calculate_region_log_probs(
                region_length=len(utr5_sequence),
                region_offset=2,
                region_name="UTR5"
            )
            final_scores['utr5'] = {
                "score": utr5_perposition_scores['score'].cpu().numpy(),
                "entropy": utr5_perposition_scores['entropy'].cpu().numpy()
            }
            
        if "cds" in regions_to_score:
            cds_perposition_scores = _calculate_region_log_probs(
                region_length=cds_length,
                region_offset=4 + utr5_length,
                region_name="CDS"
            )
            final_scores['cds'] = {
                "score": cds_perposition_scores['score'].cpu().numpy(),
                "entropy": cds_perposition_scores['entropy'].cpu().numpy()
            }
        
        if "utr3" in regions_to_score:
            
            utr3_perposition_scores = _calculate_region_log_probs(
                region_length=len(utr3_sequence),
                region_offset=6 + utr5_length + cds_length,
                region_name="UTR3"
            )
            final_scores['utr3'] = {
                "score": utr3_perposition_scores['score'].cpu().numpy(),
                "entropy": utr3_perposition_scores['entropy'].cpu().numpy()
            }
        
        return final_scores
    
    @torch.no_grad()
    def protein_motif_attribution(
            self,
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
            score_batch_size=4,
            use_cg=True,
            regions_to_score=["utr5", "cds", "utr3"],
            ablation_window=30,
            ablation_method="mask",
            ablation_mutation_token=None
        ):
        
        assert len(protein_sequence) == len(cds_sequence)//3 - 1
            
        scores_without_ablation = self.score_per_position(
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=species_id,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            score_batch_size=score_batch_size,
            use_cg=use_cg,
            retain_cg=False,
            regions_to_score=["utr3"],
            update_protein=True
        )        
        
        if ablation_method == "mask":
            ablated_protein = sequence_ablation(protein_sequence, 
                        start=0, 
                        end=-1,
                        ablation_window=ablation_window,
                        ablation_token=self.protein_tokenizer.mask_token)
        elif ablation_method == "delete":
            ablated_protein = sequence_ablation(protein_sequence, 
                        start=0, 
                        end=-1,
                        ablation_window=ablation_window,
                        ablation_token="")
        elif ablation_method == "mutate":
            assert ablation_mutation_token is not None, "Please provide the ablation_mutation_token for mutation-based ablation."
            ablated_protein = sequence_ablation(protein_sequence, 
                        start=0, 
                        end=-1,
                        ablation_window=ablation_window,
                        ablation_token=ablation_mutation_token)
        else:
            raise NotImplementedError(f"Ablation method {ablation_method} not supported. Please choose from 'mask', 'delete' or 'mutate'.")
        ablation_scores = {}
        ablation_ids = sorted(list(ablated_protein.keys()), reverse=False)
        for ablation_id in tqdm.tqdm(ablation_ids):
            ablated_protein_sequence = ablated_protein[ablation_id]
            ablation_start, ablation_end = [int(x) for x in ablation_id.split("_")]
            if ablation_method == "mask":
                ablated_cds_sequence = cds_sequence[0:ablation_start*3] + '_'*(ablation_end-ablation_start)*3 + cds_sequence[ablation_end*3:]
            elif ablation_method == "delete":
                ablated_cds_sequence = cds_sequence[0:ablation_start*3] + cds_sequence[ablation_end*3:]
            elif ablation_method == "mutate":
                raise NotImplementedError("Mutation-based ablation for CDS is not implemented yet. Please use 'mask' or 'delete' ablation methods for now.")
            scores_with_ablation = self.score_per_position(
                ablated_protein_sequence, 
                utr5_sequence, 
                ablated_cds_sequence, 
                utr3_sequence,
                species_id=species_id,
                codon_tokenizer=codon_tokenizer,
                utr_5_tokenizer=utr_5_tokenizer,
                utr_3_tokenizer=utr_3_tokenizer,
                score_batch_size=score_batch_size,
                use_cg=use_cg,
                retain_cg=True,
                regions_to_score=["utr3"],
                update_protein=True
            )
            # compute the difference in scores and attribute to the ablated region
            
            score_difference = scores_without_ablation['utr3']['score'].mean() - scores_with_ablation['utr3']['score'].mean()
            ablation_scores[ablation_id] = score_difference.item()

        # destroy cuda graph of scoring
        if self.score_cg is not None:
            del self.score_cg
            self.score_cg = None
        return ablation_scores
    
    @torch.no_grad()
    def calculate_categorical_jacobian_protein(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
            use_cg=True,
            retain_cg=False,
            protein_residue_indices=None,
            utr3_residue_indices=None
        ):
        """Compute the log-probability of a given RNA sequence conditioned on the protein sequence."""
        
        # aminoacid_list
        score_batch_size = len(aminoacid_list)
        codon_list = [ aminoacid_to_highly_used_codons[aa].replace('U', 'T') for aa in aminoacid_list ]
        codon_token_indices = torch.tensor([ codon_tokenizer.get_idx(codon) for codon in codon_list ], dtype=torch.int64).to(self.device)
        rna_vocab_indices = torch.tensor([utr_5_tokenizer.get_idx(nt) for nt in ['A', 'U', 'C', 'G']], dtype=torch.int64).to(self.device)
        
        rna_input_ids, _ = tokenize_inputs(
            protein_sequence, 
            utr5_sequence=utr5_sequence,
            cds_sequence=cds_sequence,
            utr3_sequence=utr3_sequence,
            protein_tokenizer=self.protein_tokenizer,
            protein_encoder=self.protein_encoder,
            global_tokenizer=self.global_tokenizer,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            complete_sequence=True
        )
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        
        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        
        rna_input_ids = rna_input_ids.repeat(score_batch_size, 1).to(self.device)
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = score_batch_size
        
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64, device=self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        input_batch['modality_type_ids'] = modality_type_ids
        input_batch['species_ids'] = species_ids
        if use_cg:
            if retain_cg and self.score_cg is not None:
                cg = self.score_cg
            else:
                cg = CUDAGraphForward(self.backbone, input_batch, update_protein=True)
                cg.capture(example_input_ids=rna_input_ids)
                if retain_cg:
                    self.score_cg = cg
        else:
            cg = None
        
        constraint = DiffusionDecodingConstraint()
        allowed_token_mask = constraint.get_allowed_token_mask(
            protein_sequence=protein_sequence,
            utr5_length=utr5_length,
            utr3_length=utr3_length
        )
        if protein_residue_indices is None:
            protein_residue_indices = list(range(len(protein_sequence)))
        if utr3_residue_indices is None:
            utr3_residue_indices = list(range(len(utr3_sequence)))
            utr3_residue_indices_tensor_with_offset = torch.tensor(utr3_residue_indices, device=self.device) + 6 + utr5_length + cds_length
        
        utr3_masked_rna_input_ids = rna_input_ids.clone()
        utr3_start = 6 + utr5_length + cds_length
        utr3_end = utr3_start + len(utr3_sequence)
        utr3_masked_rna_input_ids[:, utr3_start:utr3_end] = self.global_tokenizer.mask_idx
        
        raw_rna_lm_logits = cg(rna_input_ids, protein_embeddings=input_batch["protein_embeddings"])
        raw_rna_lm_logits = self.process_parameterization(
                    logits=raw_rna_lm_logits,
                    xt=utr3_masked_rna_input_ids,
                    allowed_token_mask=allowed_token_mask
        )
        log_raw_rna_lm_logits = raw_rna_lm_logits[:,:,rna_vocab_indices].log_softmax(dim=-1)
        jacobian = torch.zeros(
            (len(protein_residue_indices), 20, len(utr3_residue_indices), 4), dtype=torch.bfloat16
        ).to(self.device)
        
        _protein_sequence_list = list(protein_sequence)
        def _calculate_region_log_probs(mutation_idx):
            mutated_protein_sequence_list = []
            for aa in aminoacid_list:
                _mutated_protein_sequence = _protein_sequence_list.copy()
                _mutated_protein_sequence[mutation_idx] = aa
                mutated_protein_sequence_str = "".join(_mutated_protein_sequence)
                mutated_protein_sequence_list.append(mutated_protein_sequence_str)
            mutated_protein_input_ids = self.prepare_protein_inputs_for_model(
                mutated_protein_sequence_list,
                self.protein_tokenizer,
            ).to(self.device)
            mutated_protein_embeddings_mi = self.protein_encoder(mutated_protein_input_ids).embeddings
            corrupted_rna_input_ids = rna_input_ids.clone()
            codon_mutation_idx = 4 + utr5_length + mutation_idx
            corrupted_rna_input_ids[:, codon_mutation_idx] = codon_token_indices
            
            if use_cg:
                size_match_cg = (corrupted_rna_input_ids.shape == cg.input_size)
                # if not size_match_cg:
                #     print(f"Warning: Corrupted input size {corrupted_rna_input_ids.shape} does not match CUDAGraph input size {cg.input_size}. Disabling CUDAGraph for this scoring.")
            if use_cg and size_match_cg:
                rna_lm_logits = cg(corrupted_rna_input_ids, protein_embeddings=mutated_protein_embeddings_mi)
            else:
                rna_lm_logits = self.backbone(
                                input_ids=corrupted_rna_input_ids, 
                                protein_embeddings=mutated_protein_embeddings_mi,
                                row_wise_col_perms=input_batch["row_wise_col_perms"],
                                inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"],
                                attention_mask=input_batch["attention_mask"],
                                species_ids=species_ids,
                                modality_type_ids=modality_type_ids,
                                modality_mask=input_batch["modality_mask"]
                            )
            rna_lm_logits = self.process_parameterization(
                    logits=rna_lm_logits,
                    xt=utr3_masked_rna_input_ids,
                    allowed_token_mask=allowed_token_mask
                )
            log_probs = rna_lm_logits[:,:,rna_vocab_indices].log_softmax(dim=-1)
            log_probs_diff = log_probs - log_raw_rna_lm_logits
            log_probs_diff = log_probs_diff[:,utr3_residue_indices_tensor_with_offset,:]
            return log_probs_diff
        
        for i, mutation_idx in enumerate(protein_residue_indices):
            log_probs_diff = _calculate_region_log_probs(mutation_idx)
            jacobian[i] = log_probs_diff
            
        
        # Mean-centering: To remove that background signal, subtract the mean across every dimension.
        for dim in range(4):
            # why center?
            jacobian = jacobian - jacobian.mean(dim=dim, keepdim=True)
        # Symmetrization: To further reduce noise, symmetrize the matrix by averaging it with its transpose.
        # But we can't do this because we are uni-direction from protein to 3' UTR.
        # jacobian = (jacobian + jacobian.transpose(0, 2)) / 2
        # Frobenius norm across amino-acid channel
        interaction_map = torch.norm(jacobian, dim=(1,3))
        # apply APC
        row_sum = interaction_map.sum(dim=1, keepdim=True)
        col_sum = interaction_map.sum(dim=0, keepdim=True)
        total_sum = interaction_map.sum()
        apc_matrix = row_sum * col_sum / total_sum
        apc_interaction = interaction_map - apc_matrix
        return apc_interaction.float().cpu().numpy()  
    
    @torch.no_grad()
    def calculate_categorical_jacobian_protein_utr(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
            use_cg=True,
            retain_cg=False,
            protein_residue_indices=None,
        ):
        """Compute the log-probability of a given RNA sequence conditioned on the protein sequence."""
        
        # aminoacid_list
        score_batch_size = len(aminoacid_list)
        codon_list = [ aminoacid_to_highly_used_codons[aa].replace('U', 'T') for aa in aminoacid_list ]
        codon_token_indices = torch.tensor([ codon_tokenizer.get_idx(codon) for codon in codon_list ], dtype=torch.int64).to(self.device)
        rna_vocab_indices = torch.tensor([utr_5_tokenizer.get_idx(nt) for nt in ['A', 'U', 'C', 'G']], dtype=torch.int64).to(self.device)
        
        rna_input_ids, _ = tokenize_inputs(
            protein_sequence, 
            utr5_sequence=utr5_sequence,
            cds_sequence=cds_sequence,
            utr3_sequence=utr3_sequence,
            protein_tokenizer=self.protein_tokenizer,
            protein_encoder=self.protein_encoder,
            global_tokenizer=self.global_tokenizer,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            complete_sequence=True
        )
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        
        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        
        rna_input_ids = rna_input_ids.repeat(score_batch_size, 1).to(self.device)
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = score_batch_size
        
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64, device=self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        input_batch['modality_type_ids'] = modality_type_ids
        input_batch['species_ids'] = species_ids
        if use_cg:
            if retain_cg and self.score_cg is not None:
                cg = self.score_cg
            else:
                cg = CUDAGraphForward(self.backbone, input_batch, update_protein=True)
                cg.capture(example_input_ids=rna_input_ids)
                if retain_cg:
                    self.score_cg = cg
        else:
            cg = None
        
        constraint = DiffusionDecodingConstraint()
        allowed_token_mask = constraint.get_allowed_token_mask(
            protein_sequence=protein_sequence,
            utr5_length=utr5_length,
            utr3_length=utr3_length
        )
        if protein_residue_indices is None:
            protein_residue_indices = list(range(len(protein_sequence)))
            
        utr3_residue_indices = list(range(len(utr3_sequence)))
        utr3_residue_indices_tensor_with_offset = \
            torch.tensor(utr3_residue_indices, device=self.device) + 6 + utr5_length + cds_length
        utr3_start = 6 + utr5_length + cds_length
        utr3_end = utr3_start + len(utr3_sequence)
        utr5_residue_indices = list(range(len(utr5_sequence)))
        utr5_residue_indices_tensor_with_offset = \
            torch.tensor(utr5_residue_indices, device=self.device) + 2
        utr5_start = 2
        utr5_end = utr5_start + len(utr5_sequence)
        utr_residue_indices_tensor = torch.cat([utr5_residue_indices_tensor_with_offset, utr3_residue_indices_tensor_with_offset], dim=0)

        utr_masked_rna_input_ids = rna_input_ids.clone()
        utr_masked_rna_input_ids[:, utr5_start:utr5_end] = self.global_tokenizer.mask_idx
        utr_masked_rna_input_ids[:, utr3_start:utr3_end] = self.global_tokenizer.mask_idx
        
        raw_rna_lm_logits = cg(rna_input_ids, protein_embeddings=input_batch["protein_embeddings"])
        raw_rna_lm_logits = self.process_parameterization(
                    logits=raw_rna_lm_logits,
                    xt=utr_masked_rna_input_ids,
                    allowed_token_mask=allowed_token_mask
        )
        # log_raw_rna_lm_logits = raw_rna_lm_logits[:,:,rna_vocab_indices].log_softmax(dim=-1)
        raw_rna_lm_prob = raw_rna_lm_logits[:,:,rna_vocab_indices].softmax(dim=-1)
        raw_rna_lm_prob_utr = raw_rna_lm_prob[:,utr_residue_indices_tensor,:]
        
        
        _protein_sequence_list = list(protein_sequence)
        def _calculate_region_log_probs(mutation_idx):
            mutated_protein_sequence_list = []
            for aa in aminoacid_list:
                _mutated_protein_sequence = _protein_sequence_list.copy()
                _mutated_protein_sequence[mutation_idx] = aa
                mutated_protein_sequence_str = "".join(_mutated_protein_sequence)
                mutated_protein_sequence_list.append(mutated_protein_sequence_str)
            mutated_protein_input_ids = self.prepare_protein_inputs_for_model(
                mutated_protein_sequence_list,
                self.protein_tokenizer,
            ).to(self.device)
            mutated_protein_embeddings_mi = self.protein_encoder(mutated_protein_input_ids).embeddings
            corrupted_rna_input_ids = rna_input_ids.clone()
            codon_mutation_idx = 4 + utr5_length + mutation_idx
            corrupted_rna_input_ids[:, codon_mutation_idx] = codon_token_indices
            
            # also mask utr
            corrupted_rna_input_ids[:, utr5_start:utr5_end] = self.global_tokenizer.mask_idx
            corrupted_rna_input_ids[:, utr3_start:utr3_end] = self.global_tokenizer.mask_idx
            
            if use_cg:
                size_match_cg = (corrupted_rna_input_ids.shape == cg.input_size)
                # if not size_match_cg:
                #     print(f"Warning: Corrupted input size {corrupted_rna_input_ids.shape} does not match CUDAGraph input size {cg.input_size}. Disabling CUDAGraph for this scoring.")
            if use_cg and size_match_cg:
                rna_lm_logits = cg(corrupted_rna_input_ids, protein_embeddings=mutated_protein_embeddings_mi)
            else:
                rna_lm_logits = self.backbone(
                                input_ids=corrupted_rna_input_ids, 
                                protein_embeddings=mutated_protein_embeddings_mi,
                                row_wise_col_perms=input_batch["row_wise_col_perms"],
                                inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"],
                                attention_mask=input_batch["attention_mask"],
                                species_ids=species_ids,
                                modality_type_ids=modality_type_ids,
                                modality_mask=input_batch["modality_mask"]
                            )
            # import pdb; pdb.set_trace()
            rna_lm_logits = self.process_parameterization(
                    logits=rna_lm_logits,
                    xt=utr_masked_rna_input_ids,
                    allowed_token_mask=allowed_token_mask
                )
            # if use logits
            # log_probs = rna_lm_logits[:,:,rna_vocab_indices].log_softmax(dim=-1)
            # log_probs_diff = log_probs - log_raw_rna_lm_logits
            # log_probs_diff = log_probs_diff[:,utr_residue_indices_tensor,:]
            # return log_probs_diff
            # if use probs
            probs = rna_lm_logits[:,:,rna_vocab_indices].softmax(dim=-1)
            probs_utr = probs[:,utr_residue_indices_tensor,:]
            m_prob = 0.5 * (probs_utr + raw_rna_lm_prob_utr)
            kl_p_m = F.kl_div(m_prob.log(), probs_utr, reduction='none')
            kl_q_m = F.kl_div(m_prob.log(), raw_rna_lm_prob_utr, reduction='none')
            jsd = 0.5 * (kl_p_m.sum(dim=-1) + kl_q_m.sum(dim=-1))
            # import pdb; pdb.set_trace()
            return jsd
        
        # jacobian = torch.zeros(
        #     (len(protein_residue_indices), 20, len(utr3_residue_indices)+len(utr5_residue_indices), 4), dtype=torch.bfloat16
        # ).to(self.device)
        jacobian = torch.zeros(
            (len(protein_residue_indices), 20, len(utr3_residue_indices)+len(utr5_residue_indices)), dtype=torch.bfloat16
        ).to(self.device)
        for i, mutation_idx in enumerate(protein_residue_indices):
            log_probs_diff = _calculate_region_log_probs(mutation_idx)
            jacobian[i] = log_probs_diff
            
        # import pdb; pdb.set_trace()
        # jacobian[:,0,:utr5_length,:]
        # log_raw_rna_lm_logits[0,:utr5_length,:]
        
        # Mean-centering: To remove that background signal, subtract the mean across every dimension.
        # for dim in range(4):
        #     jacobian = jacobian - jacobian.mean(dim=dim, keepdim=True)
        # Symmetrization: To further reduce noise, symmetrize the matrix by averaging it with its transpose.
        # But we can't do this because we are uni-direction from protein to 3' UTR.
        # jacobian = (jacobian + jacobian.transpose(0, 2)) / 2
        # Frobenius norm across amino-acid channel and nucleotide channel
        # interaction_map = torch.norm(jacobian, dim=(1,3))
        # interaction_map[:, :utr5_length]
        # import pdb; pdb.set_trace()
        # apply APC
        interaction_map = jacobian.mean(dim=1) # average across amino acid channel
        row_sum = interaction_map.sum(dim=1, keepdim=True)
        col_sum = interaction_map.sum(dim=0, keepdim=True)
        total_sum = interaction_map.sum()
        apc_matrix = row_sum * col_sum / total_sum
        apc_interaction = interaction_map - apc_matrix
        # import pdb; pdb.set_trace()
        return {
            "jacobian": interaction_map.float().cpu().numpy() ,
            "corrected_jacobian": apc_interaction.float().cpu().numpy()  
        }
    
    
    @torch.no_grad()
    def score_denoising(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
            score_batch_size=4,
            use_cg=False,
            retain_cg=False,
            region_to_score="utr3"
        ):
        """Compute the log-probability of a given RNA sequence conditioned on the protein sequence."""
        
        rna_input_ids, _ = tokenize_inputs(
            protein_sequence, 
            utr5_sequence=utr5_sequence,
            cds_sequence=cds_sequence,
            utr3_sequence=utr3_sequence,
            protein_tokenizer=self.protein_tokenizer,
            protein_encoder=self.protein_encoder,
            global_tokenizer=self.global_tokenizer,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            complete_sequence=True
        )
        
        
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        
        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        
        rna_input_ids = rna_input_ids.repeat(score_batch_size, 1).to(self.device)
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = score_batch_size
        
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64, device=self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        input_batch['modality_type_ids'] = modality_type_ids
        input_batch['species_ids'] = species_ids
        if use_cg:
            if retain_cg and self.score_cg is not None:
                cg = self.score_cg
            else:
                cg = CUDAGraphForward(self.backbone, input_batch)
                cg.capture(example_input_ids=rna_input_ids)
                if retain_cg:
                    self.score_cg = cg
        else:
            cg = None
            
        def _calculate_log_probs(region_start, region_end):
            corrupted_rna_input_ids = rna_input_ids.clone()
            corrupted_rna_input_ids[:, region_start:region_end] = self.mask_index
            size_match_cg = (corrupted_rna_input_ids.shape == cg.input_size).all()
            if not size_match_cg and use_cg:
                print(f"Warning: Corrupted input size {corrupted_rna_input_ids.shape} does not match CUDAGraph input size {cg.input_size}. Disabling CUDAGraph for this scoring.")
            if use_cg and size_match_cg:
                rna_lm_logits = cg(corrupted_rna_input_ids)
            else:
                rna_lm_logits = self.backbone(
                                input_ids=corrupted_rna_input_ids, 
                                protein_embeddings=input_batch["protein_embeddings"],
                                row_wise_col_perms=input_batch["row_wise_col_perms"],
                                inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"],
                                attention_mask=input_batch["attention_mask"],
                                species_ids=species_ids,
                                modality_type_ids=modality_type_ids,
                                modality_mask=input_batch["modality_mask"]
                            )
            rna_lm_logits = self.process_parameterization(
                logits=rna_lm_logits,
                xt=corrupted_rna_input_ids,
            )
            logits_of_interest = rna_lm_logits[:, region_start:region_end, :]
            log_probs = logits_of_interest.log_softmax(dim=-1)
            # get logp based on `region_input_ids`
            target_log_probs = torch.gather(
                log_probs,
                dim=-1, 
                index=region_input_ids.unsqueeze(-1)
            ).squeeze(-1)
            return target_log_probs
        
        if region_to_score == 'utr3':
            region_start = 6 + len(utr5_sequence) + len(cds_sequence) // 3
            region_end = region_start + len(utr3_sequence)
            region_input_ids = rna_input_ids[:, region_start:region_end]
        else:
            raise NotImplementedError("Currently, only utr3 region scoring is implemented in score_denoising. Please use score_per_position for scoring other regions or implement the desired region scoring in score_denoising.")
        
        logp = _calculate_log_probs(region_start, region_end)
        return logp.float().cpu().numpy()
    
    
    @torch.no_grad()
    def get_attention_scores(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
        ):
        
        rna_input_ids, _ = tokenize_inputs(
            protein_sequence, 
            utr5_sequence=utr5_sequence,
            cds_sequence=cds_sequence,
            utr3_sequence=utr3_sequence,
            protein_tokenizer=self.protein_tokenizer,
            protein_encoder=self.protein_encoder,
            global_tokenizer=self.global_tokenizer,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            complete_sequence=True
        )
        
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        
        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = rna_input_ids.shape[0]
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )  
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64, device=self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        
        attention_scores = self.backbone.get_attention_weights(
            input_ids=rna_input_ids, 
            protein_embeddings=input_batch["protein_embeddings"],
            row_wise_col_perms=input_batch["row_wise_col_perms"],
            inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"],
            attention_mask=input_batch["attention_mask"],
            species_ids=species_ids,
            modality_type_ids=modality_type_ids,
            modality_mask=input_batch["modality_mask"]
        )
        return attention_scores
    
    def get_embeddings(
            self, 
            protein_sequence, 
            utr5_sequence, 
            cds_sequence, 
            utr3_sequence,
            species_id=85,
            codon_tokenizer=None,
            utr_5_tokenizer=None,
            utr_3_tokenizer=None,
            layer_indices=[],
            sequence_level=False
        ):
        
        rna_input_ids, _ = tokenize_inputs(
            protein_sequence, 
            utr5_sequence=utr5_sequence,
            cds_sequence=cds_sequence,
            utr3_sequence=utr3_sequence,
            protein_tokenizer=self.protein_tokenizer,
            protein_encoder=self.protein_encoder,
            global_tokenizer=self.global_tokenizer,
            codon_tokenizer=codon_tokenizer,
            utr_5_tokenizer=utr_5_tokenizer,
            utr_3_tokenizer=utr_3_tokenizer,
            complete_sequence=True
        )
        
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
        
        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = rna_input_ids.shape[0]
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size_per_gpu,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length
        )  
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64, device=self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids
        )
        
        embeddings = self.backbone.get_embeddings(
            input_ids=rna_input_ids, 
            protein_embeddings=input_batch["protein_embeddings"],
            row_wise_col_perms=input_batch["row_wise_col_perms"],
            inverse_row_wise_col_perms=input_batch["inverse_row_wise_col_perms"],
            attention_mask=input_batch["attention_mask"],
            species_ids=species_ids,
            modality_type_ids=modality_type_ids,
            modality_mask=input_batch["modality_mask"],
            layer_indices=layer_indices
        )
        return embeddings
    
    def prepare_protein_inputs_for_model(
            self,
            protein_sequence,
            protein_tokenizer,
        ):
        """Prepare inputs for the model from a protein sequence."""
        with torch.no_grad():
            # Tokenize the protein sequence
            if isinstance(protein_sequence, str):
                protein_sequence = [protein_sequence]
            elif isinstance(protein_sequence, list):
                protein_sequence = [seq for seq in protein_sequence if isinstance(seq, str)]
            else:
                raise ValueError("protein_sequence must be a string or a list of strings.")
            # Tokenize the protein sequence
            protein_input_ids = esm_tokenize(
                protein_sequence, protein_tokenizer
            )
            protein_input_ids = protein_input_ids.to(self.device)
        return protein_input_ids


class CUDAGraphForward:
    def __init__(
            self, 
            model, 
            static_kwargs, 
            warmup=3,
            update_protein=False
        ):
        """
        model: your backbone (or a wrapper around it)
        static_kwargs: dict of ALL args except input_ids, already on GPU with fixed shapes
        """
        self.model = model
        self.static_kwargs = {k: v.clone() for k, v in static_kwargs.items()}
        self.warmup = warmup

        self.graph = None
        self.static_input_ids = None
        self.static_protein_embeddings = None
        self.static_out = None
        self.input_size = None
        self.update_protein = update_protein

    @torch.no_grad()
    def capture(self, example_input_ids: torch.Tensor):
        # 1) allocate static input buffer (same shape/dtype/device)
        protein_embeddings = self.static_kwargs.get("protein_embeddings", None)
        assert protein_embeddings is not None, \
            "protein_embeddings must be provided in static_kwargs."
        if self.update_protein:
            self.static_protein_embeddings = torch.empty_like(protein_embeddings)
        else:
            self.static_protein_embeddings = protein_embeddings.clone()
        self.static_input_ids = torch.empty_like(example_input_ids)
        self.input_size = self.static_input_ids.shape
        # 2) warmup to stabilize kernels / autotune (esp. flash-attn / cudnn)
        self.model.eval()

        # 3) optional: separate capture stream
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                _ = self.model(
                    input_ids=example_input_ids,
                    protein_embeddings=protein_embeddings,
                    row_wise_col_perms=self.static_kwargs["row_wise_col_perms"],
                    inverse_row_wise_col_perms=self.static_kwargs["inverse_row_wise_col_perms"],
                    attention_mask=self.static_kwargs["attention_mask"],
                    species_ids=self.static_kwargs["species_ids"],
                    modality_type_ids=self.static_kwargs["modality_type_ids"],
                    modality_mask=self.static_kwargs["modality_mask"]
                    )
            # 4) capture
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            self.static_input_ids.copy_(example_input_ids)
            if self.update_protein:
                self.static_protein_embeddings.copy_(protein_embeddings)
            # 5) replay once to initialize static_out
            # Important: ensure no grads, no dropout randomness
            with torch.cuda.graph(self.graph):
                self.static_out = self.model(
                    input_ids=self.static_input_ids,
                    protein_embeddings=self.static_protein_embeddings,
                    row_wise_col_perms=self.static_kwargs["row_wise_col_perms"],
                    inverse_row_wise_col_perms=self.static_kwargs["inverse_row_wise_col_perms"],
                    attention_mask=self.static_kwargs["attention_mask"],
                    species_ids=self.static_kwargs["species_ids"],
                    modality_type_ids=self.static_kwargs["modality_type_ids"],
                    modality_mask=self.static_kwargs["modality_mask"]
                )
        torch.cuda.synchronize()

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, protein_embeddings=None):
        # in-place update of captured input buffer
        self.static_input_ids.copy_(input_ids)
        if protein_embeddings is not None and self.update_protein:
            self.static_protein_embeddings.copy_(protein_embeddings)
        self.graph.replay()
        return self.static_out.clone()
