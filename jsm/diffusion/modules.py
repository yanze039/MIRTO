import torch
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
from jsm.data.utils import esm_tokenize, modality_map
from jsm.data.species_specific import tokenize_inputs

LOG2 = math.log(2)
logger = utils.get_logger(__name__)

def hamming_distance_numpy(s1, s2):
    a = np.frombuffer(s1.encode(), dtype=np.uint8)
    b = np.frombuffer(s2.encode(), dtype=np.uint8)
    return np.count_nonzero(a != b)

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
                aa = codon_table.get(codon, None)
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
        utr5_nlls = (nlls * batch['utr5_mask']).sum(dim=-1)
        utr5_count = batch['utr5_mask'].sum(dim=-1)
        utr5_token_nll = (utr5_nlls / utr5_count).mean()
        
        # codon nlls
        cds_nlls = (nlls * batch['cds_mask']).sum(dim=-1)
        cds_count = batch['cds_mask'].sum(dim=-1)
        cds_token_nll = (cds_nlls / cds_count).mean()
        
        # utr3 nlls
        utr3_nlls = (nlls * batch['utr3_mask']).sum(dim=-1)
        utr3_count = batch['utr3_mask'].sum(dim=-1)
        utr3_token_nll = (utr3_nlls / utr3_count).mean()
        
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
    def _sample(self, protein_sequence, utr5_length, utr3_length, species_id=85, num_steps=None, eps=1e-5):
        batch_size_per_gpu = self.config.sampling.batch_size
        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence,
            self.protein_tokenizer,
        ).to(self.device)
                
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        sequence_length = (8 + utr5_length + utr3_length + len(protein_sequence)+1)
        
        modality_type_ids = torch.ones(batch_size_per_gpu, sequence_length, dtype=torch.int64).to(self.device)
        modality_type_ids[:, 0] = modality_map["global_special_tokens"]
        modality_type_ids[:, 1:3+utr5_length] = modality_map["utr_5"]
        modality_type_ids[:, 3+utr5_length:5+utr5_length+len(protein_sequence)+1] = modality_map["cds"]
        modality_type_ids[:, 5+utr5_length+len(protein_sequence)+1:7+utr5_length+len(protein_sequence)+1+utr3_length] = modality_map["utr_3"]
        modality_type_ids[:, 7+utr5_length+len(protein_sequence)+1+utr3_length:] = modality_map["global_special_tokens"]
        
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64).to(self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        x = self._sample_prior(
            batch_size = batch_size_per_gpu,
            seq_length = sequence_length,
            utr_5_length=utr5_length,
            utr_3_length=utr3_length,
            cds_length=len(protein_sequence)+1,
            ).to(self.device)
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        logger.info(f"starting sampling using sampler [{self.sampler}]")
        # for i in range(num_steps):
        for i in tqdm.tqdm(range(num_steps), desc=f"[{self.sampler}] Sampling", unit="step", leave=False):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt,
                                      protein_input_ids=protein_input_ids,
                    species_ids=species_ids,
                    modality_input_ids=modality_type_ids)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache, 
                    protein_input_ids=protein_input_ids,
                    species_ids=species_ids,
                    modality_input_ids=modality_type_ids
                    )
                if (not torch.allclose(x_next, x)
                    or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)
            ## respect the <eos> token, make all the tokens behind it <pad>
            # eos_token_mask = x.eq(self.eos_index)
            # eos_positions = eos_token_mask.float().masked_fill(
            #     eos_token_mask, torch.finfo(eos_token_mask.float().dtype).min
            #     ).argmin(dim=1, keepdim=True) 
            # check if the eos token is in the sequence
            # presence_eos = eos_token_mask.any(dim=1)
            # set all the tokens behind the eos token to <pad>
            # positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            # keep_mask = positions <= eos_positions
            # keep_mask[~presence_eos] = True
            # x = x.masked_fill(~keep_mask, self.padding_idx)
            
            # if all sequences have <eos>, we can just discard the tokens after them
            # if torch.all(presence_eos):
            #     # find the max position of the eos token
            #     max_eos_position = eos_positions.max()
            #     x = x[:, :max_eos_position]
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                            device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                x = self.generating_forward(x,
                                           protein_input_ids,
                                            species_ids,
                                            modality_type_ids
                                        ).argmax(dim=-1)
        return x
    
    
    def _ddpm_caching_update(self, 
                             x, 
                             t, 
                             dt, 
                             p_x0=None,
                             protein_input_ids=None,
                             species_ids=None,
                             modality_input_ids=None
                             
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
            p_x0 = self.generating_forward(
                    x, 
                    protein_input_ids,
                    species_ids,
                    modality_input_ids
                ).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt, 
                     protein_input_ids=None,
                     species_ids=None,
                     modality_input_ids=None
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
        log_p_x0 = self.generating_forward(x,
                                           protein_input_ids,
                                            species_ids,
                                            modality_input_ids
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
    
    def generating_forward(self, 
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
        
        rna_lm_logits = self.backbone(
            input_ids=rna_input_ids,
            species_ids=species_ids,                 
            protein_embeddings=protein_embeddings,
            row_wise_col_perms=row_wise_col_perms,
            inverse_row_wise_col_perms=inverse_indices,
            attention_mask=attention_mask,
            modality_type_ids=modality_input_ids,
            modality_mask=modality_mask,
        )
        rna_lm_logits = self.process_parameterization(
            logits=rna_lm_logits,
            xt=rna_input_ids,
        )
        return rna_lm_logits
    
    @torch.no_grad()
    def score(self, protein_sequence, 
              utr5_sequence, 
              cds_sequence, 
              utr3_sequence,
              species_id=85,
              codon_tokenizer=None,
              utr_5_tokenizer=None,
                utr_3_tokenizer=None
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
        
        assert sequence_length == rna_input_ids.shape[1], \
            f"Expected sequence length {sequence_length}, but got {rna_input_ids.shape[1]}"
        
        batch_size_per_gpu = rna_input_ids.shape[0]
        modality_type_ids = torch.ones(batch_size_per_gpu, sequence_length, dtype=torch.int64).to(self.device)
        modality_type_ids[:, 0] = modality_map["global_special_tokens"]
        modality_type_ids[:, 1:3+utr5_length] = modality_map["utr_5"]
        modality_type_ids[:, 3+utr5_length:5+utr5_length+cds_length] = modality_map["cds"]
        modality_type_ids[:, 5+utr5_length+cds_length:7+utr5_length+cds_length+utr3_length] = modality_map["utr_3"]
        modality_type_ids[:, 7+utr5_length+cds_length+utr3_length:] = modality_map["global_special_tokens"]
        
        species_ids = torch.ones(batch_size_per_gpu, dtype=torch.int64).to(self.device) * species_id
        species_ids = species_ids.reshape(-1, 1)
        
        # Compute log-probability using the generating_forward method
        
        
        # import pdb; pdb.set_trace()
        utr5_perposition_scores = torch.zeros(rna_input_ids.shape[0], len(utr5_sequence)
            , dtype=torch.float32).to(self.device)
        for mask_position in range(len(utr5_sequence)):
            corrupted_rna_input_ids = rna_input_ids.clone()
            corrupted_rna_input_ids[:, 2 + mask_position] = self.mask_index
            rna_lm_logits = self.generating_forward(
                            corrupted_rna_input_ids, 
                            protein_input_ids,
                            species_ids,
                            modality_type_ids
                        )
            logits_of_interest = rna_lm_logits[:, 2 + mask_position]
            logits_of_interest = logits_of_interest - logits_of_interest.logsumexp(dim=-1, keepdim=True)
            token_indices = rna_input_ids[:, 2 + mask_position]
            logp = logits_of_interest.gather(dim=-1, index=token_indices.unsqueeze(-1)).squeeze(-1)
            utr5_perposition_scores[:, mask_position] = logp
        
        cds_perposition_scores = torch.zeros(rna_input_ids.shape[0], cds_length
            , dtype=torch.float32).to(self.device)
        for mask_position in range(cds_length):
            corrupted_rna_input_ids = rna_input_ids.clone()
            corrupted_rna_input_ids[:, 4 + utr5_length + mask_position] = self.mask_index
            rna_lm_logits = self.generating_forward(
                            corrupted_rna_input_ids, 
                            protein_input_ids,
                            species_ids,
                            modality_type_ids
                        )
            logits_of_interest = rna_lm_logits[:, 4 + utr5_length + mask_position]
            logits_of_interest = logits_of_interest - logits_of_interest.logsumexp(dim=-1, keepdim=True)
            token_indices = rna_input_ids[:, 4 + utr5_length + mask_position]
            logp = logits_of_interest.gather(dim=-1, index=token_indices.unsqueeze(-1)).squeeze(-1)
            cds_perposition_scores[:, mask_position] = logp
        
        utr3_perposition_scores = torch.zeros(rna_input_ids.shape[0], len(utr3_sequence)
            , dtype=torch.float32).to(self.device)
        for mask_position in range(len(utr3_sequence)):
            corrupted_rna_input_ids = rna_input_ids.clone()
            corrupted_rna_input_ids[:, 6 + utr5_length + cds_length + mask_position] = self.mask_index
            rna_lm_logits = self.generating_forward(
                            corrupted_rna_input_ids, 
                            protein_input_ids,
                            species_ids,
                            modality_type_ids
                        )
            logits_of_interest = rna_lm_logits[:, 6 + utr5_length + cds_length + mask_position]
            logits_of_interest = logits_of_interest - logits_of_interest.logsumexp(dim=-1, keepdim=True)
            token_indices = rna_input_ids[:, 6 + utr5_length + cds_length + mask_position]
            logp = logits_of_interest.gather(dim=-1, index=token_indices.unsqueeze(-1)).squeeze(-1)
            utr3_perposition_scores[:, mask_position] = logp
        
        return {
            "utr5_perposition_scores": utr5_perposition_scores,
            "cds_perposition_scores": cds_perposition_scores,
            "utr3_perposition_scores" : utr3_perposition_scores
        }
        
    
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

