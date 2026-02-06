import itertools
import math
import os
import typing
from typing import Optional
from dataclasses import dataclass
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
import transformers
from torch import Tensor
import jsm.diffusion.noise_schedule as noise_schedule
import jsm.utils as utils
import tqdm
import pickle

LOG2 = math.log(2)
logger = utils.get_logger(__name__)

def _sample_categorical(categorical_probs):
    gumbel_norm = (
      1e-10
      - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(
                * x.shape,
                * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor
    lm_loss: Optional[torch.FloatTensor] = None
    rna_type_loss: Optional[torch.FloatTensor] = None
    go_loss: Optional[torch.FloatTensor] = None


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
        Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(
        self,
        config,
        tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        self.mask_index = self.tokenizer.mask_idx
        self.padding_idx = self.tokenizer.padding_idx
        self.cls_index = self.tokenizer.cls_idx
        self.eos_index = self.tokenizer.eos_idx
        self.unknown_index = self.tokenizer.unk_idx
        self.N_index = self.tokenizer.tok_to_idx.get('N', self.tokenizer.unk_idx)
        self.parameterization = self.config.parameterization
        self.backbone = None
        self.go_loss_weight = 0.
        self.rna_type_loss_weight = 0.
        self.lm_loss_weight = 1.

        # self.T = self.config.T
        self.T = 0

        self.subs_masking = self.config.subs_masking
        self.subs_masking = False


        self.softplus = torch.nn.Softplus()
        # metrics are automatically reset at end of epoch
        metrics = torchmetrics.MetricCollection({
          'nll': NLL(),
          'bpd': BPD(),
          'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        # self.test_metrics = metrics.clone(prefix='test/')

        # generative perplexity
        self.gen_ppl_metric = Perplexity()
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

    def _validate_configuration(self):
        assert not (self.change_of_variables
                    and self.importance_sampling)

    def _subs_parameterization(self, logits, xt):
        neg_infinity = -torch.finfo(logits.dtype).max
        logits[:, :, self.mask_index] += neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1,
                                            keepdim=True)

        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma, attention_mask=None):
        """Returns log score."""
        sigma = self._process_sigma(sigma)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:,None,None,:] * attention_mask[:,None,:,None]
            # elif len(attention_mask.shape) == 3:
            #   attention_mask = attention_mask[:,None,:,:] * attention_mask[:,None,:,:]
            assert attention_mask.shape == (x.shape[0], 1, x.shape[1], x.shape[1]),\
                f'Invalid attention mask shape: {attention_mask.shape}, expected {(x.shape[0], 1, x.shape[1], x.shape[1])}'
        logits = self.backbone(x, sigma, attention_mask)
        logits = self.process_parameterization(logits=logits,
                                            xt=x,
                                            sigma=sigma)
        return logits
    
    def process_parameterization(self, logits, xt):
        logits = self._subs_parameterization(logits=logits, xt=xt)
        return logits

    def _compute_loss(self, batch, prefix):
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']
        else:
            attention_mask = None
        
        # make attention mask
        if attention_mask is None:
            attention_mask = batch['input_ids'].ne(self.padding_idx).to(torch.bool)
        
        try:
            losses = self._loss(batch, attention_mask)
        except RuntimeError as e:
            print(batch['input_ids'].shape)
            # save the batch data
            with open("error_batch.pkl", "wb") as fp:
                pickle.dump(batch, fp)
            raise RuntimeError from e

        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        # elif prefix == 'test':
            # self.test_metrics.update(losses.nlls, losses.token_mask)
            # metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')

        self.log_dict(metrics,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True)
        return losses
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()
        self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
        print(f"Note: on epoch {self.current_epoch}")
        # number of epoch
        if self.config.enable_rna_type_prediction or self.config.enable_go_prediction:
            if self.current_epoch < self.config.enable_function_training_on_epoch:
                self.go_loss_weight = 0.
                self.rna_type_loss_weight = 0.
                self.lm_loss_weight = 1.
                print("backbone not freezed")
            elif self.current_epoch == self.config.enable_function_training_on_epoch:
                self.go_loss_weight = self.config.go_loss_weight
                self.rna_type_loss_weight = self.config.rna_type_loss_weight
                self.lm_loss_weight = 0.
                self.backbone.freeze_encoder()
                print("backbone freezed")
            else:
                self.go_loss_weight = self.config.go_loss_weight
                self.rna_type_loss_weight = self.config.rna_type_loss_weight
                self.lm_loss_weight = 1.
                self.backbone.unfreeze_encoder()
                print("backbone not freezed")
            print("go_loss_weight: ", self.go_loss_weight)
            print("rna_type_loss_weight: ", self.rna_type_loss_weight)
            print("lm_loss_weight: ", self.lm_loss_weight)   

    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='train')
        self.log(name='trainer/loss',
                value=losses.loss.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(name='trainer/lr',
                 value=lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        if hasattr(losses, "lm_loss"):
            self.log(
                "trainer/lm_loss",
                losses.lm_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        if self.config.enable_rna_type_prediction:
            self.log(
                "trainer/rna_type_loss",
                losses.rna_type_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        if self.config.enable_go_prediction:
            self.log(
                "trainer/go_loss",
                losses.go_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        return losses.loss

    def on_validation_epoch_start(self):
        self.backbone.eval()
        self.noise.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='val')
        self.log(name='validation/loss',
                    value=losses.loss.item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True)
        if self.config.enable_rna_type_prediction:
            self.log(
                "validation/rna_type_loss",
                losses.rna_type_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        if self.config.enable_go_prediction:
            self.log(
                "validation/go_loss",
                losses.go_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        if hasattr(losses, "lm_loss"):
            self.log(
                "validation/lm_loss",
                losses.lm_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        return losses.loss

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    # @property
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

    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(),
                            self.noise.parameters()),
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

    @torch.no_grad()
    def eval_retokenize(self, text_samples, max_length):
        """Retokenizes samples for the eval model.
        
        Args:
            text_samples: List of sentences generated by the model.
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        tokenizer_kwargs = {
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'truncation': True,
            'padding': True,
            'max_length': max_length,
        }
        eval_context_size = 1024
        samples = self.eval_model_tokenizer(
            text_samples, ** tokenizer_kwargs)
        attn_mask = samples['attention_mask']
        samples = samples['input_ids']
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def compute_generative_perplexity(
        self,
        text_samples: typing.List[str],
        retokenize: bool = True,
        max_length: typing.Optional[int] = None) -> None:
        """Compute the generative perplexity of the model.

        Args:
            text_samples: List of sentences generated by the model.
        
        Returns:
            Perplexity of the generated text under a different
            pre-trained AR model (e.g., GPT2).
        """
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.gen_ppl_eval_model_name_or_path).eval()
        if max_length is None:
            max_length = self.config.model.length
        if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
            eval_model = eval_model.to(self.device)
        # Re-tokenize using eval model's tokenizer
        if retokenize:
            (samples, attn_mask,
            eval_context_size) = self.eval_retokenize(
                text_samples, max_length=max_length)
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(self.device)
            eval_context_size = samples.shape[-1]
        batch_size = min(
            self.config.eval.perplexity_batch_size,
            samples.shape[0])
        num_batches = samples.shape[0] // batch_size
        for i in range(num_batches):
            _samples = torch.split(
                samples[i * batch_size: (i + 1) * batch_size],
                eval_context_size,
                dim=-1)
            _attn_mask = torch.split(
                attn_mask[i * batch_size: (i + 1) * batch_size],
                eval_context_size,
                dim=-1)
            for (sample_chunk, attn_mask_chunk) in zip(
                _samples, _attn_mask):
                logits = eval_model(
                    sample_chunk, attention_mask=attn_mask_chunk)[0]
                logits = logits.transpose(-1, -2)
                
                nlls = F.cross_entropy(logits[..., :-1],
                                        sample_chunk[..., 1:],
                                        reduction='none')
                first_eos = (sample_chunk == self.eval_model_tokenizer\
                            .eos_token_id).cumsum(-1) == 1
                token_mask = (
                    sample_chunk
                    != self.eval_model_tokenizer.eos_token_id)
                self.gen_ppl_metric.update(
                    nlls, first_eos[..., 1:] + token_mask[..., 1:])

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
            x: int torch.Tensor with shape (batch_size,
                diffusion_model_input_length), input. 
            move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(
            * batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt):
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
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
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

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.sampling.batch_size
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.sampling.sequence_length).to(self.device)
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
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x)
                    or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)
            ## respect the <eos> token, make all the tokens behind it <pad>
            eos_token_mask = x.eq(self.eos_index)
            eos_positions = eos_token_mask.float().masked_fill(
                eos_token_mask, torch.finfo(eos_token_mask.float().dtype).min
                ).argmin(dim=1, keepdim=True) 
            # check if the eos token is in the sequence
            presence_eos = eos_token_mask.any(dim=1)
            # set all the tokens behind the eos token to <pad>
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            keep_mask = positions <= eos_positions
            keep_mask[~presence_eos] = True
            x = x.masked_fill(~keep_mask, self.padding_idx)
            
            # if all sequences have <eos>, we can just discard the tokens after them
            if torch.all(presence_eos):
                # find the max position of the eos token
                max_eos_position = eos_positions.max()
                x = x[:, :max_eos_position]
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                            device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
        return x

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))
        self.backbone.train()
        self.noise.train()
        return samples

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == 'subs':
            log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1
            
            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_index] = 0
            neg_infinity = -torch.finfo(model_output.dtype).max
            unmasked_score = neg_infinity * torch.ones_like(
                model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]))
            unmasked_score[:, :, self.mask_index] = - (
                log_k[:, None] * torch.ones_like(x))
            
            masked_indices = (x == self.mask_index).to(
                model_output.dtype)[:, :, None]
            model_output = (
                masked_score * masked_indices
                + unmasked_score * (1 - masked_indices))
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
            score = score.clone()
            extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
            score *= dsigma.exp()[:, None]
            score[..., self.mask_index] += extra_const
            return score

    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = _sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
            i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen > self.config.model.length:
            assert seqlen == 2 * self.config.model.length
            # cropping is needed for text8-crop dataset
            # try the same starting point for now
            start = np.random.choice(self.config.model.length)
            end = start + self.config.model.length
            input_tokens = x0[:, start: end]
            output_tokens = x0[:, start + 1: end + 1]
            new_attention_mask = attention_mask[:, start: end]

            # Helps with validation PPL, since the val
            # examples will all start and end with BOS/EOS
            input_tokens[:, 0] = self.cls_index
            output_tokens[:, -1] = self.eos_index
        else:
            input_tokens = x0
            output_tokens = None
            new_attention_mask = attention_mask
        return input_tokens, output_tokens, new_attention_mask

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                        device=self.device)
        assert self.config.noise.type == 'loglinear'
        # The above assert is for d3pm parameterization
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self.forward(x0, unet_conditioning)
        return - torch.gather(input=model_output_t0,
                                dim=-1,
                                index=x0[:, :, None]).squeeze(-1)

    def _forward_pass_diffusion(self, x0, t, attention_mask=None):
        # t = self._sample_t(x0.shape[0], x0.device)
        sigma, dsigma = self.noise(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])    

        xt = self.q_xt(x0, move_chance)
        
        if self.config.enable_rna_type_prediction or \
            self.config.enable_go_prediction:
            model_output, go_logits, rna_type_logits = self.forward(
                xt, unet_conditioning, attention_mask=attention_mask)
        else:
            model_output = self.forward(xt, unet_conditioning, attention_mask=attention_mask)
        utils.print_nans(model_output, 'model_output')
        
        if self.T > 0:
            diffusion_loss = self._d3pm_loss(
                model_output=model_output, xt=xt, x0=x0, t=t)
            if self.parameterization == 'd3pm':
                reconstruction_loss = self._reconstruction_loss(x0)
            elif self.parameterization == 'subs':
                reconstruction_loss = 0
            return reconstruction_loss + diffusion_loss
        
        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]
        ).squeeze(-1)
        
        logits_loss =  - log_p_theta * (
            dsigma / torch.expm1(sigma))[:, None]

        if self.config.enable_rna_type_prediction or \
            self.config.enable_go_prediction:
            return logits_loss, go_logits, rna_type_logits
        else:
            return logits_loss

    def _loss(self, batch, attention_mask):
        (input_tokens, output_tokens,
        attention_mask) = self._maybe_sub_sample(
            batch['input_ids'], attention_mask)
        t = self._sample_t(input_tokens.shape[0], input_tokens.device)
        if self.config.enable_rna_type_prediction or \
            self.config.enable_go_prediction:
            lm_loss, go_logits, rna_type_logits = self._forward_pass_diffusion(
                input_tokens, t, attention_mask=attention_mask)
            function_loss_weight = (1 - t[:, None]*t[:, None])
            go_logits = (go_logits * attention_mask[:, :, None]).mean(dim=1)
            go_loss = F.binary_cross_entropy_with_logits(
                go_logits,
                batch['go_labels'].float(),
                reduction='none'
            )
            # here we use the focal loss to weight the hard predictions
            go_probs = torch.sigmoid(go_logits)
            pt = torch.where(batch['go_labels'] > 0.5, go_probs, 1 - go_probs)
            focal_weight = (1 - pt) ** 2
            go_loss = focal_weight * go_loss
            go_loss = go_loss * batch['go_mask']
            # here we weight the loss by noise conditions
            # no need to predict well if noise is high
            go_loss = go_loss * function_loss_weight
            go_loss = go_loss.sum() / (batch['go_mask'].sum()+1) / self.config.num_go_classes
            rna_type_logits = (rna_type_logits * attention_mask[:, :, None]).mean(dim=1)
            rna_type_loss = F.cross_entropy(
                rna_type_logits,
                batch['rna_type_labels'].float(),
                reduction='none'
            )
            rna_type_loss = rna_type_loss * batch['rna_type_mask']
            rna_type_loss = rna_type_loss * function_loss_weight
            rna_type_loss = rna_type_loss.sum() / (batch['rna_type_mask'].sum()+1) / self.config.num_rna_type_classes
        else:
            lm_loss = self._forward_pass_diffusion(input_tokens, t, attention_mask=attention_mask)

        nlls = lm_loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        if self.config.enable_rna_type_prediction or \
                self.config.enable_go_prediction:
            return Loss(
                loss=self.lm_loss_weight * token_nll + self.go_loss_weight * go_loss + self.rna_type_loss_weight * rna_type_loss,
                lm_loss=token_nll.detach(),
                nlls=nlls.detach(),
                token_mask=attention_mask,
                go_loss=go_loss.detach(),
                rna_type_loss=rna_type_loss.detach()
                )
        else:
            return Loss(loss=token_nll,
                        lm_loss=token_nll.detach(),
                        nlls=nlls.detach(),
                        token_mask=attention_mask)

    # @torch.no_grad
    # def sample_subs_guidance(
    #         self, n_samples, stride_length, num_strides, dt=0.001):
    #     ones = torch.ones(n_samples, dtype=self.dtype,
    #                         device=self.device)

    #     num_steps = int(1 / dt)
    #     sampling_steps = 0
    #     intermediate_tokens = []
    #     target = None
    #     for _ in range(num_strides + 1):
    #         p_x0_cache = None
    #         x = self._sample_prior(
    #             n_samples,
    #             self.config.model.length).to(self.device)
    #         if target is not None:
    #             x[:, : -stride_length] = target
    #         for i in range(num_steps + 1):
    #             p_x0_cache, x_next = self._ddpm_caching_update(
    #                 x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
    #             if (not torch.allclose(x_next, x)
    #                 or self.time_conditioning):
    #                 p_x0_cache = None
    #                 sampling_steps += 1
    #             x = x_next
    #         x = self.forward(x, 0 * ones).argmax(dim=-1)
    #         intermediate_tokens.append(
    #             x[:, :stride_length].cpu().numpy())
    #         target = x[:, stride_length:]
        
    #     intermediate_tokens.append(target.cpu().numpy())
    #     intermediate_text_samples = []
    #     sequence_lengths = ((
    #         np.concatenate(intermediate_tokens, axis=1)[:, 1:]
    #         == self.eos_index).cumsum(-1) == 0).sum(-1)
    #     for i in range(2, len(intermediate_tokens) + 1):
    #         intermediate_text_samples.append(
    #             self.tokenizer.batch_decode(
    #             np.concatenate(intermediate_tokens[:i], axis=1)))
    #     return (sampling_steps, intermediate_text_samples,
    #             sequence_lengths)

