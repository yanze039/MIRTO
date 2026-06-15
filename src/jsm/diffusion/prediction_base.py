"""Shared base class for diffusion-backbone prediction-task fine-tunes.

Collapses the 5 near-identical files
  apex_regression.py, te_regression.py, stability_regression.py,
  protein_loc_classification.py, rna_loc_classification.py
into a single parametric base. Each leaf file becomes a ~30-line subclass that
only declares: which label key in the batch, which pooling strategy, and which
task type (regression vs. multilabel classification).

Public class / module / state_dict names are preserved so existing prediction
checkpoints load without renaming.

Refactor decisions worth knowing:
  * PredictionHead signature is unified to (input_dim, hidden_dim, output_dim);
    the old rna_loc/protein_loc variant that used (hidden_size, output_dim,
    expansion=...) is folded in as hidden_dim = hidden_size * expansion. Configs
    that previously set `expansion` should now set `hidden_dim` directly. (Old
    leaf files derived hidden_dim = hidden_size * 1, so the default behavior
    matches.)
  * forward() takes only `batch` everywhere. The old stability/rna_loc/protein_loc
    call sites `self.forward(batch['rna_input_ids'], batch)` are silently fixed
    to `self.forward(batch)`. The leading positional arg was unused.
  * Pooling strategies are named in POOLING_REGISTRY and selected by the leaf.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)

import jsm.diffusion.noise_schedule as noise_schedule
import jsm.utils as utils
from jsm.data.species_specific import tokenize_inputs
from jsm.data.utils import esm_tokenize, modality_map
from jsm.diffusion.core import Diffusion
from jsm.models.transformer import SpeciesSpecificJointSequenceTransformer


LOG2 = math.log(2)
logger = utils.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Heads (state_dict-compatible with the previous leaf-file implementations).
# --------------------------------------------------------------------------- #
class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final_linear = self.prediction_head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.xavier_uniform_(final_linear.weight, gain=0.5)

    def forward(self, x):
        return self.prediction_head(x)


class ShallowEnsembleHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=5,
                 dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            PredictionHead(input_dim, hidden_dim, output_dim, dropout=dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.stack([head(x) for head in self.heads], dim=1)


# --------------------------------------------------------------------------- #
# Pooling registry.
# Each entry maps a name → (callable, input_dim_multiplier).
# The callable takes the precomputed per-modality pooled vectors and returns
# the input tensor to the prediction head.
# --------------------------------------------------------------------------- #
PoolFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _pool_concat_utr5_utr3(u5, cds, u3):
    return torch.cat([u5, u3], dim=-1)


def _pool_concat_utr5_cds_utr3(u5, cds, u3):
    return torch.cat([u5, cds, u3], dim=-1)


def _pool_utr3_only(u5, cds, u3):
    return u3


def _pool_mean_utr5_cds_utr3(u5, cds, u3):
    return (u5 + cds + u3) / 3.0


POOLING_REGISTRY: Dict[str, Tuple[PoolFn, int]] = {
    'concat_utr5_utr3': (_pool_concat_utr5_utr3, 2),
    'concat_utr5_cds_utr3': (_pool_concat_utr5_cds_utr3, 3),
    'utr3_only': (_pool_utr3_only, 1),
    'mean_utr5_cds_utr3': (_pool_mean_utr5_cds_utr3, 1),
}


# --------------------------------------------------------------------------- #
# BasePredictionDiffusion — the parametric Lightning module.
# --------------------------------------------------------------------------- #
class BasePredictionDiffusion(Diffusion):
    """Diffusion-backbone prediction-task base. See module docstring."""

    # Subclasses MUST override.
    LABEL_KEY: str = ''             # e.g. 'apex_label'
    TASK_TYPE: str = ''             # 'regression' | 'multilabel'
    POOLING: str = ''               # key into POOLING_REGISTRY

    def __init__(
        self,
        config,
        global_tokenizer,
        protein_tokenizer,
        rna_vocab_size,
        protein_vocab_size,
        protein_encoder,
    ):
        assert self.LABEL_KEY, f'{type(self).__name__} must set LABEL_KEY'
        assert self.TASK_TYPE in ('regression', 'multilabel'), \
            f'{type(self).__name__} must set TASK_TYPE'
        assert self.POOLING in POOLING_REGISTRY, \
            f'{type(self).__name__}.POOLING={self.POOLING!r} not in POOLING_REGISTRY'

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
        self.N_index = self.global_tokenizer.tok_to_idx.get(
            'N', self.global_tokenizer.unk_idx)
        self.parameterization = self.config.parameterization
        self.backbone = SpeciesSpecificJointSequenceTransformer(
            self.config.model,
            rna_vocab_size,
            protein_vocab_size,
        )
        self.T = 0
        self.subs_masking = False  # always False in the original 5 leaves

        self.softplus = torch.nn.Softplus()
        self.eval_model_tokenizer = self.tokenizer
        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
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
        self.launch_timestamp = time.time()
        self.resumed_dataloader_state_from_ckpt = None
        self.score_cg = None

        if self.protein_encoder is not None:
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
        if self.config.prediction.freeze_backbone:
            logger.info('Freezing backbone parameters.')
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Build prediction head sized by pooling multiplier.
        _, pool_mult = POOLING_REGISTRY[self.POOLING]
        hidden_size = self.config.model.hidden_size
        head_hidden = self.config.prediction.get('head_hidden_dim', hidden_size)
        self.ensemble_predictor = ShallowEnsembleHead(
            input_dim=hidden_size * pool_mult,
            hidden_dim=head_hidden,
            output_dim=self.config.prediction.output_dim,
            num_heads=self.config.prediction.num_heads,
            dropout=self.config.prediction.dropout,
        )

        # Validation metrics depend on task type.
        num_labels = self.config.prediction.output_dim
        self.label_names = list(
            self.config.prediction.get('label_names', None)
            or self.config.get('label_names', [])
            or [str(i) for i in range(num_labels)]
        )
        if len(self.label_names) != num_labels:
            raise ValueError(
                f'label_names length {len(self.label_names)} != '
                f'output_dim {num_labels}')

        self._init_metrics(num_labels)

    # ----------------------------------------------------------- training hooks
    def on_fit_start(self):
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
        if self.config.prediction.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def on_train_epoch_start(self):
        self.on_fit_start()

    # ------------------------------------------------------------------- model
    def forward(self, batch):
        """Pool the chosen modalities from the backbone and run the head."""
        with torch.no_grad():
            protein_output = self.protein_encoder(batch['protein_input_ids'])
            protein_embeddings = protein_output.embeddings

            layer_idx = self.config.prediction.layer_index
            embedding = self.backbone.get_embeddings(
                input_ids=batch['rna_input_ids'],
                species_ids=batch['species_ids'],
                protein_embeddings=protein_embeddings,
                row_wise_col_perms=batch['row_wise_col_perms'],
                inverse_row_wise_col_perms=batch['inverse_row_wise_col_perms'],
                attention_mask=batch['attention_mask'],
                modality_type_ids=batch['modality_type_ids'],
                modality_mask=batch['modality_mask'],
                layer_indices=[layer_idx],
            )

        special = batch['special_token_mask']
        utr5_mask = batch['utr5_mask'] * (1 - special)
        cds_mask = batch['cds_mask'] * (1 - special)
        utr3_mask = batch['utr3_mask'] * (1 - special)

        emb = embedding[layer_idx]
        u5 = (emb * utr5_mask.unsqueeze(-1)).sum(dim=1) / \
             (utr5_mask.sum(dim=1, keepdim=True) + 1e-8)
        cds = (emb * cds_mask.unsqueeze(-1)).sum(dim=1) / \
              (cds_mask.sum(dim=1, keepdim=True) + 1e-8)
        u3 = (emb * utr3_mask.unsqueeze(-1)).sum(dim=1) / \
             (utr3_mask.sum(dim=1, keepdim=True) + 1e-8)

        pool_fn, _ = POOLING_REGISTRY[self.POOLING]
        pooled = pool_fn(u5, cds, u3)
        return self.ensemble_predictor(pooled.float())

    # ------------------------------------------------------------------- loss
    def _loss(self, pred, labels):
        """Mean of per-ensemble-head losses."""
        if self.TASK_TYPE == 'regression':
            loss_fn = F.mse_loss
        else:  # multilabel
            loss_fn = F.binary_cross_entropy_with_logits
        per_head = [loss_fn(pred[:, h, :], labels) for h in range(pred.size(1))]
        return torch.stack(per_head).mean()

    def _ensemble_aggregate(self, pred):
        if self.TASK_TYPE == 'regression':
            return pred.mean(dim=1)
        return torch.sigmoid(pred).mean(dim=1)

    def _compute_loss(self, batch, prefix):
        output = self.forward(batch)
        labels = batch[self.LABEL_KEY].float()
        losses = self._loss(output, labels)
        on_step = (prefix == 'train')
        self.log_dict(
            {f'{prefix}/loss': losses},
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch['rna_input_ids'].shape[0],
        )
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='train')
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(
            name='trainer/lr',
            value=lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return losses

    # ----------------------------------------------------------- validation
    def _init_metrics(self, num_labels):
        if self.TASK_TYPE == 'regression':
            self.val_metrics = MetricCollection({
                'mae_macro': MeanAbsoluteError(num_outputs=num_labels),
                'rmse_macro': MeanSquaredError(num_outputs=num_labels,
                                               squared=False),
                'pearson_macro': PearsonCorrCoef(num_outputs=num_labels),
            })
            self.val_mae_per_label = MeanAbsoluteError(num_outputs=num_labels)
            self.val_rmse_per_label = MeanSquaredError(num_outputs=num_labels,
                                                      squared=False)
            self.val_pearson_per_label = PearsonCorrCoef(num_outputs=num_labels)
            self.val_r2_per_label = R2Score(multioutput='raw_values')
        else:  # multilabel
            self.threshold = self.config.get('threshold', 0.4)
            self.val_metrics = MetricCollection({
                'hamming_acc': MultilabelAccuracy(num_labels=num_labels,
                                                  threshold=self.threshold,
                                                  average='micro'),
                'precision_micro': MultilabelPrecision(num_labels=num_labels,
                                                       threshold=self.threshold,
                                                       average='micro'),
                'recall_micro': MultilabelRecall(num_labels=num_labels,
                                                 threshold=self.threshold,
                                                 average='micro'),
                'f1_micro': MultilabelF1Score(num_labels=num_labels,
                                              threshold=self.threshold,
                                              average='micro'),
                'f1_macro': MultilabelF1Score(num_labels=num_labels,
                                              threshold=self.threshold,
                                              average='macro'),
                'auroc_macro': MultilabelAUROC(num_labels=num_labels,
                                               average='macro'),
                'auroc_micro': MultilabelAUROC(num_labels=num_labels,
                                               average='micro'),
                'ap_macro': MultilabelAveragePrecision(num_labels=num_labels,
                                                       average='macro'),
                'ap_micro': MultilabelAveragePrecision(num_labels=num_labels,
                                                       average='micro'),
            })
            self.val_f1_per_label = MultilabelF1Score(
                num_labels=num_labels, threshold=self.threshold, average=None)
            self.val_auroc_per_label = MultilabelAUROC(
                num_labels=num_labels, average=None)
            self.val_ap_per_label = MultilabelAveragePrecision(
                num_labels=num_labels, average=None)

    def on_validation_epoch_start(self):
        self.backbone.eval()
        self.noise.eval()
        self.val_metrics.reset()
        if self.TASK_TYPE == 'regression':
            self.val_mae_per_label.reset()
            self.val_rmse_per_label.reset()
            self.val_pearson_per_label.reset()
            self.val_r2_per_label.reset()
        else:
            self.val_f1_per_label.reset()
            self.val_auroc_per_label.reset()
            self.val_ap_per_label.reset()

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        labels = batch[self.LABEL_KEY].float()
        losses = self._loss(output, labels)
        agg = self._ensemble_aggregate(output)

        if self.TASK_TYPE == 'regression':
            self.val_metrics.update(agg.float(), labels.float())
            self.val_mae_per_label.update(agg.float(), labels.float())
            self.val_rmse_per_label.update(agg.float(), labels.float())
            self.val_pearson_per_label.update(agg.float(), labels.float())
            self.val_r2_per_label.update(agg.float(), labels.float())
        else:
            self.val_metrics.update(agg, labels.int())
            self.val_f1_per_label.update(agg, labels.int())
            self.val_auroc_per_label.update(agg, labels.int())
            self.val_ap_per_label.update(agg, labels.int())

        self.log(
            'val/loss',
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )
        return losses

    def on_validation_epoch_end(self):
        if self.TASK_TYPE == 'regression':
            self._log_validation_regression()
        else:
            self._log_validation_multilabel()

    def _log_validation_regression(self):
        mae_vec = self.val_mae_per_label.compute()
        rmse_vec = self.val_rmse_per_label.compute()
        pearson_vec = self.val_pearson_per_label.compute()
        r2_vec = self.val_r2_per_label.compute()

        self.log_dict(
            {
                'val/mae_macro': mae_vec.mean(),
                'val/rmse_macro': rmse_vec.mean(),
                'val/pearson_macro': pearson_vec.mean(),
                'val/r2_macro': r2_vec.mean(),
            },
            on_step=False, on_epoch=True, prog_bar=False, logger=True,
            sync_dist=True,
        )
        per_label_logs = {}
        for i, name in enumerate(self.label_names):
            per_label_logs[f'val_per_label/mae_{name}'] = mae_vec[i]
            per_label_logs[f'val_per_label/rmse_{name}'] = rmse_vec[i]
            per_label_logs[f'val_per_label/pearson_{name}'] = pearson_vec[i]
            per_label_logs[f'val_per_label/r2_{name}'] = r2_vec[i]
        self.log_dict(
            per_label_logs,
            on_step=False, on_epoch=True, prog_bar=False, logger=True,
            sync_dist=True,
        )

        self.val_metrics.reset()
        self.val_mae_per_label.reset()
        self.val_rmse_per_label.reset()
        self.val_pearson_per_label.reset()
        self.val_r2_per_label.reset()

    def _log_validation_multilabel(self):
        global_metrics = self.val_metrics.compute()
        self.log_dict(
            {f'val/{k}': v for k, v in global_metrics.items()},
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
            sync_dist=True,
        )
        f1_vec = self.val_f1_per_label.compute()
        auroc_vec = self.val_auroc_per_label.compute()
        ap_vec = self.val_ap_per_label.compute()
        per_label_logs = {}
        for i, name in enumerate(self.label_names):
            per_label_logs[f'val_per_label/f1_{name}'] = f1_vec[i]
            per_label_logs[f'val_per_label/auroc_{name}'] = auroc_vec[i]
            per_label_logs[f'val_per_label/ap_{name}'] = ap_vec[i]
        self.log_dict(
            per_label_logs,
            on_step=False, on_epoch=True, prog_bar=False, logger=True,
            sync_dist=True,
        )

        self.val_metrics.reset()
        self.val_f1_per_label.reset()
        self.val_auroc_per_label.reset()
        self.val_ap_per_label.reset()

    # ----------------------------------------------------------- checkpointing
    def on_save_checkpoint(self, checkpoint):
        # Copied verbatim from the original leaf files (Lightning bookkeeping
        # so progress bars and resume work with accumulate_grad_batches > 1).
        checkpoint['_total'] = self.trainer.num_training_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total']['completed'] = (
                checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                    'optimizer']['step']['total']['completed']
                * self.trainer.accumulate_grad_batches)
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current']['completed'] = (
                checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                    'optimizer']['step']['current']['completed']
                * self.trainer.accumulate_grad_batches)
        checkpoint['loops']['fit_loop']['epoch_loop.state_dict'][
            '_batches_that_stepped'] = (
                checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                    'optimizer']['step']['total']['completed'])

    def on_load_checkpoint(self, checkpoint):
        logger.info('loading checkpoint')
        state = checkpoint['loops']['fit_loop'].get('state_dict', {})
        if 'combined_loader' in state:
            self.resumed_dataloader_state_from_ckpt = (
                state['combined_loader'][0])
        else:
            self.resumed_dataloader_state_from_ckpt = None
            logger.warning(
                'combined_loader not found in checkpoint; '
                'dataloader state will not be restored.')

    # ----------------------------------------------------------- optimizer
    def num_training_steps(self) -> int:
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        self.trainer.fit_loop.setup_data()
        if self.trainer.train_dataloader is None or self.trainer.max_epochs is None:
            raise ValueError('Trainer must have a train_dataloader.')
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = (dataset_size * self.trainer.max_epochs
                     // self.trainer.accumulate_grad_batches)
        if num_steps == 0:
            raise ValueError(
                'num_steps is zero. Check your training configuration.')
        return num_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.ensemble_predictor.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
        if self.trainer.max_epochs is None:
            raise ValueError(
                'Lightning currently does not support warmup with '
                '`max_epochs=None`. Set max_epochs to a positive integer.')

        def warmup_lr_lambda(current_step):
            return min(1.0, current_step /
                       max(1.0, self.config.optim.num_warmup_steps))

        global_step = self.trainer.global_step
        last_epoch = global_step - 1
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=warmup_lr_lambda, last_epoch=last_epoch)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_training_steps(),
            eta_min=self.config.optim.min_learning_rate,
            last_epoch=last_epoch,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            last_epoch=last_epoch,
            milestones=[self.config.optim.num_warmup_steps],
        )
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }]

    # ----------------------------------------------------------- inference API
    def predict(
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
            complete_sequence=True,
        )

        protein_input_ids = self.prepare_protein_inputs_for_model(
            protein_sequence, self.protein_tokenizer,
        ).to(self.device)

        utr5_length = len(utr5_sequence)
        utr3_length = len(utr3_sequence)
        cds_length = len(cds_sequence) // 3
        sequence_length = 8 + utr5_length + utr3_length + cds_length
        assert sequence_length == rna_input_ids.shape[1], (
            f'Expected sequence length {sequence_length}, '
            f'got {rna_input_ids.shape[1]}')

        batch_size = rna_input_ids.shape[0]
        modality_type_ids = self.create_modality_type_tensor(
            batch_size=batch_size,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length,
        )
        special_token_mask = self.create_special_token_mask(
            batch_size=batch_size,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            cds_length=cds_length,
        )
        species_ids = (torch.ones(batch_size, dtype=torch.int64,
                                  device=self.device) * species_id).reshape(-1, 1)

        input_batch = self.preprocess(
            rna_input_ids=rna_input_ids.to(self.device),
            protein_input_ids=protein_input_ids,
            species_ids=species_ids,
            modality_input_ids=modality_type_ids,
        )
        input_batch['special_token_mask'] = special_token_mask
        return self.forward(input_batch)

    def preprocess(self, rna_input_ids, protein_input_ids, species_ids,
                   modality_input_ids):
        batch_size = rna_input_ids.shape[0]
        rna_padding_mask = rna_input_ids.ne(
            self.global_tokenizer.padding_idx).to(torch.long).to(self.device)
        protein_padding_mask = protein_input_ids.ne(
            self.protein_tokenizer.vocab['<pad>']).to(torch.long).to(
                self.device).repeat(batch_size, 1)
        species_padding_mask = torch.ones(
            (batch_size, 1), dtype=torch.long).to(self.device)
        L_total = (rna_padding_mask.shape[1] + protein_padding_mask.shape[1]
                   + species_ids.shape[1])
        joint_masking = torch.cat(
            [species_padding_mask, protein_padding_mask, rna_padding_mask],
            dim=1)
        arange_tensor = torch.arange(L_total).unsqueeze(0).expand(
            batch_size, L_total).to(self.device)
        product = joint_masking * (arange_tensor + 1)
        product[product == 0] = L_total + 1
        row_wise_col_perms = torch.argsort(
            product, dim=1, descending=False, stable=True).to(self.device)
        inverse_indices = torch.empty_like(row_wise_col_perms).to(self.device)
        inverse_indices.scatter_(1, row_wise_col_perms, arange_tensor)
        attention_mask = torch.gather(
            joint_masking, dim=1, index=row_wise_col_perms).to(torch.int64)
        utr5_mask = (modality_input_ids == modality_map['utr_5']).to(torch.long)
        utr3_mask = (modality_input_ids == modality_map['utr_3']).to(torch.long)
        cds_mask = (modality_input_ids == modality_map['cds']).to(torch.long)
        modality_mask = utr5_mask + utr3_mask + cds_mask

        protein_embeddings = self.protein_encoder(
            protein_input_ids).embeddings
        protein_embeddings = protein_embeddings.repeat(
            batch_size, 1, 1).to(self.device)
        return {
            'rna_input_ids': rna_input_ids,
            'protein_input_ids': protein_input_ids,
            'row_wise_col_perms': row_wise_col_perms,
            'inverse_row_wise_col_perms': inverse_indices,
            'attention_mask': attention_mask,
            'modality_mask': modality_mask,
            'protein_embeddings': protein_embeddings,
            'utr5_mask': utr5_mask,
            'utr3_mask': utr3_mask,
            'cds_mask': cds_mask,
            'species_ids': species_ids,
            'modality_type_ids': modality_input_ids,
        }

    def create_modality_type_tensor(self, batch_size, utr5_length,
                                    utr3_length, cds_length):
        seq_len = 8 + utr5_length + utr3_length + cds_length
        m = torch.ones(batch_size, seq_len, dtype=torch.int64).to(self.device)
        m[:, 0] = modality_map['global_special_tokens']
        m[:, 1:3 + utr5_length] = modality_map['utr_5']
        m[:, 3 + utr5_length:5 + utr5_length + cds_length] = modality_map['cds']
        m[:, 5 + utr5_length + cds_length
          :7 + utr5_length + cds_length + utr3_length] = modality_map['utr_3']
        m[:, 7 + utr5_length + cds_length + utr3_length:] = (
            modality_map['global_special_tokens'])
        return m

    def create_special_token_mask(self, batch_size, utr5_length,
                                  utr3_length, cds_length):
        seq_len = 8 + utr5_length + utr3_length + cds_length
        m = torch.zeros(batch_size, seq_len, dtype=torch.int64).to(self.device)
        for idx in (
            0,
            1,
            2 + utr5_length,
            3 + utr5_length,
            4 + utr5_length + cds_length,
            5 + utr5_length + cds_length,
            6 + utr5_length + cds_length + utr3_length,
            7 + utr5_length + cds_length + utr3_length,
        ):
            m[:, idx] = 1
        return m

    def prepare_protein_inputs_for_model(self, protein_sequence,
                                         protein_tokenizer):
        with torch.no_grad():
            if isinstance(protein_sequence, str):
                protein_sequence = [protein_sequence]
            elif isinstance(protein_sequence, list):
                protein_sequence = [
                    s for s in protein_sequence if isinstance(s, str)]
            else:
                raise ValueError(
                    'protein_sequence must be a string or a list of strings.')
            protein_input_ids = esm_tokenize(
                protein_sequence, protein_tokenizer).to(self.device)
        return protein_input_ids
