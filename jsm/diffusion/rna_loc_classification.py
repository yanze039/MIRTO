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
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from jsm.diffusion.decoding_utils import DiffusionDecodingConstraint
import torch.nn.functional as F


LOG2 = math.log(2)
logger = utils.get_logger(__name__)


class PredictionHead(nn.Module):
    def __init__(self, hidden_size, output_dim, expansion=1, dropout=0.1):
        super().__init__()
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * expansion, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform is a good default for GELU networks
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Optional: scale down final layer for stability
        final_linear = self.prediction_head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.xavier_uniform_(final_linear.weight, gain=0.5)

    def forward(self, x):
        # x: [batch, hidden] or [batch, seq_len, hidden]
        logits = self.prediction_head(x)
        # shape:
        # [batch, output_dim]
        # or [batch, seq_len, output_dim]
        return logits


class ShallowEnsembleHead(nn.Module):
    def __init__(self, hidden_size, output_dim, num_heads=5, expansion=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            PredictionHead(hidden_size, output_dim, expansion=expansion, dropout=dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # x: [batch, hidden] or [batch, seq_len, hidden]
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        # shape:
        # [batch, num_heads, output_dim]
        # or [batch, num_heads, seq_len, output_dim]
        return logits


class JointSequenceDiffusionClassifier(Diffusion):
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
        if self.config.prediction.freeze_backbone:
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        self.translation_lm_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.launch_timestamp = time.time()
        self.resumed_dataloader_state_from_ckpt = None
        self.score_cg = None
        self.ensemble_predictor = ShallowEnsembleHead(
            hidden_size=self.config.model.hidden_size,
            output_dim=self.config.prediction.output_dim,
            num_heads=self.config.prediction.num_heads,
            expansion=self.config.prediction.expansion,
            dropout=self.config.prediction.dropout
        )
        num_labels = self.config.prediction.output_dim
        self.label_names = self.config.label_names
        if len(self.label_names) != num_labels:
            raise ValueError("label_names length must match num_labels")
        self.threshold = self.config.get("threshold", 0.4)
        self.val_metrics = MetricCollection({
            "hamming_acc": MultilabelAccuracy(
                num_labels=num_labels,
                threshold=self.threshold,
                average="micro",
            ),
            "precision_micro": MultilabelPrecision(
                num_labels=num_labels,
                threshold=self.threshold,
                average="micro",
            ),
            "recall_micro": MultilabelRecall(
                num_labels=num_labels,
                threshold=self.threshold,
                average="micro",
            ),
            "f1_micro": MultilabelF1Score(
                num_labels=num_labels,
                threshold=self.threshold,
                average="micro",
            ),
            "f1_macro": MultilabelF1Score(
                num_labels=num_labels,
                threshold=self.threshold,
                average="macro",
            ),
            "auroc_macro": MultilabelAUROC(
                num_labels=num_labels,
                average="macro",
            ),
            "auroc_micro": MultilabelAUROC(
                num_labels=num_labels,
                average="micro",
            ),
            "ap_macro": MultilabelAveragePrecision(
                num_labels=num_labels,
                average="macro",
            ),
            "ap_micro": MultilabelAveragePrecision(
                num_labels=num_labels,
                average="micro",
            ),
        })
        self.val_f1_per_label = MultilabelF1Score(
            num_labels=num_labels,
            threshold=self.threshold,
            average=None,
        )
        self.val_auroc_per_label = MultilabelAUROC(
            num_labels=num_labels,
            average=None,
        )
        self.val_ap_per_label = MultilabelAveragePrecision(
            num_labels=num_labels,
            average=None,
        )
    
    def on_fit_start(self):
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
        if self.config.prediction.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def on_train_epoch_start(self):
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
        if self.config.prediction.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
    
    
    def forward(self, 
                x, 
                batch,
        ):
        """Returns log score."""
        with torch.no_grad():
            protein_output = self.protein_encoder(batch["protein_input_ids"])
            protein_embeddings = protein_output.embeddings
            
            embedding = self.backbone.get_embeddings(
                input_ids=batch['rna_input_ids'],
                species_ids=batch['species_ids'],                 
                protein_embeddings=protein_embeddings,
                row_wise_col_perms=batch["row_wise_col_perms"],
                inverse_row_wise_col_perms=batch["inverse_row_wise_col_perms"],
                attention_mask=batch['attention_mask'],
                modality_type_ids=batch['modality_type_ids'],
                modality_mask=batch['modality_mask'],
                layer_indices=[self.config.prediction.layer_index]
            )
        
        utr5_mask = batch['utr5_mask'] * (1-batch['special_token_mask'])
        cds_mask = batch['cds_mask'] * (1-batch['special_token_mask'])
        utr3_mask = batch['utr3_mask'] * (1-batch['special_token_mask'])
        utr5_embedding = embedding[self.config.prediction.layer_index] * utr5_mask.unsqueeze(-1)
        cds_embedding = embedding[self.config.prediction.layer_index] * cds_mask.unsqueeze(-1)
        utr3_embedding = embedding[self.config.prediction.layer_index] * utr3_mask.unsqueeze(-1)
        pooled_utr5_embedding = utr5_embedding.sum(dim=1) / (utr5_mask.sum(dim=1, keepdim=True) + 1e-8)
        pooled_cds_embedding = cds_embedding.sum(dim=1) / (cds_mask.sum(dim=1, keepdim=True) + 1e-8)
        pooled_utr3_embedding = utr3_embedding.sum(dim=1) / (utr3_mask.sum(dim=1, keepdim=True) + 1e-8)
        pooled_embeddings = (pooled_utr5_embedding + pooled_cds_embedding + pooled_utr3_embedding) / 3
        output = self.ensemble_predictor(pooled_embeddings.float())
        return output  # [batchsize, n_predictor, hidden_dim]
    
    def _loss(self, logits, labels):
        
        
        losses = []
        for h in range(logits.size(1)):
            loss_h = F.binary_cross_entropy_with_logits(logits[:, h, :], labels)
            losses.append(loss_h)
        return torch.stack(losses).mean()

    def _compute_loss(self, batch, prefix):
        output = self.forward(batch['rna_input_ids'], batch)
        labels = batch['rnaloc_label'].float()
        
        losses = self._loss(output, labels)
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
                f"{prefix}/loss": losses
            },
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        return losses
    
    def _ensemble_probs(self, logits):
        return torch.sigmoid(logits).mean(dim=1)  # [B, C]
    
    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='train')
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(name='trainer/lr',
                 value=lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return losses

    def on_validation_epoch_start(self):
        self.backbone.eval()
        self.noise.eval()
        self.val_metrics.reset()
        self.val_f1_per_label.reset()
        self.val_auroc_per_label.reset()
        self.val_ap_per_label.reset()

    def validation_step(self, batch, batch_idx):
        # losses = self._compute_loss(batch, prefix='val')
        output = self.forward(batch['rna_input_ids'], batch)
        labels = batch['rnaloc_label'].float()
        
        losses = self._loss(output, labels)
        probs = self._ensemble_probs(output)
        
        self.val_metrics.update(probs, labels.int())
        self.val_f1_per_label.update(probs, labels.int())
        self.val_auroc_per_label.update(probs, labels.int())
        self.val_ap_per_label.update(probs, labels.int())
        
        self.log(
            "val/loss",
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
        global_metrics = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in global_metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        f1_per_label = self.val_f1_per_label.compute()
        auroc_per_label = self.val_auroc_per_label.compute()
        ap_per_label = self.val_ap_per_label.compute()

        per_label_logs = {}
        for i, name in enumerate(self.label_names):
            per_label_logs[f"val_per_label/f1_{name}"] = f1_per_label[i]
            per_label_logs[f"val_per_label/auroc_{name}"] = auroc_per_label[i]
            per_label_logs[f"val_per_label/ap_{name}"] = ap_per_label[i]
        
        self.log_dict(
            per_label_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        # Optional: print a compact summary to stdout
        print("\nPer-label validation metrics:")
        for i, name in enumerate(self.label_names):
            f1_i = float(f1_per_label[i].detach().cpu())
            auroc_i = float(auroc_per_label[i].detach().cpu())
            ap_i = float(ap_per_label[i].detach().cpu())
            print(
                f"  {name:>12s} | "
                f"F1={f1_i:.4f} | "
                f"AUROC={auroc_i:.4f} | "
                f"AP={ap_i:.4f}"
            )

        self.val_metrics.reset()
        self.val_f1_per_label.reset()
        self.val_auroc_per_label.reset()
        self.val_ap_per_label.reset()
    
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

    
    def on_load_checkpoint(self, checkpoint):
        print("loading checkpoint")
        if "combined_loader" in checkpoint['loops']['fit_loop']['state_dict']:
            self.resumed_dataloader_state_from_ckpt = checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]
        else:
            self.resumed_dataloader_state_from_ckpt = None
            print("Warning: combined_loader not found in checkpoint. Dataloader state will not be restored.")
    
    
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
    
    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
            self.ensemble_predictor.parameters(),
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
        special_token_mask = self.create_special_token_mask(
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
        input_batch['special_token_mask'] = special_token_mask
        output = self.forward(input_batch)
        return output
        
    
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
            "rna_input_ids": rna_input_ids,
            "protein_input_ids": protein_input_ids,
            "row_wise_col_perms": row_wise_col_perms,
            "inverse_row_wise_col_perms": inverse_indices,
            "attention_mask": attention_mask,
            "modality_mask": modality_mask,
            "protein_embeddings": protein_embeddings,
            "utr5_mask": utr5_mask,
            "utr3_mask": utr3_mask,
            "cds_mask": cds_mask,    
            "species_ids": species_ids,
            "modality_type_ids": modality_input_ids
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


    def create_special_token_mask(
            self,
            batch_size,
            utr5_length,
            utr3_length,
            cds_length
        ):
        sequence_length = (8 + utr5_length + utr3_length + cds_length)
        special_token_mask = torch.zeros(batch_size, sequence_length, dtype=torch.int64).to(self.device)
        special_token_mask[:,0] = 1
        special_token_mask[:,1] = 1
        special_token_mask[:,2+utr5_length] = 1
        special_token_mask[:,3+utr5_length] = 1
        special_token_mask[:,4+utr5_length+cds_length] = 1
        special_token_mask[:,5+utr5_length+cds_length] = 1
        special_token_mask[:,6+utr5_length+cds_length+utr3_length] = 1
        special_token_mask[:,7+utr5_length+cds_length+utr3_length] = 1
        
        return special_token_mask
    
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