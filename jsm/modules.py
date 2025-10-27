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
from jsm.data.utils import codon_table, modality_map, ConcatenatedAlphabet
from jsm.data.joint_sequence import prepare_inputs_for_model, tokenize_inputs
try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None
    
from pytorch_lightning.utilities.model_summary import ModelSummary

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
    perplexity: Optional[torch.FloatTensor] = None
    rna_lm_loss_codon: Optional[torch.FloatTensor] = None
    rna_lm_loss_utr5: Optional[torch.FloatTensor] = None
    rna_lm_loss_utr3: Optional[torch.FloatTensor] = None


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
        utr_3_tokenizer = None
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
        if self.config.backbone == 'mamba2':
            from jsm.models.mamba import JointSequenceMambaModel
            self.backbone = JointSequenceMambaModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
                num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
            )
        elif self.config.backbone == 'hyena':
            from jsm.models.vortex_striped_hyena import JointSequenceStripedHyenaModel
            self.backbone = JointSequenceStripedHyenaModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
                num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
            )
        elif self.config.backbone == 'hyena_nemo':
            # We init model in the setup stage, because nemo needs the megatron initialization 
            # to be done after the trainer is setup
            self.backbone = None
        elif self.config.backbone == 'hybrid_mamba':
            from jsm.models.hybrid_mamba import JointSequenceMambaModel
            self.backbone = JointSequenceMambaModel(
                self.config.model,
                rna_vocab_size=rna_vocab_size,
                protein_vocab_size=protein_vocab_size,
                num_modalities=5,  # 5 modalities: 5' UTR, CDS, 3' UTR, padding, others)
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
        self.translation_lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.protein_tokenizer.vocab["<pad>"],
        )
        self.rna_lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
        )
        self.modality_prediction_loss_fn = nn.CrossEntropyLoss(
            ignore_index=modality_map["padding"],
        )
        self.launch_timestamp = time.time()
        self.codon_tokenizer = codon_tokenizer
        self.utr_5_tokenizer = utr_5_tokenizer
        self.utr_3_tokenizer = utr_3_tokenizer
        self.strict_loading = False
        self.resumed_dataloader_state_from_ckpt = None
    
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
        
        rna_logits, codon_protein_translation_logits, modality_logits = self.backbone(
            input_ids=batch["rna_input_ids"],                 
            protein_embeddings=protein_embeddings,
            row_wise_col_perms=batch["row_wise_col_perms"],
            inverse_row_wise_col_perms=batch["inverse_row_wise_col_perms"],
            attention_mask=batch["attention_mask"],
            seq_idx=batch["seq_idx"],
            inference_params=None, 
        )
        
        return rna_logits, codon_protein_translation_logits, modality_logits
    
    def _loss(self, batch, prefix="train"):
        rna_logits, codon_protein_translation_logits, modality_logits = self.forward(batch)
        
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

        # utils.print_nans(codon_protein_translation_logits, 'codon_protein_translation_logits')
        # utils.print_nans(modality_logits, 'modality_logits')
        
        labels = batch["rna_input_ids"]
        labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], self.rna_lm_loss_fn.ignore_index)), 1)
        rna_lm_loss = self.rna_lm_loss_fn(rna_logits.view(labels.numel(), -1), labels.view(-1))
        
        if hasattr(self.config, "utr_only"):
            if self.config.utr_only:
                return Loss(
                    loss=rna_lm_loss,
                    rna_lm_loss=rna_lm_loss.detach(),
                    perplexity=torch.exp(rna_lm_loss).detach(),
                    translation_loss=0,
                    num_codon_aa_errors=1,
                    total_aa_length=1,
                    modality_prediction_loss=0,
                    rna_lm_loss_codon=0,
                    rna_lm_loss_utr5=0,
                    rna_lm_loss_utr3=0
                )
        
        if codon_protein_translation_logits is not None:
            codon_protein_translation_lm_loss = self.translation_lm_loss_fn(
                codon_protein_translation_logits[batch["translation_rna_mask"].bool()].view(-1, self.protein_vocab_size),
                batch["protein_input_ids"][batch["translation_protein_mask"].bool()].view(-1)
            )
        else:
            codon_protein_translation_lm_loss = 0
        if modality_logits is not None:
            modality_prediction_loss = self.modality_prediction_loss_fn(
                modality_logits.view(-1, modality_logits.size(-1)),
                batch["modality_type_ids"].view(-1)
            )
        else:
            modality_prediction_loss = 0
        
        with torch.no_grad():
            rna_perplexity = torch.exp(rna_lm_loss)
            if prefix != "train":
                codon_labels = labels[batch["translation_rna_mask"].bool()]
                utr_5_mask = batch["modality_type_ids"] == modality_map["utr_5"]
                utr_3_mask = batch["modality_type_ids"] == modality_map["utr_3"]
                utr_5_labels = labels[utr_5_mask]
                utr_3_labels = labels[utr_3_mask]
                codon_labels = labels[batch["translation_rna_mask"].bool()]
                rna_lm_loss_codon = self.rna_lm_loss_fn(
                    rna_logits[batch["translation_rna_mask"].bool()].view(codon_labels.numel(), -1), 
                    codon_labels.view(-1)
                )

                rna_lm_loss_utr5 = self.rna_lm_loss_fn(
                    rna_logits[utr_5_mask].view(utr_5_labels.numel(), -1), 
                    utr_5_labels.view(-1)
                )
                rna_lm_loss_utr3 = self.rna_lm_loss_fn(
                    rna_logits[utr_3_mask].view(utr_3_labels.numel(), -1), 
                    utr_3_labels.view(-1)
                )
            else:
                rna_lm_loss_codon = None
                rna_lm_loss_utr5 = None
                rna_lm_loss_utr3 = None
        
        return Loss(
            loss=rna_lm_loss + codon_protein_translation_lm_loss + modality_prediction_loss,
            # loss=codon_protein_translation_lm_loss,
            rna_lm_loss=rna_lm_loss.detach(),
            translation_loss=codon_protein_translation_lm_loss.detach() if codon_protein_translation_logits is not None else None,
            modality_prediction_loss=modality_prediction_loss.detach() if modality_logits is not None else None,
            perplexity=rna_perplexity.detach(),
            num_codon_aa_errors=n_errors,
            total_aa_length=total_length,
            rna_lm_loss_codon=rna_lm_loss_codon,
            rna_lm_loss_utr5=rna_lm_loss_utr5,
            rna_lm_loss_utr3=rna_lm_loss_utr3
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
            
        
        self.log(
            f"{prefix}/loss",
            losses.loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log(
            f"{prefix}/perplexity",
            losses.perplexity,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log(
            f"{prefix}/rna_lm_loss",
            losses.rna_lm_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        if losses.translation_loss is not None:
            self.log(
                f"{prefix}/translation_loss",
                losses.translation_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size
            )
        if losses.modality_prediction_loss is not None:
            self.log(
                f"{prefix}/modality_prediction_loss",
                losses.modality_prediction_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size
            )
        
        if prefix == 'train':
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log(name='trainer/lr',
                    value=lr,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True)
        
        if prefix == 'val':
            if hasattr(losses, "num_codon_aa_errors"):
                self.codon_aa_errors.append(losses.num_codon_aa_errors)
            if hasattr(losses, "total_aa_length"):
                self.total_length.append(losses.total_aa_length)
            
            self.log(name="rna_lm_loss_codon",
                     value=losses.rna_lm_loss_codon,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
            self.log(name="rna_lm_loss_utr5",
                     value=losses.rna_lm_loss_utr5,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
            self.log(name="rna_lm_loss_utr3",
                     value=losses.rna_lm_loss_utr3,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
        return losses
        
    def on_fit_start(self):
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()

    def on_train_epoch_start(self):
        # freeze protein encoder
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
        print(f"Elapsed time: {elapsed_time // 3600} hours, "
                  f"{(elapsed_time % 3600) // 60} minutes, "
                  f"{elapsed_time % 60} seconds")
        if elapsed_time // 3600 > 1:
            self.trainer.should_stop = True
        self.backbone.train()
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        self.protein_encoder.eval()
    
    # def on_optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        
    
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
            expected_utr_3_length=None
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
        
        
        ctokenizer = ConcatenatedAlphabet(
            [
                global_tokenizer, codon_tokenizer, utr_tokenizer
            ]
        )
        if self.config.backbone == 'hybrid_mamba':
            from jsm.generation.flash_attention_generation import decode
        else:
            from jsm.generation.mamba_generation import decode
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