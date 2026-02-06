import torch
import torch.nn.functional as F
from jsm.modules import JointSequenceModeling

def grad_norms_for_loss(loss, params, *, norm_type=2.0):
    """
    Returns global grad norm for `loss` wrt `params` (like clip_grad_norm_),
    computed via autograd.grad (does NOT populate .grad).
    """
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,   # keep graph so you can compute another loss' grads
        create_graph=False,
        allow_unused=True,
    )
    total = 0.0
    for g in grads:
        if g is None:
            continue
        total += g.detach().norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def get_batch_logps(logits, labels, label_pad_token_id = -100):
    """
    Get log probabilities summed over all tokens for a given batch.
    
    Parameters
    ----------
    logits: [batch size, seq length, vocab size]
    labels: [batch size, seq length]

    Returns
    -------
    logps: [batch size]
    """
    # next token prediction: labels are inputs shifted by one
    targets = labels[:, 1:].clone()
    # truncate logits to match labels' number of tokens
    logits = logits[:, :-1, :]

    loss_mask = targets != label_pad_token_id
    
    # dummy token; we'll ignore loss on these tokens later
    targets[targets == label_pad_token_id] = 0
    
    log_probs = logits.log_softmax(dim=-1)
    # [batch size, seq length]
    per_token_logps = torch.gather(log_probs, dim=2, index=targets.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)



class DPOJointSequenceModeling(JointSequenceModeling):
    """Joint Sequence Modeling with DPO loss."""

    def __init__(self, beta=None, reference_model_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["protein_encoder", "tokenizer", "datamodule", "reference_model"])
        assert self.training_style == "species_specific", "DPOJointSequenceModeling only supports species_specific training style."
        self.beta = beta
        self.reference_model_path = reference_model_path
        self.reference_model = None

    def setup(self, stage: str):
        # Called on every rank; safe place to create/load big things.
        print("Setting up Reference Model DPOJointSequenceModeling...")
        if self.reference_model is None:
            ref = JointSequenceModeling.load_from_checkpoint(
                self.reference_model_path,
                config=self.config,
                global_tokenizer=self.global_tokenizer,
                protein_tokenizer=self.protein_tokenizer,
                rna_vocab_size=self.rna_vocab_size,
                protein_vocab_size=self.protein_vocab_size,
                protein_encoder=None,
            )
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)

            # IMPORTANT: put it on the same device + dtype as policy
            ref.to(device=self.device)

            # If you use bf16/fp16 mixed precision, reference should follow too
            # (Lightning will autocast policy forward; reference forward should run under autocast as well)
            self.reference_model = ref
        print("Reference Model setup complete.")
    
    def forward(
            self, 
            batch,
        ):
        """DPO forward pass.
            For each sample (x, y_w, y_l):
                compute log-prob of y_w given x
                compute log-prob of y_l given x
                do this for both policy and reference
                DPO loss pushes policy to prefer y_w over y_l relative to reference
        Args:
            batch: Input batch containing RNA and protein sequences.
        Returns:
            Dictionary with model outputs.
        """
        
        # ====> DPO Block <==== #
        dpo_batch = batch['dpo']
        with torch.no_grad():
            dpo_protein_output = self.protein_encoder(dpo_batch['chosen']["protein_input_ids"])
            dpo_protein_embeddings = dpo_protein_output.embeddings
        
        batch_chosen = dpo_batch['chosen']
        batch_rejected = dpo_batch['rejected']
        
        rna_logits_policy_chosen = self.backbone(
            input_ids=batch_chosen["rna_input_ids"],    
            species_ids=batch_chosen["species_ids"],             
            protein_embeddings=dpo_protein_embeddings,
            row_wise_col_perms=batch_chosen["row_wise_col_perms"],
            inverse_row_wise_col_perms=batch_chosen["inverse_row_wise_col_perms"],
            attention_mask=batch_chosen["attention_mask"],
            seq_idx=batch_chosen["seq_idx"],
            inference_params=None,
            modality_type_ids=batch_chosen['modality_type_ids'],
            modality_mask=batch_chosen['modality_mask'],
            return_middle_hidden_states=False,
        )
        rna_logits_policy_chosen_logps = get_batch_logps(
            rna_logits_policy_chosen,
            batch_chosen["rna_input_ids"],
            label_pad_token_id=self.padding_index,
        )
        
        rna_logits_policy_rejected = self.backbone(
            input_ids=batch_rejected["rna_input_ids"],    
            species_ids=batch_rejected["species_ids"],             
            protein_embeddings=dpo_protein_embeddings,
            row_wise_col_perms=batch_rejected["row_wise_col_perms"],
            inverse_row_wise_col_perms=batch_rejected["inverse_row_wise_col_perms"],
            attention_mask=batch_rejected["attention_mask"],
            seq_idx=batch_rejected["seq_idx"],
            inference_params=None,
            modality_type_ids=batch_rejected['modality_type_ids'],
            modality_mask=batch_rejected['modality_mask'],
            return_middle_hidden_states=False,
        )
        rna_logits_policy_rejected_logps = get_batch_logps(
            rna_logits_policy_rejected,
            batch_rejected["rna_input_ids"],
            label_pad_token_id=self.padding_index,
        )
        
        with torch.no_grad():
            rna_logits_reference_chosen = self.reference_model.backbone(
                input_ids=batch_chosen["rna_input_ids"],    
                species_ids=batch_chosen["species_ids"],             
                protein_embeddings=dpo_protein_embeddings,
                row_wise_col_perms=batch_chosen["row_wise_col_perms"],
                inverse_row_wise_col_perms=batch_chosen["inverse_row_wise_col_perms"],
                attention_mask=batch_chosen["attention_mask"],
                seq_idx=batch_chosen["seq_idx"],
                inference_params=None,
                modality_type_ids=batch_chosen['modality_type_ids'],
                modality_mask=batch_chosen['modality_mask'],
                return_middle_hidden_states=False,
            )
            rna_logits_reference_chosen_logps = get_batch_logps(
                rna_logits_reference_chosen,
                batch_chosen["rna_input_ids"],
                label_pad_token_id=self.padding_index,
            )
            
            rna_logits_reference_rejected = self.reference_model.backbone(
                input_ids=batch_rejected["rna_input_ids"],    
                species_ids=batch_rejected["species_ids"],             
                protein_embeddings=dpo_protein_embeddings,
                row_wise_col_perms=batch_rejected["row_wise_col_perms"],
                inverse_row_wise_col_perms=batch_rejected["inverse_row_wise_col_perms"],
                attention_mask=batch_rejected["attention_mask"],
                seq_idx=batch_rejected["seq_idx"],
                inference_params=None,
                modality_type_ids=batch_rejected['modality_type_ids'],
                modality_mask=batch_rejected['modality_mask'],
                return_middle_hidden_states=False,
            )
            rna_logits_reference_rejected_logps = get_batch_logps(
                rna_logits_reference_rejected,
                batch_rejected["rna_input_ids"],
                label_pad_token_id=self.padding_index,
            )
        # ====> End DPO Block <==== #
        
        # ====> CE Block <==== #   
        ce_batch = batch['ce']         
        with torch.no_grad():
            ce_protein_output = self.protein_encoder(ce_batch["protein_input_ids"])
            ce_protein_embeddings = ce_protein_output.embeddings
        rna_logits = self.backbone(
            input_ids=ce_batch["rna_input_ids"],    
            species_ids=ce_batch["species_ids"],             
            protein_embeddings=ce_protein_embeddings,
            row_wise_col_perms=ce_batch["row_wise_col_perms"],
            inverse_row_wise_col_perms=ce_batch["inverse_row_wise_col_perms"],
            attention_mask=ce_batch["attention_mask"],
            seq_idx=ce_batch["seq_idx"],
            inference_params=None,
            modality_type_ids=ce_batch['modality_type_ids'],
            modality_mask=ce_batch['modality_mask'],
            return_middle_hidden_states=False,
        )
        # ====> End CE Block <==== #
            
        
        return (
            rna_logits_policy_chosen_logps,
            rna_logits_policy_rejected_logps,
            rna_logits_reference_chosen_logps,
            rna_logits_reference_rejected_logps,
            rna_logits,
        )
    
    def _compute_loss(self, batch, prefix=''):
        """Compute DPO loss."""
        rna_logits_policy_chosen_logps, rna_logits_policy_rejected_logps, \
            rna_logits_reference_chosen_logps, rna_logits_reference_rejected_logps, rna_logits = self(batch)
        
        if self.config.use_ipo_loss:
            dpo_loss = self.ipo_loss(
                policy_chosen_logps=rna_logits_policy_chosen_logps,
                policy_rejected_logps=rna_logits_policy_rejected_logps,
                reference_chosen_logps=rna_logits_reference_chosen_logps,
                reference_rejected_logps=rna_logits_reference_rejected_logps,
            )
        else:
            dpo_loss = self.dpo_loss(
                policy_chosen_logps=rna_logits_policy_chosen_logps,
                policy_rejected_logps=rna_logits_policy_rejected_logps,
                reference_chosen_logps=rna_logits_reference_chosen_logps,
                reference_rejected_logps=rna_logits_reference_rejected_logps,
            )
        
        ce_batch = batch['ce']
        labels = ce_batch["rna_input_ids"]
        labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], self.rna_lm_loss_fn.ignore_index)), 1)
        rna_lm_loss_per_token = self.rna_lm_loss_fn(rna_logits.reshape(labels.numel(), -1), labels.reshape(-1)).reshape(labels.shape)
        
        utr5_mask = ce_batch["utr5_mask"]
        cds_mask = ce_batch["cds_mask"]
        utr3_mask = ce_batch["utr3_mask"]
        
        rna_lm_loss_utr5 = (rna_lm_loss_per_token * utr5_mask).sum() / torch.clamp(utr5_mask.sum(), min=1.0)
        rna_lm_loss_codon = (rna_lm_loss_per_token * cds_mask).sum() / torch.clamp(cds_mask.sum(), min=1.0)
        rna_lm_loss_utr3 = (rna_lm_loss_per_token * utr3_mask).sum() / torch.clamp(utr3_mask.sum(), min=1.0)
        rna_lm_loss = (rna_lm_loss_utr5 + rna_lm_loss_codon + rna_lm_loss_utr3) / 3.0
        
        # import pdb; pdb.set_trace()
        # # params = [p for p in self.backbone.parameters() if p.requires_grad]

        # dpo_gn = grad_norms_for_loss(dpo_loss, params)
        # rna_lm_gn  = grad_norms_for_loss(rna_lm_loss,  params)
        
        loss = dpo_loss * self.config.dpo_weight + rna_lm_loss * self.config.rna_lm_weight
        
        
        if prefix == 'train':
            on_step = True
            on_epoch = False
            batch_size = batch['dpo']['chosen']['rna_input_ids'].shape[0]
        else:
            on_step = False
            on_epoch = True
            batch_size = batch['dpo']['chosen']['rna_input_ids'].shape[0]
            
        
        self.log_dict(
            {
                f"{prefix}/loss": loss,
                f"{prefix}/dpo_loss": dpo_loss,
                f"{prefix}/rna_lm_loss": rna_lm_loss,
                f"{prefix}/rna_lm_loss_utr5": rna_lm_loss_utr5,
                f"{prefix}/rna_lm_loss_codon": rna_lm_loss_codon,
                f"{prefix}/rna_lm_loss_utr3": rna_lm_loss_utr3,
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
        
        return loss
        
    def dpo_loss(
            self,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        ):
            """logps shape [batch size,]"""
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            logits = chosen_logratios - rejected_logratios
            loss = -F.logsigmoid(self.beta * logits).mean(dim=-1)
            return loss
    
    def ipo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        """
        IPO Loss implementation for RNA sequence alignment.
        logps shape: [batch size,]
        """
        # 1. Calculate the log-ratios relative to the reference model
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        # 2. Compute the difference in log-ratios
        # This is essentially the model's "preference margin"
        logits = chosen_logratios - rejected_logratios
        
        # 3. Apply the IPO Quadratic Loss
        # Instead of -logsigmoid(beta * logits), we use (logits - 1/(2*beta))^2
        # This forces the margin to stay close to the target rather than growing infinitely.
        loss = (logits - 1 / (2 * self.beta)) ** 2
        
        return loss.mean()
    
    def on_fit_start(self):
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()

    def on_train_epoch_start(self):
        print("Starting training epoch...")
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
    
    def on_train_epoch_end(self):
        pass
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix='train')        
        return loss
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        if self.config.backbone == 'hyena_nemo':
            torch.compiler.cudagraph_mark_step_begin()
        loss = self._compute_loss(batch, prefix='val')
        return loss
    
    def on_validation_epoch_end(self):
        pass
    
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
        self.resumed_dataloader_state_from_ckpt = checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]
    
    
