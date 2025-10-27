from lightning.pytorch.callbacks import TQDMProgressBar
import lightning.pytorch as L

class LitProgressBar(TQDMProgressBar):
    
    def __init__(self, refresh_rate, leave=True):
        super().__init__(refresh_rate=refresh_rate, leave=leave)
        self._total = None
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        resumed_batches = checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped']
        self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(total=checkpoint.get('_total', None))
        self.train_progress_bar.n = resumed_batches
        self._total = checkpoint.get('_total', None)
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(total=trainer.num_training_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        self._total = None
        return super().on_train_epoch_end(trainer, pl_module)
        
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if self._total is not None:
            bar.reset(total=self._total)
        return bar
    
    # def on_before_backward(self, trainer, pl_module):
    #     print("on_before_backward called")
    #     import pdb; pdb.set_trace()  # Debugging line, can be removed later
    #     pass

    # def on_after_backward(self, trainer, pl_module):
    #     print("on_after_backward called")
    #     import pdb; pdb.set_trace()  # Debugging line, can be removed later
    #     pass