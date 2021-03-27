from egg.core import CheckpointSaver
from egg.core.interaction import Interaction

class BestEpochCheckpointSaver(CheckpointSaver):
    def __init__(self, checkpoint_path, checkpoint_freq=1, prefix='', max_checkpoints=1000, metric = "acc"):
        super().__init__(checkpoint_path, 
                         checkpoint_freq=checkpoint_freq, 
                         prefix=prefix, 
                         max_checkpoints=max_checkpoints)
        self.metric = metric
        self.best_epoch_score = 0.0
        
    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        """
            Save checkpoint of the model if new high score obtained.
        """
        self.epoch_counter = epoch
        current_epoch_score = logs.aux[self.metric].mean()
        if current_epoch_score > self.best_epoch_score:
            filename = f"best_{self.prefix}" if self.prefix else str(epoch)
            self.save_checkpoint(filename=filename)
            self.best_epoch_score = current_epoch_score
            
            