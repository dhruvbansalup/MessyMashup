import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from abc import ABC, abstractmethod
import wandb
import time

from src.config import AudioConfig

# BaseModel: Abstract base class for all models in the project

class BaseModel(pl.LightningModule, ABC):
    def __init__(self, lr:float):
        super().__init__()
        
        #Automatically saves hyperparameters
        self.save_hyperparameters() 
        
        self.lr=lr
        self.num_classes = len(AudioConfig.GENRES)
        
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.val_acc=MulticlassAccuracy(num_classes=self.num_classes)

        #Predictions for confusion matrix
        self.y_true = []
        self.y_pred = []

        self.epoch_start_time = None

    @abstractmethod
    def forward(self, x):
        # Forward pass method to be implemented by all subclasses.
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Forward Pass

        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Forward Pass

        loss= F.cross_entropy(y_hat, y)
        preds=torch.argmax(y_hat, dim=1)
        
        # update metrics
        self.val_f1.update(preds, y)
        self.val_acc.update(preds, y)

        # For confusion matrix
        self.y_pred.append(preds.detach()) # Detach to avoid memory issues
        self.y_true.append(y.detach()) 

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        acc = self.val_acc.compute()

        self.log("val_macro_f1", f1, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

        #  Concatenate all predictions and true labels for the confusion matrix
        preds = torch.cat(self.y_pred).cpu().tolist()
        trues = torch.cat(self.y_true).cpu().tolist()
        
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=preds,
                    y_true=trues,
                    class_names=AudioConfig.GENRES
                )
            })
        
        # Reset metrics & stored predictions for the next epoch
        self.val_f1.reset()
        self.val_acc.reset()
        self.y_true.clear()
        self.y_pred.clear()

        # Log epoch duration
        if self.epoch_start_time is not None:
            epoch_time_sec = time.time() - self.epoch_start_time
            self.log("epoch_time_sec", epoch_time_sec, prog_bar=False)