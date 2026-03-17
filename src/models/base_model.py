import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchaudio.transforms as T
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from abc import ABC, abstractmethod
import wandb
import time

from src.config import AudioConfig

class BaseModel(pl.LightningModule, ABC):
    """
    Abstract base class for all models in this project.

    Handles the full training loop:
        - training_step with optional Mixup
        - validation_step with F1, accuracy, confusion matrix
        - metric logging to W&B
        - epoch timing
    """

    def __init__(self, lr:float):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        self.lr=lr
        self.num_classes = len(AudioConfig.GENRES)


        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.val_acc=MulticlassAccuracy(num_classes=self.num_classes)

        # Loging train_f1 to analyse overfitting signals
        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")

        #Predictions for confusion matrix
        self.y_true = []
        self.y_pred = []

        # For epoch timing
        self.train_epoch_start_time = None
        self.val_epoch_start_time = None

    @abstractmethod
    def forward(self, x):
        # Forward pass method to be implemented by all subclasses.
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x) # Forward Pass
        loss = F.cross_entropy(logits, y)
        
        # Compute train predictions for F1 tracking
        preds = torch.argmax(logits, dim=1)
        self.train_f1.update(preds, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self):
        # Log train F1 at end of training phase (before validation runs)
        train_f1 = self.train_f1.compute()
        self.log("train_macro_f1", train_f1, prog_bar=True)
        self.train_f1.reset()

        # Log training epoch time
        if self.train_epoch_start_time is not None:
            train_time = time.time() - self.train_epoch_start_time
            self.log("train_epoch_time_sec", train_time, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x) # Forward Pass
        loss= F.cross_entropy(logits, y)
        preds=torch.argmax(logits, dim=1)

        # update metrics
        self.val_f1.update(preds, y)
        self.val_acc.update(preds, y)

        # For confusion matrix
        self.y_pred.append(preds.detach()) # Detach to avoid memory issues
        self.y_true.append(y.detach()) 

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_epoch_start_time = time.time()

    def on_validation_epoch_end(self):
        # final metrics from batch results
        f1 = self.val_f1.compute()
        acc = self.val_acc.compute()

        # val_macro_f1 is monitored by ModelCheckpoint for saving best model
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
        if self.val_epoch_start_time is not None:
            self.log("val_epoch_time_sec",
                     time.time() - self.val_epoch_start_time,
                     prog_bar=False)