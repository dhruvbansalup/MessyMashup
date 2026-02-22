def train(MODEL):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from src.utils import time_now_ist
    from src.data.datamodule import MessyMashDataModule
    from src.config import TrainConfig, EnvConfig, setup_environment

    setup_environment()

    # Seed for reproducibility
    pl.seed_everything(TrainConfig.SEED)

    DATA_MODULE=MessyMashDataModule(
        data_dir=EnvConfig.DATA_DIR,
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=TrainConfig.NUM_WORKERS,
        val_split=TrainConfig.VAL_SPLIT,
    )

    # Model Checkpoint Callback Initialization
    checkpoint_callback=ModelCheckpoint(
        monitor="val_macro_f1",
        dirpath="checkpoints",
        mode="max",
        filename="{MODEL.__class__.__name__}-{epoch:02d}-{val_macro_f1:.4f}",
        save_top_k=3,
    )

    # Wandb Logger Initialization
    wandb_logger = WandbLogger(
        project=EnvConfig.WANDB_PROJECT,
        name=f"{MODEL.__class__.__name__}-{time_now_ist()}",
        log_model=True # Automatically log the best model checkpoint to W&B
    )

    # Pytorch Lightning Trainer Initialization
    trainer = pl.Trainer(
        max_epochs=TrainConfig.MAX_EPOCHS,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        log_every_n_steps=10,
        precision=TrainConfig.PRECISION,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(MODEL, datamodule=DATA_MODULE)
    wandb_logger.experiment.finish()

    # Uploading to KaggleHub
    from utils import kagglehub_upload_model
    kagglehub_upload_model(MODEL, trainer)

if __name__ == "__main__":
    from src.models.simple_cnn_01 import SimpleCNN01

    MODEL=SimpleCNN01()
    train(MODEL)