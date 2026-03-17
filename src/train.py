from src.config import setup_environment, EnvConfig, TrainConfig

def train(MODEL, log=True,batch_size=TrainConfig.BATCH_SIZE, num_workers=TrainConfig.NUM_WORKERS, processed_data_dir=EnvConfig.PROCESSED_DATA_DIR, test_wav_dir=EnvConfig.TEST_WAV_DIR, test_csv=EnvConfig.TEST_CSV):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from src.utils import time_now_ist
    from src.data.datamodule import MashupDataModule
    
    
    # Setting high precision for matrix multiplications to speed up training
    torch.set_float32_matmul_precision("high")

    # Seed for reproducibility
    pl.seed_everything(TrainConfig.SEED)

    DATA_MODULE=MashupDataModule(
        processed_data_dir=processed_data_dir,
        test_wav_dir=test_wav_dir,
        test_csv=test_csv,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_class_name=MODEL.__class__.__name__

    # Model Checkpoint Callback Initialization
    checkpoint_callback=ModelCheckpoint(
        monitor="val_macro_f1",
        dirpath=EnvConfig.CHECKPOINT_DIR,
        mode="max",
        filename="{model_class_name}-{epoch:02d}-{val_macro_f1:.4f}",
        save_top_k=3,
    )

    wandb_logger = None
    if log:
        # Wandb Logger Initialization
        wandb_logger = WandbLogger(
            project=EnvConfig.WANDB_PROJECT,
            name=f"{model_class_name}-{time_now_ist()}",
            log_model=True, # Automatically log the best model checkpoint to W&B
            save_dir=EnvConfig.OUTPUT_DIR
        )

    # Pytorch Lightning Trainer Initialization
    trainer = pl.Trainer(
        max_epochs=TrainConfig.MAX_EPOCHS,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        callbacks=[checkpoint_callback]
    )

    # Train the model
    print(f"Starting training for model: {model_class_name}")
    trainer.fit(MODEL, datamodule=DATA_MODULE)
    if log:
        wandb_logger.experiment.finish()
    print(f"Training completed for model: {model_class_name}")

    # Uploading to KaggleHub
    from src.utils import kagglehub_upload_model
    kagglehub_upload_model(MODEL, trainer)

if __name__ == "__main__":
    
    setup_environment()

    from src.models.simple_cnn_01 import SimpleCNN01
    from src.models.simple_cnn_02 import SimpleCNN02

    MODEL=SimpleCNN02(lr=TrainConfig.LR)
    train(MODEL, log=True, batch_size=20, num_workers=2)