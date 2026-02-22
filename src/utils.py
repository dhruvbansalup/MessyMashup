import kagglehub
import os

from src.config import AudioConfig

def genre_to_idx(genre):
    return AudioConfig.GENRES.index(genre)

def idx_to_genre(idx):
    return AudioConfig.GENRES[idx]

def time_now_ist():
    from datetime import datetime
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

def kagglehub_upload_model(model, trainer):
    #getting the best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path

    if not best_ckpt:
        raise ValueError("No checkpoint saved!")

    #handle
    VARIATION=model.__class__.__name__.lower()
    handle =  f"{os.environ['KAGGLE_USERNAME']}/{os.environ['KAGGLEHUB_MODEL_REPO']}/pytorch/{VARIATION}"
    
    model_ref = kagglehub.model_upload(handle,
                                       local_model_dir=best_ckpt,
                                       version_notes=f"{model.__class__.__name__} trained at {time_now_ist}"
                                      )
    print(f"{VARIATION} Model Uploaded successdully")

def kagglehub_download_model(model_handle, ckpt_name):
    print(f"Downloading model from KaggleHub: {model_handle}")
    model_path = kagglehub.model_download(model_handle)
    
    local_path = os.path.join(model_path, ckpt_name) 
    print(f"Model downloaded to: {local_path}")

    return local_path

def load_model_from_checkpoint(model_class, checkpoint_path, load_to_device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    model = model_class.load_from_checkpoint(checkpoint_path).to(load_to_device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully from checkpoint: {checkpoint_path} to device: {load_to_device}")
    return model