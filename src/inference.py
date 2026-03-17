import torchaudio
from tqdm import tqdm
import pandas as pd
import torch
from src.data.datamodule import MashupDataModule
from src.utils import kagglehub_download_model, load_model_from_checkpoint
from src.config import EnvConfig, AudioConfig
from src.utils import idx_to_genre

def inference(MODEL_CLASS, MODEL_HANDLE, CKPT_NAME, batch_size, num_workers, test_wav_dir=EnvConfig.TEST_WAV_DIR, test_csv=EnvConfig.TEST_CSV):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running infrence on: {device}")

    # Downloading the model from KaggleHub
    model_path=kagglehub_download_model(MODEL_HANDLE, CKPT_NAME)

    #Loading Model
    model=load_model_from_checkpoint(MODEL_CLASS, model_path, load_to_device="cuda" if torch.cuda.is_available() else "cpu")

    # Building test dataset
    from src.data.dataset import TestDataset
    
    data_module = MashupDataModule(
        processed_data_dir=None, # Not needed for inference
        test_wav_dir=test_wav_dir,
        test_csv=test_csv,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            spec, file_names = batch
            spec = spec.to(device)
                        
            logits = model(spec) # Forward Pass                   
            preds  = torch.argmax(logits, dim=1)     
            genres = [idx_to_genre(p.item()) for p in preds.cpu()]
            all_predictions.extend(genres)
    
    submission_df = pd.DataFrame({
        "id": data_module.test_dataset.ids,
        "genre": all_predictions
    })

    return submission_df

if __name__ == "__main__":
    from src.models.simple_cnn_02 import SimpleCNN02

    MODEL_CLASS=SimpleCNN02
    MODEL_HANDLE='dhruvbansalup/dl-genai-project-26-t1-messy-mashup/pytorch/simplecnn02'
    CKPT_NAME='model_class_name0-epoch02-val_macro_f10.6853.ckpt'

    submission_df=inference(MODEL_CLASS, MODEL_HANDLE, CKPT_NAME, batch_size=32, num_workers=4)
    submission_df.to_csv("outputs/submissions/submission.csv", index=False)
    print("Inference completed. Submission file 'submission.csv' created.")