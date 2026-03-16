from tqdm import tqdm
import pandas as pd
import torch
from src.utils import kagglehub_download_model, load_model_from_checkpoint
from src.config import EnvConfig, TrainConfig
from src.utils import idx_to_genre

def inference(MODEL_CLASS, MODEL_HANDLE, CKPT_NAME, batch_size, num_workers, test_wav_dir=EnvConfig.TEST_WAV_DIR, test_csv=EnvConfig.TEST_CSV):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running infrence on: {device}")

    # Downloading the model from KaggleHub
    model_path=kagglehub_download_model(MODEL_HANDLE, CKPT_NAME)

    #Loading Model
    model=load_model_from_checkpoint(MODEL_CLASS, model_path, load_to_device="gpu" if torch.cuda.is_available() else "cpu")

    # Building test dataset
    from src.data.dataset import TestDataset
    test_dataset = TestDataset(test_wav_dir, test_csv)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device=="cuda")
    )

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            waveforms, file_names = batch
            waveforms = waveforms.to(device)

            #Spectrogram conversion
            specs  = model.waveform_to_spectogram(waveforms)

            logits = model(specs)                      
            preds  = torch.argmax(logits, dim=1)     
            genres = [idx_to_genre(p.item()) for p in preds.cpu()]
            all_predictions.extend(genres)
    
    submission_df = pd.DataFrame({
        "id": test_dataset.ids,
        "genre": all_predictions
    })

    return submission_df

if __name__ == "__main__":

    MODEL_CLASS=None
    MODEL_HANDLE=None
    CKPT_NAME=None

    submission_df=inference(MODEL_CLASS, MODEL_HANDLE, CKPT_NAME, batch_size=32, num_workers=4)
    submission_df.to_csv("outputs/submissions/submission.csv", index=False)
    print("Inference completed. Submission file 'submission.csv' created.")