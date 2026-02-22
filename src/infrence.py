def inference(MODEL_CLASS, MODEL_HANDLE, CKPT_NAME):
    import torch
    from src.utils import kagglehub_download_model, load_model_from_checkpoint

    # Downloading the model from KaggleHub
    model_path=kagglehub_download_model(MODEL_HANDLE, CKPT_NAME)

    #Loading Model
    model=load_model_from_checkpoint(MODEL_CLASS, model_path, load_to_device="gpu" if torch.cuda.is_available() else "cpu")

    @torch.no_grad() # Disable gradient calculation
    def predict():
        pass

if __name__ == "__main__":

    MODEL_CLASS=None
    MODEL_HANDLE=None
    CKPT_NAME=None

    inference(MODEL_CLASS, MODEL_HANDLE, CKPT_NAME)