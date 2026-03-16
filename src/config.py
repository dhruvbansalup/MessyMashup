from dotenv import load_dotenv
import os

load_dotenv()

class AudioConfig:
    SAMPLE_RATE=22050
    DURATION=30
    SAMPLES_PER_TRACK=SAMPLE_RATE * DURATION

    # Spectogram parameters
    N_FFT=2048
    HOP_LENGTH=512
    N_MELS=128


    GENRES=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    STEM_FILES=["vocals.wav", "drums.wav", "bass.wav", "other.wav"]

class TrainConfig:
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    VAL_SPLIT = 0.1
    MAX_EPOCHS = 10
    SEED = 42
    LR=1e-3

def setup_kaggle():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username is None or key is None:
        raise ValueError("KAGGLE_USERNAME or KAGGLE_KEY not found in .env")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    print("Kaggle credentials set")

class EnvConfig:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")

    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")

    DATA_DIR = os.getenv("DATA_DIR", "data/raw/messy_mashup")
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    TEST_WAV_DIR = os.getenv("TEST_WAV_DIR", "data/raw/messy_mashup/mashups")
    TEST_CSV = os.getenv("TEST_CSV", "data/raw/messy_mashup/test.csv")

    # Output directories
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "outputs/checkpoints")

def setup_environment():
    # Wandb
    if EnvConfig.WANDB_API_KEY:
        import wandb
        wandb.login(key=EnvConfig.WANDB_API_KEY)
        print("Logged in to Weights & Biases successfully!")
    
    # Kaggle
    if EnvConfig.KAGGLE_USERNAME and EnvConfig.KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = EnvConfig.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = EnvConfig.KAGGLE_KEY
        print("Kaggle credentials set")
