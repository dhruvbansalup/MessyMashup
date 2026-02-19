from dotenv import load_dotenv
import os

load_dotenv()

GENRES=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
STEM_FILES=["vocals.wav", "drums.wav", "bass.wav", "other.wav"]

class Config:
    pass