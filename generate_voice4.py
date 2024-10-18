import os
from TTS.api import TTS
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm
import time

# Initialiser le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lire le texte à partir du fichier
with open('story_3.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Initialiser l'instance TTS
tts = TTS("tts_models/de/thorsten/tacotron2-DDC")

# Simuler une barre de progression
for i in tqdm(range(100), desc="Conversion TTS en cours"):
    time.sleep(0.1)  # Simuler le temps de traitement

# Convertir le texte en audio
tts.tts_with_vc_to_file(
    text,
    speaker_wav="voice/audio.wav",
    file_path="voice/output4.wav"
)