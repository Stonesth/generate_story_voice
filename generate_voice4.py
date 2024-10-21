import os
from TTS.api import TTS
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm
import time
import re

# Initialiser le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lire le texte à partir du fichier
with open('./story/story_5.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Diviser le texte en phrases toutes les 3 phrases
sentences = re.split(r'((?:[^.!?]*[.!?]){3})', text)
# Filtrer les phrases vides résultantes du split
sentences = [sentence for sentence in sentences if sentence.strip()]

# Initialiser l'instance TTS avec un modèle français
# tts = TTS("tts_models/fr/mai/tacotron2-DDC")
# tts = TTS("tts_models/fr/css10/vits")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")





# # Simuler une barre de progression
# for i in tqdm(range(100), desc="Conversion TTS en cours"):
#     time.sleep(0.1)  # Simuler le temps de traitement

# Convertir chaque phrase en audio
for i, sentence in enumerate(sentences):
    if sentence.strip():  # Ignorer les phrases vides
        # tts.tts_with_vc_to_file(
        #     sentence.strip(),
        #     speaker_wav="voice/audio.wav",
        #     file_path=f"voice/output4_{i}.wav"
        # )
        # generate speech by cloning a voice using default settings
        tts.tts_to_file(text=sentence.strip(),
                file_path=f"voice/output4_{i}.wav",
                speaker_wav=["voice/audio.wav"],
                language="fr",
                split_sentences=True
                )



# Lire tous les fichiers audio générés et les combiner en un seul fichier
combined_audio = []

for i in range(len(sentences)):
    if sentences[i].strip():  # Ignorer les phrases vides
        data, samplerate = sf.read(f"voice/output4_{i}.wav")
        combined_audio.append(data)

# Combiner tous les fichiers audio en un seul tableau numpy
combined_audio = np.concatenate(combined_audio)

# Écrire le fichier audio combiné
sf.write("voice/combined_output.wav", combined_audio, samplerate)