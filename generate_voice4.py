import os
import logging
import re
import time
import numpy as np
import soundfile as sf
from TTS.api import TTS

# Initialiser le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOICE_OF_ME = "voice/recorded_voice_Dorota.wav"
STORY_FILE = "./story/guild_meeting_2.txt"
STORY_FILE_OUTPUT = "./story_output/guild_meeting_2_output.txt"
OUTPUT_FILE = "voice/guild_meeting_2.wav"
OUTPUT_LANGUAGE = "en"

# Définir une variable pour contrôler la suppression des fichiers temporaires
DELETE_TEMP_FILES = False

# Créer une fonction "split_sentences" qui permet de faire un split des phrases. 
# Cette fonction permet de diviser le texte en phrases toutes les 3 phrases.
# Mais il faut considéré les phrases entre guillemets comme une seule phrase.
def split_sentences(text, split_every=3):
    sentences = []
    current_sentence = []
    quote_open = False

    for part in re.split(r'(\s*["“”]\s*|\s*[.!?]\s*)', text):
        if part.strip() in ['"', '“', '”']:
            quote_open = not quote_open
            current_sentence.append(part)
        elif not quote_open and re.match(r'\s*[.!?]\s*', part):
            current_sentence.append(part)
            if len(current_sentence) >= split_every * 2 - 1:
                sentences.append(''.join(current_sentence).strip())
                current_sentence = []
        else:
            current_sentence.append(part)

    if current_sentence:
        sentences.append(''.join(current_sentence).strip())

    return sentences

# Lire le texte à partir du fichier
with open(STORY_FILE, 'r', encoding='utf-8') as file:
    text = file.read()

# Supprimer les retours à la ligne et les espaces en trop
text = re.sub(r'\n', ' ', text)
text = re.sub(r'\s+', ' ', text)

# # Supprimer les espaces avant les ponctuations
# text = re.sub(r'\s([.,!?])', r'\1', text)

# # Supprimer les espaces après les ponctuations
# text = re.sub(r'([.,!?])\s', r'\1', text)

# # Supprimer les espaces avant et après les ponctuations
# text = re.sub(r'\s([.,!?])\s', r'\1', text)

# # Supprimer les espaces avant les parenthèses ouvrantes
# text = re.sub(r'\s(\()', r'\1', text)

# # Supprimer les espaces après les parenthèses fermantes
# text = re.sub(r'(\))\s', r'\1', text)

# # Supprimer les espaces avant et après les parenthèses
# text = re.sub(r'\s(\()', r'\1', text)

# # Supprimer les espaces avant les crochets ouvrants
# text = re.sub(r'\s(\[)', r'\1', text)

# # Supprimer les espaces après les crochets fermants
# text = re.sub(r'(\])\s', r'\1', text)

# # Supprimer les espaces avant et après les crochets
# text = re.sub(r'\s(\[)', r'\1', text)

# # Supprimer les espaces avant les accolades ouvrantes
# text = re.sub(r'\s(\{)', r'\1', text)

# # Supprimer les espaces après les accolades fermantes
# text = re.sub(r'(\})\s', r'\1', text)

# # Supprimer les espaces avant et après les accolades
# text = re.sub(r'\s(\{)', r'\1', text)

# # Supprimer les espaces avant les chevrons ouvrants
# text = re.sub(r'\s(\<)', r'\1', text)

# # Supprimer les espaces après les chevrons fermants
# text = re.sub(r'(\>)\s', r'\1', text)

# # Supprimer les espaces avant et après les chevrons
# text = re.sub(r'\s(\<)', r'\1', text)

# # Supprimer les espaces avant les deux-points
# text = re.sub(r'\s(:)', r'\1', text)

# # Supprimer les espaces après les deux-points
# text = re.sub(r'(:)\s', r'\1', text)

# # Supprimer les espaces avant et après les deux-points
# text = re.sub(r'\s(:)', r'\1', text)

# # Supprimer les espaces avant les points-virgules
# text = re.sub(r'\s(;)', r'\1', text)

# # Supprimer les espaces après les points-virgules
# text = re.sub(r'(;)\s', r'\1', text)

# # Supprimer les espaces avant et après les points-virgules
# text = re.sub(r'\s(;)', r'\1', text)

# # Supprimer les espaces avant les guillemets ouvrants
# text = re.sub(r'\s(\")', r'\1', text)

# # Supprimer les espaces après les guillemets fermants
# text = re.sub(r'(\")\s', r'\1', text)

# # Supprimer les espaces avant et après les guillemets
# text = re.sub(r'\s(\")', r'\1', text)

# # Supprimer les espaces avant les apostrophes
# text = re.sub(r'\s(\')', r'\1', text)

# # Supprimer les espaces après les apostrophes
# text = re.sub(r'(\')\s', r'\1', text)

# # Supprimer les espaces avant et après les apostrophes
# text = re.sub(r'\s(\')', r'\1', text)

# # Supprimer les espaces avant les tirets
# text = re.sub(r'\s(-)', r'\1', text)

# # Supprimer les espaces après les tirets
# text = re.sub(r'(-)\s', r'\1', text)

# # Supprimer les espaces avant et après les tirets
# text = re.sub(r'\s(-)', r'\1', text)

# place le nouveau fichier texte dans le dossier story
with open(STORY_FILE_OUTPUT, 'w', encoding='utf-8') as file:
    file.write(text)

# Diviser le texte en phrases toutes les 3 phrases
# sentences = re.split(r'((?:[^.!?]*[.!?]){3})', text)
sentences = split_sentences(text, split_every=3)

# Filtrer les phrases vides résultantes du split
sentences = [sentence for sentence in sentences if sentence.strip()]

# Sauvegarder chaque sentence dans un fichier .txt séparé.
for i, sentence in enumerate(sentences):
    with open(f"story_output/sentence_{i}.txt", 'w', encoding='utf-8') as file:
        file.write(sentence.strip())

# Initialiser l'instance TTS avec un modèle multilingue
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Convertir chaque phrase en audio et mise du texte dans un fichier séparé
for i, sentence in enumerate(sentences):
    if sentence.strip():  # Ignorer les phrases vides
        # generate speech by cloning a voice using default settings
        tts.tts_to_file(
            text=sentence.strip(),
            file_path=f"story_output/sentence_{i}.wav",
            speaker_wav=[VOICE_OF_ME],
            language=OUTPUT_LANGUAGE,
            split_sentences=True
        )

# Lire tous les fichiers audio générés et les combiner en un seul fichier
combined_audio = []

for i in range(len(sentences)):
    if sentences[i].strip():  # Ignorer les phrases vides
        data, samplerate = sf.read(f"story_output/sentence_{i}.wav")
        combined_audio.append(data)

# Combiner tous les fichiers audio en un seul tableau numpy
combined_audio = np.concatenate(combined_audio)

# Écrire le fichier audio combiné
sf.write(OUTPUT_FILE, combined_audio, samplerate)



if DELETE_TEMP_FILES:
    # Supprimer les fichiers temporaires après lecture
    for i in range(len(sentences)):
        if sentences[i].strip():  # Ignorer les phrases vides
            file_path = f"story_output/sentence_{i}.wav"
            os.remove(file_path)

    # Supprimer les fichiers temporaires après lecture
    for i in range(len(sentences)):
        if sentences[i].strip():  # Ignorer les phrases vides
            file_path = f"story_output/sentence_{i}.txt"
            os.remove(file_path)
