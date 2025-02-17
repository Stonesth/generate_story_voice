import os
import logging
import re
import time
import numpy as np
import soundfile as sf
from TTS.api import TTS
from scipy.io import wavfile
from scipy import signal

# Initialiser le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des voix pour chaque personnage
VOICE_CONFIG = {
    "NARRATOR": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/recorded_voice_Pierre.wav",
        "speed": 1.0,
        "effects": {"reverb": 0.1}
    },
    "SANTA": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/santa_voice.wav",  # Voix plus grave
        "speed": 0.85,
        "effects": {"reverb": 0.3}  # Plus de réverbération pour Santa
    },
    "ELF1": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/elf1_voice.wav",  # Voix plus aiguë
        "speed": 1.2,
        "effects": {"reverb": 0.1}
    },
    "ELF2": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/elf2_voice.wav",
        "speed": 1.15,
        "effects": {"reverb": 0.1}
    },
    "ELF3": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/elf3_voice.wav",
        "speed": 1.1,
        "effects": {"reverb": 0.1}
    },
    "ELF_TINSEL": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/elf3_voice.wav",
        "speed": 1.1,
        "effects": {"reverb": 0.1}
    },
    "CHILD1": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/child_voice.wav",  # Voix d'enfant
        "speed": 1.1,
        "effects": {"reverb": 0.05}
    },
    "SHADOW": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_file": "voice/recorded_voice_Dimitri.wav",  # Voix d'enfant
        "speed": 1.1,
        "effects": {"reverb": 0.05}
    }
}

# Configuration des fichiers
STORY_FILE = "./story/story_14.txt"
OUTPUT_FILE = "story_output/long_christmas_story.wav"
OUTPUT_NAME = "_character_voice"
OUTPUT_LANGUAGE = "fr"

# Définir une variable pour contrôler la suppression des fichiers temporaires
DELETE_TEMP_FILES = False

def extract_tagged_sections(text):
    """
    Extrait les sections de texte avec leurs tags correspondants.
    Retourne une liste de tuples (tag, texte).
    """
    pattern = r'\[(.*?)\](.*?)(?=\[|$)'
    matches = re.finditer(pattern, text, re.DOTALL)
    sections = []
    
    for match in matches:
        tag = match.group(1).strip()
        content = match.group(2).strip()
        sections.append((tag, content))
    
    return sections

def get_voice_for_tag(tag):
    """
    Retourne la configuration de voix correspondant au tag.
    Si aucune voix n'est définie, utilise la voix du narrateur par défaut.
    """
    return VOICE_CONFIG.get(tag, VOICE_CONFIG["NARRATOR"])

def apply_audio_effects(audio, sample_rate, effects):
    """
    Applique des effets audio au signal.
    """
    if "reverb" in effects:
        # Simulation simple de réverbération
        reverb_amount = effects["reverb"]
        delay = int(sample_rate * 0.1)  # 100ms delay
        decay = np.exp(-6.0 * np.arange(delay) / sample_rate)
        reverb = np.zeros_like(audio)
        reverb[delay:] = audio[:-delay] * decay[-1]
        audio = audio + reverb_amount * reverb
    
    return audio

def process_audio(audio_path, config):
    """
    Traite l'audio avec les effets configurés
    """
    sample_rate, audio = wavfile.read(audio_path)
    
    # Normaliser l'audio
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    
    # Appliquer les effets
    if "effects" in config:
        audio = apply_audio_effects(audio, sample_rate, config["effects"])
    
    # Reconvertir en int16
    audio = np.int16(audio * np.iinfo(np.int16).max)
    return audio, sample_rate

# Lire le texte à partir du fichier
with open(STORY_FILE, 'r', encoding='utf-8') as file:
    text = file.read()

# Extraire les sections taguées
tagged_sections = extract_tagged_sections(text)

# Initialiser l'instance TTS avec un modèle multilingue
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Générer l'audio pour chaque section
audio_segments = []
samplerate = None

for i, (tag, content) in enumerate(tagged_sections):
    if content.strip():
        voice_config = get_voice_for_tag(tag)
        output_file = f"story_output/section_{i}{OUTPUT_NAME}.wav"
        
        # Générer l'audio avec la voix appropriée
        tts.tts_to_file(
            text=content.strip(),
            file_path=output_file,
            speaker_wav=[voice_config["voice_file"]],
            language=OUTPUT_LANGUAGE,
            speed=voice_config["speed"]
        )
        
        # Traiter l'audio avec les effets
        audio, sample_rate = process_audio(output_file, voice_config)
        
        # Lire le segment audio généré
        if samplerate is None:
            samplerate = sample_rate
        audio_segments.append(audio)

# Combiner tous les segments audio
combined_audio = np.concatenate(audio_segments)

# Écrire le fichier audio final
sf.write(OUTPUT_FILE, combined_audio, samplerate)

# Nettoyage des fichiers temporaires si nécessaire
if DELETE_TEMP_FILES:
    for i in range(len(tagged_sections)):
        temp_file = f"story_output/section_{i}{OUTPUT_NAME}.wav"
        if os.path.exists(temp_file):
            os.remove(temp_file)

logger.info("Génération de l'histoire terminée avec succès !")
