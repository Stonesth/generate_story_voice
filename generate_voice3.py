import os
from TTS.api import TTS
import logging
import soundfile as sf
import numpy as np

# Initialiser le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser l'instance TTS
tts = TTS("tts_models/de/thorsten/tacotron2-DDC")

# Convertir le texte en audio
tts.tts_with_vc_to_file(
    "Il était une fois, dans un royaume lointain, un petit village paisible niché au pied d'une montagne majestueuse.",
    speaker_wav="voice/audio.wav",
    file_path="voice/ouptut3.wav"
)