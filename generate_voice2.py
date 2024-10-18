import os
from TTS.api import TTS
import logging
import soundfile as sf
import numpy as np

# Initialiser le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser l'instance TTS
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)

# Chemin absolu ou relatif correct vers le fichier audio
speaker_wav = "voice/audio.wav"

# Vérifiez si le fichier existe
if not os.path.exists(speaker_wav):
    raise FileNotFoundError(f"Le fichier {speaker_wav} n'existe pas.")

# Générer des fichiers audio temporaires pour chaque phrase
# temp_files = [
#     ("Bonjour Pierre, comment tu vas ?", "fr-fr", "voice/temp_1_fr.wav"),
#     ("C'est le clonage de ta voix que on va tester de faire.", "fr-fr", "voice/temp_2_fr.wav"),
#     ("Mais pour ça on a encore beaucoup de travail", "fr-fr", "voice/temp_3_fr.wav")
# ]

# Générer des fichiers audio temporaires pour chaque phrase
temp_files = [
    ("Il était une fois, dans un royaume lointain, un petit village paisible niché au pied d'une montagne majestueuse.", "fr-fr", "voice/temp_1_fr.wav"),
    ("Dans ce village vivait un jeune garçon nommé Léo.", "fr-fr", "voice/temp_2_fr.wav"),
    ("Léo rêvait de devenir un chevalier courageux, comme ceux des contes que sa grand-mère lui racontait le soir.", "fr-fr", "voice/temp_3_fr.wav"),
    ("Un jour, alors que Léo jouait près de la rivière, il entendit un bruit étrange venant des montagnes.", "fr-fr", "voice/temp_4_fr.wav"),
    ("Intrigué, il décida de suivre le son et grimpa la montagne avec détermination.", "fr-fr", "voice/temp_5_fr.wav"),
    ("Après une longue ascension, il découvrit une grotte cachée derrière un rideau de lierre.", "fr-fr", "voice/temp_6_fr.wav"),
    ("À l'intérieur de la grotte, Léo trouva un petit dragon blessé.", "fr-fr", "voice/temp_7_fr.wav"),
    ("Le dragon, nommé Flamme, avait une aile cassée et semblait très faible.", "fr-fr", "voice/temp_8_fr.wav"),
    ("Léo, avec son cœur plein de compassion, décida de l'aider.", "fr-fr", "voice/temp_9_fr.wav"),
    ("Il utilisa son mouchoir pour bander l'aile du dragon et lui donna à boire de l'eau fraîche de la rivière.", "fr-fr", "voice/temp_10_fr.wav"),
    ("Les jours suivants, Léo rendit visite à Flamme tous les jours, apportant de la nourriture et des soins.", "fr-fr", "voice/temp_11_fr.wav"),
    ("Petit à petit, le dragon retrouva ses forces et son aile guérit.", "fr-fr", "voice/temp_12_fr.wav"),
    ("Flamme fut émerveillé par la gentillesse de Léo et décida de lui offrir un cadeau en retour.", "fr-fr", "voice/temp_13_fr.wav"),
    ("'Je vais t'apprendre à voler,' dit Flamme avec un sourire.", "fr-fr", "voice/temp_14_fr.wav"),
    ("'Mais en échange, tu devras promettre de toujours aider ceux qui en ont besoin.'", "fr-fr", "voice/temp_15_fr.wav"),
    ("Léo accepta avec joie et, ensemble, ils s'envolèrent dans le ciel.", "fr-fr", "voice/temp_16_fr.wav"),
    ("Léo apprit à voler sur le dos de Flamme, survolant les montagnes et les vallées.", "fr-fr", "voice/temp_17_fr.wav"),
    ("Il découvrit des paysages magnifiques et des créatures étonnantes.", "fr-fr", "voice/temp_18_fr.wav"),
    ("De retour au village, Léo raconta son aventure à ses amis et à sa famille.", "fr-fr", "voice/temp_19_fr.wav"),
    ("Tous étaient émerveillés par son courage et sa gentillesse.", "fr-fr", "voice/temp_20_fr.wav"),
    ("Léo devint un héros pour les habitants du village, et il continua à aider ceux qui en avaient besoin, fidèle à sa promesse.", "fr-fr", "voice/temp_21_fr.wav"),
    ("Et ainsi, Léo réalisa son rêve de devenir un chevalier courageux, non pas avec une épée et une armure, mais avec son cœur.", "fr-fr", "voice/temp_22_fr.wav")
]

# Générer les fichiers audio temporaires
for text, lang, temp_file in temp_files:
    logger.info(f"Génération du fichier audio en {lang}")
    tts.tts_to_file(text, speaker_wav=speaker_wav, language=lang, file_path=temp_file)

# Concaténer les fichiers audio en un seul fichier
output_path = "voice/output.wav"
data = []
for _, _, temp_file in temp_files:
    temp_data, samplerate = sf.read(temp_file)
    data.append(temp_data)

# Écrire les données concaténées dans le fichier de sortie
sf.write(output_path, np.concatenate(data), samplerate)

# Supprimer les fichiers temporaires
for _, _, temp_file in temp_files:
    os.remove(temp_file)

logger.info("Fichier audio généré et sauvegardé avec succès")