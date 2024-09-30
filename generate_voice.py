import os
from gtts import gTTS

# Texte à convertir en audio
text = "Bonjour, ceci est une voix clonée!"

# Initialiser gTTS
tts = gTTS(text=text, lang='fr')

# Chemin du fichier de sortie
output_path = 'voice/output.wav'

# Créer le répertoire s'il n'existe pas
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Enregistrer le fichier audio
tts.save(output_path)