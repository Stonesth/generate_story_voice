import os
from gtts import gTTS

# Texte à convertir en audio
text = """Il était une fois, dans un royaume lointain, un petit village paisible niché au pied d'une montagne majestueuse. Dans ce village vivait un jeune garçon nommé Léo. Léo rêvait de devenir un chevalier courageux, comme ceux des contes que sa grand-mère lui racontait le soir.

Un jour, alors que Léo jouait près de la rivière, il entendit un bruit étrange venant des montagnes. Intrigué, il décida de suivre le son et grimpa la montagne avec détermination. Après une longue ascension, il découvrit une grotte cachée derrière un rideau de lierre.

À l'intérieur de la grotte, Léo trouva un petit dragon blessé. Le dragon, nommé Flamme, avait une aile cassée et semblait très faible. Léo, avec son cœur plein de compassion, décida de l'aider. Il utilisa son mouchoir pour bander l'aile du dragon et lui donna à boire de l'eau fraîche de la rivière.

Les jours suivants, Léo rendit visite à Flamme tous les jours, apportant de la nourriture et des soins. Petit à petit, le dragon retrouva ses forces et son aile guérit. Flamme fut émerveillé par la gentillesse de Léo et décida de lui offrir un cadeau en retour.

"Je vais t'apprendre à voler," dit Flamme avec un sourire. "Mais en échange, tu devras promettre de toujours aider ceux qui en ont besoin."

Léo accepta avec joie et, ensemble, ils s'envolèrent dans le ciel. Léo apprit à voler sur le dos de Flamme, survolant les montagnes et les vallées. Il découvrit des paysages magnifiques et des créatures étonnantes.

De retour au village, Léo raconta son aventure à ses amis et à sa famille. Tous étaient émerveillés par son courage et sa gentillesse. Léo devint un héros pour les habitants du village, et il continua à aider ceux qui en avaient besoin, fidèle à sa promesse.

Et ainsi, Léo réalisa son rêve de devenir un chevalier courageux, non pas avec une épée et une armure, mais avec son cœur."""

# Initialiser gTTS
tts = gTTS(text=text, lang='fr')

# Chemin du fichier de sortie
output_path = 'voice/output1.wav'

# Créer le répertoire s'il n'existe pas
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Enregistrer le fichier audio
tts.save(output_path)