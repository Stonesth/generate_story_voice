# Simple_TTS

Script de synthèse vocale utilisant différents modèles pour générer de l'audio en français et en anglais.

## Installation

```bash
# Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install TTS
```

## Utilisation

Le script prend en charge différents modèles et voix. Voici les principales options :

### Modèles anglais (--lang 0 ou 2)

1. VITS (voix féminine) :
```bash
python Simple_TTS.py --lang 2 --en-model 1 --text-file text_en.txt --use-cuda
```

2. Tacotron2 (voix féminine) :
```bash
python Simple_TTS.py --lang 2 --en-model 2 --text-file text_en.txt --use-cuda
```

### Voix VCTK recommandées (--lang 2)

Le modèle VCTK offre plusieurs voix de haute qualité :

- VCTK_p232 (homme, bien)
- VCTK_p273 (femme, bien)
- VCTK_p278 (femme, bien)
- VCTK_p279 (homme, bien)
- VCTK_p304 (femme, voix préférée)

Pour utiliser une voix VCTK spécifique :
```bash
python Simple_TTS.py --lang 2 --en-model 3 --text-file text_en.txt --use-cuda --speaker VCTK_p304
```

### Modèle français (--lang 1)

Pour générer de l'audio en français avec le modèle VITS CSS10 :
```bash
python Simple_TTS.py --lang 1 --text-file text_fr.txt --use-cuda
```

## Options

- `--lang` : Langue (0: Anglais, 1: Français, 2: Anglais avec VCTK)
- `--text-file` : Chemin vers le fichier texte à lire
- `--use-cuda` : Utiliser CUDA si disponible
- `--en-model` : Modèle anglais (0: Tacotron2-DDC, 1: VITS, 2: Tacotron2, 3: VCTK)
- `--speaker` : ID du speaker pour VCTK (ex: VCTK_p304)
- `--length-scale` : Vitesse de la parole (< 1.0 plus rapide, > 1.0 plus lent)
