"""
Script de synthèse vocale utilisant différents modèles pour générer de l'audio en français et en anglais.
"""

import os
import argparse
import torch
import torch.nn as nn
from TTS.api import TTS
from TTS.utils.radam import RAdam
from torch.serialization import add_safe_globals, safe_globals
from collections import defaultdict, OrderedDict
import numpy
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import XttsAudioConfig, Xtts, XttsArgs
from TTS.tts.configs.bark_config import BarkConfig
from TTS.config.shared_configs import BaseDatasetConfig
from pathlib import Path
import importlib.util
import sys
import shutil
import numpy as np

# Ajouter les globals sécurisés pour PyTorch 2.6
add_safe_globals([
    RAdam,
    defaultdict,
    OrderedDict,
    numpy.ndarray,
    torch.nn.Parameter,
    torch._utils._rebuild_tensor_v2,
    torch.Tensor,
    dict,
    list,
    tuple,
    int,
    float,
    str,
    bool,
    type(None),  # NoneType
    BaseTTSConfig,
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    Xtts,
    np.core.multiarray.scalar,
    np.ndarray,
    np._globals._NoValue,
    np.dtype,
    np.ufunc,
    np.generic,
    BaseTTS,
    nn.Module,
    BarkConfig,
    BaseDatasetConfig,
])

# Configurer le chemin de cache pour Bark
cache_dir = str(Path.home() / ".cache" / "bark_tts")
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
os.environ["SUNO_OFFLOAD_CPU"] = "1"
os.environ["BARK_CACHE_DIR"] = cache_dir

# Créer le dossier de cache
os.makedirs(cache_dir, exist_ok=True)

# Copier les fichiers de modèle si nécessaire
def setup_bark_cache():
    try:
        bark_dir = Path(cache_dir) / "bark"
        os.makedirs(bark_dir, exist_ok=True)
        
        # Créer les sous-dossiers nécessaires
        for subdir in ["text_2.0", "coarse_2.0", "fine_2.0"]:
            os.makedirs(bark_dir / subdir, exist_ok=True)
        
        print(f"Cache Bark configuré dans : {cache_dir}")
        return True
    except Exception as e:
        print(f"Erreur lors de la configuration du cache Bark : {e}")
        return False

# Configurer le cache Bark
setup_bark_cache()

def read_text_file(file_path: str) -> str | None:
    """
    Lit le contenu d'un fichier texte.
    
    Args:
        file_path: Chemin vers le fichier texte à lire
        
    Returns:
        Le contenu du fichier ou None si une erreur survient
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {str(e)}")
        return None

def setup_argparse() -> argparse.ArgumentParser:
    """
    Configure et retourne le parseur d'arguments.
    
    Returns:
        Le parseur d'arguments configuré
    """
    parser = argparse.ArgumentParser(
        description='Générer de l\'audio à partir de texte en français ou en anglais'
    )
    parser.add_argument(
        '--lang', 
        type=int, 
        choices=[0, 1], 
        default=0,
        help='0 pour anglais, 1 pour français'
    )
    parser.add_argument(
        '--en-model',
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help='Modèle anglais à utiliser : 0 pour Tacotron2-DDC, 1 pour Glow-TTS, 2 pour Speedy-Speech, 3 pour VITS, 4 pour Jenny'
    )
    parser.add_argument(
        '--fr-model',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help='Modèle français à utiliser : 0 pour VITS, 1 pour Tacotron2-DDC, 2 pour YourTTS, 3 pour YourTTS avec speaker, 4 pour Bark, 5 pour XTTS v2'
    )
    parser.add_argument(
        '--text-file', 
        type=str, 
        required=True,
        help='Chemin vers le fichier texte à lire'
    )
    parser.add_argument(
        '--yourtts-speaker',
        type=str,
        choices=['male-en-2', 'female-en-5', 'female-pt-4', 'male-pt-3'],
        default='male-en-2',
        help='Speaker à utiliser pour YourTTS (uniquement pour --fr-model 3)'
    )
    parser.add_argument(
        '--bark-speaker',
        type=str,
        choices=['v2/fr_speaker_0', 'v2/fr_speaker_1', 'v2/fr_speaker_2', 'v2/fr_speaker_3', 'v2/fr_speaker_4', 'v2/fr_speaker_5'],
        default='v2/fr_speaker_3',
        help='Speaker à utiliser pour Bark (uniquement pour --fr-model 4)'
    )
    parser.add_argument(
        '--reference-audio',
        type=str,
        help='Fichier audio de référence pour XTTS v2'
    )
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Utiliser CUDA si disponible'
    )
    parser.add_argument(
        '--speaker',
        type=str,
        help='ID du speaker pour les modèles multi-voix (ex: p229 pour une voix masculine)',
        default='p229'
    )
    parser.add_argument(
        '--length-scale',
        type=float,
        default=1.0,
        help='Contrôle la vitesse de la parole (< 1.0 plus rapide, > 1.0 plus lent)'
    )
    parser.add_argument(
        '--vocoder',
        type=str,
        choices=['fullband-melgan', 'hifigan_v2'],
        default='fullband-melgan',
        help='Vocoder à utiliser pour la synthèse'
    )
    return parser

def get_model_name(lang: int, en_model: int = 0, fr_model: int = 0) -> str:
    """
    Retourne le nom du modèle en fonction de la langue choisie.
    
    Args:
        lang: 0 pour anglais, 1 pour français
        en_model: 0 pour Tacotron2-DDC, 1 pour Glow-TTS, 2 pour Speedy-Speech, 3 pour VITS, 4 pour Jenny
        fr_model: 0 pour VITS, 1 pour Tacotron2-DDC, 2 pour YourTTS, 3 pour YourTTS avec speaker, 4 pour Bark, 5 pour XTTS v2
        
    Returns:
        Le nom du modèle à utiliser
    """
    if lang == 0:  # Anglais
        models = {
            0: "tts_models/en/ljspeech/tacotron2-DDC",
            1: "tts_models/en/ljspeech/glow-tts",
            2: "tts_models/en/ljspeech/speedy-speech",
            3: "tts_models/en/vctk/vits",
            4: "tts_models/en/jenny/jenny"
        }
        return models.get(en_model, models[0])
    else:  # Français
        models = {
            0: "tts_models/fr/css10/vits",
            1: "tts_models/fr/css10/tacotron2-DDC",
            2: "tts_models/multilingual/multi-dataset/your_tts",
            3: "tts_models/multilingual/multi-dataset/your_tts",
            4: "tts_models/multilingual/multi-dataset/bark",
            5: "tts_models/multilingual/multi-dataset/xtts_v2"
        }
        return models.get(fr_model, models[0])

def get_model_suffix(lang: int, en_model: int = 0, fr_model: int = 0, speaker: str = None) -> str:
    """
    Retourne un suffixe distinctif pour le nom du fichier.
    
    Args:
        lang: 0 pour anglais, 1 pour français
        en_model: 0 pour Tacotron2-DDC, 1 pour Glow-TTS, 2 pour Speedy-Speech, 3 pour VITS, 4 pour Jenny
        fr_model: 0 pour VITS, 1 pour Tacotron2-DDC, 2 pour YourTTS, 3 pour YourTTS avec speaker, 4 pour Bark, 5 pour XTTS v2
        speaker: ID du speaker pour VCTK
        
    Returns:
        Un suffixe distinctif pour le fichier
    """
    if lang == 0:  # Anglais
        if en_model == 0:
            return "_en_tacotron"
        elif en_model == 1:
            return "_en_glowtts"
        elif en_model == 2:
            return "_en_speedyspeech"
        elif en_model == 3:
            return "_en_vits"
        elif en_model == 4:
            return "_en_jenny"
    else:  # Français
        if fr_model == 0:
            return "_fr_vits"
        elif fr_model == 1:
            return "_fr_tacotron"
        elif fr_model == 2:
            return "_fr_yourtts"
        elif fr_model == 3:
            return "_fr_yourtts"
        elif fr_model == 4:
            return "_fr_bark"
        elif fr_model == 5:
            return "_fr_xtts_v2"

def main():
    """Fonction principale du script."""
    # Parser les arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Vérifier si CUDA est disponible
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Utilisation du device : {device}")
    
    # Obtenir le nom du modèle
    model_name = get_model_name(args.lang, args.en_model, args.fr_model)
    print(f"Chargement du modèle : {model_name}")
    
    # Initialiser le modèle TTS avec les options appropriées
    if args.lang == 1:
        if args.fr_model == 3:  # YourTTS
            tts = TTS(model_name).to(device)
            print("\nSpeakers disponibles pour YourTTS :")
            print(tts.speakers)
            speaker = args.yourtts_speaker
            language = "fr-fr"
        elif args.fr_model == 4:  # Bark
            print(f"Utilisation du cache Bark dans : {cache_dir}")
            tts = TTS(model_name).to(device)
            print("\nSpeakers disponibles pour Bark :")
            print(tts.speakers)
            speaker = args.bark_speaker
            language = "fr"
        elif args.fr_model == 5:  # XTTS v2
            print("Utilisation de XTTS v2")
            if not args.reference_audio:
                print("Erreur : XTTS v2 nécessite un fichier audio de référence (--reference-audio)")
                sys.exit(1)
                
            # Patch la fonction load_fsspec pour désactiver weights_only
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            tts = TTS(model_name).to(device)
            # Restaurer la fonction originale
            torch.load = original_load
            speaker = None
            language = "fr"
            speaker_wav = args.reference_audio
        else:
            tts = TTS(model_name).to(device)
            speaker = None
            language = None
            speaker_wav = None
    else:
        tts = TTS(model_name).to(device)
        speaker = None
        language = None
        speaker_wav = None
    
    # Si c'est le modèle VCTK, afficher les speakers disponibles
    if args.en_model == 3:
        print("\nSpeakers disponibles :")
        print(tts.speakers)
    
    # Configurer les options spécifiques pour les différents modèles
    if args.lang == 1:
        if args.fr_model == 2:  # Tacotron2-DDC
            tts.synthesizer.length_scale = args.length_scale
        elif args.fr_model == 3:  # YourTTS
            tts.synthesizer.length_scale = args.length_scale
    
    # Lire le contenu du fichier texte
    text = read_text_file(args.text_file)
    if text is None:
        print(f"Erreur : Impossible de lire le fichier {args.text_file}")
        return
    
    # Générer le nom du fichier de sortie
    output_path = f"story_output/output{get_model_suffix(args.lang, args.en_model, args.fr_model, speaker)}.wav"
    
    # Générer l'audio
    if args.fr_model == 5:  # XTTS v2
        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=language)
    else:
        tts.tts_to_file(text=text, file_path=output_path, speaker=speaker, language=language)
    
    print(f"Fichier audio généré avec succès : {output_path}")

if __name__ == "__main__":
    main()