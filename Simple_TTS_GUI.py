"""
Interface graphique pour Simple_TTS utilisant PyQt6
"""

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QTextEdit, QPushButton,
                            QFileDialog, QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

# Configuration pour PyTorch 2.6+
torch.serialization.add_safe_globals([XttsConfig])

class TTSWorker(QThread):
    """Thread worker pour la génération TTS"""
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # Configuration du device
            device = "cuda" if torch.cuda.is_available() and self.params['use_cuda'] else "cpu"
            self.progress.emit(f"Utilisation du device : {device}")

            # Obtention du modèle
            model_name = self.get_model_name()
            self.progress.emit(f"Chargement du modèle : {model_name}")

            # Patch temporaire pour PyTorch 2.6 si c'est XTTS v2
            if "xtts_v2" in model_name:
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load

            # Initialisation TTS
            tts = TTS(model_name).to(device)
            
            # Restauration de torch.load si c'était XTTS v2
            if "xtts_v2" in model_name:
                torch.load = original_load

            # Configuration des paramètres selon le modèle
            kwargs = {}
            
            if self.params['lang'] == 1:  # Français
                if self.params['fr_model'] == 4:  # XTTS v2
                    kwargs['speaker_wav'] = self.params.get('reference_audio')
                    kwargs['language'] = 'fr'
                elif self.params['fr_model'] in [2, 3]:  # YourTTS
                    kwargs['speaker'] = 'male-en-2' if self.params['fr_model'] == 2 else 'female-en-5'
                    kwargs['language'] = 'fr'
            elif self.params['lang'] == 2:  # VCTK
                speaker = self.params['speaker']
                if speaker.startswith("VCTK_"):
                    speaker = speaker[5:]
                kwargs['speaker'] = speaker

            # Génération de l'audio
            output_path = os.path.join(self.params['output_dir'], f"output{self.get_model_suffix()}.wav")
            
            # Appel TTS avec les paramètres appropriés
            tts.tts_to_file(text=self.params['text'], 
                           file_path=output_path,
                           **kwargs)
            
            self.progress.emit(f"Audio généré : {output_path}")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))

    def get_model_name(self):
        """Retourne le nom du modèle en fonction de la langue choisie."""
        models = {
            0: {  # Anglais
                0: "tts_models/en/ljspeech/tacotron2",  
                1: "tts_models/en/ljspeech/glow-tts",
                2: "tts_models/en/ljspeech/speedy-speech",
                3: "tts_models/en/vctk/vits",
                4: "tts_models/en/jenny/jenny"
            },
            1: {  # Français
                0: "tts_models/fr/css10/vits",
                1: "tts_models/fr/css10/tacotron2-DDC",
                2: "tts_models/multilingual/multi-dataset/your_tts",  
                3: "tts_models/multilingual/multi-dataset/your_tts",  
                4: "tts_models/multilingual/multi-dataset/xtts_v2"    
            },
            2: {  # Anglais avec VCTK
                3: "tts_models/en/vctk/vits"  
            }
        }
        return models[self.params['lang']].get(
            self.params['en_model'] if self.params['lang'] != 1 else self.params['fr_model'],
            models[self.params['lang']][0]
        )

    def get_model_suffix(self):
        """Retourne un suffixe distinctif pour le nom du fichier."""
        suffixes = {
            0: {  # Anglais
                0: "_en_tacotron2",
                1: "_en_glowtts",
                2: "_en_speedyspeech",
                3: "_en_fastpitch",
                4: "_en_jenny"
            },
            1: {  # Français
                0: "_fr_vits",
                1: "_fr_tacotron2",
                2: "_fr_yourtts",
                3: "_fr_yourtts",
                4: "_fr_xtts_v2"
            },
            2: {  # Anglais avec VCTK
                3: "_en_vctk_vits"  
            }
        }
        return suffixes[self.params['lang']].get(
            self.params['en_model'] if self.params['lang'] != 1 else self.params['fr_model'],
            "_unknown"
        )

class CustomTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.paste()
        elif event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.copy()
        elif event.key() == Qt.Key.Key_X and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.cut()
        elif event.key() == Qt.Key.Key_A and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.selectAll()
        else:
            super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple TTS GUI")
        self.setMinimumSize(600, 400)
        
        # Dossier de sortie par défaut
        self.output_dir = os.path.join(os.getcwd(), "story_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Dossier de sortie
        output_layout = QHBoxLayout()
        output_label = QLabel("Dossier de sortie:")
        self.output_path_label = QLabel(self.output_dir)
        self.output_path_label.setStyleSheet("background-color: white; padding: 5px; border: 1px solid gray;")
        output_button = QPushButton("Choisir...")
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_label, stretch=1)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        # Langue
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Langue:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Anglais", "Français", "Anglais (VCTK)"])
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        layout.addLayout(lang_layout)

        # Modèle
        model_layout = QHBoxLayout()
        model_label = QLabel("Modèle:")
        self.model_combo = QComboBox()
        self.update_model_list(0)  # Par défaut: Anglais
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # VCTK Speakers
        speaker_layout = QHBoxLayout()
        speaker_label = QLabel("Voix VCTK:")
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems([
            "VCTK_p232 (homme, bien)",
            "VCTK_p273 (femme, bien)",
            "VCTK_p278 (femme, bien)",
            "VCTK_p279 (homme, bien)",
            "VCTK_p304 (femme, voix préférée)"
        ])
        self.speaker_combo.setEnabled(False)
        speaker_layout.addWidget(speaker_label)
        speaker_layout.addWidget(self.speaker_combo)
        layout.addLayout(speaker_layout)

        # XTTS Reference Audio
        ref_audio_layout = QHBoxLayout()
        ref_audio_label = QLabel("Audio de référence (XTTS):")
        self.ref_audio_path = QLabel("Non sélectionné")
        self.ref_audio_path.setStyleSheet("background-color: white; padding: 5px; border: 1px solid gray;")
        ref_audio_button = QPushButton("Choisir...")
        ref_audio_layout.addWidget(ref_audio_label)
        ref_audio_layout.addWidget(self.ref_audio_path, stretch=1)
        ref_audio_layout.addWidget(ref_audio_button)
        layout.addLayout(ref_audio_layout)

        # CUDA
        cuda_layout = QHBoxLayout()
        self.cuda_check = QCheckBox("Utiliser CUDA (si disponible)")
        self.cuda_check.setChecked(True)
        cuda_layout.addWidget(self.cuda_check)
        layout.addLayout(cuda_layout)

        # Zone de texte avec support copier-coller
        self.text_edit = CustomTextEdit()
        self.text_edit.setPlaceholderText("Entrez votre texte ici... (Ctrl+V pour coller)")
        layout.addWidget(self.text_edit)

        # Boutons
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Générer")
        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Log (QTextEdit en lecture seule)
        self.log_text = CustomTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setPlaceholderText("Les messages de log apparaîtront ici...")
        layout.addWidget(self.log_text)

        # Connexions
        self.lang_combo.currentIndexChanged.connect(self.on_lang_changed)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        output_button.clicked.connect(self.choose_output_dir)
        ref_audio_button.clicked.connect(self.choose_ref_audio)
        self.generate_button.clicked.connect(self.generate_audio)
        self.cancel_button.clicked.connect(self.cancel_generation)

        # Worker
        self.worker = None

    def update_model_list(self, lang_index):
        """Met à jour la liste des modèles en fonction de la langue."""
        self.model_combo.clear()
        if lang_index == 0:  # Anglais
            self.model_combo.addItems([
                "Tacotron2",
                "Glow-TTS",
                "Speedy-Speech",
                "VITS",
                "Jenny"
            ])
        elif lang_index == 1:  # Français
            self.model_combo.addItems([
                "VITS",
                "Tacotron2-DDC",
                "YourTTS (voix masculine)",
                "YourTTS (voix féminine)",
                "XTTS v2"
            ])
        else:  # VCTK
            self.model_combo.addItems(["VITS"])
            self.model_combo.setCurrentIndex(0)

    def on_lang_changed(self, index):
        """Gère le changement de langue."""
        self.update_model_list(index)
        self.speaker_combo.setEnabled(index == 2)  # Active VCTK speakers uniquement pour VCTK
        # Active le choix du fichier audio de référence uniquement pour XTTS v2
        self.ref_audio_path.setEnabled(index == 1 and self.model_combo.currentIndex() == 4)

    def on_model_changed(self, index):
        """Gère le changement de modèle."""
        # Active le choix du fichier audio de référence uniquement pour XTTS v2
        is_xtts = self.lang_combo.currentIndex() == 1 and index == 4
        self.ref_audio_path.setEnabled(is_xtts)

    def choose_output_dir(self):
        """Ouvre une boîte de dialogue pour choisir le dossier de sortie."""
        dir_path = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie", self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.output_path_label.setText(dir_path)

    def choose_ref_audio(self):
        """Ouvre une boîte de dialogue pour choisir le fichier audio de référence."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier audio de référence",
            "",
            "Fichiers audio (*.wav *.mp3)"
        )
        if file_path:
            self.ref_audio_path.setText(file_path)

    def generate_audio(self):
        """Lance la génération audio."""
        if not self.text_edit.toPlainText():
            self.log_text.append("Erreur : Veuillez entrer du texte")
            return

        # Vérification pour XTTS v2
        if (self.lang_combo.currentIndex() == 1 and 
            self.model_combo.currentIndex() == 4 and 
            self.ref_audio_path.text() == "Non sélectionné"):
            self.log_text.append("Erreur : Veuillez sélectionner un fichier audio de référence pour XTTS v2")
            return

        # Préparation des paramètres
        params = {
            'text': self.text_edit.toPlainText(),
            'lang': self.lang_combo.currentIndex(),
            'en_model': self.model_combo.currentIndex(),
            'fr_model': self.model_combo.currentIndex(),
            'speaker': self.speaker_combo.currentText() if self.speaker_combo.isEnabled() else None,
            'use_cuda': self.cuda_check.isChecked(),
            'output_dir': self.output_dir
        }

        # Ajout du fichier audio de référence pour XTTS v2
        if self.lang_combo.currentIndex() == 1 and self.model_combo.currentIndex() == 4:
            params['reference_audio'] = self.ref_audio_path.text()

        # Démarrage du worker
        self.worker = TTSWorker(params)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.generation_finished)
        
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.worker.start()

    def cancel_generation(self):
        """Annule la génération en cours."""
        if hasattr(self, 'worker'):
            self.worker.terminate()
            self.worker.wait()
            self.generation_finished()
            self.log_text.append("Génération annulée")

    def update_progress(self, message):
        """Met à jour le message de progression."""
        self.log_text.append(message)

    def show_error(self, error_message):
        """Affiche un message d'erreur."""
        self.log_text.append(f"Erreur : {error_message}")
        self.generation_finished()

    def generation_finished(self):
        """Gère la fin de la génération."""
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
