import pyaudio
import wave
import keyboard

# Paramètres d'enregistrement
FORMAT = pyaudio.paInt16
CHANNELS = 1 # Utiliser un seul canal (mono)
RATE = 44100
CHUNK = 1024
OUTPUT_FILE = "voice/recorded_voice.wav"

# Initialiser PyAudio
audio = pyaudio.PyAudio()

# Démarrer l'enregistrement
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Enregistrement en cours... Appuyez sur ENTER pour terminer.")

frames = []

# Enregistrer jusqu'à ce que ENTER soit pressé
while True:
    data = stream.read(CHUNK)
    frames.append(data)
    if keyboard.is_pressed('enter'):
        print("Enregistrement terminé.")
        break

# Arrêter et fermer le flux
stream.stop_stream()
stream.close()
audio.terminate()

# Sauvegarder l'enregistrement dans un fichier WAV
with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Enregistrement sauvegardé sous {OUTPUT_FILE}")