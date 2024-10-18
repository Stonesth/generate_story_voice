# generate_story_voice

Try to generate a story.
Try to clone my voice.
Try to read the story with my voice.

python -m venv venv

<!-- to activate the test environment -->

MAC
source venv/bin/activate
Windows

<!-- To go out the test environment -->
MAC
deactivate
Windows

pip install --upgrade pip

<!-- For generate_voice.py -->
pip install gtts

<!-- For generate_voice2.py -->
pip install torch

<!-- Need to give the accord for the licene Xcode (MAC) -->
sudo xcodebuild -license
pip install numpy
pip install soundfile numpy

CFLAGS="-I$(python -c 'import numpy; print(numpy.get_include())')" pip install git+https://github.com/coqui-ai/TTS


<!-- For record_voice.py -->
pip install pyaudio keyboard
pip install sounddevice
pip install wavio

<!-- For generate_voice3.py -->
brew install espeak
espeak "Hello, this is a test."
pip install --upgrade aiohttp fsspec