import os

# add ffmpeg path
os.environ["PATH"] += r";D:\fir_ai_local\ffmpeg-8.0.1-essentials_build\bin"

import whisper

MODEL = whisper.load_model("base")

def speech_to_text(audio_file):

    print("Processing audio:", audio_file)

    result = MODEL.transcribe(audio_file)

    return result["text"]