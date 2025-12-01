import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import config

def load_audio(audio_path):
    try:
        audio, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        try:
            audio, sr = sf.read(str(audio_path))
            if sr != config.SAMPLE_RATE:
                audio = signal.resample(audio, int(len(audio) * config.SAMPLE_RATE / sr))
            if len(audio.shape) > 1:
                audio = audio[:, 0]
        except Exception as e2:
            raise Exception(f"Error loading audio: {e}, {e2}")
    
    audio = audio.astype(np.float32)
    return audio

