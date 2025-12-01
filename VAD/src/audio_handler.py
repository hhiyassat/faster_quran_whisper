import numpy as np
import wave
import torch

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False

def resample_audio(audio_chunk, device_rate, target_rate):
    if device_rate == target_rate:
        return audio_chunk
    
    if TORCHAUDIO_AVAILABLE:
        try:
            wav = torch.from_numpy(audio_chunk).unsqueeze(0)
            resampled_chunk = torchaudio.functional.resample(
                wav, orig_freq=device_rate, new_freq=target_rate
            ).squeeze(0).numpy()
            return resampled_chunk
        except Exception:
            pass
    
    import math
    old_n = audio_chunk.shape[0]
    new_n = int(math.ceil(old_n * target_rate / float(device_rate)))
    resampled_chunk = np.interp(
        np.linspace(0, old_n, new_n, endpoint=False),
        np.arange(old_n),
        audio_chunk
    ).astype(np.float32)
    return resampled_chunk

def save_audio_wav(filepath, audio_data, sample_rate):
    audio_float = audio_data.astype(np.float32)
    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    
    if TORCHAUDIO_AVAILABLE:
        try:
            tensor = torch.from_numpy(audio_clipped).unsqueeze(0).contiguous()
            torchaudio.save(filepath, tensor, sample_rate, format='wav')
            return
        except Exception:
            pass
    
    pcm = (audio_clipped * 32767.0).astype(np.int16).tobytes()
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)

