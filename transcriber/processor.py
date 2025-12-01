import os
import soundfile as sf
from pathlib import Path
import config

def transcribe_segment(whisper_model, segment_audio, segment_idx, temp_dir):
    temp_audio_path = temp_dir / f"chunk_{segment_idx:03d}.wav"
    sf.write(str(temp_audio_path), segment_audio, config.SAMPLE_RATE)
    
    segments_whisper, info = whisper_model.transcribe(
        str(temp_audio_path),
        beam_size=config.TRANSCRIBE_CONFIG["beam_size"],
        language=config.TRANSCRIBE_CONFIG["language"],
        task=config.TRANSCRIBE_CONFIG["task"],
        max_new_tokens=config.TRANSCRIBE_CONFIG["max_new_tokens"],
        condition_on_previous_text=config.TRANSCRIBE_CONFIG["condition_on_previous_text"],
        best_of=config.TRANSCRIBE_CONFIG["best_of"],
        temperature=config.TRANSCRIBE_CONFIG["temperature"],
        vad_filter=config.TRANSCRIBE_CONFIG["vad_filter"]
    )
    
    segment_texts = []
    for seg in segments_whisper:
        segment_texts.append(seg.text)
    
    transcription = " ".join(segment_texts)
    
    os.remove(temp_audio_path)
    
    return transcription

