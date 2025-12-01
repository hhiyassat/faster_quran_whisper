import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000

WHISPER_CONFIG = {
    "model_dir": BASE_DIR / "model-ct2",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 4,
}

VAD_CONFIG = {
    "model_dir": BASE_DIR / "VAD",
    "model_path": BASE_DIR / "VAD" / "silero_vad.onnx",
    "model_url": "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx",
    "chunk_duration_ms": 32,
    "threshold": 0.25,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 400,
    "silence_pad_ms": 500,
}

TRANSCRIBE_CONFIG = {
    "beam_size": 5,
    "language": "ar",
    "task": "transcribe",
    "max_new_tokens": 445,
    "condition_on_previous_text": False,
    "best_of": 5,
    "temperature": 0.0,
    "vad_filter": False,
    "repetition_penalty": 1.0,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 3.0,
    "log_prob_threshold": -2.0,
    "word_timestamps": False,
}

AUDIO_CONFIG = {
    "default_file": BASE_DIR / "quran_test_audio.mp3",
    "supported_formats": [".mp3", ".wav", ".m4a", ".aac", ".flac"],
}

