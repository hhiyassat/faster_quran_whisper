import sys
from pathlib import Path
import onnxruntime as ort
from faster_whisper import WhisperModel
import config

BASE_DIR = Path(__file__).resolve().parent.parent
VAD_DIR = BASE_DIR / "VAD"

if str(VAD_DIR) not in sys.path:
    sys.path.insert(0, str(VAD_DIR))

def load_vad_model():
    from src.model import load_silero_vad_onnx
    vad_session = load_silero_vad_onnx(
        str(config.VAD_CONFIG["model_path"]),
        config.VAD_CONFIG["model_url"],
        str(config.VAD_CONFIG["model_dir"]),
    )
    return vad_session

def load_whisper_model():
    whisper_model = WhisperModel(
        str(config.WHISPER_CONFIG["model_dir"]),
        device=config.WHISPER_CONFIG["device"],
        compute_type=config.WHISPER_CONFIG["compute_type"],
        cpu_threads=config.WHISPER_CONFIG["cpu_threads"],
    )
    return whisper_model

