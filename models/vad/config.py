import os

SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION_MS = 32
DEFAULT_VAD_THRESHOLD = 0.25
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 400
DEFAULT_SILENCE_PAD_MS = 500
DEFAULT_PRE_ROLL_MS = 500

ONNX_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_URL = "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx"
ONNX_MODEL_PATH = os.path.join(ONNX_MODEL_DIR, "silero_vad.onnx")

RECORDINGS_DIR = "chunks"

