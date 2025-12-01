import argparse
import config
from src.realtime_vad import realtime_vad_chunks

def main():
    parser = argparse.ArgumentParser(
        description="Real-time VAD Chunk Collector - ONNX ONLY"
    )
    
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_VAD_THRESHOLD,
                        help=f"VAD threshold (default={config.DEFAULT_VAD_THRESHOLD})")
    parser.add_argument("--min-speech-ms", type=int, default=config.DEFAULT_MIN_SPEECH_DURATION_MS,
                        help=f"Min speech duration ms (default={config.DEFAULT_MIN_SPEECH_DURATION_MS})")
    parser.add_argument("--min-silence-ms", type=int, default=config.DEFAULT_MIN_SILENCE_DURATION_MS,
                        help=f"Min silence (default={config.DEFAULT_MIN_SILENCE_DURATION_MS})")
    parser.add_argument("--silence-pad-ms", type=int, default=config.DEFAULT_SILENCE_PAD_MS,
                        help=f"Silence padding (default={config.DEFAULT_SILENCE_PAD_MS})")
    
    args = parser.parse_args()

    realtime_vad_chunks(
        vad_threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        silence_pad_ms=args.silence_pad_ms,
        sample_rate=config.SAMPLE_RATE,
        chunk_duration_ms=config.DEFAULT_CHUNK_DURATION_MS,
        model_path=config.ONNX_MODEL_PATH,
        model_url=config.ONNX_MODEL_URL,
        model_dir=config.ONNX_MODEL_DIR,
        recordings_dir=config.RECORDINGS_DIR
    )

if __name__ == "__main__":
    main()

