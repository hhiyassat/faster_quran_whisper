# faster_quran_whisper

## Faster-Whisper Speech-to-Text Pipeline

Complete speech-to-text pipeline using **Faster-Whisper** with **Voice Activity Detection (VAD)** for processing long audio files.

## 📋 Overview

This implementation provides:
- **Faster-Whisper** (CTranslate2 optimized Whisper) for fast transcription
- **Silero VAD** for automatic speech segmentation
- Support for long audio files (10+ minutes) through chunking
- Full Arabic transcription support
- Modular, maintainable code structure

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Virtual Environment** (recommended)

### Installation

1. Create and activate virtual environment:
```bash
cd faster-whisper
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Models

The repository includes both models:

1. **Faster-Whisper Model** (CTranslate2 format)
   - Location: `model-ct2/`
   - Quantization: `int8` (CPU optimized)

2. **VAD Model** (Silero VAD ONNX)
   - Location: `VAD/silero_vad.onnx`
   - Automatically downloaded if missing

## 📁 Directory Structure

```
faster-whisper/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
├── main.py                   # Main entry point
├── models/                   # Model loading module
│   ├── __init__.py
│   └── loader.py
├── audio/                    # Audio loading module
│   ├── __init__.py
│   └── loader.py
├── vad/                      # VAD processing module
│   ├── __init__.py
│   └── processor.py
├── transcriber/              # Transcription module
│   ├── __init__.py
│   └── processor.py
├── model-ct2/                # Faster-Whisper model files
│   ├── config.json
│   ├── model.bin
│   ├── tokenizer.json
│   └── ...
├── VAD/                      # VAD model and utilities
│   ├── silero_vad.onnx
│   ├── config.py
│   └── src/
└── quran_test_audio.mp3      # Test audio file (optional)
```

## 🎯 Usage

### Basic Usage

Run with default audio file:
```bash
source venv/bin/activate
python main.py
```

Run with custom audio file:
```bash
python main.py /path/to/your/audio.mp3
```

### What it does:

1. Loads VAD model (Silero VAD)
2. Loads Faster-Whisper model
3. Processes audio with VAD to detect speech segments
4. Transcribes each segment separately
5. Combines all transcriptions

### Output:

- Number of detected segments
- Transcription for each segment with timestamps
- Full combined transcription
- Summary statistics

## 🔧 Configuration

All configuration is centralized in `config.py`:

### Audio Configuration

```python
AUDIO_CONFIG = {
    "default_file": BASE_DIR / "quran_test_audio.mp3",
    "supported_formats": [".mp3", ".wav", ".m4a", ".aac", ".flac"],
}
```

### Whisper Model Configuration

```python
WHISPER_CONFIG = {
    "model_dir": BASE_DIR / "model-ct2",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 4,
}
```

### VAD Configuration

```python
VAD_CONFIG = {
    "model_dir": BASE_DIR / "VAD",
    "model_path": BASE_DIR / "VAD" / "silero_vad.onnx",
    "threshold": 0.25,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 400,
    "silence_pad_ms": 500,
}
```

### Transcription Configuration

```python
TRANSCRIBE_CONFIG = {
    "beam_size": 5,
    "language": "ar",
    "task": "transcribe",
    "max_new_tokens": 445,
    "temperature": 0.0,
    ...
}
```

Edit `config.py` to customize any settings.

## 📦 Dependencies

See `requirements.txt` for full list:

- `faster-whisper` - Faster-Whisper library (CTranslate2 backend)
- `ctranslate2` - CTranslate2 inference engine
- `transformers[torch]` - HuggingFace transformers
- `librosa` - Audio loading (supports many formats)
- `soundfile` - Audio I/O
- `onnxruntime` - VAD model inference

## 🔬 Technical Details

### Model Information

- **Base Model**: Fine-tuned Whisper Tiny (Arabic)
- **Format**: CTranslate2 (converted from HuggingFace)
- **Quantization**: int8 (CPU optimized)
- **Max Length**: 448 tokens
- **Language**: Arabic (ar)
- **Task**: Transcribe

### VAD Information

- **Model**: Silero VAD v4
- **Format**: ONNX
- **Window Size**: 512 samples (32ms at 16kHz)
- **Sample Rate**: 16000 Hz
- **State Management**: Maintains internal state across chunks

### Processing Pipeline

1. **Audio Loading** (`audio/loader.py`)
   - Loads audio file using `librosa` or `soundfile`
   - Resamples to 16kHz if needed
   - Converts to mono channel

2. **VAD Processing** (`vad/processor.py`)
   - Splits audio into 512-sample chunks
   - Processes each chunk with VAD model
   - Detects speech vs silence
   - Groups speech chunks into segments

3. **Transcription** (`transcriber/processor.py`)
   - Each speech segment is transcribed separately
   - Uses Faster-Whisper with optimized settings
   - Combines all segment transcriptions

## 🎛️ Advanced Configuration

### Adjust CPU Threads

Edit `config.py`:
```python
WHISPER_CONFIG = {
    ...
    "cpu_threads": 8,
}
```

### Change Device (GPU)

Edit `config.py`:
```python
WHISPER_CONFIG = {
    ...
    "device": "cuda",
    "compute_type": "float16",
}
```

### Customize VAD Threshold

Edit `config.py`:
```python
VAD_CONFIG = {
    ...
    "threshold": 0.3,
    "min_silence_duration_ms": 300,
}
```

## ⚠️ Troubleshooting

### Model Not Found

If you get model errors:
```bash
# Check model directory exists
ls -la model-ct2/
ls -la VAD/silero_vad.onnx
```

### VAD Model Missing

VAD model should be at `VAD/silero_vad.onnx`. If missing, it will be downloaded automatically on first run.

### Audio Loading Errors

If audio format is not supported:
- Install additional codecs: `sudo apt-get install ffmpeg libavcodec-extra`
- Or convert audio to WAV format

### Out of Memory

For very long audio files:
- Reduce `cpu_threads` in `config.py`
- Process audio in smaller chunks
- Use lower quantization (if available)

## 📈 Performance Tips

1. **Use int8 quantization** for best CPU performance
2. **Set appropriate CPU threads** (usually 4-8)
3. **Use VAD** for long audio files to split into manageable chunks
4. **Disable Faster-Whisper's built-in VAD** when using external VAD

## 📄 License

This implementation uses:
- **Faster-Whisper**: MIT License
- **Silero VAD**: Apache 2.0 License
- **CTranslate2**: MIT License

## 👤 Usage Example

```bash
# Activate environment
cd faster-whisper
source venv/bin/activate

# Run with default audio
python main.py

# Run with custom audio
python main.py /path/to/audio.mp3
```

## 📊 Test Results

Tested on:
- **Audio**: 20-minute Quran recitation (Surah Al-Kahf)
- **Device**: CPU only (4 threads)
- **Model**: Fine-tuned Whisper Tiny (Arabic, int8)
- **Performance**: ~82 seconds total processing time
- **Real-time Factor**: ~14.6x faster than real-time

---

For questions or issues, refer to the source code in the module directories.
