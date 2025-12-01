# Real-Time Voice Activity Detection (VAD)

Real-time voice activity detection system using Silero VAD ONNX model for efficient speech detection and audio chunk collection.

## Features

- Real-time speech detection using Silero VAD ONNX model
- Automatic audio chunk saving when speech segments are detected
- Configurable VAD threshold and timing parameters
- Automatic model downloading on first run
- Efficient ONNX runtime for low latency

## Installation

### Prerequisites

- Python 3.7 or higher
- Microphone access

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirments.txt
```

Note: On some systems, you may need to install `pyaudio` separately:
- Windows: `pip install pipwin && pipwin install pyaudio`
- Linux: `sudo apt-get install portaudio19-dev python3-pyaudio`
- macOS: `brew install portaudio && pip install pyaudio`

## Usage

### Basic Usage

Run the application with default settings:
```bash
python entry.py
```

### Command Line Options

```bash
python entry.py [OPTIONS]
```

Options:
- `--threshold FLOAT`: VAD threshold (default: 0.25)
  - Lower values = more sensitive to speech
  - Higher values = less sensitive, fewer false positives

- `--min-speech-ms INT`: Minimum speech duration in milliseconds (default: 250)
  - Minimum duration of speech before detection starts

- `--min-silence-ms INT`: Minimum silence duration in milliseconds (default: 400)
  - Minimum silence duration after speech to save chunk

- `--silence-pad-ms INT`: Silence padding in milliseconds (default: 500)
  - Additional silence included with each saved chunk

### Examples

High sensitivity mode:
```bash
python entry.py --threshold 0.15 --min-speech-ms 200
```

Low sensitivity mode (fewer false positives):
```bash
python entry.py --threshold 0.4 --min-silence-ms 600
```

## Project Structure

```
test_vad/
├── entry.py                 # Entry point
├── config.py                # Configuration settings
├── requirments.txt         # Python dependencies
├── silero_vad.onnx         # VAD model (auto-downloaded)
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── model.py            # Model loading and downloading
│   ├── vad_state.py        # VAD state management
│   ├── processor.py        # VAD audio processing
│   ├── audio_handler.py    # Audio I/O operations
│   └── realtime_vad.py     # Main real-time VAD loop
├── scripts/                 # Utility scripts
│   ├── delete_short_segments.py
│   └── inspect_wav.py
└── chunks/                 # Output directory for saved audio chunks
```

## Configuration

Edit `config.py` to modify default settings:

- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `DEFAULT_CHUNK_DURATION_MS`: Audio chunk duration (default: 32 ms)
- `DEFAULT_VAD_THRESHOLD`: Default VAD threshold (default: 0.25)
- `DEFAULT_MIN_SPEECH_DURATION_MS`: Minimum speech duration (default: 250 ms)
- `DEFAULT_MIN_SILENCE_DURATION_MS`: Minimum silence duration (default: 400 ms)
- `DEFAULT_SILENCE_PAD_MS`: Silence padding (default: 500 ms)
- `RECORDINGS_DIR`: Output directory for saved chunks (default: "chunks")

## How It Works

1. **Initialization**: Loads Silero VAD ONNX model (downloads automatically if not present)
2. **Audio Capture**: Captures audio from default microphone input
3. **VAD Processing**: Processes audio chunks in real-time to detect speech
4. **Speech Detection**: Tracks speech segments and silence periods
5. **Chunk Saving**: Saves audio chunks when speech ends and silence threshold is reached
6. **Output**: Saves WAV files to the `chunks/` directory with timestamps

## Output

Saved audio chunks are stored in the `chunks/` directory with the format:
```
chunk_XXX_YYYYMMDD_HHMMSS_microseconds.wav
```

Where:
- `XXX`: Sequential chunk number
- `YYYYMMDD_HHMMSS_microseconds`: Timestamp

## Stopping the Application

Press `Ctrl+C` to stop the application. Statistics will be displayed before exit.

## Dependencies

- `torch>=2.0.0`: PyTorch framework
- `torchaudio>=2.0.0`: Audio processing utilities
- `onnxruntime>=1.15.0`: ONNX model inference
- `numpy>=1.21.0`: Numerical operations
- `pyaudio>=0.2.11`: Audio I/O
- `packaging>=23.0`: Package utilities

## Model

The project uses the Silero VAD ONNX model, which is automatically downloaded from Hugging Face on first run. The model file is saved as `silero_vad.onnx` in the project root directory.

## Troubleshooting

### No audio input detected
- Check microphone permissions
- Verify microphone is connected and working
- Try adjusting the VAD threshold

### Model download fails
- Check internet connection
- Manually download from: https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx
- Save as `silero_vad.onnx` in the project root

### PyAudio installation issues
- Windows: Use `pipwin install pyaudio`
- Linux: Install `portaudio19-dev` first
- macOS: Install `portaudio` via Homebrew first

## License

This project uses the Silero VAD model. Please refer to the original model's license for usage terms.

