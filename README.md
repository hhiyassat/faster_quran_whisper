# faster_quran_whisper

Faster-Whisper speech-to-text pipeline with VAD for **Quran audio transcription** in Arabic.

## 🚀 Quick Start

### 1. Create Virtual Environment
```bash
cd faster-whisper
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python main.py
```

To use custom audio file:
```bash
python main.py /path/to/your/audio.mp3
```

## 📥 Input Example

**Input:** Quran audio file (MP3, WAV, M4A, AAC, FLAC)

Example audio file: `sample/quran_test_audio.mp3`
- Format: MP3
- Duration: ~20 minutes (Surah Al-Kahf)
- Sample Rate: 16kHz (auto-resampled if needed)

## 📤 Output Example

**Output:** Arabic text transcription

Example output:
```
الحمد لله الذي أنزل على عبده الكتاب ولم يجعل له عوجا
قيما لينذر بأسا شديدا من لدنه ويبشر المؤمنين الذين يعملون الصالحات أن لهم أجرا حسنا
```

The script processes the audio, detects speech segments, transcribes each segment, and combines them into full text.

## 📁 Folder Structure

```
faster-whisper/
├── README.md
├── requirements.txt
├── config.py              # Configuration
├── main.py                # Entry point
├── models/                # All models
│   ├── whisper/          # Faster-Whisper model
│   ├── vad/              # VAD model and utilities
│   └── loader.py         # Model loading
├── audio/                # Audio processing
│   └── loader.py
├── vad/                  # VAD processing module
│   └── processor.py
├── transcriber/          # Transcription module
│   └── processor.py
├── sample/               # Sample audio files
│   └── quran_test_audio.mp3
└── venv/                 # Virtual environment
```

## 🔄 How It Works

1. **VAD Processing**: Detects speech segments in audio
2. **Transcription**: Each segment transcribed using Faster-Whisper
3. **Combination**: All transcriptions merged into full text

## ⚙️ Configuration

Edit `config.py` to customize:
- Audio file path
- Model settings (device, threads, quantization)
- VAD parameters (threshold, duration)
- Transcription settings

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## 🔧 Troubleshooting

**Model not found?**
```bash
ls -la models/whisper/
ls -la models/vad/silero_vad.onnx
```

**Audio format error?**
Install codecs: `sudo apt-get install ffmpeg libavcodec-extra`

## 📄 License

- Faster-Whisper: MIT
- Silero VAD: Apache 2.0
- CTranslate2: MIT
