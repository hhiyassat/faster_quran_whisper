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

**Output:** Structured transcription with timestamps for each spoken segment

Example output format:
```
============================================================
🎤 TRANSCRIBING 106 SEGMENTS
============================================================

[Segment 1/106]
   Duration: 12.85s
   Time range: 1.25s - 14.10s
   ✅ Transcription: ءَلحَمدُوولِللَااهِللَ ذِييييي ءَڻزَلَ عَلَاا عَپدِهِ لكِتَاابَ وَلَم يَچعَللَهُووعِ وَجَاا

[Segment 2/106]
   Duration: 19.64s
   Time range: 15.20s - 34.84s
   ✅ Transcription: ڨَييِمَللِ يُڻذِڗَبَءَ سَڻ شَدِييدَممِللَدُنهُ وَ يُبَششِڗَلمُءُمِنِيينَللَذِيينَ يَعمَلُوونَ ڝصَلِحَااتِ

============================================================
📊 SUMMARY
============================================================
Total segments: 106
Successfully transcribed: 106
Failed: 0

============================================================
📄 FULL TRANSCRIPTION:
============================================================
ءَلحَمدُوولِللَااهِللَ ذِييييي ءَڻزَلَ عَلَاا عَپدِهِ لكِتَاابَ...
   Total length: 6995 characters
   Total tokens (approx): ~608 words
```

The output includes:
- **Per-segment transcription** with timestamps (start/end time, duration)
- **Full combined transcription** with all segments merged
- **Summary statistics** (total segments, success rate)

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
3. **Combination**: All transcriptions merged into full text with timestamps

## ⚡ Performance

- **CPU optimized**: Uses int8 quantization for fast CPU inference
- **Real-time factor**: ~14.6x faster than real-time (20 min audio in ~82 seconds)
- **Multi-threaded**: Configurable CPU threads (default: 4)
- **Efficient**: CTranslate2 backend for optimized inference

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
