#!/usr/bin/env python3
import sys
from pathlib import Path
import config
from models.loader import load_vad_model, load_whisper_model
from audio.loader import load_audio
from vad.processor import extract_speech_segments
from transcriber.processor import transcribe_segment

BASE_DIR = Path(__file__).resolve().parent
VAD_DIR = BASE_DIR / "VAD"

if str(VAD_DIR) not in sys.path:
    sys.path.insert(0, str(VAD_DIR))

from src.vad_state import VADState

print("="*60)
print("Faster-Whisper Full Transcription with VAD")
print("="*60)

print(f"\n📦 Loading models...")
vad_session = load_vad_model()
print("✅ VAD model loaded")

print(f"\n📦 Loading Faster-Whisper model from: {config.WHISPER_CONFIG['model_dir']}")
whisper_model = load_whisper_model()
print("✅ Faster-Whisper model loaded")

audio_file = config.AUDIO_CONFIG["default_file"]
if len(sys.argv) > 1:
    audio_file = Path(sys.argv[1])

print(f"\n📁 Loading audio file: {audio_file}")

if not audio_file.exists():
    print(f"❌ Audio file not found: {audio_file}")
    sys.exit(1)

try:
    audio = load_audio(audio_file)
    audio_duration = len(audio) / config.SAMPLE_RATE
    print(f"✅ Audio loaded: {audio_duration:.2f} seconds, {len(audio)} samples")
except Exception as e:
    print(f"❌ Error loading audio: {e}")
    sys.exit(1)

vad_state = VADState(sampling_rate=config.SAMPLE_RATE)
speech_segments = extract_speech_segments(audio, vad_session, vad_state)

if len(speech_segments) == 0:
    print("\n❌ No audio segments to process")
    sys.exit(0)

print(f"\n{'='*60}")
print(f"🎤 TRANSCRIBING {len(speech_segments)} SEGMENTS")
print(f"{'='*60}\n")

results = []
temp_dir = Path("/tmp/faster_whisper_chunks")
temp_dir.mkdir(exist_ok=True)

for idx, segment in enumerate(speech_segments, 1):
    print(f"\n[Segment {idx}/{len(speech_segments)}]")
    print(f"   Duration: {segment['duration']:.2f}s")
    print(f"   Time range: {segment['start']/config.SAMPLE_RATE:.2f}s - {segment['end']/config.SAMPLE_RATE:.2f}s")
    
    try:
        transcription = transcribe_segment(whisper_model, segment['audio'], idx, temp_dir)
        
        results.append({
            'segment': idx,
            'start_time': segment['start']/config.SAMPLE_RATE,
            'end_time': segment['end']/config.SAMPLE_RATE,
            'duration': segment['duration'],
            'transcription': transcription
        })
        print(f"   ✅ Transcription: {transcription}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'segment': idx,
            'start_time': segment['start']/config.SAMPLE_RATE,
            'end_time': segment['end']/config.SAMPLE_RATE,
            'duration': segment['duration'],
            'transcription': f"ERROR: {str(e)}"
        })

print(f"\n{'='*60}")
print("📊 SUMMARY")
print(f"{'='*60}")
print(f"Total segments: {len(speech_segments)}")
print(f"Successfully transcribed: {sum(1 for r in results if not r['transcription'].startswith('ERROR'))}")
print(f"Failed: {sum(1 for r in results if r['transcription'].startswith('ERROR'))}")

print(f"\n{'='*60}")
print("📝 ALL TRANSCRIPTIONS:")
print(f"{'='*60}")
for result in results:
    print(f"\n[Segment {result['segment']}] ({result['start_time']:.2f}s - {result['end_time']:.2f}s, {result['duration']:.2f}s)")
    print(f"Transcription: {result['transcription']}")

full_text = " ".join([r['transcription'] for r in results if not r['transcription'].startswith('ERROR')])
print(f"\n{'='*60}")
print("📄 FULL TRANSCRIPTION:")
print(f"{'='*60}")
print(full_text)
print(f"\n   Total length: {len(full_text)} characters")
print(f"   Total tokens (approx): ~{len(full_text.split())} words")
print(f"   Effective max tokens: {len(speech_segments) * config.TRANSCRIBE_CONFIG['max_new_tokens']} (across {len(speech_segments)} chunks)")
print(f"{'='*60}")

