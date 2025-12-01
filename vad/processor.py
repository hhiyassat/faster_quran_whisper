import sys
import numpy as np
from pathlib import Path
import config

BASE_DIR = Path(__file__).resolve().parent.parent
VAD_DIR = BASE_DIR / "VAD"

if str(VAD_DIR) not in sys.path:
    sys.path.insert(0, str(VAD_DIR))

from src.vad_state import VADState
from src.processor import process_audio_chunk_onnx

def extract_speech_segments(audio, vad_session, vad_state):
    print(f"\n🎤 Processing audio with VAD...")
    print(f"   Audio length: {len(audio)} samples ({len(audio)/config.SAMPLE_RATE:.2f} seconds)")
    
    chunk_duration_ms = config.VAD_CONFIG["chunk_duration_ms"]
    chunk_size = int(config.SAMPLE_RATE * chunk_duration_ms // 1000)
    frames_per_buffer = chunk_size
    
    min_silence_duration_ms = config.VAD_CONFIG["min_silence_duration_ms"]
    min_speech_duration_ms = config.VAD_CONFIG["min_speech_duration_ms"]
    silence_pad_ms = config.VAD_CONFIG["silence_pad_ms"]
    vad_threshold = config.VAD_CONFIG["threshold"]
    
    silence_threshold_frames = max(1, int((min_silence_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms))
    min_speech_frames = max(1, int((min_speech_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms))
    silence_pad_samples = int(silence_pad_ms / 1000.0 * config.SAMPLE_RATE)
    
    print(f"   Chunk size: {chunk_size} samples ({chunk_duration_ms}ms)")
    print(f"   Min speech frames: {min_speech_frames}")
    print(f"   Silence threshold frames: {silence_threshold_frames}")
    print(f"   Silence pad: {silence_pad_samples} samples")
    
    total_frames = 0
    speech_frame_count = 0
    silent_frame_count = 0
    is_in_speech = False
    speech_start_frame = None
    last_speech_frame = None
    
    segments = []
    silence_periods = []
    current_silence_start = None
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        if len(chunk) >= vad_state.window_size_samples:
            vad_audio = chunk[-vad_state.window_size_samples:] if len(chunk) > vad_state.window_size_samples else chunk
            speech_prob = process_audio_chunk_onnx(vad_audio, vad_session, vad_state)
        else:
            speech_prob = 0.0
        
        is_speech_now = speech_prob > vad_threshold
        
        if is_speech_now:
            speech_frame_count += 1
            silent_frame_count = 0
            last_speech_frame = total_frames
            
            if not is_in_speech:
                if speech_frame_count >= min_speech_frames:
                    is_in_speech = True
                    speech_start_frame = total_frames - speech_frame_count
        else:
            if not is_in_speech:
                speech_frame_count = 0
        
        if is_in_speech:
            if not is_speech_now:
                silent_frame_count += 1
                if current_silence_start is None:
                    current_silence_start = total_frames
            else:
                if current_silence_start is not None:
                    silence_duration_ms = (total_frames - current_silence_start) * chunk_duration_ms
                    if silence_duration_ms >= min_silence_duration_ms:
                        silence_periods.append((current_silence_start, total_frames - 1))
                current_silence_start = None
                silent_frame_count = 0
            
            if silent_frame_count >= silence_threshold_frames:
                if speech_start_frame is not None and last_speech_frame is not None:
                    start_sample = speech_start_frame * frames_per_buffer
                    end_sample = min(len(audio), (last_speech_frame + 1) * frames_per_buffer + silence_pad_samples)
                    
                    segment_audio = audio[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        segments.append({
                            'start': start_sample,
                            'end': end_sample,
                            'audio': segment_audio.copy(),
                            'duration': len(segment_audio) / config.SAMPLE_RATE
                        })
                
                vad_state.reset()
                is_in_speech = False
                silent_frame_count = 0
                speech_frame_count = 0
                speech_start_frame = None
                last_speech_frame = None
        
        total_frames += 1
    
    if is_in_speech and speech_start_frame is not None and last_speech_frame is not None:
        start_sample = speech_start_frame * frames_per_buffer
        end_sample = min(len(audio), (last_speech_frame + 1) * frames_per_buffer + silence_pad_samples)
        
        segment_audio = audio[start_sample:end_sample]
        
        if len(segment_audio) > 0:
            segments.append({
                'start': start_sample,
                'end': end_sample,
                'audio': segment_audio.copy(),
                'duration': len(segment_audio) / config.SAMPLE_RATE
            })
    
    print(f"   Processed {total_frames} chunks")
    print(f"   ✅ Detected {len(segments)} speech segments")
    
    if len(segments) <= 1:
        print(f"\n   🔍 Debug: Silence period analysis:")
        if len(silence_periods) > 0:
            print(f"      Found {len(silence_periods)} silence periods >= {min_silence_duration_ms}ms:")
            for idx, (start_frame, end_frame) in enumerate(silence_periods[:5], 1):
                start_time = (start_frame * chunk_size) / config.SAMPLE_RATE
                end_time = ((end_frame + 1) * chunk_size) / config.SAMPLE_RATE
                duration_ms = (end_frame - start_frame + 1) * chunk_duration_ms
                print(f"         Silence {idx}: {start_time:.2f}s - {end_time:.2f}s ({duration_ms:.0f}ms)")
            if len(silence_periods) > 5:
                print(f"      ... ({len(silence_periods) - 5} more)")
        else:
            print(f"      ⚠️  No silence periods >= {min_silence_duration_ms}ms detected")
            print(f"      The audio appears to have continuous speech without long enough gaps")
            print(f"      Try reducing MIN_SILENCE_DURATION_MS (currently {min_silence_duration_ms}ms)")
    
    if len(segments) == 0 and total_frames > 0:
        print(f"   ⚠️  Warning: No segments detected. This might mean:")
        print(f"      - Audio has no speech above threshold {vad_threshold}")
        print(f"      - Speech segments are too short (< {min_speech_duration_ms}ms)")
        print(f"      - Silence gaps are too short (< {min_silence_duration_ms}ms)")
    
    return segments

