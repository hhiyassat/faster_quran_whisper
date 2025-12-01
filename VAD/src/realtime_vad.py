import os
import sys
import datetime
import numpy as np

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("pyaudio is required")
    sys.exit(1)

from src.model import load_silero_vad_onnx
from src.vad_state import VADState
from src.processor import process_audio_chunk_onnx
from src.audio_handler import resample_audio, save_audio_wav
import config

def check_pyaudio_available():
    if not PYAUDIO_AVAILABLE:
        print("pyaudio is required")
        sys.exit(1)

def initialize_vad_session(model_path, model_url, model_dir, sample_rate):
    vad_session = load_silero_vad_onnx(model_path, model_url, model_dir)
    vad_state = VADState(sampling_rate=sample_rate)
    return vad_session, vad_state

def get_device_sample_rate(p, sample_rate):
    try:
        default_device = p.get_default_input_device_info()
        device_rate = int(default_device.get('defaultSampleRate', sample_rate))
    except Exception:
        device_rate = sample_rate
    return device_rate

def create_audio_stream(p, device_rate, chunk_duration_ms):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    frames_per_buffer = int(device_rate * chunk_duration_ms // 1000)
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=device_rate,
        input=True,
        frames_per_buffer=frames_per_buffer
    )
    return stream, frames_per_buffer

def calculate_threshold_frames(min_silence_duration_ms, min_speech_duration_ms, chunk_duration_ms):
    silence_threshold_frames = max(1, int((min_silence_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms))
    min_speech_frames = max(1, int((min_speech_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms))
    return silence_threshold_frames, min_speech_frames

def initialize_buffers():
    resampled_buffer = np.zeros(0, dtype=np.float32)
    original_buffer = np.zeros(0, dtype=np.float32)
    buffer_start_frame = 0
    return resampled_buffer, original_buffer, buffer_start_frame

def initialize_speech_state():
    chunk_count = 0
    total_frames = 0
    silent_frame_count = 0
    is_in_speech = False
    speech_start_time = None
    last_speech_time = None
    speech_frame_count = 0
    return {
        'chunk_count': chunk_count,
        'total_frames': total_frames,
        'silent_frame_count': silent_frame_count,
        'is_in_speech': is_in_speech,
        'speech_start_time': speech_start_time,
        'last_speech_time': last_speech_time,
        'speech_frame_count': speech_frame_count
    }

def read_audio_chunk(stream, frames_per_buffer):
    data = stream.read(frames_per_buffer, exception_on_overflow=False)
    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_chunk

def update_audio_buffers(audio_chunk, original_buffer, resampled_buffer, buffer_start_frame, total_frames, device_rate, sample_rate):
    if audio_chunk.size:
        if original_buffer.size == 0:
            buffer_start_frame = total_frames
        original_buffer = np.concatenate([original_buffer, audio_chunk])
    
    resampled_chunk = resample_audio(audio_chunk, device_rate, sample_rate)
    
    if resampled_chunk.size:
        resampled_buffer = np.concatenate([resampled_buffer, resampled_chunk])
    
    return original_buffer, resampled_buffer, resampled_chunk, buffer_start_frame

def detect_speech(resampled_chunk, vad_session, vad_state, vad_threshold):
    speech_prob = 0.0
    if len(resampled_chunk) >= vad_state.window_size_samples:
        vad_audio = resampled_chunk[-vad_state.window_size_samples:]
        speech_prob = process_audio_chunk_onnx(
            vad_audio,
            vad_session,
            vad_state,
            threshold=vad_threshold
        )
    return speech_prob

def handle_speech_detection(speech_prob, vad_threshold, state, min_speech_frames):
    is_speech_now = speech_prob > vad_threshold
    speech_started = False
    
    if is_speech_now:
        state['speech_frame_count'] += 1
        state['silent_frame_count'] = 0
        state['last_speech_time'] = state['total_frames']
        
        if not state['is_in_speech']:
            if state['speech_frame_count'] >= min_speech_frames:
                state['is_in_speech'] = True
                state['speech_start_time'] = state['total_frames'] - state['speech_frame_count']
                speech_started = True
    else:
        if not state['is_in_speech']:
            state['speech_frame_count'] = 0
    
    return is_speech_now, speech_started

def extract_audio_segment(original_buffer, buffer_start_frame, speech_start_time, last_speech_time, 
                         frames_per_buffer, silence_pad_samples, speech_frame_count):
    start_frame_offset = int(speech_start_time - buffer_start_frame)
    end_frame_offset = int(last_speech_time - buffer_start_frame)
    
    start_sample = max(0, start_frame_offset * frames_per_buffer)
    end_sample = min(len(original_buffer), end_frame_offset * frames_per_buffer + silence_pad_samples)
    
    audio_segment = original_buffer[start_sample:end_sample]
    
    if audio_segment.size == 0:
        fallback_samples = frames_per_buffer * max(1, int(speech_frame_count)) + silence_pad_samples
        if fallback_samples > 0 and len(original_buffer) >= fallback_samples:
            audio_segment = original_buffer[-fallback_samples:]
        else:
            return None
    
    return audio_segment

def save_chunk(audio_segment, chunk_count, recordings_dir, device_rate):
    chunk_count += 1
    filename_ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    chunk_filename = f"chunk_{chunk_count:03d}_{filename_ts}.wav"
    chunk_filepath = os.path.join(recordings_dir, chunk_filename)
    
    try:
        duration_sec = len(audio_segment) / float(device_rate)
        print(f"[SAVING] Speech detected and saved...", end='\r', flush=True)
        save_audio_wav(chunk_filepath, audio_segment, device_rate)
        print(f"[SAVED] Chunk #{chunk_count} ({duration_sec:.2f}s) - {chunk_filename}")
        return chunk_count
    except Exception as e:
        print(f"Failed to save: {e}")
        return chunk_count

def reset_buffers_and_state(resampled_buffer, original_buffer, vad_state, state, buffer_start_frame):
    resampled_buffer = np.zeros(0, dtype=np.float32)
    original_buffer = np.zeros(0, dtype=np.float32)
    buffer_start_frame = 0
    vad_state.reset()
    state['is_in_speech'] = False
    state['silent_frame_count'] = 0
    state['speech_frame_count'] = 0
    state['speech_start_time'] = None
    state['last_speech_time'] = None
    return resampled_buffer, original_buffer, state, buffer_start_frame

def process_silence_detection(state, silence_threshold_frames, original_buffer, buffer_start_frame,
                             frames_per_buffer, silence_pad_samples, device_rate, recordings_dir,
                             resampled_buffer, vad_state):
    if state['silent_frame_count'] >= silence_threshold_frames:
        if state['speech_start_time'] is None or state['last_speech_time'] is None:
            return reset_buffers_and_state(resampled_buffer, original_buffer, vad_state, state, buffer_start_frame)
        
        audio_segment = extract_audio_segment(
            original_buffer,
            buffer_start_frame,
            state['speech_start_time'],
            state['last_speech_time'],
            frames_per_buffer,
            silence_pad_samples,
            state['speech_frame_count']
        )
        
        if audio_segment is None:
            return reset_buffers_and_state(resampled_buffer, original_buffer, vad_state, state, buffer_start_frame)
        
        if len(audio_segment) > 0:
            state['chunk_count'] = save_chunk(
                audio_segment,
                state['chunk_count'],
                recordings_dir,
                device_rate
            )
        
        return reset_buffers_and_state(resampled_buffer, original_buffer, vad_state, state, buffer_start_frame)
    
    return resampled_buffer, original_buffer, state, buffer_start_frame

def cleanup_audio_resources(stream, p):
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()

def realtime_vad_chunks(vad_threshold, min_speech_duration_ms, min_silence_duration_ms, silence_pad_ms, 
                        sample_rate, chunk_duration_ms, model_path, model_url, model_dir, recordings_dir):
    
    check_pyaudio_available()
    
    vad_session, vad_state = initialize_vad_session(model_path, model_url, model_dir, sample_rate)
    
    p = pyaudio.PyAudio()
    os.makedirs(recordings_dir, exist_ok=True)
    
    device_rate = get_device_sample_rate(p, sample_rate)
    
    try:
        stream, frames_per_buffer = create_audio_stream(p, device_rate, chunk_duration_ms)
        
        resampled_buffer, original_buffer, buffer_start_frame = initialize_buffers()
        state = initialize_speech_state()
        
        silence_threshold_frames, min_speech_frames = calculate_threshold_frames(
            min_silence_duration_ms, min_speech_duration_ms, chunk_duration_ms
        )
        
        silence_pad_samples = int(silence_pad_ms / 1000.0 * device_rate)
        
        try:
            while True:
                audio_chunk = read_audio_chunk(stream, frames_per_buffer)
                
                original_buffer, resampled_buffer, resampled_chunk, buffer_start_frame = update_audio_buffers(
                    audio_chunk, original_buffer, resampled_buffer, buffer_start_frame,
                    state['total_frames'], device_rate, sample_rate
                )
                
                state['total_frames'] += 1
                
                speech_prob = detect_speech(resampled_chunk, vad_session, vad_state, vad_threshold)
                
                is_speech_now, speech_started = handle_speech_detection(speech_prob, vad_threshold, state, min_speech_frames)
                
                if speech_started:
                    print(f"[SPEECH DETECTED] Recording... (prob: {speech_prob:.2f})")
                
                if state['is_in_speech']:
                    if is_speech_now:
                        print(f"[SPEAKING] Recording speech... (prob: {speech_prob:.2f})", end='\r', flush=True)
                    state['silent_frame_count'] += 1
                    resampled_buffer, original_buffer, state, buffer_start_frame = process_silence_detection(
                        state, silence_threshold_frames, original_buffer, buffer_start_frame,
                        frames_per_buffer, silence_pad_samples, device_rate, recordings_dir,
                        resampled_buffer, vad_state
                    )
                
        except KeyboardInterrupt:
            print(f"\nTotal chunks saved: {state['chunk_count']}")
            print(f"Output directory: {recordings_dir}")
        finally:
            cleanup_audio_resources(stream, p)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        p.terminate()
