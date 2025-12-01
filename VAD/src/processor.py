import numpy as np

def process_audio_chunk_onnx(audio_chunk, model_session, vad_state, threshold=0.5):
    audio_chunk = audio_chunk.astype(np.float32)
    
    if len(audio_chunk) < vad_state.window_size_samples:
        audio_chunk = np.pad(audio_chunk, (0, vad_state.window_size_samples - len(audio_chunk)))
    else:
        audio_chunk = audio_chunk[:vad_state.window_size_samples]
    
    audio_input = audio_chunk.reshape(1, -1)
    sr_tensor = np.array([vad_state.sampling_rate], dtype=np.int64)
    
    try:
        inputs = {
            'input': audio_input,
            'state': vad_state.state,
            'sr': sr_tensor
        }
        
        output_names = [out.name for out in model_session.get_outputs()]
        outputs = model_session.run(output_names, inputs)
        
        speech_prob = outputs[0][0][0]
        vad_state.state = outputs[1]
        
        return float(speech_prob)
        
    except Exception as e:
        print(f"VAD inference error: {e}")
        return 0.0

