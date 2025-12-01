# Audio Chunking Explanation - VAD Model

## How Audio Chunking Works

**Answer: The model does NOT automatically separate audio. You must manually split audio into 512-sample chunks.**

### How It Works:

#### 1. **Real-Time Streaming (Microphone)**

```
Microphone → Read 512 samples (32ms) → Process → Get speech_prob
     ↓
Microphone → Read next 512 samples → Process → Get speech_prob
     ↓
Microphone → Read next 512 samples → Process → Get speech_prob
```

**Code Flow:**
```python
# Read audio from microphone
data = stream.read(frames_per_buffer)  # frames_per_buffer = 512 samples
audio_chunk = convert_to_float32(data)  # 512 samples

# Process each chunk
speech_prob = process_audio_chunk_onnx(audio_chunk, model, state)
# Each call processes exactly 512 samples
```

#### 2. **Batch Processing (Audio File)**

If you have a long audio file, you must split it manually:

```python
# Audio file: 10 seconds = 160,000 samples at 16kHz

# Split into chunks
chunk_size = 512
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]  # Take 512 samples
    speech_prob = process_audio_chunk_onnx(chunk, model, state)
```

### Key Points:

1. **Model requires exactly 512 samples per call**
   - Input shape: `[1, 512]`
   - Fixed window size
   
2. **If audio is shorter than 512 samples:**
   ```python
   if len(audio_chunk) < 512:
       audio_chunk = pad_with_zeros(audio_chunk, 512)  # Pad to 512
   ```

3. **If audio is longer than 512 samples:**
   ```python
   if len(audio_chunk) > 512:
       audio_chunk = audio_chunk[:512]  # Take first 512 samples
   ```

4. **State is maintained between chunks:**
   - State preserves context across chunks
   - Each chunk uses previous chunk's state
   - This allows streaming detection

### Example: Processing 5 seconds of audio

```python
# 5 seconds = 80,000 samples at 16kHz
audio = load_audio("5_seconds.wav")  # 80,000 samples

# Split into chunks
state = VADState()
chunks_needed = 80000 / 512 = 156.25 → 157 chunks

for i in range(0, len(audio), 512):
    chunk = audio[i:i+512]  # Get 512 samples
    if len(chunk) < 512:
        chunk = pad_to_512(chunk)  # Pad if needed
    
    speech_prob = process_audio_chunk_onnx(chunk, model, state)
    # State is automatically updated for next iteration
```

### For Kotlin:

```kotlin
// Real-time: Read from AudioRecord
val bufferSize = 512
val audioBuffer = ShortArray(bufferSize)
audioRecord.read(audioBuffer, 0, bufferSize)

// Convert to FloatArray[512]
val audioChunk = FloatArray(512) { 
    audioBuffer[it].toFloat() / 32768.0f 
}

// Process chunk
val speechProb = vadProcessor.processAudioChunk(audioChunk)

// For next iteration, read next 512 samples from microphone
```

### Summary:

- **Model does NOT split audio automatically**
- **You must provide 512-sample chunks manually**
- **Each inference call processes exactly 512 samples**
- **State connects chunks together for continuous detection**
- **For streaming: Read 512 samples → Process → Repeat**
- **For files: Split into 512-sample chunks → Process each → Repeat**

