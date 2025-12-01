# Kotlin Port Guide - Real-Time VAD for Flutter/Android

This guide explains how to create a Kotlin version of the Python VAD system for Flutter mobile applications.

## Overview

The Python VAD system uses:
- Silero VAD ONNX model
- Real-time audio capture
- Speech detection and chunk saving
- ONNX Runtime for inference

## Architecture for Kotlin/Android

### 1. Project Structure

```
android/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/yourpackage/
│   │   │   │   ├── vad/
│   │   │   │   │   ├── VADManager.kt
│   │   │   │   │   ├── AudioCapture.kt
│   │   │   │   │   ├── VADProcessor.kt
│   │   │   │   │   ├── VADState.kt
│   │   │   │   │   └── AudioSaver.kt
│   │   │   │   └── MainActivity.kt
│   │   │   ├── assets/
│   │   │   │   └── silero_vad.onnx
│   │   │   └── res/
│   │   └── build.gradle
│   └── build.gradle
└── build.gradle
```

### 2. Dependencies (build.gradle)

Add these dependencies to your `app/build.gradle`:

```gradle
dependencies {
    // ONNX Runtime for Android
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.0'
    
    // Audio processing
    implementation 'androidx.media:media:1.6.0'
    
    // Coroutines for async operations
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // Lifecycle components
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.6.2'
}
```

### 3. Android Permissions

Add to `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

Request runtime permissions in your Activity/Fragment.

## Component Breakdown

### 1. VADState.kt

**Purpose**: Maintains VAD model state (similar to Python's VADState)

**Key Elements**:
- Hidden state array: `Array<FloatArray>` (2x1x128 dimensions)
- Window size: 512 samples
- Sample rate: 16000 Hz
- Reset function to clear state

**Implementation Notes**:
- Use `FloatArray` for state storage
- Initialize as zeros: `Array(2) { FloatArray(128) }`
- Store as class properties

### 2. AudioCapture.kt

**Purpose**: Captures audio from microphone in real-time

**Key Elements**:
- Use `AudioRecord` class for audio capture
- Sample rate: 16000 Hz (or resample if device uses different rate)
- Audio format: `AudioFormat.ENCODING_PCM_16BIT`
- Channel config: `AudioFormat.CHANNEL_IN_MONO`
- Buffer size: Calculate based on chunk duration (e.g., 32ms = 512 samples at 16kHz)

**Implementation Notes**:
- Request `RECORD_AUDIO` permission before starting
- Use `AudioRecord` in a separate thread or coroutine
- Read audio data in chunks matching your processing window
- Convert PCM 16-bit to Float32 normalized (-1.0 to 1.0)

### 3. VADProcessor.kt

**Purpose**: Processes audio chunks with ONNX model

**Key Elements**:
- Load ONNX model from assets: `OrtEnvironment.getEnvironment().createSession()`
- Input shape: `[1, window_size]` where window_size = 512
- Input names: `["input", "state", "sr"]`
- Output names: `["output", "state"]`
- Process each audio chunk through model

**Implementation Notes**:
- Use `OnnxTensor.createTensor()` to create input tensors
- Pad or trim audio to exactly 512 samples
- Update VADState after each inference
- Extract speech probability from first output
- Update state from second output

### 4. VADManager.kt

**Purpose**: Main orchestrator (similar to Python's realtime_vad_chunks)

**Key Elements**:
- Initialize AudioCapture
- Initialize VADProcessor with ONNX model
- Initialize VADState
- Main processing loop:
  - Read audio chunk
  - Process with VAD
  - Track speech/silence state
  - Save chunks when speech ends

**State Machine**:
- Track `isInSpeech: Boolean`
- Track `speechFrameCount: Int`
- Track `silentFrameCount: Int`
- Track `speechStartTime: Long`
- Track `lastSpeechTime: Long`

**Implementation Notes**:
- Use coroutines or background thread for processing
- Maintain audio buffers (original and resampled)
- Calculate thresholds in frames based on milliseconds
- Handle state transitions (speech start, speech end, silence detection)

### 5. AudioSaver.kt

**Purpose**: Saves audio chunks to WAV files

**Key Elements**:
- Write WAV file format (header + PCM data)
- File naming: `chunk_XXX_YYYYMMDD_HHMMSS_microseconds.wav`
- Save to app's external storage or internal storage
- Convert Float32 array back to PCM 16-bit

**Implementation Notes**:
- Use `FileOutputStream` to write files
- Write WAV header (44 bytes) before PCM data
- Convert normalized float (-1.0 to 1.0) to Int16
- Use `File` API to create directory structure

## Configuration

Create a `VADConfig.kt` object:

```kotlin
object VADConfig {
    const val SAMPLE_RATE = 16000
    const val CHUNK_DURATION_MS = 32
    const val DEFAULT_VAD_THRESHOLD = 0.25f
    const val DEFAULT_MIN_SPEECH_DURATION_MS = 250
    const val DEFAULT_MIN_SILENCE_DURATION_MS = 400
    const val DEFAULT_SILENCE_PAD_MS = 500
    const val WINDOW_SIZE_SAMPLES = 512
}
```

## Model Setup

### Download ONNX Model

1. Download Silero VAD ONNX model:
   - URL: `https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx`
   - Save as `silero_vad.onnx`

2. Place in Android assets:
   - Copy to `app/src/main/assets/silero_vad.onnx`

3. Load in code:
   ```kotlin
   val modelBytes = assets.open("silero_vad.onnx").readBytes()
   val session = ortEnvironment.createSession(modelBytes)
   ```

## Audio Processing Flow

### 1. Audio Capture Flow

```
AudioRecord.startRecording()
    ↓
Read audio data in chunks (32ms = 512 samples at 16kHz)
    ↓
Convert Int16 PCM to Float32 normalized (-1.0 to 1.0)
    ↓
Pass to VAD Processor
```

### 2. VAD Processing Flow

```
Receive audio chunk (Float32 array)
    ↓
Pad/trim to exactly 512 samples
    ↓
Create ONNX input tensors:
  - input: [1, 512]
  - state: [2, 1, 128]
  - sr: [16000]
    ↓
Run ONNX inference
    ↓
Extract speech probability (0.0 to 1.0)
    ↓
Update VADState with new state
```

### 3. Speech Detection Flow

```
Compare speech_prob > threshold
    ↓
If speech detected:
  - Increment speech_frame_count
  - Reset silent_frame_count
  - Update last_speech_time
  - If not in speech and frames >= min_speech_frames:
      → Set is_in_speech = true
      → Record speech_start_time
    ↓
If silence detected:
  - If in speech:
      → Increment silent_frame_count
      → If silent_frame_count >= silence_threshold:
          → Extract audio segment
          → Save chunk
          → Reset buffers and state
```

## Buffer Management

### Original Buffer
- Stores audio at device sample rate
- Used for saving final chunks
- Grows during speech detection
- Cleared after chunk save

### Resampled Buffer
- Stores audio at 16kHz (VAD sample rate)
- Used for VAD processing
- Grows during speech detection
- Cleared after chunk save

### Buffer Start Frame
- Tracks which frame index corresponds to buffer start
- Used to calculate correct audio segment boundaries
- Reset when buffers are cleared

## Chunk Saving Logic

### When to Save
1. Speech detected and `is_in_speech = true`
2. Silence detected for `min_silence_duration_ms`
3. Extract audio segment from `speech_start_time` to `last_speech_time + silence_pad_ms`

### Audio Segment Extraction
1. Calculate start/end sample indices from frame times
2. Extract from original buffer (device sample rate)
3. Add silence padding at the end
4. Convert Float32 to Int16 PCM
5. Write WAV file with proper header

## Threading Model

### Recommended Approach

```
Main Thread
    ↓
Background Coroutine/Thread
    ├── Audio Capture Thread
    ├── VAD Processing Thread
    └── File Saving Thread
```

### Implementation
- Use Kotlin Coroutines with `Dispatchers.IO` for audio processing
- Use `Flow` or `Channel` for audio data pipeline
- Use `suspend` functions for async operations
- Post results to main thread for UI updates

## Flutter Integration

### Method Channel Setup

1. Create Kotlin class implementing `MethodCallHandler`
2. Register in `MainActivity.kt`
3. Expose methods:
   - `startVAD()` - Start VAD processing
   - `stopVAD()` - Stop VAD processing
   - `getChunkCount()` - Get number of saved chunks
   - `getChunkPaths()` - Get list of saved chunk file paths
   - `configure(threshold, minSpeech, minSilence, silencePad)` - Configure parameters

### Flutter Side

```dart
// Example usage
final platform = MethodChannel('vad_channel');
await platform.invokeMethod('startVAD', {
  'threshold': 0.25,
  'minSpeechMs': 250,
  'minSilenceMs': 400,
  'silencePadMs': 500,
});
```

## File Storage

### Storage Options

1. **Internal Storage** (Recommended for privacy):
   - Path: `context.filesDir / "chunks"`
   - No permissions needed
   - Files deleted when app is uninstalled

2. **External Storage** (For user access):
   - Path: `context.getExternalFilesDir(null) / "chunks"`
   - Requires storage permission
   - Files accessible via file manager

### File Naming
- Format: `chunk_XXX_YYYYMMDD_HHMMSS_microseconds.wav`
- Use `SimpleDateFormat` for timestamp
- Increment chunk counter

## Error Handling

### Common Issues

1. **Permission Denied**:
   - Check runtime permissions
   - Show permission request dialog
   - Handle permission denial gracefully

2. **Model Loading Failed**:
   - Check if model file exists in assets
   - Verify ONNX Runtime version compatibility
   - Handle file read errors

3. **Audio Capture Failed**:
   - Check if microphone is available
   - Verify audio format compatibility
   - Handle device-specific issues

4. **Storage Full**:
   - Check available storage space
   - Implement chunk cleanup/rotation
   - Notify user when storage is low

## Performance Considerations

### Optimization Tips

1. **Audio Buffer Size**:
   - Use appropriate buffer size (512 samples = 32ms at 16kHz)
   - Balance latency vs CPU usage

2. **ONNX Inference**:
   - Run inference on background thread
   - Consider using NNAPI delegate if available
   - Batch processing if possible

3. **Memory Management**:
   - Clear buffers after chunk save
   - Limit buffer growth
   - Use object pooling for arrays

4. **Battery Optimization**:
   - Stop processing when app is backgrounded
   - Use wake locks sparingly
   - Optimize processing frequency

## Testing

### Test Scenarios

1. **Basic Speech Detection**:
   - Speak into microphone
   - Verify chunks are saved
   - Check file format and duration

2. **Silence Handling**:
   - Test with various silence durations
   - Verify chunks are saved correctly
   - Check silence padding

3. **Edge Cases**:
   - Very short speech (< min_speech_duration)
   - Very long speech segments
   - Rapid speech/silence transitions
   - Background noise handling

4. **Performance**:
   - Test on different Android versions
   - Test on low-end devices
   - Monitor CPU and memory usage

## Resources

### ONNX Runtime Android
- Documentation: https://onnxruntime.ai/docs/tutorials/mobile/
- GitHub: https://github.com/microsoft/onnxruntime

### Silero VAD Model
- Model Repository: https://huggingface.co/onnx-community/silero-vad
- Original Paper: Check Silero VAD documentation

### Android Audio
- AudioRecord: https://developer.android.com/reference/android/media/AudioRecord
- AudioFormat: https://developer.android.com/reference/android/media/AudioFormat

## Migration Checklist

- [ ] Set up Android project with ONNX Runtime dependency
- [ ] Add audio recording permissions
- [ ] Download and add ONNX model to assets
- [ ] Implement VADState class
- [ ] Implement AudioCapture class
- [ ] Implement VADProcessor with ONNX inference
- [ ] Implement VADManager with state machine
- [ ] Implement AudioSaver for WAV file writing
- [ ] Create configuration object
- [ ] Set up file storage directory
- [ ] Implement error handling
- [ ] Add logging/debugging output
- [ ] Test on physical device
- [ ] Integrate with Flutter (if needed)
- [ ] Optimize for performance
- [ ] Handle edge cases

## Notes

- The ONNX model expects input at 16kHz sample rate
- If device uses different sample rate, implement resampling
- State must be maintained between inference calls
- Buffer management is critical for correct chunk extraction
- Test thoroughly on different Android devices and versions

