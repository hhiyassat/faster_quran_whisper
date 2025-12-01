# Kotlin VAD - Local Usage Guide

This guide explains how to use the Kotlin VAD implementation locally in your Flutter/Android app without any API calls.

## Quick Start

### 1. Setup Dependencies

Add to `app/build.gradle`:

```gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

### 2. Add Model to Assets

1. Download `silero_vad.onnx` model
2. Place in `app/src/main/assets/silero_vad.onnx`
3. Model is loaded locally from assets (no download needed)

### 3. Add Permissions

In `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Request permission at runtime before starting VAD.

### 4. Initialize VAD Manager

```kotlin
val vadManager = VADManager(
    context = this,
    threshold = 0.25f,
    minSpeechMs = 250,
    minSilenceMs = 400,
    silencePadMs = 500
)
```

### 5. Start VAD Processing

```kotlin
vadManager.startVAD()
```

### 6. Stop VAD Processing

```kotlin
vadManager.stopVAD()
```

## How It Works Locally

### No Internet Required
- Model is bundled in app assets
- All processing happens on-device
- No API calls or network requests
- Works completely offline

### Processing Flow
1. **Audio Capture**: Microphone captures audio at device sample rate
2. **Resampling**: Audio resampled to 16kHz if needed
3. **VAD Inference**: ONNX model processes audio locally
4. **Speech Detection**: Detects speech vs silence
5. **Chunk Saving**: Saves audio chunks to local storage

## Configuration

### VAD Parameters

- **threshold** (0.0-1.0): Speech detection sensitivity
  - Lower = more sensitive (detects quieter speech)
  - Higher = less sensitive (fewer false positives)
  - Default: 0.25

- **minSpeechMs**: Minimum speech duration before detection starts
  - Default: 250ms

- **minSilenceMs**: Minimum silence after speech to save chunk
  - Default: 400ms

- **silencePadMs**: Additional silence included with each chunk
  - Default: 500ms

### Example Configuration

```kotlin
val vadManager = VADManager(
    context = this,
    threshold = 0.3f,        // Less sensitive
    minSpeechMs = 300,       // Longer minimum speech
    minSilenceMs = 500,      // Longer silence wait
    silencePadMs = 600       // More padding
)
```

## File Storage

### Default Location
Chunks are saved to:
```
/data/data/your.package.name/files/chunks/
```

### Accessing Saved Chunks

```kotlin
val chunksDir = File(context.filesDir, "chunks")
val chunkFiles = chunksDir.listFiles()?.filter { it.name.endsWith(".wav") }
```

### File Naming
Format: `chunk_XXX_YYYYMMDD_HHMMSS_microseconds.wav`

Example: `chunk_001_20241129_143022_123456.wav`

## Flutter Integration

### Setup Method Channel

In `MainActivity.kt`:

```kotlin
class MainActivity: FlutterActivity() {
    private val CHANNEL = "vad_channel"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        MethodChannel(flutterEngine?.dartExecutor?.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "startVAD" -> {
                        // Start VAD
                        result.success(true)
                    }
                    "stopVAD" -> {
                        // Stop VAD
                        result.success(true)
                    }
                    "getChunkCount" -> {
                        // Return chunk count
                        result.success(count)
                    }
                    else -> result.notImplemented()
                }
            }
    }
}
```

### Flutter Side Usage

```dart
import 'package:flutter/services.dart';

class VADService {
  static const platform = MethodChannel('vad_channel');
  
  Future<void> startVAD({
    double threshold = 0.25,
    int minSpeechMs = 250,
    int minSilenceMs = 400,
    int silencePadMs = 500,
  }) async {
    await platform.invokeMethod('startVAD', {
      'threshold': threshold,
      'minSpeechMs': minSpeechMs,
      'minSilenceMs': minSilenceMs,
      'silencePadMs': silencePadMs,
    });
  }
  
  Future<void> stopVAD() async {
    await platform.invokeMethod('stopVAD');
  }
  
  Future<int> getChunkCount() async {
    final count = await platform.invokeMethod('getChunkCount');
    return count as int;
  }
}
```

## Usage Example

### Basic Usage

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var vadManager: VADManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        vadManager = VADManager(
            context = this,
            threshold = 0.25f
        )
        
        requestAudioPermission {
            vadManager.startVAD()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        vadManager.stopVAD()
    }
}
```

### With Callbacks

```kotlin
vadManager.setOnChunkSavedListener { chunkPath, duration ->
    Log.d("VAD", "Chunk saved: $chunkPath ($duration seconds)")
    // Update UI or process chunk
}

vadManager.setOnSpeechDetectedListener { isSpeaking ->
    if (isSpeaking) {
        // Show "Speaking..." indicator
    } else {
        // Hide indicator
    }
}
```

## Local Storage Management

### Check Storage Space

```kotlin
val chunksDir = File(context.filesDir, "chunks")
val totalSize = chunksDir.walkTopDown()
    .filter { it.isFile }
    .map { it.length() }
    .sum()
    
Log.d("Storage", "Total chunks size: ${totalSize / 1024 / 1024} MB")
```

### Clean Old Chunks

```kotlin
fun cleanOldChunks(keepLastN: Int = 10) {
    val chunksDir = File(context.filesDir, "chunks")
    val chunks = chunksDir.listFiles()
        ?.filter { it.name.endsWith(".wav") }
        ?.sortedBy { it.lastModified() }
    
    chunks?.dropLast(keepLastN)?.forEach { it.delete() }
}
```

### Delete All Chunks

```kotlin
fun deleteAllChunks() {
    val chunksDir = File(context.filesDir, "chunks")
    chunksDir.listFiles()?.forEach { it.delete() }
}
```

## Troubleshooting

### Model Not Found
- Verify model file is in `app/src/main/assets/`
- Check file name is exactly `silero_vad.onnx`
- Ensure file is included in build (check build.gradle)

### No Audio Captured
- Check microphone permission is granted
- Verify microphone is not used by another app
- Check device audio settings

### Chunks Not Saved
- Verify storage permissions (if using external storage)
- Check available storage space
- Ensure VAD is actually detecting speech (check threshold)

### Performance Issues
- Reduce processing frequency
- Use smaller buffer sizes
- Check device capabilities
- Monitor CPU usage

## Best Practices

1. **Request Permissions Early**: Request audio permission before user needs to use VAD

2. **Handle Background**: Stop VAD when app goes to background to save battery

3. **Storage Management**: Implement chunk cleanup to prevent storage overflow

4. **Error Handling**: Always handle exceptions in audio capture and processing

5. **User Feedback**: Show visual indicators when speech is detected

6. **Configuration**: Allow users to adjust VAD parameters for their environment

## Offline Operation

### Complete Offline Support
- ✅ Model bundled in app (no download)
- ✅ All processing on-device
- ✅ No network calls
- ✅ Works in airplane mode
- ✅ No external dependencies

### First Run
- Model loads from assets (instant)
- No internet connection needed
- Ready to use immediately

## Model Information

- **Model**: Silero VAD ONNX
- **Size**: ~1-2 MB
- **Input**: 16kHz mono audio
- **Window**: 512 samples (32ms)
- **Output**: Speech probability (0.0-1.0)
- **State**: Maintained between calls

## Performance

### Typical Performance
- **Latency**: < 50ms per chunk
- **CPU Usage**: Low (ONNX optimized)
- **Memory**: ~10-20 MB
- **Battery**: Minimal impact

### Optimization Tips
- Process in background thread
- Use appropriate buffer sizes
- Clear buffers after saving
- Stop when not needed

## Security & Privacy

### Local Processing
- All audio processed on-device
- No data sent to servers
- Chunks stored locally
- User has full control

### Permissions
- Only requires microphone permission
- No internet permission needed
- No location or other sensitive permissions

## Next Steps

1. Implement the Kotlin classes (see KOTLIN_PORT_GUIDE.md)
2. Add model to assets folder
3. Set up permissions
4. Test on physical device
5. Integrate with Flutter UI
6. Add error handling
7. Optimize for your use case

