# Kotlin VAD Model Parameters

## ONNX Model Input/Output Structure for Silero VAD

### ⚠️ Important: Manual Audio Chunking Required

**The model does NOT automatically split audio. You must manually provide 512-sample chunks.**

- **Each inference call processes exactly 512 samples** (32ms at 16kHz)
- **You must split audio manually into 512-sample chunks**
- **For real-time: Read 512 samples from microphone → Process → Repeat**
- **For batch: Split audio file into sequential 512-sample windows**

See `AUDIO_CHUNKING_EXPLANATION.md` for detailed explanation.

---

### Model Inputs

```kotlin
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

// Input 1: Audio Data
// Name: "input"
// Shape: [1, 512]
// Type: FloatArray
val audioChunk = FloatArray(512) // 512 samples at 16kHz = 32ms
// Normalized to -1.0 to 1.0 range
val audioInput = OnnxTensor.createTensor(
    ortEnvironment,
    arrayOf(audioChunk) // Shape: [1, 512]
)

// Input 2: VAD State
// Name: "state"  
// Shape: [2, 1, 128]
// Type: Array<FloatArray>
val state = Array(2) { 
    Array(1) { 
        FloatArray(128) { 0.0f } 
    } 
}
// Flatten to single dimension: [2, 1, 128] = [256] total elements
val stateFlat = FloatArray(256)
var idx = 0
for (i in 0 until 2) {
    for (j in 0 until 1) {
        for (k in 0 until 128) {
            stateFlat[idx++] = state[i][j][k]
        }
    }
}
val stateTensor = OnnxTensor.createTensor(
    ortEnvironment,
    arrayOf(stateFlat),
    longArrayOf(2, 1, 128) // Original shape
)

// Input 3: Sample Rate
// Name: "sr"
// Shape: [1]
// Type: LongArray (int64)
val sampleRate = longArrayOf(16000L)
val srTensor = OnnxTensor.createTensor(
    ortEnvironment,
    sampleRate
)
```

### Complete Inference Example

```kotlin
class VADProcessor(private val context: Context) {
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var vadState = Array(2) { Array(1) { FloatArray(128) { 0.0f } } }
    
    private val SAMPLE_RATE = 16000
    private val WINDOW_SIZE = 512
    
    fun initialize() {
        ortEnvironment = OrtEnvironment.getEnvironment()
        
        // Load model from assets
        val modelBytes = context.assets.open("silero_vad.onnx").readBytes()
        ortSession = ortEnvironment.createSession(modelBytes)
    }
    
    fun processAudioChunk(audioChunk: FloatArray, threshold: Float = 0.25f): Float {
        // Ensure audio is exactly 512 samples
        val processedAudio = if (audioChunk.size < WINDOW_SIZE) {
            audioChunk + FloatArray(WINDOW_SIZE - audioChunk.size) { 0.0f }
        } else {
            audioChunk.take(WINDOW_SIZE).toFloatArray()
        }
        
        // Prepare inputs
        val audioInput = OnnxTensor.createTensor(
            ortEnvironment,
            arrayOf(processedAudio) // Shape: [1, 512]
        )
        
        val stateFlat = FloatArray(256)
        var idx = 0
        for (i in 0 until 2) {
            for (j in 0 until 1) {
                for (k in 0 until 128) {
                    stateFlat[idx++] = vadState[i][j][k]
                }
            }
        }
        val stateTensor = OnnxTensor.createTensor(
            ortEnvironment,
            arrayOf(stateFlat),
            longArrayOf(2, 1, 128)
        )
        
        val srTensor = OnnxTensor.createTensor(
            ortEnvironment,
            longArrayOf(SAMPLE_RATE.toLong())
        )
        
        // Prepare input map
        val inputs = mapOf(
            "input" to audioInput,
            "state" to stateTensor,
            "sr" to srTensor
        )
        
        // Run inference
        val outputs = ortSession.run(inputs)
        
        // Extract outputs
        val speechProbTensor = outputs[0] as OnnxTensor
        val speechProbArray = speechProbTensor.floatBuffer.array()
        val speechProbability = speechProbArray[0] // Extract [0][0][0]
        
        val newStateTensor = outputs[1] as OnnxTensor
        val newStateArray = newStateTensor.floatBuffer.array()
        
        // Update state: reshape from [256] back to [2, 1, 128]
        idx = 0
        for (i in 0 until 2) {
            for (j in 0 until 1) {
                for (k in 0 until 128) {
                    vadState[i][j][k] = newStateArray[idx++]
                }
            }
        }
        
        // Clean up tensors
        audioInput.close()
        stateTensor.close()
        srTensor.close()
        speechProbTensor.close()
        newStateTensor.close()
        outputs.close()
        
        return speechProbability
    }
    
    fun resetState() {
        vadState = Array(2) { Array(1) { FloatArray(128) { 0.0f } } }
    }
}
```

### Data Type Summary

| Parameter | Python Type | Kotlin Type | Shape | Example |
|-----------|-------------|-------------|-------|---------|
| **input** | `np.float32` | `FloatArray` | `[1, 512]` | `FloatArray(512)` |
| **state** | `np.float32` | `Array<FloatArray>` | `[2, 1, 128]` | `Array(2) { Array(1) { FloatArray(128) } }` |
| **sr** | `np.int64` | `LongArray` | `[1]` | `longArrayOf(16000L)` |
| **output[0]** | `float` | `Float` | `[0][0][0]` | Speech probability `0.0-1.0` |
| **output[1]** | `np.float32` | `Array<FloatArray>` | `[2, 1, 128]` | Updated state |

### Key Points

1. **Audio Input**: Must be exactly 512 samples (32ms at 16kHz), normalized to -1.0 to 1.0
2. **State**: 3D array `[2, 1, 128]` = 256 total float values, must be maintained between calls
3. **Sample Rate**: Single int64 value `[16000]`
4. **Speech Probability**: Single float value from `outputs[0][0][0]`
5. **State Update**: Replace old state with `outputs[1]` for next iteration

### Memory Management

```kotlin
// Always close tensors after use to free memory
audioInput.close()
stateTensor.close()
srTensor.close()
outputs.close()
```

### Example Usage in Real-Time Loop

```kotlin
val vadProcessor = VADProcessor(context)
vadProcessor.initialize()

while (isRecording) {
    val audioChunk = readAudioChunk() // Get 512 samples
    val speechProb = vadProcessor.processAudioChunk(audioChunk, threshold = 0.25f)
    
    if (speechProb > 0.25f) {
        // Speech detected
        handleSpeech()
    } else {
        // Silence detected
        handleSilence()
    }
}
```

