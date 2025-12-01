import os
import sys
import wave
import numpy as np

rec_dir = os.path.join(os.getcwd(), "recordings")
if not os.path.isdir(rec_dir):
    print("No recordings/ directory found")
    sys.exit(1)

# find first file named segment_1_*.wav, else pick earliest segment_*.wav
files = [f for f in os.listdir(rec_dir) if f.lower().endswith('.wav')]
if not files:
    print('No wav files in recordings/')
    sys.exit(1)

seg1 = None
for f in files:
    if f.startswith('segment_1_'):
        seg1 = f
        break
if seg1 is None:
    # find by segment_ prefix and smallest number
    segs = [f for f in files if f.startswith('segment_')]
    if not segs:
        seg1 = files[0]
    else:
        # parse segment number
        def seg_key(fn):
            try:
                num = int(fn.split('_')[1])
            except Exception:
                num = 999999
            return num, fn
        segs.sort(key=seg_key)
        seg1 = segs[0]

path = os.path.join(rec_dir, seg1)
print('Inspecting:', path)
try:
    with wave.open(path, 'rb') as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        duration = nframes / float(fr)
        raw = wf.readframes(nframes)
        # convert to numpy
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if nch > 1:
            data = data.reshape(-1, nch)
            mono = data.mean(axis=1)
        else:
            mono = data
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        print(f"channels: {nch}")
        print(f"sample rate: {fr}")
        print(f"sample width bytes: {sampwidth}")
        print(f"frames: {nframes}")
        print(f"duration (s): {duration:.3f}")
        print(f"peak: {peak:.6f}")
        print(f"rms: {rms:.6f}")
except Exception as e:
    print('Error reading file:', e)
    sys.exit(1)
