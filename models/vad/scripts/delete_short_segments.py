import os
import wave
import sys

REC_DIR = os.path.join(os.getcwd(), 'recordings')
if not os.path.isdir(REC_DIR):
    print('No recordings/ directory found')
    sys.exit(0)

threshold = 0.5
deleted = []
for fn in sorted(os.listdir(REC_DIR)):
    if not fn.lower().endswith('.wav'):
        continue
    path = os.path.join(REC_DIR, fn)
    try:
        with wave.open(path, 'rb') as wf:
            fr = wf.getframerate()
            n = wf.getnframes()
            duration = n / float(fr) if fr else 0.0
        if duration < threshold:
            os.remove(path)
            deleted.append((fn, duration))
    except Exception as e:
        print(f'Error reading {fn}: {e}')

if deleted:
    print('Deleted files:')
    for fn, d in deleted:
        print(f"  {fn}  ({d:.3f}s)")
else:
    print('No files deleted (none shorter than {0:.3f}s)'.format(threshold))
