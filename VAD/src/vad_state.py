import numpy as np

class VADState:
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.window_size_samples = 512
    
    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

