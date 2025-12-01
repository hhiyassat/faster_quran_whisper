import os
import sys
import urllib.request
import onnxruntime

def download_onnx_model(model_path, model_url, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        return model_path
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        return model_path
    except Exception as e:
        print(f"Failed to download ONNX model: {e}")
        print(f"Try downloading manually from: {model_url}")
        print(f"And save to: {model_path}")
        sys.exit(1)

def load_silero_vad_onnx(model_path, model_url, model_dir):
    try:
        model_path = download_onnx_model(model_path, model_url, model_dir)
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        print(f"Could not load ONNX model: {e}")
        sys.exit(1)

