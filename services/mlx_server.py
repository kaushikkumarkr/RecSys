"""
MLX Inference Service for RAG.
This script runs a simple HTTP server that accepts prompts and returns LLM completions.
Designed to run on Apple Silicon (M1/M2/M3) for GPU-accelerated inference.

Usage (local):
    python services/mlx_server.py

The server listens on port 8502 and exposes POST /generate endpoint.
"""
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Check if MLX is available (only works on Apple Silicon)
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available. Install with: pip install mlx mlx-lm")

# Default model - Phi-3 Mini is small and fast
MODEL_NAME = os.getenv("MLX_MODEL", "mlx-community/Phi-3-mini-4k-instruct-4bit")
MODEL = None
TOKENIZER = None

def load_model():
    global MODEL, TOKENIZER
    if MLX_AVAILABLE and MODEL is None:
        print(f"Loading model: {MODEL_NAME}")
        MODEL, TOKENIZER = load(MODEL_NAME)
        print("Model loaded successfully!")

class MLXHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 256)
            
            if not MLX_AVAILABLE:
                response = {"error": "MLX not available on this system"}
            elif MODEL is None:
                response = {"error": "Model not loaded"}
            else:
                try:
                    output = generate(MODEL, TOKENIZER, prompt=prompt, max_tokens=max_tokens, verbose=False)
                    response = {"response": output}
                except Exception as e:
                    response = {"error": str(e)}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "mlx_available": MLX_AVAILABLE, "model": MODEL_NAME if MODEL else None}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8502):
    load_model()
    server_address = ('', port)
    httpd = HTTPServer(server_address, MLXHandler)
    print(f"MLX Server running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8502
    run_server(port)
