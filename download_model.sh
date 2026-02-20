#!/bin/bash
set -e

MODEL_DIR="models/all-MiniLM-L6-v2"

mkdir -p "$MODEL_DIR"

echo "Downloading model files to $MODEL_DIR..."

curl -L -o "$MODEL_DIR/model.safetensors" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"
curl -L -o "$MODEL_DIR/tokenizer.json" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
curl -L -o "$MODEL_DIR/config.json" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"

echo "Model files downloaded successfully!"
