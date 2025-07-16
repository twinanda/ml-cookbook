#!/bin/bash
set -e

: "${HF_TOKEN:?Environment variable HF_TOKEN is not set}"

python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

# install torchft
pip install torchft-nightly==2025.7.15

# download llama tokenizer
wget https://raw.githubusercontent.com/pytorch/torchtitan/refs/tags/v0.1.0/scripts/download_tokenizer.py
python download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"

# download model config
wget https://raw.githubusercontent.com/pytorch/torchtitan/refs/tags/v0.1.0/torchtitan/models/llama3/train_configs/llama3_8b.toml