#!/bin/bash

mkdir -p output
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

wget https://raw.githubusercontent.com/huggingface/accelerate/refs/heads/main/examples/slurm/fsdp_config.yaml
wget https://raw.githubusercontent.com/huggingface/accelerate/refs/heads/main/examples/complete_nlp_example.py
