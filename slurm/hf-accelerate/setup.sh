#!/bin/bash

mkdir -p output
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

wget https://raw.githubusercontent.com/huggingface/accelerate/refs/heads/main/examples/slurm/fsdp_config.yaml
wget https://raw.githubusercontent.com/huggingface/accelerate/refs/heads/main/examples/complete_nlp_example.py

# download, dataset, metric and and model to cache
python -c "from datasets import load_dataset; load_dataset('glue', 'mrpc')" 
python -c "import evaluate; evaluate.load('glue', 'mrpc')"
python -c "from transformers import pipeline; pipeline('fill-mask', model='bert-base-cased')"
