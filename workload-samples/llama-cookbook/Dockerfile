FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN pip install wandb==0.18.7
RUN git clone --no-checkout https://github.com/meta-llama/llama-cookbook.git \
  && cd llama-cookbook \
  && git fetch --depth 1 origin faae2fd877995430906e1d0904131ecdaa89a604 \
  && git checkout faae2fd877995430906e1d0904131ecdaa89a604 \
  && pip install -U pip setuptools \
  && pip install -e .

RUN wget https://raw.githubusercontent.com/meta-llama/llama-cookbook/faae2fd877995430906e1d0904131ecdaa89a604/src/llama_cookbook/finetuning.py