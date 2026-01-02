#!/bin/bash

# save git credentials
git config --global credential.helper store 

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# create and activate venv
uv venv
source .venv/bin/activate
export UV_LINK_MODE=copy

# install dependencies
uv pip install --no-cache-dir "vllm==0.11.0" "torch==2.8.0" "deepspeed==0.17.6" \
    "transformers[hf_xet]>=4.51.0" "trl==0.24.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" "grpcio>=1.62.1" "optree>=0.13.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb liger-kernel mathruler \
    pytest yapf py-spy pre-commit ruff

uv pip install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation
