#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=1

# llama2 直接推理
EXP_NO=0
llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference.yaml

llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference_exchange.yaml


# cutoff lens exp
EXP_NO=6


# llama3