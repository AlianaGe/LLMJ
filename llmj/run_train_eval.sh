#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=1

# # llama2 直接推理
# EXP_NO=0
# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference.yaml

# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference_exchange.yaml

# # 3+origin推理
EXP_NO=5
llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference.yaml

llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference_exchange.yaml

# # test4model和rule推理
# EXP_NO=4
# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference.yaml

# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference_exchange.yaml

# # 2个原始数据集训练和rule推理
# EXP_NO=3

# # Run the first llamafactory-cli command
# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/train.yaml

# # Run the second llamafactory-cli command
# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference.yaml

# # Run the third llamafactory-cli command
# llamafactory-cli train saves/LLaMA2-7B-Chat/lora/${EXP_NO}/inference_exchange.yaml

