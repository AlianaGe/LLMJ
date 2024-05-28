#!/bin/bash

# 定义变量
BASE_DIR="llmj/saves/LLaMA2-7B-Chat/lora/"
EXP_NO="4"
INPUT_JSONL_1="${BASE_DIR}${EXP_NO}eval/generated_predictions.jsonl"
OUTPUT_JSONL_1="${BASE_DIR}${EXP_NO}eval/autoj.jsonl"
INPUT_JSONL_2="${BASE_DIR}${EXP_NO}eval_exchange/generated_predictions.jsonl"
OUTPUT_JSONL_2="${BASE_DIR}${EXP_NO}eval_exchange/autoj.jsonl"
SOURCE_FILE_PATH="./benchmarks/auto_j/data/test/testdata_pairwise.jsonl"
# 运行第一个命令
python ali_autoj.py --input_jsonl $INPUT_JSONL_1 --output_jsonl $OUTPUT_JSONL_1

# 运行第二个命令
python ali_autoj.py --input_jsonl $INPUT_JSONL_2 --output_jsonl $OUTPUT_JSONL_2

# 运行第三个命令
python benchmarks/auto_j/codes/leaderboard/pairwise_eval.py --type=pairwise --pred_file_path=$OUTPUT_JSONL_1 --exchange_pred_file_path=$OUTPUT_JSONL_2 --source_file_path=$SOURCE_FILE_PATH


# INPUT_JSONL_1="llmj/saves/LLaMA2-7B-Chat/lora/1eval/generated_predictions.jsonl"
# OUTPUT_JSONL_1="llmj/saves/LLaMA2-7B-Chat/lora/1eval/autoj.jsonl"
# INPUT_JSONL_2="llmj/saves/LLaMA2-7B-Chat/lora/1eval_exchange/generated_predictions.jsonl"
# OUTPUT_JSONL_2="llmj/saves/LLaMA2-7B-Chat/lora/1eval_exchange/autoj.jsonl"
# PRED_FILE_PATH="./llmj/saves/LLaMA2-7B-Chat/lora/1eval/autoj.jsonl"
# EXCHANGE_PRED_FILE_PATH="./llmj/saves/LLaMA2-7B-Chat/lora/1eval_exchange/autoj.jsonl"


