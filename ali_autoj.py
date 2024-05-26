from benchmarks.auto_j.codes.usage.constants_prompt import build_autoj_input
from jsonl2json import jsonl_to_json
import json
from vllm import LLM, SamplingParams
import torch
from transformers import pipeline
import os
import argparse

model_name_or_path = "/home/user/wuxie/ali/LLaMA-Factory/trained_models/test"  # or "local path to auto-j"

def extract_pairwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            pred_label = 0
        elif pred_rest.startswith('response 2'):
            pred_label = 1
        elif pred_rest.startswith('tie'):
            pred_label = 2
    return pred_label


def build_autoj_testset(exchange=False):
    # 遍历jsonl文件，将每一行的数据提取出来，然后调用build_autoj_input函数，将数据传入，然后将返回的数据写入到新的jsonl文件中
    origin = "/home/user/wuxie/ali/auto-j/data/test/testdata_pairwise.jsonl"
    if exchange:
        targetl = "/home/user/wuxie/ali/auto-j/data/test/autoj_test_exchange.jsonl"
        target = "/home/user/wuxie/ali/auto-j/data/test/autoj_test_exchange.json"
    else:
        targetl = "/home/user/wuxie/ali/auto-j/data/test/autoj_test.jsonl"
        target = "/home/user/wuxie/ali/auto-j/data/test/autoj_test.json"
    with open(origin, "r") as f:
        lines = f.readlines()
        with open(targetl, "w",encoding="utf-8",errors="ignore") as t:
            for line in lines:
                line = eval(line)
                prompt = line["prompt"]
                resp1 = line["response 1"]
                resp2 = line["response 2"]
                resp1,resp2 = resp2,resp1 if exchange else resp1,resp2
                input_pairwise = build_autoj_input(prompt=prompt,
                                        resp1=resp1,
                                        resp2=resp2,
                                        protocol="pairwise_tie")
                t.write(json.dumps({"input": input_pairwise},ensure_ascii=False) + "\n")
    jsonl_to_json(targetl, target)
    print("Done!")


def extract_autoj_testset_output(input_jsonl,output_jsonl): 
    # input_jsonl = '/home/user/wuxie/ali/LLaMA-Factory/saves/LLaMA2-7B-Chat/lora/eval_2024-05-24-22-35-54/generated_predictions.jsonl'
    # output_jsonl = '/home/user/wuxie/ali/LLaMA-Factory/saves/LLaMA2-7B-Chat/lora/eval_2024-05-24-22-35-54/output.jsonl'
    with open(input_jsonl, "r") as f:
        lines = f.readlines()
        with open(output_jsonl, "w",encoding="utf-8",errors="ignore") as t:
            for line in lines:
                # line = eval(line)
                evaluation_result = extract_pairwise_result(line)
                t.write(json.dumps({"output": evaluation_result},ensure_ascii=False) + "\n")
    print("Done!")



def inference_vllm():
    # num_gpus = torch.cuda.device_count()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    llm = LLM(model=model_name_or_path, tensor_parallel_size=1,device=torch.device("cuda:1"))
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    input_pairwise = build_autoj_input(prompt="What's the capital of the United States?",
                                       resp1="The capital of the United States is New York.",
                                       resp2="The captical of the United States is Washington D.C.",
                                       protocol="pairwise_tie")  # for pairwise response comparison
    input = input_pairwise  # or input_single

    outputs = llm.generate(input, sampling_params)

    judgment = outputs[0].outputs[0].text

    print(judgment)

    evaluation_result = extract_pairwise_result(judgment)

    print(evaluation_result)


if __name__ == "__main__":
    # inference_vllm()
    # build_autoj_testset()
    # inference_hf()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default='llmj/saves/LLaMA2-7B-Chat/lora/eval_test_exchange/generated_predictions.jsonl', help="path to the prediction file")
    parser.add_argument("--output_jsonl", type=str, default='llmj/saves/LLaMA2-7B-Chat/lora/eval_test_exchange/autoj.jsonl', help="path to the autoj-formed jsonl file")
    args = parser.parse_args()
    extract_autoj_testset_output(args.input_jsonl,args.output_jsonl)
    print("extract output Done!")