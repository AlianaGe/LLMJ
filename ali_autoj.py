from benchmarks.auto_j.codes.usage.ali_constants_prompt import build_autoj_input, RULE
from jsonl2json import jsonl_to_json
import json

# from vllm import LLM, SamplingParams
import torch
from transformers import pipeline
import os
import argparse

model_name_or_path = "/home/user/wuxie/ali/LLaMA-Factory/trained_models/test"  # or "local path to auto-j"


def create_symlink(source_path, target_path):
    # 检查目标路径是否已存在
    if os.path.exists(target_path) or os.path.islink(target_path):
        os.remove(target_path)  # 删除现有的文件或链接

    # 创建软链接
    os.symlink(source_path, target_path)
    print(f"软链接已创建: {source_path} -> {target_path}")


def extract_pairwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind("final decision is ")
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len("final decision is ") :].strip().lower()
        if pred_rest.startswith("response 1"):
            pred_label = 0
        elif pred_rest.startswith("response 2"):
            pred_label = 1
        elif pred_rest.startswith("tie"):
            pred_label = 2
    return pred_label


def build_autoj_testset(
    save_name="autoj_test",
    exchange=False,
    addition_prompt="",
    origin="/home/user/wuxie/ali/benchmarks/auto_j/data/test/testdata_pairwise.jsonl",
    output_dir="/home/user/wuxie/ali/benchmarks/auto_j/data/test/",
    llmj_data_dir="/home/user/wuxie/ali/llmj/data/",
):
    # 读取JSONL文件并处理每一行
    save_name = save_name + "_exchange" + ".json" if exchange else save_name + ".json"
    target = output_dir + save_name
    targetl = target + "l"
    with open(origin, "r") as f:
        lines = f.readlines()
        with open(targetl, "w", encoding="utf-8", errors="ignore") as t:
            for line in lines:
                line = eval(line)
                prompt = line["prompt"]
                resp1 = line["response 1"]
                resp2 = line["response 2"]
                if exchange:
                    resp1, resp2 = resp2, resp1
                input_pairwise = build_autoj_input(
                    prompt=prompt,
                    resp1=resp1,
                    resp2=resp2,
                    rule=RULE + addition_prompt,
                    protocol="pairwise_tie",
                )  # for pairwise response comparison
                ###
                t.write(
                    json.dumps({"input": input_pairwise}, ensure_ascii=False) + "\n"
                )
                ###
    jsonl_to_json(targetl, target)
    print("处理完成，结果已写入", target)
    create_symlink(target, llmj_data_dir + save_name)


def build_autoj_trainset(
    save_name,
    exchange=False,
    addition_prompt="",
    origin="/home/user/wuxie/ali/benchmarks/auto_j/data/training/pairwise_traindata_recover.jsonl",
    output_dir="/home/user/wuxie/ali/benchmarks/auto_j/data/training/",
    llmj_data_dir="/home/user/wuxie/ali/llmj/data/",
):
    # 读取JSONL文件并处理每一行
    save_name = save_name + "_exchange" + ".json" if exchange else save_name + ".json"
    target = output_dir + save_name
    targetl = target + "l"
    with open(origin, "r", encoding="utf-8") as infile, open(
        targetl, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            line = eval(line)
            prompt = line["prompt"]
            resp1 = line["response 1"]
            resp2 = line["response 2"]
            if exchange:
                resp1, resp2 = resp2, resp1
            usrmsg = build_autoj_input(
                prompt=prompt,
                resp1=resp1,
                resp2=resp2,
                rule=RULE + addition_prompt,
                protocol="pairwise_tie",
            )  # for pairwise response comparison
            ###
            line["usrmsg"] = usrmsg
            del line["prompt"]
            del line["response 1"]
            del line["response 2"]
            outfile.write(json.dumps(line) + "\n")
            ###
    jsonl_to_json(targetl, target)
    print("处理完成，结果已写入", target)
    create_symlink(target, llmj_data_dir + save_name)


def recover_autoj_trainset(input_file_path, output_file_path):
    import re

    # pattern = re.compile(
    #     r"\[Query\]:\s*(.*?)\s*\*\*\*\s*\[Response 1\]:\s*(.*?)\s*\*\*\*\s*\[Response 2\]:\s*(.*?)\s*\*\*\*",
    #     re.DOTALL,
    # )
    pattern = re.compile(
        r"\[Query\]:\s*(.*?)\s*\*\*\*\s*\[Response 1\]:\s*(.*?)\s*\*\*\*\s*\[Response 2\]:\s*(.*?)\s*\*\*\*\s*\[END DATA\]\s*(.*)",
        re.DOTALL,
    )
    # 使用正则表达式从字符串中提取信息
    # match = pattern.search(data)
    with open(input_file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line.strip())
            match = pattern.search(data["usrmsg"])
            if match:
                query = match.group(1).strip()
                response1 = match.group(2).strip()
                response2 = match.group(3).strip()
                instructions = match.group(4).strip()
                data["prompt"] = query
                data["response 1"] = response1
                data["response 2"] = response2
                data["instructions"] = instructions
                del data["usrmsg"]
                outfile.write(json.dumps(data) + "\n")
            else:
                print("No match found")


def extract_autoj_testset_output(input_jsonl, output_jsonl):
    # input_jsonl = '/home/user/wuxie/ali/LLaMA-Factory/saves/LLaMA2-7B-Chat/lora/eval_2024-05-24-22-35-54/generated_predictions.jsonl'
    # output_jsonl = '/home/user/wuxie/ali/LLaMA-Factory/saves/LLaMA2-7B-Chat/lora/eval_2024-05-24-22-35-54/output.jsonl'
    with open(input_jsonl, "r") as f:
        lines = f.readlines()
        with open(output_jsonl, "w", encoding="utf-8", errors="ignore") as t:
            for line in lines:
                # line = eval(line)
                evaluation_result = extract_pairwise_result(line)
                t.write(
                    json.dumps({"output": evaluation_result}, ensure_ascii=False) + "\n"
                )
    print("Done!")


def inference_vllm():
    # num_gpus = torch.cuda.device_count()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # llm = LLM(model=model_name_or_path, tensor_parallel_size=1,device=torch.device("cuda:1"))
    # sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    # input_pairwise = build_autoj_input(prompt="What's the capital of the United States?",
    #                                    resp1="The capital of the United States is New York.",
    #                                    resp2="The captical of the United States is Washington D.C.",
    #                                    protocol="pairwise_tie")  # for pairwise response comparison
    # input = input_pairwise  # or input_single

    # outputs = llm.generate(input, sampling_params)

    # judgment = outputs[0].outputs[0].text

    # print(judgment)

    # evaluation_result = extract_pairwise_result(judgment)

    # print(evaluation_result)
    pass


def process_criteria():
    os.chdir("./benchmarks/auto_j")
    import yaml
    from yamlinclude import YamlIncludeConstructor

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

    def read_yaml(yaml_file_path):
        with open(yaml_file_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data

    yaml_content = read_yaml(
        "./other_resources/scenario_criteria/specials/analyzing_general.yaml"
    )
    print(yaml_content)


if __name__ == "__main__":
    # process_criteria()

    # recover_autoj_trainset(
    #     input_file_path="/home/user/wuxie/ali/benchmarks/auto_j/data/training/pairwise_traindata.jsonl",
    #     output_file_path="/home/user/wuxie/ali/benchmarks/auto_j/data/training/pairwise_traindata_recover.jsonl",
    # )

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--input_jsonl",
    #     type=str,
    #     default="llmj/saves/LLaMA2-7B-Chat/lora/eval_test_exchange/generated_predictions.jsonl",
    #     help="path to the prediction file",
    # )
    # parser.add_argument(
    #     "--output_jsonl",
    #     type=str,
    #     default="llmj/saves/LLaMA2-7B-Chat/lora/eval_test_exchange/autoj.jsonl",
    #     help="path to the autoj-formed jsonl file",
    # )
    # args = parser.parse_args()
    # extract_autoj_testset_output(args.input_jsonl, args.output_jsonl)
    # print("extract output Done!")

    rule_prompt = """
Here are some rules of the evaluation:

1. You should prioritize evaluating whether the response honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
2. You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the responses were presented should NOT affect your judgment, as Response 1 and Response 2 are **equally likely** to be the better.
"""
#     metric_prompt = """

# """
# build_autoj_trainset(
#     save_name="autoj_rule",
#     exchange=False,
#     addition_prompt=rule_prompt,
# )
# build_autoj_trainset(
#     save_name="autoj_rule",
#     exchange=True,
#     addition_prompt=rule_prompt,
# )
# build_autoj_testset(
#     save_name="autoj_test_rule",
#     exchange=False,
#     addition_prompt=rule_prompt,
# )
# build_autoj_testset(
#     save_name="autoj_test_rule",
#     exchange=True,
#     addition_prompt=rule_prompt,
# )
