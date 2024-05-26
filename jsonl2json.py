import json
import sys


# 读取JSONL文件并将每一行解析为JSON对象
def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        json_list = [json.loads(line) for line in f]

    # 将JSON对象列表写入JSON文件
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 从命令行获取JSONL文件和JSON文件路径
    if len(sys.argv) < 3:
        print("Usage: python jsonl2json.py <jsonl_file> <json_file>")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    json_file = sys.argv[2]

    jsonl_to_json(jsonl_file, json_file)
