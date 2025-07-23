#!/usr/bin/env python
# coding: utf-8
"""
cube_parser_v3_fixed.py
=======================
使用v2的完整正则逻辑 + LLM备用解析
用法：
python cube_parser_v3.py --input ./predictions_2cube/predictions_img_init_2obj_v2.jsonl --output cube_parser_v4_fixed.jsonl --model_path ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf --n_gpu_layers 30
"""
import re
import json
import argparse
from pathlib import Path
from typing import Optional

# -------- LLM 依赖 -------- #
try:
    from llama_cpp import Llama  # pip install llama-cpp-python>=0.2.40
except ImportError as e:
    raise ImportError("请先安装 llama-cpp-python: pip install --upgrade llama-cpp-python") from e

# -------- 色彩归一映射（使用v2的逻辑）-------- #
color_map = {
    "turquoise":"green",
    "cyan": "green",  # v2中cyan映射到green
    "blue": "blue",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "magenta": "magenta",
    "black": "black",
    "white": "white",
}


def normalize_color(color):
    """完全使用v2的normalize_color函数"""
    color = color.lower()
    return color_map.get(color, color)


# -------- v2的完整正则逻辑 -------- #
def parse_prediction_regex(text: str) -> Optional[dict]:
    """完全使用v2的parse_prediction逻辑"""

    # 1. 检查是否有"on top of"字样
    if "on top of" in text:
        # 抓取top和bottom的颜色
        m = re.search(r'The (\w+) (?:cube|object) is on top of the (\w+) (?:cube|object)', text, re.IGNORECASE)
        if m:
            top = normalize_color(m.group(1))
            bottom = normalize_color(m.group(2))
            return {"relationship": "stacked", "top": {"color": top}, "bottom": {"color": bottom}}

        # 或支持"A is on top of B"的句式
        m2 = re.search(r'(\w+) (?:cube|object) is on top of (\w+) (?:cube|object)', text, re.IGNORECASE)
        if m2:
            top = normalize_color(m2.group(1))
            bottom = normalize_color(m2.group(2))
            return {"relationship": "stacked", "top": {"color": top}, "bottom": {"color": bottom}}

        # 如果捕获失败，返回空结构（这会导致正则失败，LLM接入）
        return {"relationship": "stacked", "top": {}, "bottom": {}}

    else:
        # 没有"on top of"，就检测分开（左右）
        m_left = re.search(r'The (\w+) (?:cube|object) is on the left side of the image', text, re.IGNORECASE)
        m_right = re.search(r'The (\w+) (?:cube|object) is on the right side of the image', text, re.IGNORECASE)
        if m_left and m_right:
            left = normalize_color(m_left.group(1))
            right = normalize_color(m_right.group(1))
            return {"relationship": "separated", "left": {"color": left}, "right": {"color": right}}

        # 或支持"A is on the left ... B is on the right ..."的句式
        m3 = re.search(r'(\w+) (?:cube|object) is on the left.*?(\w+) (?:cube|object) is on the right', text,
                       re.IGNORECASE)
        if m3:
            left = normalize_color(m3.group(1))
            right = normalize_color(m3.group(2))
            return {"relationship": "separated", "left": {"color": left}, "right": {"color": right}}

        # 如果左右都找不到，返回空结构（这会导致正则失败，LLM接入）
        return {"relationship": "separated", "left": {}, "right": {}}


def is_structured(parsed):
    """完全使用v2的is_structured逻辑"""
    # 检查结构化信息是否完整（不能有空字典）
    if parsed["relationship"] == "stacked":
        return bool(parsed.get("top")) and bool(parsed.get("bottom")) and "color" in parsed["top"] and "color" in \
            parsed["bottom"]
    elif parsed["relationship"] == "separated":
        return bool(parsed.get("left")) and bool(parsed.get("right")) and "color" in parsed["left"] and "color" in \
            parsed["right"]
    else:
        return False


# -------- LLM 封装 -------- #
class LLMParser:
    """封装 llama‑cpp 模型加载与 JSON 解析"""

    def __init__(self, model_path: str, n_gpu_layers: int, n_threads: int = 8):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,  # GPU 层数，可调整以适配显存
            n_threads=n_threads,
            verbose=False
        )

    def parse(self, text: str) -> dict:
        """调用模型，把自由英文描述转成结构化 JSON"""
        prompt = (
            "<|im_start|>system\n"
            "You are a converter that ONLY outputs a SINGLE valid JSON object (no markdown, no arrays). "
            "Analyze the text about two cubes and determine their relationship and colors. "
            "Output EXACTLY ONE JSON object in this format:\n"
            "For stacked cubes: "
            '{"relationship":"stacked","top":{"color":"[color]"},"bottom":{"color":"[color]"}}\n'
            "For separated cubes: "
            '{"relationship":"separated","left":{"color":"[color]"},"right":{"color":"[color]"}}\n'
            "Valid colors: red, blue, green, yellow, cyan, magenta, black, white\n"
            "Do NOT output arrays or multiple objects. Output only ONE object."
            "\n<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant"
        )

        out = self.llm(prompt, max_tokens=128, stop=["<|im_end|>"])["choices"][0]["text"].strip()
        # 清理可能带的代码块或多余字符
        out = out.replace("```json", "").replace("```", "").strip()

        try:
            js = json.loads(out)

            # 如果LLM返回了数组，取第一个元素
            if isinstance(js, list):
                print(f"警告: LLM返回了数组，取第一个元素: {js}")
                if len(js) > 0:
                    js = js[0]
                else:
                    raise ValueError("LLM返回了空数组")

            # 确保返回的是字典
            if not isinstance(js, dict):
                raise ValueError(f"LLM返回的不是有效的对象: {type(js)}")

            # 对LLM输出的颜色也进行标准化处理
            if "top" in js and isinstance(js["top"], dict) and "color" in js["top"]:
                js["top"]["color"] = normalize_color(js["top"]["color"])
            if "bottom" in js and isinstance(js["bottom"], dict) and "color" in js["bottom"]:
                js["bottom"]["color"] = normalize_color(js["bottom"]["color"])
            if "left" in js and isinstance(js["left"], dict) and "color" in js["left"]:
                js["left"]["color"] = normalize_color(js["left"]["color"])
            if "right" in js and isinstance(js["right"], dict) and "color" in js["right"]:
                js["right"]["color"] = normalize_color(js["right"]["color"])

            return js

        except json.JSONDecodeError:
            raise ValueError(f"LLM 输出无法解析为 JSON:\n{out}")


# -------- 主流程 -------- #
def main():
    ap = argparse.ArgumentParser(description="v2正则逻辑+LLM 解析两方块关系，输出结构化 jsonl")
    ap.add_argument("--input", required=True, help="输入 jsonl（含 prediction 字段）")
    ap.add_argument("--output", required=True, help="输出 jsonl（仅结构化结果）")
    ap.add_argument("--model_path", default="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    help="GGUF 模型路径")
    ap.add_argument("--n_gpu_layers", type=int, default=30,
                    help="加载到 GPU 的层数，8GB 显存推荐 25~32")
    args = ap.parse_args()

    # 初始化 LLM
    print("正在初始化 LLM 模型...")
    llm_parser = LLMParser(args.model_path, args.n_gpu_layers)
    print("LLM 模型初始化完成")

    processed, llm_calls, regex_success = 0, 0, 0

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prediction_txt = raw.get("prediction", "")

            # 首先尝试正则解析（使用v2完整逻辑）
            parsed = parse_prediction_regex(prediction_txt)

            # 检查正则解析是否成功
            if parsed and is_structured(parsed):
                # 正则解析成功
                regex_success += 1
            else:
                # 正则失败，调用LLM
                llm_calls += 1
                try:
                    parsed = llm_parser.parse(prediction_txt)
                    print(f"LLM解析: {prediction_txt[:50]}... -> {parsed}")
                except Exception as e:
                    # 若 LLM 仍失败，记录错误信息但保持原文本
                    parsed = {"error": str(e), "original": prediction_txt}
                    print(f"LLM解析失败: {e}")

            # 输出结果
            fout.write(json.dumps({
                "image": raw.get("image"),
                "parsed": parsed
            }, ensure_ascii=False) + "\n")

            processed += 1
            if processed % 100 == 0:
                print(f"已处理 {processed} 条... (正则成功: {regex_success}, LLM调用: {llm_calls})")

    print(f"完成！")
    print(f"总计处理: {processed} 条")
    print(f"正则成功: {regex_success} 条 ({regex_success / processed * 100:.1f}%)")
    print(f"LLM调用: {llm_calls} 次 ({llm_calls / processed * 100:.1f}%)")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()

# 使用示例：
# python cube_parser_v3_fixed.py --input ./predictions_2cube/predictions_img_init_2obj_v1.jsonl --output cube_parser_v3_fixed.jsonl --model_path ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf --n_gpu_layers 30