# smolvlm_finetuned_batch_infer.py
"""
SmolVLM 微调模型批量图片推理脚本 - 优化版本，添加格式输出最优参数和HSV预处理

"""

import argparse
import json
import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel


def optimize_for_green_attention(image):
    """
    HSV混合处理策略：提升绿色区域的视觉显著性

    Args:
        image: PIL Image对象 (RGB格式)

    Returns:
        processed_image: PIL Image对象 (处理后的RGB格式)
    """
    # 1. PIL Image转换为numpy数组，然后转换为OpenCV格式
    image_array = np.array(image)

    # 2. RGB转HSV
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # 3. 先进行全图轻度对比度增强（温和参数）
    clahe_mild = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe_mild.apply(hsv[:, :, 2])

    # 4. 识别绿色/青绿色区域
    # 针对RGB(32,117,107)转换的HSV(175°, 73%, 46%)进行优化
    # 色调范围：160-185° (覆盖青绿色区域)
    # 饱和度范围：50-255 (确保包含中等饱和度)
    # 明度范围：30-255 (确保包含较暗的绿色)
    green_mask = cv2.inRange(hsv, (160, 50, 30), (185, 255, 255))

    # 5. 对绿色区域进行额外的饱和度和明度增强
    # 温和的增强避免过度处理
    # 处理前：较暗、较灰的青绿色
    # 处理后：更鲜艳、更亮的青绿色
    hsv[green_mask, 1] = np.clip(hsv[green_mask, 1] * 1.2, 0, 255)  # 饱和度+20%
    hsv[green_mask, 2] = np.clip(hsv[green_mask, 2] * 1.15, 0, 255)  # 明度+15%

    # 6. 转回RGB
    processed_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 7. 转换回PIL Image
    processed_image = Image.fromarray(processed_array)

    return processed_image


def load_model(model_name, adapter_path=None):
    """加载SmolVLM模型和处理器 - 支持微调模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"正在加载基础模型: {model_name}")

    # 初始化处理器和模型
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # 使用eager attention避免flash_attn依赖
    )

    # 如果提供了adapter路径，加载微调权重
    if adapter_path:
        print(f"正在加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("LoRA适配器加载完成！")

    model = model.to(device)

    print("模型加载完成！")
    return model, processor, device


def run_inference(model, processor, device, image_path, prompt, max_new_tokens, enable_hsv_preprocessing=False):
    """对单张图片进行推理 - 添加格式输出最优参数和HSV预处理选项"""

    # 检查图片是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB")

        # HSV预处理（可选）
        if enable_hsv_preprocessing:
            image = optimize_for_green_attention(image)

        # 注释掉打印信息以避免批量处理时输出过多
        # print(f"成功加载图片: {image_path} (尺寸: {image.size})")
    except Exception as e:
        raise ValueError(f"图片加载失败: {e}")

    # 创建消息（按照官方格式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # 预处理（官方步骤）
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    # 将输入移到正确的设备上
    inputs = inputs.to(device)

    # 格式输出最优参数 - 针对transformers 4.52.3优化
    optimal_params = {
        "do_sample": False,  # 确定性生成，不使用采样
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.2,  # 降低重复惩罚，避免过度约束
        "length_penalty": 1.2,  # 长度惩罚
        "num_beams": 2,  # 使用贪婪解码
        "pad_token_id": processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id
    }

    # 生成（使用优化参数）
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **optimal_params)

    # # 使用默认参数生成（只设置max_new_tokens）
    # with torch.no_grad():
    #     generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 解码输出
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # 清理GPU显存
    if device == "cuda":
        torch.cuda.empty_cache()

    # 提取只包含Assistant回答的部分
    full_output = generated_texts[0]

    # 查找Assistant:后的内容
    if "Assistant:" in full_output:
        assistant_response = full_output.split("Assistant:")[-1].strip()
        return assistant_response
    else:
        # 如果没有找到Assistant:标记，返回原始输出
        return full_output


def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def batch_inference(model, processor, device, input_file, output_file, image_dir, max_new_tokens=50,
                    enable_hsv_preprocessing=False):
    """批量推理主函数"""

    # Part1，目标数量识别

    # prompt = "How many objects are there in the image? Only answer with a number."

    ###################################################################################################
    # Part2，相对位置关系识别，堆叠或是分开

    # prompt = """Describe the position relationship between the objects in this image: are they stacked or separated? Only answer with a word.
    # """

    ####################################################################################################
    # Part3，详细的目标识别，如果是堆叠，谁上谁下？如果是分开，各自是什么？

    # prompt = """Look at the TWO objects in this image, describe what you see in this image.
    #
    # Format: I found two objects, a [actual_color] cube and a [actual_color] cube. The [actual_color] cube is on top of the [actual_color] cube.
    #
    # Attention: Look carefully at BOTH objects! Pay special attention to the green cube, and make sure to identify clearly if the green cube is on top or at the bottom.
    #
    # Answer:"""


    prompt = """
    Look at the TWO objects in this image, and carefully describe what you see.

    If the two cubes are stacked, use this format:
    I found two objects, a [color1] cube and a [color2] cube. The [color1] cube is on top of the [color2] cube.

    If the two cubes are not stacked and are placed separately on the table, use this format:
    I found two objects, a [color1] cube and a [color2] cube. The cubes are placed separately on the table.

    Attention: Look carefully at BOTH objects! Pay special attention to the green cube, and make sure to clearly state its position (on top, on bottom, or placed separately).
    Answer:
    """

    # prompt = """
    # Look at the TWO objects in this image, and carefully describe what you see.
    #
    # If the two cubes are stacked, use this format:
    # I found two objects, a [color1] cube and a [color2] cube. The [color1] cube is on top of the [color2] cube.
    #
    # Attention: Look carefully at BOTH objects! Pay special attention to the green cube, and make sure to clearly state its position (on top, on bottom, or placed separately).
    # Answer:
    # """


    # prompt = "This image shows the workspace before a robot arm performs a grasping task. There are exactly two objects in the workspace. Please describe the color of each object and their spatial relationship.(for example, whether they are stacked or separated, and which one is on top if they are stacked)."
    # prompt = "This image shows the workspace before a robot arm performs a grasping task. There are exactly two objects! Please describe the color of each object and their spatial relationship.(for example, whether they are stacked or separated, and which one is on top if they are stacked)."


    #
    # prompt = """Look at the object in this image.
    #
    # Describe its color in this exact format: I found one object, it's a [color] cube.
    #
    # Attention: Look carefully at the cube!!!
    #
    # Your answer:"""

    #############################################################################################
    # prompt = """
    # This image shows the scene before a robot grasping task. Please analyze the cubes (blocks) in the image and answer the following, following the required format:
    #
    # 1. How many cubes are there in the image?
    # 2. What are the colors of the cubes?
    # 3. What is the spatial arrangement of the cubes? For example:
    #     - If there is one cube, it must be placed on the table.
    #     - If there are two cubes, they may be placed separately on the table or stacked on top of each other.
    #     - If there are three cubes, they may all be placed separately on the table, two cubes stacked and one placed separately, or all three cubes stacked on top of each other.
    #
    # Format your answer like this:
    # There are [number] cubes in the image: a [color1] cube, a [color2] cube (and a [color3] cube, ...).
    # The spatial arrangement: [Describe clearly, e.g., "the green cube is on top of the blue cube," "the three cubes are stacked: red on top of green, green on top of blue," "the red and green cubes are stacked, and the blue cube is placed separately," or "all three cubes are placed separately on the table"].
    #
    # Attention: Please be precise in describing both the color and spatial relationship of each cube.
    # """

    ##########################################################################################################
    print(f"使用的prompt: {prompt}")
    if enable_hsv_preprocessing:
        print("HSV预处理: 已启用 (全图对比度增强 + 绿色区域饱和度/明度增强)")
    else:
        print("HSV预处理: 未启用")

    # 加载输入数据
    print(f"正在加载输入文件: {input_file}")
    input_data = load_jsonl(input_file)
    print(f"共找到 {len(input_data)} 条数据")

    # 准备输出数据
    output_data = []

    # 使用tqdm显示进度
    for item in tqdm(input_data, desc="批量推理中"):
        try:
            # 获取图片路径
            image_filename = item["image"]

            # 构建完整图片路径
            image_path = os.path.join(image_dir, image_filename)

            # 检查图片是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图片文件不存在 {image_path}")
                # 添加空预测结果
                item["prediction"] = ""
                output_data.append(item)
                continue

            # 执行推理 - 使用代码内定义的prompt和优化参数
            prediction = run_inference(
                model, processor, device, image_path, prompt, max_new_tokens, enable_hsv_preprocessing
            )

            # 添加预测结果
            item["prediction"] = prediction
            output_data.append(item)

        except Exception as e:
            print(f"处理失败 {item.get('image', 'unknown')}: {e}")
            # 添加空预测结果，确保输出完整性
            item["prediction"] = ""
            output_data.append(item)
            continue

    # 保存结果
    print(f"正在保存结果到: {output_file}")
    save_jsonl(output_data, output_file)
    print(f"批量推理完成！处理了 {len(output_data)} 条数据")


def main():
    parser = argparse.ArgumentParser(description="SmolVLM 微调模型批量图片推理脚本")
    parser.add_argument('--input', type=str, required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出JSONL文件路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图片文件夹路径')
    parser.add_argument('--model', type=str,
                        default="HuggingFaceTB/SmolVLM-Instruct",
                        help='基础模型名称 (可选: HuggingFaceTB/SmolVLM-256M-Instruct 或 HuggingFaceTB/SmolVLM-500M-Instruct 或 HuggingFaceTB/SmolVLM-Instruct)')

    parser.add_argument('--adapter_path', type=str,
                        help='LoRA适配器路径 (如果不提供，则使用原始模型)')

    parser.add_argument('--max_new_tokens', type=int, default=50, help='生成最大 token 数')
    parser.add_argument('--enable_hsv_preprocessing', action='store_true',
                        help='启用HSV预处理，提升绿色区域显著性')

    args = parser.parse_args()

    try:
        # 检查OpenCV依赖（只有启用HSV预处理时才需要）
        if args.enable_hsv_preprocessing:
            try:
                import cv2
            except ImportError:
                raise ImportError("启用HSV预处理需要安装OpenCV: pip install opencv-python")

        # 检查输入文件和图片目录
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"输入文件不存在: {args.input}")

        if not os.path.exists(args.image_dir):
            raise FileNotFoundError(f"图片目录不存在: {args.image_dir}")

        # 检查adapter路径（如果提供）
        if args.adapter_path and not os.path.exists(args.adapter_path):
            raise FileNotFoundError(f"LoRA适配器路径不存在: {args.adapter_path}")

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载模型（一次性加载） - 支持微调模型
        model, processor, device = load_model(args.model, args.adapter_path)

        # 执行批量推理
        print(f"\n开始批量推理...")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"图片目录: {args.image_dir}")
        print(f"基础模型: {args.model}")
        if args.adapter_path:
            print(f"LoRA适配器: {args.adapter_path}")
        else:
            print("LoRA适配器: 未使用 (使用原始模型)")
        print(f"最大生成tokens: {args.max_new_tokens}")
        print(f"格式输出优化: 已启用 (do_sample=False, repetition_penalty=1.1, num_beams=2)")

        print("-" * 50)

        batch_inference(
            model, processor, device,
            args.input, args.output, args.image_dir,
            args.max_new_tokens, args.enable_hsv_preprocessing
        )

    except Exception as e:
        print(f"批量推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 使用示例：

# 使用原始模型（不使用微调权重）
# python smolvlm_batch_infer_qlora.py --input ./annotations/annotations_img_test.jsonl --output ./predictions_v2/predictions_img_test_original.jsonl --image_dir ./image_test_batch/image_test

# 使用微调模型
# python smolvlm_batch_infer_qlora.py --input ./annotations/annotations_img_test.jsonl --output ./predictions_v2/predictions_img_test_finetuned.jsonl --image_dir ./image_test_batch/image_test --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1

# 测试
# python smolvlm_batch_infer_hsv_qlora.py --input ./annotations/annotations_img_test_2obj.jsonl --output ./predictions_2cube/predictions_img_test_2obj_hsv_v5.jsonl --image_dir ./image_test_batch/image_test_2obj  --enable_hsv_preprocessing --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1

# img_init 启用HSV预处理模式 + 微调模型
# python smolvlm_batch_infer_hsv_qlora.py --input ./annotations/annotations_img_init_3obj.jsonl --output ./predictions_3cube/predictions_img_init_3obj_v1.jsonl --image_dir ./image_test_batch/image_init_3obj  --enable_hsv_preprocessing --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1







# 测试用
# python smolvlm_batch_infer_hsv_qlora.py --input ./annotations/annotations_img_test_2obj.jsonl --output ./predictions_2cube/predictions_img_test_2obj_v1.jsonl --image_dir ./image_test_batch/image_test_2obj --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1