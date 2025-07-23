# smolvlm_finetuned_batch_infer_local_tta.py
"""
SmolVLM 微调模型批量图片推理脚本 - 使用本地权重版本 + TTA增强
该脚本只进行方块数量推理，使用本地存储的模型权重，支持Test-Time Augmentation
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
from collections import Counter

# 本地模型路径配置
LOCAL_MODEL_PATH = "./models--HuggingFaceTB--SmolVLM-Instruct/snapshots/81cd9a775a4d644f2faf4e7becff4559b46b14c7"


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


def apply_tta_transforms(image):
    """
    应用TTA变换：原图 + 水平翻转 + 垂直翻转

    Args:
        image: PIL Image对象

    Returns:
        dict: 包含三种变换的字典
    """
    transforms = {
        'original': image,
        'hflip': image.transpose(Image.FLIP_LEFT_RIGHT),
        'vflip': image.transpose(Image.FLIP_TOP_BOTTOM)
    }
    return transforms


def smart_majority_vote(predictions):
    """
    智能多数投票，处理平票情况

    Args:
        predictions: 预测结果列表

    Returns:
        最终投票结果
    """
    counter = Counter(predictions)
    most_common = counter.most_common()

    # 有明确多数
    if most_common[0][1] > 1:
        return most_common[0][0]

    # 完全平票情况（三方平票）- 取中位数
    if len(set(predictions)) == len(predictions):
        median_result = sorted(predictions)[len(predictions) // 2]
        print(f"Warning: 完全平票 {predictions}, 使用中位数策略: {median_result}")
        return median_result

    # 其他情况（部分平票）
    return most_common[0][0]


def load_model(adapter_path=None):
    """加载SmolVLM模型和处理器 - 使用本地权重"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"正在加载本地模型: {LOCAL_MODEL_PATH}")

    # 检查本地模型路径是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"本地模型路径不存在: {LOCAL_MODEL_PATH}")

    # 从本地路径加载处理器和模型
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # 使用eager attention避免flash_attn依赖
    )

    # 如果提供了adapter路径，加载微调权重
    if adapter_path:
        print(f"正在加载LoRA适配器: {adapter_path}")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"LoRA适配器路径不存在: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("LoRA适配器加载完成！")

    model = model.to(device)

    print("模型加载完成！")
    return model, processor, device


def run_single_inference(model, processor, device, image, prompt, max_new_tokens, enable_hsv_preprocessing=False):
    """
    对单张PIL图片进行推理 - 内部函数，用于TTA

    Args:
        image: PIL Image对象
        其他参数同原函数

    Returns:
        推理结果字符串
    """
    # HSV预处理（可选）
    if enable_hsv_preprocessing:
        image = optimize_for_green_attention(image)

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


def run_inference_with_tta(model, processor, device, image_path, prompt, max_new_tokens,
                           enable_hsv_preprocessing=False, enable_tta=False):
    """
    TTA增强版推理函数

    Args:
        enable_tta: 是否启用TTA
        其他参数同原函数

    Returns:
        tuple: (final_result, detailed_info) 如果enable_tta=True
        str: prediction 如果enable_tta=False
    """
    # 检查图片是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"图片加载失败: {e}")

    # 如果不启用TTA，使用原始逻辑
    if not enable_tta:
        return run_single_inference(model, processor, device, image, prompt,
                                    max_new_tokens, enable_hsv_preprocessing)

    # TTA逻辑
    transforms = apply_tta_transforms(image)
    predictions = []
    raw_predictions = []

    for transform_name, transformed_image in transforms.items():
        try:
            raw_pred = run_single_inference(model, processor, device, transformed_image,
                                            prompt, max_new_tokens, enable_hsv_preprocessing)
            raw_predictions.append(f"{transform_name}: {raw_pred}")

            # 尝试提取数字
            pred_clean = raw_pred.strip()
            # 尝试解析数字（处理可能的非数字输出）
            try:
                # 提取字符串中的第一个数字
                import re
                numbers = re.findall(r'\d+', pred_clean)
                if numbers:
                    pred_num = int(numbers[0])
                    predictions.append(pred_num)
                else:
                    print(f"Warning: 无法从 '{pred_clean}' 中提取数字 (变换: {transform_name})")
            except:
                print(f"Warning: 无法解析预测结果: '{raw_pred}' (变换: {transform_name})")

        except Exception as e:
            print(f"Warning: 变换 {transform_name} 推理失败: {e}")
            continue

    if not predictions:
        return "解析失败", raw_predictions

    # 智能投票
    final_result = smart_majority_vote(predictions)

    # 详细信息
    detailed_info = {
        "final_result": final_result,
        "individual_predictions": predictions,
        "raw_outputs": raw_predictions,
        "vote_distribution": dict(Counter(predictions))
    }

    return final_result, detailed_info


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
                    enable_hsv_preprocessing=False, enable_tta=False):
    """批量推理主函数"""
    #########################################################################################################
    # Part1，目标数量识别

    # prompt = "How many objects are there in the image? Only answer with a number."
    prompt = "Think step by step, but only answer with a number: How many objects are there in the image?"

    ##########################################################################################################
    print(f"使用的prompt: {prompt}")
    print(f"TTA增强: {'已启用' if enable_tta else '未启用'}")
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
    tta_stats = {"tie_cases": 0, "successful_predictions": 0, "failed_predictions": 0}

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
                if enable_tta:
                    item["tta_details"] = {}
                output_data.append(item)
                tta_stats["failed_predictions"] += 1
                continue

            # 执行推理 - TTA或普通推理
            if enable_tta:
                prediction, tta_details = run_inference_with_tta(
                    model, processor, device, image_path, prompt, max_new_tokens,
                    enable_hsv_preprocessing, enable_tta=True
                )

                # 添加预测结果和TTA详细信息
                item["prediction"] = prediction
                item["tta_details"] = tta_details

                # 统计
                if len(set(tta_details["individual_predictions"])) == len(tta_details["individual_predictions"]):
                    tta_stats["tie_cases"] += 1
                tta_stats["successful_predictions"] += 1

            else:
                prediction = run_inference_with_tta(
                    model, processor, device, image_path, prompt, max_new_tokens,
                    enable_hsv_preprocessing, enable_tta=False
                )
                item["prediction"] = prediction
                tta_stats["successful_predictions"] += 1

            output_data.append(item)

        except Exception as e:
            print(f"处理失败 {item.get('image', 'unknown')}: {e}")
            # 添加空预测结果，确保输出完整性
            item["prediction"] = ""
            if enable_tta:
                item["tta_details"] = {}
            output_data.append(item)
            tta_stats["failed_predictions"] += 1
            continue

    # 保存结果
    print(f"正在保存结果到: {output_file}")
    save_jsonl(output_data, output_file)

    # 打印统计信息
    print(f"批量推理完成！处理了 {len(output_data)} 条数据")
    if enable_tta:
        print(f"TTA统计: 成功={tta_stats['successful_predictions']}, "
              f"失败={tta_stats['failed_predictions']}, "
              f"平票={tta_stats['tie_cases']}")


def main():
    parser = argparse.ArgumentParser(description="SmolVLM 微调模型批量图片推理脚本 - 使用本地权重 + TTA增强")
    parser.add_argument('--input', type=str, required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出JSONL文件路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图片文件夹路径')

    parser.add_argument('--adapter_path', type=str,
                        help='LoRA适配器路径 (如果不提供，则使用原始模型)')

    parser.add_argument('--max_new_tokens', type=int, default=50, help='生成最大 token 数')
    parser.add_argument('--enable_hsv_preprocessing', action='store_true',
                        help='启用HSV预处理，提升绿色区域显著性')

    # 新增TTA参数
    parser.add_argument('--enable_tta', action='store_true',
                        help='启用Test-Time Augmentation (原图+水平翻转+垂直翻转+多数投票)')

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

        # 加载模型（一次性加载） - 使用本地权重
        model, processor, device = load_model(args.adapter_path)

        # 执行批量推理
        print(f"\n开始批量推理...")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"图片目录: {args.image_dir}")
        print(f"本地模型路径: {LOCAL_MODEL_PATH}")
        if args.adapter_path:
            print(f"LoRA适配器: {args.adapter_path}")
        else:
            print("LoRA适配器: 未使用 (使用原始模型)")
        print(f"最大生成tokens: {args.max_new_tokens}")
        print(f"格式输出优化: 已启用 (do_sample=False, repetition_penalty=1.2, num_beams=2)")

        if args.enable_tta:
            print("TTA模式: 原图 + 水平翻转 + 垂直翻转 + 智能多数投票")
            print("预期推理时间: 约为普通模式的3倍")

        print("-" * 50)

        batch_inference(
            model, processor, device,
            args.input, args.output, args.image_dir,
            args.max_new_tokens, args.enable_hsv_preprocessing, args.enable_tta
        )

    except Exception as e:
        print(f"批量推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 使用示例（新增TTA参数）：

# 不使用TTA（与原版本兼容）
# python smolvlm_finetuned_batch_infer_local_tta.py \
#     --input ./annotations/annotations_img_test.jsonl \
#     --output ./predictions_normal/predictions_img_test_normal.jsonl \
#     --image_dir ./image_test_batch/image_test \
#     --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1 \
#     --enable_hsv_preprocessing

# 只使用TTA，不使用HSV预处理和微调
# python smolvlm_finetuned_batch_infer_local_tta.py \
#     --input ./annotations/annotations_img_test.jsonl \
#     --output ./predictions_tta_only/predictions_img_test_tta_only.jsonl \
#     --image_dir ./image_test_batch/image_test \
#     --enable_tta

# 测试，使用TTA + 微调模型 + HSV预处理
"""
python smolvlm_batch_infer_hsv_qlora_num_tta.py --input ./annotations/annotations_img_test.jsonl --output ./predictions_tta/test/predictions_img_test_tta_v1.jsonl --image_dir ./image_test_batch/image_test --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1 --enable_hsv_preprocessing --enable_tta
"""

# 使用TTA + 微调模型 + HSV预处理
"""
python smolvlm_batch_infer_hsv_qlora.py --input ./annotations/annotations_img_2obj.jsonl --output ./predictions_2cube/num_obj/predictions_img_2obj_v4.jsonl --image_dir ./image_test_batch/image_2obj  --enable_hsv_preprocessing --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1

python smolvlm_batch_infer_hsv_qlora_num_tta.py --input ./annotations/annotations_img_2obj.jsonl --output ./predictions_tta/num_obj/predictions_img_2obj_v1.jsonl --image_dir ./image_test_batch/image_2obj --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1 --enable_hsv_preprocessing --enable_tta
"""
