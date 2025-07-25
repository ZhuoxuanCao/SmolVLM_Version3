# smolvlm_single_infer.py
"""
SmolVLM 单张图片推理脚本
用法示例：
    python smolvlm_single_infer.py --image ./image_train/img1 --prompt "描述这个图片"
    python smolvlm_single_infer.py --image ./test.jpg --prompt "这张图片中有什么？" --model HuggingFaceTB/SmolVLM-256M-Instruct
"""

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import os


def load_model(model_name):
    """加载SmolVLM模型和处理器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"正在加载模型: {model_name}")

    # 初始化处理器和模型
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # 使用eager attention避免flash_attn依赖
    )
    model = model.to(device)

    print("模型加载完成！")
    return model, processor, device


def run_inference(model, processor, device, image_path, prompt):
    """对单张图片进行推理"""

    # 检查图片是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"成功加载图片: {image_path} (尺寸: {image.size})")
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

    # 生成（按照官方示例）
    print("开始推理...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)

    # 解码输出
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


def main():
    parser = argparse.ArgumentParser(description="SmolVLM 单张图片推理脚本")
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--prompt', type=str, default="描述这个图片", help='输入提示词')
    parser.add_argument('--model', type=str,
                        default="HuggingFaceTB/SmolVLM-Instruct",
                        help='模型名称 (可选: HuggingFaceTB/SmolVLM-256M-Instruct 或 HuggingFaceTB/SmolVLM-500M-Instruct 或 HuggingFaceTB/SmolVLM-2B-Instruct)')

    args = parser.parse_args()

    try:
        # 加载模型
        model, processor, device = load_model(args.model)

        # 运行推理
        print(f"\n推理参数:")
        print(f"  图片: {args.image}")
        print(f"  提示词: {args.prompt}")
        print(f"  模型: {args.model}")
        print("-" * 50)

        result = run_inference(model, processor, device, args.image, args.prompt)

        print("\n==== 推理结果 ====")
        print(result)
        print("=" * 50)

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 使用示例:
# python model_infer_single.py --image ./image_test_single/BlueUp2.jpg --prompt "描述这个图片"
"""
2cube
python smolvlm_single_infer.py --image ./image_test_single/red_and_blue_0123.jpg --prompt "This image shows the workspace before a robot arm performs a grasping task. There are exactly two objects in the workspace. Please describe the color of each object and their spatial relationship (for example, whether they are stacked or separated, and which one is on top if they are stacked)." --model HuggingFaceTB/SmolVLM-Instruct
"""

"""
3cube
python smolvlm_single_infer.py --image ./image_test_single/pyramid_green_on_red_and_blue_0109.jpg --prompt "This image shows the workspace before a robot arm performs a grasping task. There are exactly three objects in the workspace. Please describe the color of each object and their spatial relationship (for example, whether they are stacked, arranged in a pyramid, or separated, and which one is on top or at the bottom if they are stacked)." --model HuggingFaceTB/SmolVLM-Instruct

python smolvlm_single_infer.py --image ./image_test_single/red_and_green_and_blue_0001.jpg --prompt "  " --model HuggingFaceTB/SmolVLM-Instruct

"""