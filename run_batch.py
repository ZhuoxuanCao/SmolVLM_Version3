import subprocess
import sys
import os


def run_command(cmd, description):
    """运行命令并检查返回状态"""
    print(f"\n🚀 {description}")
    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误代码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return False


def main():
    # 命令1：2obj推理
    cmd1 = [
        "python", "smolvlm_batch_infer_hsv_qlora.py",
        "--input", "./annotations/annotations_img_init_2obj.jsonl",
        "--output", "./predictions_2cube/num_obj/predictions_img_init_2obj_v2.jsonl",
        "--image_dir", "./image_test_batch/image_init_2obj",
        "--enable_hsv_preprocessing",
        "--adapter_path", "./output_smolvlm_lora/output_smolvlm_lora_V1"
    ]

    # 命令2：1obj推理
    cmd2 = [
        "python", "smolvlm_batch_infer_hsv_qlora.py",
        "--input", "./annotations/annotations_img_init_1obj.jsonl",
        "--output", "./predictions_1cube/num_obj/predictions_img_init_1obj_v2.jsonl",
        "--image_dir", "./image_test_batch/image_init_1obj",
        "--enable_hsv_preprocessing",
        "--adapter_path", "./output_smolvlm_lora/output_smolvlm_lora_V1"
    ]

    print("🎯 开始批量推理任务")

    # 执行第一个命令
    if not run_command(cmd1, "2obj推理任务"):
        print("❌ 第一个任务失败，停止执行")
        sys.exit(1)

    # 执行第二个命令
    if not run_command(cmd2, "1obj推理任务"):
        print("❌ 第二个任务失败")
        sys.exit(1)

    print("\n🎉 所有任务完成！")


if __name__ == "__main__":
    main()
