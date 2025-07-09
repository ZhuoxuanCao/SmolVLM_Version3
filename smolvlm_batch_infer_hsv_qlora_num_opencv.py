# # opencv_contour_batch_infer.py
# """
# 不好用！！！！！！！
# OpenCV轮廓检测批量图片推理脚本 - 替代VLM进行方块数量检测
# 基于轮廓检测和HSV预处理，保持与原VLM脚本相同的输入输出格式
# """
#
# import argparse
# import json
# import os
# from tqdm import tqdm
# import numpy as np
# import cv2
# from PIL import Image
#
#
# def optimize_for_green_attention(image):
#     """
#     HSV混合处理策略：提升绿色区域的视觉显著性
#
#     Args:
#         image: PIL Image对象 (RGB格式)
#
#     Returns:
#         processed_image: PIL Image对象 (处理后的RGB格式)
#     """
#     # 1. PIL Image转换为numpy数组，然后转换为OpenCV格式
#     image_array = np.array(image)
#
#     # 2. RGB转HSV
#     hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
#
#     # 3. 先进行全图轻度对比度增强（温和参数）
#     clahe_mild = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
#     hsv[:, :, 2] = clahe_mild.apply(hsv[:, :, 2])
#
#     # 4. 识别绿色/青绿色区域
#     # 针对RGB(32,117,107)转换的HSV(175°, 73%, 46%)进行优化
#     # 色调范围：160-185° (覆盖青绿色区域)
#     # 饱和度范围：50-255 (确保包含中等饱和度)
#     # 明度范围：30-255 (确保包含较暗的绿色)
#     green_mask = cv2.inRange(hsv, (160, 50, 30), (185, 255, 255))
#
#     # 5. 对绿色区域进行额外的饱和度和明度增强
#     # 温和的增强避免过度处理
#     hsv[green_mask > 0, 1] = np.clip(hsv[green_mask > 0, 1] * 1.2, 0, 255)  # 饱和度+20%
#     hsv[green_mask > 0, 2] = np.clip(hsv[green_mask > 0, 2] * 1.15, 0, 255)  # 明度+15%
#
#     # 6. 转回RGB
#     processed_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#
#     # 7. 转换回PIL Image
#     processed_image = Image.fromarray(processed_array)
#
#     return processed_image
#
#
# def preprocess_image_for_contours(image, enable_hsv_preprocessing=False, contour_params=None):
#     """
#     为轮廓检测预处理图像
#
#     Args:
#         image: PIL Image对象
#         enable_hsv_preprocessing: 是否启用HSV预处理
#         contour_params: 预处理参数
#
#     Returns:
#         binary: 二值化图像（用于轮廓检测）
#         processed_bgr: 处理后的BGR图像（用于结果显示）
#     """
#     if contour_params is None:
#         contour_params = {
#             'blur_kernel': 5,
#             'adaptive_block_size': 11,
#             'adaptive_c': 2,
#             'morph_kernel_size': 3
#         }
#
#     # 可选的HSV预处理
#     if enable_hsv_preprocessing:
#         image = optimize_for_green_attention(image)
#
#     # 转换为OpenCV格式（BGR）
#     image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
#     # 转换为灰度图
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#
#     # 高斯模糊去噪
#     blur_kernel = contour_params['blur_kernel']
#     blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
#
#     # 尝试多种二值化方法
#     # 方法1：自适应阈值
#     binary1 = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, contour_params['adaptive_block_size'], contour_params['adaptive_c']
#     )
#
#     # 方法2：自适应阈值（反转）
#     binary2 = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, contour_params['adaptive_block_size'], contour_params['adaptive_c']
#     )
#
#     # 方法3：Otsu阈值
#     _, binary3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 方法4：Otsu阈值（反转）
#     _, binary4 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     # 选择产生最多轮廓的二值化方法
#     methods = [binary1, binary2, binary3, binary4]
#     method_names = ['adaptive', 'adaptive_inv', 'otsu', 'otsu_inv']
#     best_binary = binary1
#     max_contours = 0
#
#     for binary, name in zip(methods, method_names):
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) > max_contours:
#             max_contours = len(contours)
#             best_binary = binary
#
#     # 形态学操作去除噪声
#     kernel_size = contour_params['morph_kernel_size']
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     best_binary = cv2.morphologyEx(best_binary, cv2.MORPH_CLOSE, kernel)
#     best_binary = cv2.morphologyEx(best_binary, cv2.MORPH_OPEN, kernel)
#
#     return best_binary, image_bgr
#
#
# def count_objects_by_contours(image_path, enable_hsv_preprocessing=False, debug_mode=False, contour_params=None):
#     """
#     使用轮廓检测计算图像中的物体数量
#
#     Args:
#         image_path: 图像文件路径
#         enable_hsv_preprocessing: 是否启用HSV预处理
#         debug_mode: 是否启用调试模式（保存中间结果）
#         contour_params: 轮廓检测参数字典
#
#     Returns:
#         count: 检测到的物体数量
#     """
#     # 默认参数（更宽松的设置）
#     if contour_params is None:
#         contour_params = {
#             'min_area_ratio': 0.0001,  # 最小面积比例（更小）
#             'max_area_ratio': 0.5,  # 最大面积比例（更大）
#             'min_circularity': 0.1,  # 最小圆形度（更宽松）
#             'max_circularity': 2.0,  # 最大圆形度（更宽松）
#             'max_aspect_ratio': 5.0,  # 最大长宽比（更宽松）
#             'blur_kernel': 5,  # 高斯模糊核大小
#             'adaptive_block_size': 11,  # 自适应阈值块大小
#             'adaptive_c': 2,  # 自适应阈值常数
#             'morph_kernel_size': 3  # 形态学操作核大小
#         }
#
#     try:
#         # 加载图像
#         image = Image.open(image_path).convert("RGB")
#
#         # 预处理
#         binary, image_bgr = preprocess_image_for_contours(image, enable_hsv_preprocessing, contour_params)
#
#         # 查找轮廓
#         contours, hierarchy = cv2.findContours(
#             binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#
#         # 调试信息
#         if debug_mode:
#             print(f"\n=== 调试信息 {os.path.basename(image_path)} ===")
#             print(f"图像尺寸: {binary.shape}")
#             print(f"总轮廓数: {len(contours)}")
#
#         # 过滤轮廓 - 基于面积和形状特征
#         valid_contours = []
#         filtered_reasons = []
#         image_area = binary.shape[0] * binary.shape[1]
#
#         for i, contour in enumerate(contours):
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#
#             # 面积过滤
#             min_area = image_area * contour_params['min_area_ratio']
#             max_area = image_area * contour_params['max_area_ratio']
#
#             if area < min_area:
#                 if debug_mode:
#                     filtered_reasons.append(f"轮廓{i}: 面积太小 ({area:.1f} < {min_area:.1f})")
#                 continue
#
#             if area > max_area:
#                 if debug_mode:
#                     filtered_reasons.append(f"轮廓{i}: 面积太大 ({area:.1f} > {max_area:.1f})")
#                 continue
#
#             # 形状过滤：计算圆形度
#             circularity = 0
#             if perimeter > 0:
#                 circularity = 4 * np.pi * area / (perimeter * perimeter)
#                 if circularity < contour_params['min_circularity']:
#                     if debug_mode:
#                         filtered_reasons.append(
#                             f"轮廓{i}: 圆形度太小 ({circularity:.3f} < {contour_params['min_circularity']})")
#                     continue
#                 if circularity > contour_params['max_circularity']:
#                     if debug_mode:
#                         filtered_reasons.append(
#                             f"轮廓{i}: 圆形度太大 ({circularity:.3f} > {contour_params['max_circularity']})")
#                     continue
#
#             # 长宽比过滤
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
#             if aspect_ratio > contour_params['max_aspect_ratio']:
#                 if debug_mode:
#                     filtered_reasons.append(
#                         f"轮廓{i}: 长宽比太大 ({aspect_ratio:.2f} > {contour_params['max_aspect_ratio']})")
#                 continue
#
#             # 通过所有过滤条件
#             valid_contours.append(contour)
#             if debug_mode:
#                 print(f"✓ 轮廓{i}: 面积={area:.1f}, 圆形度={circularity:.3f}, 长宽比={aspect_ratio:.2f}")
#
#         # 调试模式：保存详细信息和图像
#         if debug_mode:
#             print(f"有效轮廓数: {len(valid_contours)}")
#             print(f"过滤原因:")
#             for reason in filtered_reasons[:10]:  # 只显示前10个
#                 print(f"  {reason}")
#             if len(filtered_reasons) > 10:
#                 print(f"  ... 还有 {len(filtered_reasons) - 10} 个被过滤")
#
#             # 保存多个调试图像
#             base_name = os.path.splitext(image_path)[0]
#
#             # 1. 保存二值化图像
#             cv2.imwrite(f"{base_name}_debug_binary.png", binary)
#
#             # 2. 保存所有轮廓
#             all_contours_img = image_bgr.copy()
#             cv2.drawContours(all_contours_img, contours, -1, (0, 0, 255), 1)  # 红色所有轮廓
#             cv2.imwrite(f"{base_name}_debug_all_contours.png", all_contours_img)
#
#             # 3. 保存有效轮廓
#             valid_contours_img = image_bgr.copy()
#             cv2.drawContours(valid_contours_img, valid_contours, -1, (0, 255, 0), 2)  # 绿色有效轮廓
#
#             # 为每个有效轮廓添加编号和信息
#             for i, contour in enumerate(valid_contours):
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])
#                     area = cv2.contourArea(contour)
#                     cv2.putText(valid_contours_img, f"{i + 1}", (cx - 10, cy + 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#                     cv2.putText(valid_contours_img, f"A:{area:.0f}", (cx - 20, cy + 25),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
#
#             cv2.imwrite(f"{base_name}_debug_valid_contours.png", valid_contours_img)
#
#         return len(valid_contours)
#
#     except Exception as e:
#         print(f"处理图像 {image_path} 时出错: {e}")
#         return 0
#
#
# def load_jsonl(file_path):
#     """加载JSONL文件"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 data.append(json.loads(line))
#     return data
#
#
# def save_jsonl(data, file_path):
#     """保存数据到JSONL文件"""
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')
#
#
# def batch_inference(input_file, output_file, image_dir, enable_hsv_preprocessing=False, debug_mode=False,
#                     contour_params=None):
#     """
#     批量轮廓检测主函数
#
#     Args:
#         input_file: 输入JSONL文件路径
#         output_file: 输出JSONL文件路径
#         image_dir: 图片目录路径
#         enable_hsv_preprocessing: 是否启用HSV预处理
#         debug_mode: 是否启用调试模式
#         contour_params: 轮廓检测参数
#     """
#     print("=== OpenCV轮廓检测批量物体计数 ===")
#
#     if enable_hsv_preprocessing:
#         print("HSV预处理: 已启用 (全图对比度增强 + 绿色区域饱和度/明度增强)")
#     else:
#         print("HSV预处理: 未启用")
#
#     if debug_mode:
#         print("调试模式: 已启用 (将保存详细的调试图像和信息)")
#
#     # 打印当前参数
#     if contour_params:
#         print("当前轮廓检测参数:")
#         for key, value in contour_params.items():
#             print(f"  {key}: {value}")
#
#     # 加载输入数据
#     print(f"正在加载输入文件: {input_file}")
#     input_data = load_jsonl(input_file)
#     print(f"共找到 {len(input_data)} 条数据")
#
#     # 准备输出数据
#     output_data = []
#
#     # 统计信息
#     total_processed = 0
#     successful_detections = 0
#     detection_counts = {}
#
#     # 使用tqdm显示进度
#     for item in tqdm(input_data, desc="轮廓检测处理中"):
#         try:
#             # 获取图片路径
#             image_filename = item["image"]
#
#             # 构建完整图片路径
#             image_path = os.path.join(image_dir, image_filename)
#
#             # 检查图片是否存在
#             if not os.path.exists(image_path):
#                 print(f"警告: 图片文件不存在 {image_path}")
#                 # 添加空预测结果
#                 item["prediction"] = "0"
#                 output_data.append(item)
#                 continue
#
#             # 执行轮廓检测计数
#             count = count_objects_by_contours(
#                 image_path, enable_hsv_preprocessing, debug_mode, contour_params
#             )
#
#             # 添加预测结果（转换为字符串格式，保持与VLM输出一致）
#             item["prediction"] = str(count)
#             output_data.append(item)
#
#             total_processed += 1
#             if count > 0:
#                 successful_detections += 1
#
#             # 统计检测数量分布
#             detection_counts[count] = detection_counts.get(count, 0) + 1
#
#         except Exception as e:
#             print(f"处理失败 {item.get('image', 'unknown')}: {e}")
#             # 添加空预测结果，确保输出完整性
#             item["prediction"] = "0"
#             output_data.append(item)
#             continue
#
#     # 保存结果
#     print(f"正在保存结果到: {output_file}")
#     save_jsonl(output_data, output_file)
#
#     # 输出详细统计信息
#     print(f"\n=== 处理完成统计 ===")
#     print(f"总处理数量: {len(output_data)}")
#     print(f"成功处理: {total_processed}")
#     print(f"检测到物体的图像数: {successful_detections}")
#     print(f"成功率: {total_processed / len(output_data) * 100:.1f}%")
#
#     print(f"\n=== 检测数量分布 ===")
#     for count in sorted(detection_counts.keys()):
#         percentage = detection_counts[count] / total_processed * 100 if total_processed > 0 else 0
#         print(f"检测到 {count} 个物体: {detection_counts[count]} 张图像 ({percentage:.1f}%)")
#
#     if debug_mode:
#         print(f"\n调试图像已保存到图片目录中:")
#         print(f"  *_debug_binary.png - 二值化图像")
#         print(f"  *_debug_all_contours.png - 所有轮廓（红色）")
#         print(f"  *_debug_valid_contours.png - 有效轮廓（绿色）")
#
#     # 如果检测率很低，给出建议
#     if successful_detections / total_processed < 0.3 if total_processed > 0 else True:
#         print(f"\n⚠️  检测率较低，建议:")
#         print(f"1. 使用 --debug_mode 查看调试图像")
#         print(f"2. 调整参数: 降低 min_area_ratio, 增加 max_circularity")
#         print(f"3. 尝试不同的预处理方法")
#         print(f"4. 检查图像质量和目标物体特征")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="OpenCV轮廓检测批量物体计数脚本")
#     parser.add_argument('--input', type=str, required=True, help='输入JSONL文件路径')
#     parser.add_argument('--output', type=str, required=True, help='输出JSONL文件路径')
#     parser.add_argument('--image_dir', type=str, required=True, help='图片文件夹路径')
#     parser.add_argument('--enable_hsv_preprocessing', action='store_true',
#                         help='启用HSV预处理，提升绿色区域显著性')
#     parser.add_argument('--debug_mode', action='store_true',
#                         help='启用调试模式，保存带轮廓标注的图像')
#
#     args = parser.parse_args()
#
#     try:
#         # 检查输入文件和图片目录
#         if not os.path.exists(args.input):
#             raise FileNotFoundError(f"输入文件不存在: {args.input}")
#
#         if not os.path.exists(args.image_dir):
#             raise FileNotFoundError(f"图片目录不存在: {args.image_dir}")
#
#         # 创建输出目录（如果不存在）
#         output_dir = os.path.dirname(args.output)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         # 执行批量轮廓检测
#         print(f"\n开始批量轮廓检测...")
#         print(f"输入文件: {args.input}")
#         print(f"输出文件: {args.output}")
#         print(f"图片目录: {args.image_dir}")
#         print(f"HSV预处理: {'已启用' if args.enable_hsv_preprocessing else '未启用'}")
#         print(f"调试模式: {'已启用' if args.debug_mode else '未启用'}")
#         print("-" * 50)
#
#         batch_inference(
#             args.input, args.output, args.image_dir,
#             args.enable_hsv_preprocessing, args.debug_mode, None
#         )
#
#     except Exception as e:
#         print(f"批量推理失败: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     main()
#
# # 使用示例：
#
# # 基础轮廓检测（不使用HSV预处理）
# # python opencv_contour_batch_infer.py --input ./annotations/annotations_img_test.jsonl --output ./predictions_opencv/predictions_img_test_contours.jsonl --image_dir ./image_test_batch/image_test
#
# # 启用HSV预处理的轮廓检测
# # python opencv_contour_batch_infer.py --input ./annotations/annotations_img_test_2obj.jsonl --output ./predictions_opencv/predictions_img_test_2obj_hsv_contours.jsonl --image_dir ./image_test_batch/image_test_2obj --enable_hsv_preprocessing
#
# # 启用调试模式（保存带轮廓标注的图像）
# # python opencv_contour_batch_infer.py --input ./annotations/annotations_img_init_2obj.jsonl --output ./predictions_opencv/predictions_img_init_2obj_debug.jsonl --image_dir ./image_test_batch/image_init_2obj --enable_hsv_preprocessing --debug_mode
#
# # 与你的原始VLM命令对应的OpenCV版本：
# # 原命令: python smolvlm_batch_infer_hsv_qlora.py --input ./annotations/annotations_img_init_2obj.jsonl --output ./predictions_2cube/num_obj/predictions_img_init_2obj_v3.jsonl --image_dir ./image_test_batch/image_init_2obj --enable_hsv_preprocessing --adapter_path ./output_smolvlm_lora/output_smolvlm_lora_V1
# # 对应的OpenCV命令: python smolvlm_batch_infer_hsv_qlora_num_opencv.py --input ./annotations/annotations_img_init_2obj.jsonl --output ./predictions_2cube/num_obj/predictions_img_init_2obj_opencv.jsonl --image_dir ./image_test_batch/image_init_2obj --enable_hsv_preprocessing