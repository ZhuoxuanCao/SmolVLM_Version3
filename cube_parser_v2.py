import re
import json
import argparse

# 色彩归一映射
color_map = {
    "cyan": "green",
    "blue": "blue",
    "red": "red",
    "green": "green"
}

def normalize_color(color):
    color = color.lower()
    return color_map.get(color, color)

def parse_prediction(text):
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
        # 如果捕获失败，返回空结构
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
        m3 = re.search(r'(\w+) (?:cube|object) is on the left.*?(\w+) (?:cube|object) is on the right', text, re.IGNORECASE)
        if m3:
            left = normalize_color(m3.group(1))
            right = normalize_color(m3.group(2))
            return {"relationship": "separated", "left": {"color": left}, "right": {"color": right}}
        # 如果左右都找不到，返回空结构
        return {"relationship": "separated", "left": {}, "right": {}}

def is_structured(parsed):
    # 检查结构化信息是否完整（不能有空字典）
    if parsed["relationship"] == "stacked":
        return bool(parsed.get("top")) and bool(parsed.get("bottom")) and "color" in parsed["top"] and "color" in parsed["bottom"]
    elif parsed["relationship"] == "separated":
        return bool(parsed.get("left")) and bool(parsed.get("right")) and "color" in parsed["left"] and "color" in parsed["right"]
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description="正则归一化模型预测的jsonl批处理脚本（带兜底机制）")
    parser.add_argument("--input", type=str, required=True, help="输入jsonl文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出jsonl文件路径")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pred = data.get("prediction", "")
            parsed = parse_prediction(pred)
            # 如果解析失败，直接返回原始预测文本
            if is_structured(parsed):
                new_data = {
                    "image": data.get("image"),
                    "parsed": parsed
                }
            else:
                new_data = {
                    "image": data.get("image"),
                    "parsed": pred
                }
            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    print(f"处理完成，结果保存到：{args.output}")

if __name__ == "__main__":
    main()



# def parse_prediction(text):
#     # [1] 堆叠关系: on top of
#     m = re.search(r'The (\w+) (?:cube|object) is on top of the (\w+) (?:cube|object)', text, re.IGNORECASE)
#     if m:
#         top = normalize_color(m.group(1))
#         bottom = normalize_color(m.group(2))
#         return {"relationship": "stacked", "top": {"color": top}, "bottom": {"color": bottom}}
#
#     # [2] 堆叠关系: top object is ... bottom object is ...
#     m = re.search(
#         r'top object is (?:a )?(\w+) (?:cube|object)[, ]+and the bottom object is (?:a )?(\w+) (?:cube|object)', text,
#         re.IGNORECASE)
#     if m:
#         top = normalize_color(m.group(1))
#         bottom = normalize_color(m.group(2))
#         return {"relationship": "stacked", "top": {"color": top}, "bottom": {"color": bottom}}
#
#     # [3] 分开关系: left...right...
#     m = re.search(r'(\w+) object is on the left side.*?(\w+) object is on the right side', text, re.IGNORECASE)
#     if m:
#         left = normalize_color(m.group(1))
#         right = normalize_color(m.group(2))
#         return {"relationship": "separated", "left": {"color": left}, "right": {"color": right}}
#
#     # [4] 分开关系: the (\w+) object is on the left... the (\w+) object is on the right...
#     m = re.search(r'the (\w+) object is on the left.*the (\w+) object is on the right', text, re.IGNORECASE)
#     if m:
#         left = normalize_color(m.group(1))
#         right = normalize_color(m.group(2))
#         return {"relationship": "separated", "left": {"color": left}, "right": {"color": right}}
#
#     # [5] first object / second object（作为辅助，极端情况可选）
#     m1 = re.search(r'first object is (?:a )?(\w+).*second object is (?:a )?(\w+)', text, re.IGNORECASE)
#     m2 = re.search(r'on top of the (\w+)', text, re.IGNORECASE)
#     if m1 and m2:
#         # 如果同时出现first/second + on top of，用on top of确定top/bottom
#         top_color = re.search(r'(\w+) (?:cube|object) is on top of', text, re.IGNORECASE)
#         if top_color:
#             top = normalize_color(top_color.group(1))
#             # 取first/second object中剩下的为bottom
#             first = normalize_color(m1.group(1))
#             second = normalize_color(m1.group(2))
#             bottom = second if top == first else first
#             return {"relationship": "stacked", "top": {"color": top}, "bottom": {"color": bottom}}
#     if m1:
#         # 如果只有first/second，不确定空间关系，给出未知关系
#         first = normalize_color(m1.group(1))
#         second = normalize_color(m1.group(2))
#         return {"relationship": "unknown", "first": {"color": first}, "second": {"color": second}}
#
#     # [6] 无法解析的
#     return {"relationship": "unparsed"}

# python cube_parser_v2.py --input ./predictions_2cube/predictions_img_init_2obj_v1.jsonl --output cube_parser.jsonl
