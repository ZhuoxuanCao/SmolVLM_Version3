import re

# 色彩归一字典
color_map = {
    "cyan": "green",  # cyan归为green
    "blue": "blue",
    "red": "red",
    "green": "green"
}

def normalize_color(color):
    color = color.lower()
    return color_map.get(color, color)  # fallback到原色

def analyze(text):
    # 堆叠关系1：“on top of”
    m = re.search(r'The (\w+) (?:cube|object) is on top of the (\w+) (?:cube|object)', text, re.IGNORECASE)
    if m:
        top = normalize_color(m.group(1))
        bottom = normalize_color(m.group(2))
        print(f"[stacked] top: {top}, bottom: {bottom}")
        return

    # 堆叠关系2：“top object is ... bottom object is ...”
    m = re.search(r'top object is (?:a )?(\w+) (?:cube|object)[, ]+and the bottom object is (?:a )?(\w+) (?:cube|object)', text, re.IGNORECASE)
    if m:
        top = normalize_color(m.group(1))
        bottom = normalize_color(m.group(2))
        print(f"[stacked] top: {top}, bottom: {bottom}")
        return

    # 分开关系：“left...right...”
    m = re.search(r'(\w+) object is on the left side.*?(\w+) object is on the right side', text, re.IGNORECASE)
    if m:
        left = normalize_color(m.group(1))
        right = normalize_color(m.group(2))
        print(f"[separated] left: {left}, right: {right}")
        return

    # 辅助1：first/second对象 + 分开关系
    m1 = re.search(r'first object is (?:a )?(\w+) (?:cube|object).*second object is (?:a )?(\w+) (?:cube|object)', text, re.IGNORECASE)
    m2 = re.search(r'on the left.*?(\w+) object.*on the right.*?(\w+) object', text, re.IGNORECASE)
    if m1 and m2:
        # 优先用空间关系
        left = normalize_color(m2.group(1))
        right = normalize_color(m2.group(2))
        print(f"[separated] left: {left}, right: {right}")
        return
    elif m1:
        first = normalize_color(m1.group(1))
        second = normalize_color(m1.group(2))
        print(f"[unknown/first-second] first: {first}, second: {second} (需人工判定)")
        return

    # 默认
    print("[unparsed]", text)

# ==== 以下为批量测试 ====

texts = [
    "The workspace contains two objects. The top object is a blue cube, and the bottom object is a green cube. The blue cube is smaller than the green cube. The blue cube is on top of the green cube.",
    "The first object is a blue cube and the second object is a green cube. The blue cube is on top of the green cube.",
    "The first object is a blue cube and the second object is a red cube. The blue cube is on top of the red cube.",
    "The workspace contains a blue cube and a red cube. The blue cube is on top of the red cube.",
    "The first object is green and the second object is blue. The green object is on the left side of the image and the blue object is on the right side of the image.",
    "The first object is green and the second object is blue. The green object is on the left side of the image and the blue object is on the right side of the image.",
    "The first object is a green cube and the second object is a blue cube. The green cube is on top of the blue cube.",
    "The first object is a blue cube. The second object is a green cube. The green cube is on top of the blue cube.",
    "The first object is a red cube. The second object is a green cube. The green cube is on top of the red cube.",
    "The first object is a red cube. The second object is a cyan cube. The cyan cube is on top of the red cube.",
    "The workspace has a blue object and a red object. The blue object is on the left side of the workspace and the red object is on the right side of the workspace. The blue object is smaller than the red object.",
    "The workspace has a blue object and a red object. The blue object is on the left side of the workspace and the red object is on the right side of the workspace.",
    "The first object is red and the second object is green. The red object is on the left side of the image and the green object is on the right side of the image",
    "The first object is red and the second object is green. The red object is on the left side of the image and the green object is on the right side of the image",
    "The workspace has a red and a blue object. The red object is on top of the blue object.",
    "Yes, there are two objects in the workspace. The first object is a red cube and the second object is a blue cube. The red cube is on top of the blue cube.",
    "The workspace has a red object and a green object. The red object is on top of the green object.",
    "The first object is red and the second object is green. The red object is on top of the green object."
]

for t in texts:
    analyze(t)
