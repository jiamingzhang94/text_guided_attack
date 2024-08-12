import json

# 读取原始 JSON 文件
input_file = 'json/coco_karpathy_val.json'  # 替换为您的输入文件名
output_file = 'json/coco_karpathy_val_0.json'  # 替换为您想要的输出文件名

# 读取 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 修改每个元素，只保留第一个 caption
for item in data:
    if 'caption' in item and isinstance(item['caption'], list) and len(item['caption']) > 0:
        item['caption'] = [item['caption'][0]]  # 只保留第一个 caption

# 将修改后的数据写入新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"处理完成。结果已保存到 {output_file}")