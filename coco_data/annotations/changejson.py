import json

# 加载 JSON 文件
with open('annotations_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历 images 列表，并将 .png 后缀改为 .jpg
for image in data['images']:
    if image['file_name'].endswith('.png'):
        image['file_name'] = image['file_name'].replace('.png', '.jpg')

# 保存修改后的 JSON 文件
with open('annotations_modified.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("文件后缀已修改为 .jpg")
