import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from PIL import Image

# 可视化COCO标注的函数，适用于一张图片
def visualize_coco_annotations(coco, image_dir, image_id):
    # 获取图像信息
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])

    # 打开图像
    img = Image.open(img_path)

    # 创建一个绘图
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # 获取该图片的所有标注
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    # 绘制每个标注的矩形框
    for ann in anns:
        bbox = ann['bbox']
        # COCO的bbox格式是 [x, y, width, height]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 显示类别标签
        cat_id = ann['category_id']
        cat_info = coco.loadCats(cat_id)[0]
        plt.text(bbox[0], bbox[1] - 2, cat_info['name'], color='red', fontsize=12, weight='bold')

    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 遍历文件夹中的所有图片并可视化
def visualize_folder_images(image_dir, annotation_file):
    # 初始化 COCO api
    coco = COCO(annotation_file)

    # 获取所有的图片ID
    img_ids = coco.getImgIds()

    # 遍历每一张图片
    for image_id in img_ids:
        print(f"正在可视化图片ID: {image_id}")
        visualize_coco_annotations(coco, image_dir, image_id)

# 使用示例
image_dir = r'C:\Users\Lenovo\OneDrive\桌面\Split\coco\val2017'
annotation_file = r'C:\Users\Lenovo\OneDrive\桌面\Split\coco\annotations\instances_val2017.json'

visualize_folder_images(image_dir, annotation_file)
