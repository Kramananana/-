#数据增强 Python 脚本
import os
import cv2
import numpy as np
import albumentations as A
import random

# --- 1. 请在这里配置您的路径和参数 ---
# 【请修改】指向您数据集的根目录
DATASET_BASE_DIR = r'C:\tuxiangxuexi\yachi\code\datasets\data' # 示例路径

# 【请修改】原始的训练图片和标注文件夹路径
ORIGINAL_IMAGES_DIR = os.path.join(DATASET_BASE_DIR, 'train/images')
ORIGINAL_LABELS_DIR = os.path.join(DATASET_BASE_DIR, 'train/labels')

# 【请修改】您想增强的类别在原始标注文件中的索引号
# 根据您的 data.yaml (source: 30)，'Periapical lesion' 的索引是 2
TARGET_CLASS_ID = 2

# 【请修改】您希望为每个包含目标类别的原始图片生成多少张增强后的新图片
NUM_AUGMENTATIONS_PER_IMAGE = 1 # 您可以根据需要调整这个数字

# -----------------------------------------------

def read_yolo_segmentation_label(label_path):
    """读取YOLO分割标注文件，返回所有物体的类别和归一化多边形坐标。"""
    if not os.path.exists(label_path):
        return [], []
    
    classes = []
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                poly = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                classes.append(class_id)
                polygons.append(poly)
            except (ValueError, IndexError) as e:
                print(f"警告：跳过格式错误的行 '{line.strip()}' 在文件 '{os.path.basename(label_path)}': {e}")
    return classes, polygons

def save_yolo_segmentation_label(label_path, classes, polygons):
    """将增强后的类别和多边形坐标保存到新的YOLO标注文件中。"""
    with open(label_path, 'w') as f:
        for class_id, poly in zip(classes, polygons):
            line = f"{class_id}"
            for point in poly:
                line += f" {point[0]:.6f} {point[1]:.6f}"
            f.write(line + "\n")

# --- 2. 定义我们的数据增强“魔法药水” ---
# Albumentations 的变换组合
# 这里定义了一些常见的、对分割任务比较友好的增强方法
# 您可以根据需要添加、删除或修改这些增强的参数
transform = A.Compose([
    # 几何变换
    A.HorizontalFlip(p=0.5), # 50%的概率水平翻转
    A.RandomRotate90(p=0.5), # 50%的概率随机旋转90/180/270度
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7), # 轻微的平移、缩放、旋转
    A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), # 弹性变形

    # 颜色变换
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7), # 随机调整亮度和对比度
    A.RandomGamma(p=0.2), # 随机Gamma校正

    # 模糊和噪声 (对于医学图像要谨慎使用，强度不宜过大)
    # A.GaussNoise(p=0.2),
    # A.Blur(p=0.2),

],
# Albumentations 需要知道我们是在处理分割掩码
# keypoint_params (如果需要处理关键点), mask_params (处理掩码), bbox_params (处理边界框)
# 对于 YOLO 分割 (多边形)，我们实际上是在变换坐标点，所以用 keypoint_params
# 每个多边形被视为一组关键点
keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_classes'])
)

# --- 3. 主逻辑 ---
if __name__ == "__main__":
    if not os.path.isdir(ORIGINAL_IMAGES_DIR) or not os.path.isdir(ORIGINAL_LABELS_DIR):
        print(f"错误：找不到原始图片或标签文件夹。请检查路径配置。")
        exit()

    print("开始查找包含目标类别的图像...")
    
    # 找到所有包含目标类别的图片和标签
    images_to_augment = []
    for filename in os.listdir(ORIGINAL_LABELS_DIR):
        if filename.endswith('.txt'):
            label_path = os.path.join(ORIGINAL_LABELS_DIR, filename)
            classes, _ = read_yolo_segmentation_label(label_path)
            if TARGET_CLASS_ID in classes:
                base_name = os.path.splitext(filename)[0]
                # 寻找对应的图片文件 (尝试 .jpg 和 .png)
                img_path_jpg = os.path.join(ORIGINAL_IMAGES_DIR, base_name + '.jpg')
                img_path_png = os.path.join(ORIGINAL_IMAGES_DIR, base_name + '.png')
                
                if os.path.exists(img_path_jpg):
                    images_to_augment.append((img_path_jpg, label_path))
                elif os.path.exists(img_path_png):
                    images_to_augment.append((img_path_png, label_path))

    if not images_to_augment:
        print(f"在 '{ORIGINAL_LABELS_DIR}' 中没有找到任何包含类别ID {TARGET_CLASS_ID} 的标注文件。脚本将退出。")
        exit()

    print(f"找到了 {len(images_to_augment)} 张包含目标类别 {TARGET_CLASS_ID} ('Periapical lesion') 的图像。")
    print(f"将为每张图片生成 {NUM_AUGMENTATIONS_PER_IMAGE} 个增强样本...")

    generated_count = 0
    # 开始生成增强数据
    for img_path, label_path in images_to_augment:
        try:
            # 读取原始图像和标注
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告：无法读取图片 {img_path}，跳过。")
                continue
            
            classes, polygons = read_yolo_segmentation_label(label_path)

            for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                # 准备 albumentations 需要的输入格式
                # 将所有多边形点和类别ID整合
                keypoints_to_transform = []
                keypoint_classes_to_transform = []
                for class_id, poly in zip(classes, polygons):
                    keypoints_to_transform.extend(poly.tolist()) # 将多边形的点展平添加
                    # 为每个点都标记上它所属的类别ID
                    keypoint_classes_to_transform.extend([class_id] * len(poly))

                # 应用数据增强
                transformed = transform(image=image, keypoints=keypoints_to_transform, keypoint_classes=keypoint_classes_to_transform)
                transformed_image = transformed['image']
                transformed_keypoints = transformed['keypoints']
                transformed_keypoint_classes = transformed['keypoint_classes']

                # 将变换后的关键点重新组合成多边形
                new_polygons = []
                new_classes = []
                current_poly = []
                # 假设每个多边形的类别ID是连续的，通过类别ID变化来区分不同多边形
                if transformed_keypoints:
                    current_class_id = transformed_keypoint_classes[0]
                    for kp, cls_id in zip(transformed_keypoints, transformed_keypoint_classes):
                        if cls_id == current_class_id:
                            current_poly.append(list(kp))
                        else:
                            # 一个多边形结束，保存它
                            if current_poly:
                                new_polygons.append(np.array(current_poly))
                                new_classes.append(current_class_id)
                            # 开始新的多边形
                            current_class_id = cls_id
                            current_poly = [list(kp)]
                    # 保存最后一个多边形
                    if current_poly:
                        new_polygons.append(np.array(current_poly))
                        new_classes.append(current_class_id)

                # 生成新的文件名
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                new_base_name = f"{base_name}_aug_periapical_{i}"
                new_img_path = os.path.join(ORIGINAL_IMAGES_DIR, new_base_name + '.jpg') # 新图片保存为jpg
                new_label_path = os.path.join(ORIGINAL_LABELS_DIR, new_base_name + '.txt')

                # 保存新的图片和标注文件
                # 确保坐标值在0-1之间 (albumentations有时会超出一点点)
                for poly in new_polygons:
                    np.clip(poly[:, 0], 0, 1, out=poly[:, 0]) # x坐标
                    np.clip(poly[:, 1], 0, 1, out=poly[:, 1]) # y坐标

                if new_polygons:
                    cv2.imwrite(new_img_path, transformed_image)
                    save_yolo_segmentation_label(new_label_path, new_classes, new_polygons)
                    generated_count += 1
        except Exception as e:
            print(f"处理图片 {img_path} 时发生错误: {e}")

    print(f"\n--- 数据增强完成 ---")
    print(f"总共生成了 {generated_count} 个新的增强样本（图片和标注文件）并添加到了您的训练集中。")