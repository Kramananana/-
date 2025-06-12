import os
import shutil

# --- 请在这里配置您的核心路径 ---
# 【【【 您需要修改这个路径为您项目中 'data' 文件夹的实际绝对路径 】】】
dataset_base_dir = r'C:\tuxiangxuexi\yachi\code\datasets\data' # 指向包含 test, train, val 文件夹的 "data" 目录
# ----------------------------------------------------------------

# 定义要处理的子集文件夹 (比如 train, val, test)
subsets_to_process = ['train', 'val', 'test'] # 根据您的实际情况修改

def delete_unannotated_images_in_subset(subset_name, base_data_dir):
    """
    删除在指定图像子文件夹中没有对应有效标注文件的图像。
    并删除空的标注文件。
    """
    images_subset_dir = os.path.join(base_data_dir, subset_name, 'images')
    
    # 优先尝试 'labels'，如果找不到再尝试 'labes'
    labels_subset_dir_standard = os.path.join(base_data_dir, subset_name, 'labels')
    labels_subset_dir_alternative = os.path.join(base_data_dir, subset_name, 'labes') # 对应您截图中的可能拼写

    chosen_labels_subset_dir = None
    if os.path.isdir(labels_subset_dir_standard):
        chosen_labels_subset_dir = labels_subset_dir_standard
    elif os.path.isdir(labels_subset_dir_alternative):
        chosen_labels_subset_dir = labels_subset_dir_alternative
    
    if not os.path.isdir(images_subset_dir):
        print(f"警告：找不到图像子文件夹 '{images_subset_dir}'，跳过子集 '{subset_name}'。")
        return 0, 0
    if not chosen_labels_subset_dir:
        print(f"警告：在 '{subset_name}' 子集中，既找不到 'labels' 也找不到 'labes' 标注文件夹，无法进行比对，跳过清理 '{images_subset_dir}'。")
        return 0, 0

    print(f"\n正在检查并清理图像文件夹 '{images_subset_dir}' (基于 '{chosen_labels_subset_dir}' 中的标注)...")
    
    deleted_images_count = 0
    deleted_empty_label_files_count = 0
    
    # 首先，清理标签文件夹中空的 .txt 文件
    if chosen_labels_subset_dir: # 确保标签目录存在
        for label_filename in os.listdir(chosen_labels_subset_dir):
            if label_filename.endswith(".txt"):
                label_file_path = os.path.join(chosen_labels_subset_dir, label_filename)
                if os.path.getsize(label_file_path) == 0:
                    try:
                        os.remove(label_file_path)
                        print(f"  已删除空的标注文件: {os.path.join(subset_name, os.path.basename(chosen_labels_subset_dir) ,label_filename)}")
                        deleted_empty_label_files_count +=1
                    except Exception as e:
                        print(f"  错误：删除空的标注文件 {label_file_path} 失败: {e}")


    # 然后，根据标签文件夹中的情况清理图像文件夹
    image_files = os.listdir(images_subset_dir)
    for image_filename in image_files:
        image_file_path = os.path.join(images_subset_dir, image_filename)
        
        if not os.path.isfile(image_file_path):
            continue

        image_filename_base, image_ext = os.path.splitext(image_filename)
        label_filename = image_filename_base + ".txt"
        label_file_path = os.path.join(chosen_labels_subset_dir, label_filename) # 使用已确定的标签文件夹路径

        # 检查对应的标注文件是否存在 (并且在上面一步清理后，如果为空也已被删除)
        if not os.path.exists(label_file_path): # 如果标注文件不存在（可能因为原始就没有，或因为是空的已被删除）
            print(f"  未找到有效标注文件 '{label_filename}' 对于图像 '{image_filename}'。准备删除图像...")
            try:
                os.remove(image_file_path)
                print(f"    已删除图像: {os.path.join(subset_name,'images',image_filename)}")
                deleted_images_count += 1
            except Exception as e:
                print(f"    错误：删除图像 {image_file_path} 失败: {e}")
                
    print(f"图像子文件夹 '{images_subset_dir}' 清理完成。")
    return deleted_images_count, deleted_empty_label_files_count

# --- 主逻辑 ---
if __name__ == "__main__":
    if not os.path.isdir(dataset_base_dir):
        print(f"错误：找不到数据集基础路径 '{dataset_base_dir}'")
        exit()

    total_deleted_images_overall = 0
    total_deleted_empty_labels_overall = 0

    for subset in subsets_to_process:
        print(f"\n--- 开始处理子集: {subset} ---")
        img_del_count, empty_lbl_del_count = delete_unannotated_images_in_subset(subset, dataset_base_dir)
        total_deleted_images_overall += img_del_count
        total_deleted_empty_labels_overall += empty_lbl_del_count

    print(f"\n--- 所有子集清理完毕 ---")
    print(f"总共删除了 {total_deleted_images_overall} 张没有对应有效标注文件的图片。")
    print(f"总共删除了 {total_deleted_empty_labels_overall} 个空的标注文件。")