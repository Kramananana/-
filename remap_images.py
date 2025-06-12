import os
import shutil
import yaml

# --- 请在这里配置您的核心路径和新的 data.yaml 文件路径 ---
# 【【【 您需要修改这些路径为您自己的实际路径 】】】
dataset_base_dir = r'C:\tuxiangxuexi\yachi\code\datasets\data' # 指向包含 train, val, test 文件夹的 "data" 目录
new_labels_parent_dir_name = 'labels_filtered_caries_periapical' # 【修改！】处理后，新的labels文件夹的名称
new_data_yaml_path = os.path.join(dataset_base_dir, 'data_caries_periapical.yaml') # 【修改！】指向您新的data.yaml文件
# ----------------------------------------------------------------

# 定义我们想要保留的旧类别索引，以及它们对应的新类别名称 (这些名称必须在新data.yaml的names列表中)
# 旧类别索引 -> 新类别名称 (脚本会从new_data.yaml中查找这个名称对应的新索引)
class_mapping_old_idx_to_new_name = {
    0: 'Caries',             # 原始的索引0 (假设是Caries)
    7: 'Periapical lesion'   # 原始的索引7 (假设是Periapical lesion)
}

def remap_annotations_in_folder(original_labels_folder, new_labels_folder, new_class_names_list, old_idx_to_new_name_map):
    """
    处理单个文件夹内的所有.txt标注文件。
    """
    if not os.path.isdir(original_labels_folder):
        print(f"警告：找不到原始标注文件夹 '{original_labels_folder}'，跳过。")
        return 0, 0

    # 创建新的标注文件夹 (如果已存在，先删除再创建，确保是干净的)
    if os.path.exists(new_labels_folder):
        shutil.rmtree(new_labels_folder)
    os.makedirs(new_labels_folder)
    print(f"已创建新的标注子文件夹: {new_labels_folder}")

    processed_files_count = 0
    generated_files_count = 0

    print(f"\n开始处理标注文件从 '{original_labels_folder}' 到 '{new_labels_folder}'...")

    for filename in os.listdir(original_labels_folder):
        if filename.endswith(".txt"):
            original_file_path = os.path.join(original_labels_folder, filename)
            new_file_path = os.path.join(new_labels_folder, filename)

            new_annotations = []
            has_desired_annotations = False

            try:
                with open(original_file_path, 'r', encoding='utf-8') as f_orig:
                    for line in f_orig:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            old_class_index = int(parts[0])
                            coordinates_and_rest = parts[1:]

                            if old_class_index in old_idx_to_new_name_map:
                                target_class_name = old_idx_to_new_name_map[old_class_index]
                                if target_class_name in new_class_names_list:
                                    new_class_index = new_class_names_list.index(target_class_name)
                                    new_annotations.append(f"{new_class_index} {' '.join(coordinates_and_rest)}\n")
                                    has_desired_annotations = True
                        except ValueError:
                            print(f"警告：文件 {filename} 中行 '{line.strip()}' 的类别索引格式不正确，已跳过该行。")
                            continue
                        except IndexError:
                            print(f"警告：文件 {filename} 中行 '{line.strip()}' 格式不完整，已跳过该行。")
                            continue
                
                if has_desired_annotations:
                    with open(new_file_path, 'w', encoding='utf-8') as f_new:
                        f_new.writelines(new_annotations)
                    generated_files_count += 1
                
                processed_files_count += 1
            except Exception as e:
                print(f"处理文件 {original_file_path} 时发生错误: {e}")
    
    print(f"文件夹 '{original_labels_folder}' 处理完成！")
    print(f"共检查了: {processed_files_count} 个原始标注文件。")
    print(f"生成了包含目标类别的新标注文件: {generated_files_count} 个。")
    if processed_files_count > generated_files_count:
        print(f"有 {processed_files_count - generated_files_count} 个原始文件因不包含目标类别或内容为空而未生成对应的新文件。")
    return processed_files_count, generated_files_count

# --- 主逻辑 ---
if __name__ == "__main__":
    # 1. 从新的 data.yaml 文件中加载新的类别名称列表
    try:
        print(f"尝试读取配置文件: {new_data_yaml_path}")
        with open(new_data_yaml_path, 'r', encoding='utf-8') as f_yaml:
            new_data_config = yaml.safe_load(f_yaml)

        if new_data_config is None:
            print(f"错误：配置文件 '{os.path.basename(new_data_yaml_path)}' 内容为空或格式完全错误，导致无法解析。请检查文件内容。")
            exit()

        if 'names' not in new_data_config or not isinstance(new_data_config['names'], list):
            print(f"错误：配置文件 '{os.path.basename(new_data_yaml_path)}' 中没有找到 'names' 列表，或者 'names' 字段不是一个列表。")
            print(f"读取到的配置内容是: {new_data_config}")
            exit()
        
        target_new_class_names = new_data_config['names']
        print(f"目标训练类别 (来自 {os.path.basename(new_data_yaml_path)}): {target_new_class_names}")
        
        # 检查映射中定义的名称是否都在新的yaml文件中
        for old_idx, mapped_name in class_mapping_old_idx_to_new_name.items():
            if mapped_name not in target_new_class_names:
                print(f"警告：class_mapping中定义的类别名称 '{mapped_name}' (来自旧索引 {old_idx}) 在 '{os.path.basename(new_data_yaml_path)}' 的names列表中找不到！请检查配置。")


    except FileNotFoundError:
        print(f"错误：找不到配置文件 '{new_data_yaml_path}'。请确认文件路径和名称是否正确。")
        exit()
    except yaml.YAMLError as e:
        print(f"错误：解析配置文件 '{new_data_yaml_path}' 时发生 YAML 格式错误: {e}")
        exit()
    except Exception as e:
        print(f"错误：读取或解析配置文件 '{new_data_yaml_path}' 时发生其他未知错误: {e}")
        exit()

    # 2. 定义要处理的文件夹集合 (train, val, test)
    subsets_to_process = ['train', 'val', 'test'] # 根据您的实际情况增删

    total_processed_overall = 0
    total_generated_overall = 0

    # 3. 循环处理每个子集
    for subset_name in subsets_to_process:
        # 优先尝试 'labes' (考虑到截图中的拼写)，如果找不到再尝试 'labels'
        original_labels_subfolder_primary = os.path.join(dataset_base_dir, subset_name, 'labes') # 对应您截图中的拼写
        original_labels_subfolder_secondary = os.path.join(dataset_base_dir, subset_name, 'labels') # 正确的拼写

        chosen_original_labels_subfolder = None
        if os.path.isdir(original_labels_subfolder_primary):
            chosen_original_labels_subfolder = original_labels_subfolder_primary
        elif os.path.isdir(original_labels_subfolder_secondary):
            chosen_original_labels_subfolder = original_labels_subfolder_secondary
        else:
            print(f"信息：在 '{subset_name}' 子集中，既找不到 'labes' 也找不到 'labels' 文件夹，跳过处理 '{subset_name}'。")
            print(f"  曾尝试路径1: {original_labels_subfolder_primary}")
            print(f"  曾尝试路径2: {original_labels_subfolder_secondary}")
            continue # 跳过这个子集的处理

        # 新的标注文件会保存到 new_labels_parent_dir_name 下对应的子文件夹中
        new_labels_subfolder = os.path.join(dataset_base_dir, new_labels_parent_dir_name, subset_name)
        
        p_count, g_count = remap_annotations_in_folder(chosen_original_labels_subfolder, 
                                                       new_labels_subfolder, 
                                                       target_new_class_names, 
                                                       class_mapping_old_idx_to_new_name)
        total_processed_overall += p_count
        total_generated_overall += g_count

    print(f"\n--- 所有子集处理完毕 ---")
    print(f"总共检查了: {total_processed_overall} 个原始标注文件。")
    print(f"总共生成了包含目标类别的新标注文件: {total_generated_overall} 个。")