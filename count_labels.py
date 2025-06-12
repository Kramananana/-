#统计类别数量的 Python 脚本
import os
from collections import defaultdict

# --- 1. 请在这里配置您的路径和类别信息 ---
# 【请修改】指向包含 train, val, test 标注文件夹的父目录
# 例如，如果您的标注在 datasets/data/labels/ 下，那么这里就是 datasets/data/
# 根据您之前的信息，您的标注文件夹在 datasets/data/ 下名为 'labels' 或 'labes'
# 我们假设原始标注在 'labels_original_backup'
dataset_labels_base_dir = r'C:\tuxiangxuexi\yachi\code\datasets\data'

# 【请修改】定义您原始标注中所有类别的名称，顺序必须和索引 (0, 1, 2...) 一致
ORIGINAL_CLASS_NAMES = ['Caries', 'Periapical lesion', ]

# 定义要处理的子集文件夹
subsets_to_process = ['train', 'val', 'test'] # 根据您的实际情况修改
# ----------------------------------------------------------------

def count_instances_in_folder(labels_folder, class_names):
    """
    统计单个标注文件夹中每个类别的实例数量。
    """
    if not os.path.isdir(labels_folder):
        print(f"警告：找不到标注文件夹 '{labels_folder}'，跳过。")
        return None

    class_counts = defaultdict(int) # 创建一个默认值为0的字典来计数

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            class_index = int(parts[0])
                            if 0 <= class_index < len(class_names):
                                class_name = class_names[class_index]
                                class_counts[class_name] += 1
                            else:
                                print(f"警告：在文件 {filename} 中发现无效的类别索引 {class_index}。")
                        except ValueError:
                            print(f"警告：文件 {filename} 中行 '{line.strip()}' 的类别索引格式不正确，已跳过该行。")
                            continue
                        except IndexError:
                             print(f"警告：文件 {filename} 中行 '{line.strip()}' 格式不完整，已跳过该行。")
                             continue
            except Exception as e:
                print(f"处理文件 {file_path} 时发生错误: {e}")

    return class_counts

# --- 主逻辑 ---
if __name__ == "__main__":
    if not os.path.isdir(dataset_labels_base_dir):
        print(f"错误：找不到数据集标签的基础路径 '{dataset_labels_base_dir}'")
        exit()

    overall_counts = defaultdict(int)

    print("开始统计数据集中各个类别的实例数量...")

    for subset in subsets_to_process:
        print(f"\n--- 正在处理子集: {subset} ---")
        current_labels_dir = os.path.join(dataset_labels_base_dir, subset)
        
        subset_counts = count_instances_in_folder(current_labels_dir, ORIGINAL_CLASS_NAMES)
        
        if subset_counts:
            print(f"'{subset}' 子集统计结果:")
            # 排序后打印，方便查看
            sorted_subset_counts = sorted(subset_counts.items(), key=lambda item: item[1], reverse=True)
            for name, count in sorted_subset_counts:
                print(f"  - {name}: {count} 个实例")
            
            # 累加到总数
            for name, count in subset_counts.items():
                overall_counts[name] += count
    
    print("\n--- 所有子集统计完毕 ---")
    print("整个数据集中各个类别的总实例数量：")
    if overall_counts:
        # 按数量从多到少排序后打印
        sorted_overall_counts = sorted(overall_counts.items(), key=lambda item: item[1], reverse=True)
        for name, count in sorted_overall_counts:
            print(f"  - {name}: {count} 个实例")
        
        print("\n--- 您特别关注的类别数量 ---")
        caries_count = overall_counts.get('Caries', 0)
        periapical_lesion_count = overall_counts.get('Periapical lesion', 0)
        
        print(f"龋齿 (Caries): {caries_count} 个实例")
        print(f"牙周病 (Periapical lesion): {periapical_lesion_count} 个实例")
    else:
        print("在指定路径下没有统计到任何标注信息。")