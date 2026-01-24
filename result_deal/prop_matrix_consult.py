import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的目录
save_dir = 'prop_matrix_analysis'
os.makedirs(save_dir, exist_ok=True)

# 日志文件列表 自定义
log_files = ['/home/hlt/WORK/HYPER/HEMM/logs/20251109_230355.log',      # [0.25, 0.25, 0.25, 0.25]
            '/home/hlt/WORK/HYPER/HEMM/logs/20251111_122019.log',       # [1.0, 0.0, 0.0, 0.0]
            '/home/hlt/WORK/HYPER/HEMM/logs/20251111_122026.log',       # [0.0, 1.0, 0.0, 0.0]
            '/home/hlt/WORK/HYPER/HEMM/logs/20251111_122039.log',       # [0.0, 0.0, 1.0, 0.0]
            '/home/hlt/WORK/HYPER/HEMM/logs/20251111_122045.log']       # [0.0, 0.0, 0.0, 1.0]

# prop_matrix权重组合 自定义
prop_matrices = [
    [0.25, 0.25, 0.25, 0.25],  # 均匀权重
    [1.0, 0.0, 0.0, 0.0],      # 仅度数
    [0.0, 1.0, 0.0, 0.0],      # 仅特征差异
    [0.0, 0.0, 1.0, 0.0],      # 仅节点度数
    [0.0, 0.0, 0.0, 1.0]       # 仅特征重要性
]

# 权重组合描述
weight_group_desc = {
    0: "Uniform [0.25, 0.25, 0.25, 0.25]",
    1: "Degree Only [1.0, 0.0, 0.0, 0.0]",
    2: "Feature Difference Only [0.0, 1.0, 0.0, 0.0]",
    3: "Node Degree Only [0.0, 0.0, 1.0, 0.0]",
    4: "Feature Importance Only [0.0, 0.0, 0.0, 1.0]"
}

# 限定只分析这8个数据集
target_datasets = {'hiv', 'bace', 'bbbp', 'tox21', 'sider', 'clintox', 'toxcast', 'muv'}

# 存储所有数据集的结果
# 数据结构: {dataset_name: {weight_group_index: (mean, std)}}
results = {}

# 从日志文件中提取prop_matrix信息
def extract_prop_matrix_from_log(log_content):
    # 查找权重组合信息
    match = re.search(r'✓ Weight Combination: (\d+)', log_content)
    if match:
        return int(match.group(1))
    return None

# 从日志中提取数据集结果
def extract_dataset_results(log_content, weight_group_index):
    dataset_results = {}
    
    # 查找数据集开始的行
    dataset_pattern = r'Starting training on ([A-Z0-9]+) dataset'
    dataset_matches = list(re.finditer(dataset_pattern, log_content))
    
    for i, match in enumerate(dataset_matches):
        dataset_name = match.group(1).lower()
        # 只处理指定的数据集
        if dataset_name not in target_datasets:
            continue
            
        dataset_start = match.start()
        
        # 确定结束位置（下一个数据集开始或文件结束）
        if i + 1 < len(dataset_matches):
            dataset_end = dataset_matches[i + 1].start()
        else:
            dataset_end = len(log_content)
        
        # 提取当前数据集的内容
        dataset_content = log_content[dataset_start:dataset_end]
        
        # 首先尝试匹配最佳三个结果的列表
        best_results_pattern = r'All results: \[([0-9.,\s]+)\]'
        best_results_match = re.search(best_results_pattern, dataset_content)
        
        if best_results_match:
            # 从最佳三个结果计算平均值和标准差
            best_results_str = best_results_match.group(1)
            best_results = [float(x.strip()) for x in best_results_str.split(',')]
            
            # 取最好的三个结果
            sorted_results = sorted(best_results, reverse=True)   # 降序排列，取最好的
            
            mean_val = np.mean(sorted_results[:3])  # 取前3个
            std_val = np.std(sorted_results[:3])
            dataset_results[dataset_name] = (mean_val, std_val)
        else:
            # 查找平均值和标准差
            mean_pattern = r'Average of top 3 results: ([0-9.]+)'
            std_pattern = r'Standard deviation of top 3 results: ([0-9.]+)'
            
            mean_match = re.search(mean_pattern, dataset_content)
            std_match = re.search(std_pattern, dataset_content)
            
            if mean_match and std_match:
                mean_val = float(mean_match.group(1))
                std_val = float(std_match.group(1))
                dataset_results[dataset_name] = (mean_val, std_val)
    
    return dataset_results

# 处理每个日志文件
for i, log_file in enumerate(log_files):
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取权重组合索引
        weight_group_index = i
        
        # 提取数据集结果，传递权重组合索引
        dataset_results = extract_dataset_results(content, weight_group_index)
        
        # 将结果存储到总结果中
        for dataset_name, (mean_val, std_val) in dataset_results.items():
            if dataset_name not in results:
                results[dataset_name] = {}
            results[dataset_name][weight_group_index] = (mean_val, std_val)
            
    except Exception as e:
        print(f"处理文件 {log_file} 时出错: {e}")

# 打印提取的结果
print("Extracted Results:")
for dataset_name, dataset_data in results.items():
    print(f"\n{dataset_name.upper()} Dataset:")
    sorted_weights = sorted(dataset_data.keys())
    for weight_idx in sorted_weights:
        mean_val, std_val = dataset_data[weight_idx]
        print(f"  Weight Combination {weight_idx} ({weight_group_desc[weight_idx]}): Mean={mean_val:.4f}, Std={std_val:.4f}")

# 保存提取的结果到文件
results_file = os.path.join(save_dir, 'extracted_prop_matrix_results.txt')
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("Extracted Weight Combination Results:\n")
    for dataset_name, dataset_data in results.items():
        f.write(f"\n{dataset_name.upper()} Dataset:\n")
        sorted_weights = sorted(dataset_data.keys())
        for weight_idx in sorted_weights:
            mean_val, std_val = dataset_data[weight_idx]
            f.write(f"  Weight Combination {weight_idx} ({weight_group_desc[weight_idx]}): Mean={mean_val:.4f}, Std={std_val:.4f}\n")

print(f"\nResults saved to file: {results_file}")

# 创建每个数据集的单独图表
for dataset_name, dataset_data in results.items():
    sorted_weights = sorted(dataset_data.keys())
    means = [dataset_data[weight_idx][0] for weight_idx in sorted_weights]
    stds = [dataset_data[weight_idx][1] for weight_idx in sorted_weights]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(sorted_weights, means, yerr=stds, marker='o', capsize=3, capthick=1, elinewidth=1, 
                markersize=8, linewidth=2, color='#1f77b4', ecolor='red', alpha=0.8)
    plt.xlabel('Weight Combination Index', fontsize=12, fontweight='bold')
    plt.ylabel('ROC-AUC (Top 3 Average)', fontsize=12, fontweight='bold')
    plt.title(f'{dataset_name.upper()} Dataset Performance with Different Weight Combinations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(sorted_weights, [str(i) for i in sorted_weights], fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0.6, 1.0)
    
    # 添加数值标签
    for i, (weight_idx, mean_val) in enumerate(zip(sorted_weights, means)):
        plt.annotate(f'{mean_val:.4f}', (weight_idx, mean_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_prop_matrix_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 计算所有数据集的均值和标准差
# 数据结构: {weight_group_index: (mean_of_means, std_of_means, mean_of_stds)}
overall_results = {}
sorted_weights = list(range(len(prop_matrices)))

for weight_idx in sorted_weights:
    means_at_weight = []
    stds_at_weight = []
    
    for dataset_name, dataset_data in results.items():
        if weight_idx in dataset_data:
            means_at_weight.append(dataset_data[weight_idx][0])
            stds_at_weight.append(dataset_data[weight_idx][1])
    
    if means_at_weight:
        mean_of_means = np.mean(means_at_weight)
        std_of_means = np.std(means_at_weight)
        mean_of_stds = np.mean(stds_at_weight)
        overall_results[weight_idx] = (mean_of_means, std_of_means, mean_of_stds)

# 创建所有数据集均值的图表
plt.figure(figsize=(12, 7))
sorted_overall_weights = sorted(overall_results.keys())
overall_means = [overall_results[weight_idx][0] for weight_idx in sorted_overall_weights]
overall_stds_of_means = [overall_results[weight_idx][1] for weight_idx in sorted_overall_weights]
overall_means_of_stds = [overall_results[weight_idx][2] for weight_idx in sorted_overall_weights]

plt.errorbar(sorted_overall_weights, overall_means, yerr=overall_stds_of_means, 
            marker='o', linewidth=2.5, markersize=8, color='#1f77b4',
            capsize=4, capthick=1.5, elinewidth=1.5, markeredgecolor='black',
            markeredgewidth=0.5, label='Mean Performance ± Std of Means')

# 添加第二组误差条表示各数据集标准差的平均值
plt.errorbar(sorted_overall_weights, overall_means, yerr=overall_means_of_stds,
            fmt='none', ecolor='orange', capsize=3, capthick=1, elinewidth=1,
            alpha=0.7, label='Mean Performance ± Mean of Dataset Stds')

plt.xlabel('Weight Combination Index', fontsize=14, fontweight='bold')
plt.ylabel('ROC-AUC (Top 3 Average)', fontsize=14, fontweight='bold')
plt.title('Overall Performance Across All Datasets with Different Weight Combinations', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)
plt.xticks(sorted_overall_weights, [str(i) for i in sorted_overall_weights], fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.6, 1.0)

# 添加数值标签
for i, (weight_idx, mean_val) in enumerate(zip(sorted_overall_weights, overall_means)):
    plt.annotate(f'{mean_val:.4f}', (weight_idx, mean_val), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='black')

# 设置坐标轴边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overall_prop_matrix_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

# 创建所有数据集在同一图中的图表
plt.figure(figsize=(12, 8))
sorted_weights = list(range(len(prop_matrices)))

# 定义颜色列表以区分不同数据集
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# 定义不同的标记形状以更好地区分数据集
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']

# 为每个数据集绘制一条线
for idx, (dataset_name, dataset_data) in enumerate(results.items()):
    means = []
    stds = []
    available_weights = []
    
    for weight_idx in sorted_weights:
        if weight_idx in dataset_data:
            means.append(dataset_data[weight_idx][0])
            stds.append(dataset_data[weight_idx][1])
            available_weights.append(weight_idx)
    
    if means:  # 只有当有数据时才绘制
        plt.errorbar(available_weights, means, yerr=stds, 
                    label=dataset_name.upper(), 
                    linewidth=2.5, 
                    marker=markers[idx % len(markers)], 
                    markersize=8, 
                    color=colors[idx % len(colors)],
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    markeredgecolor='black',
                    markeredgewidth=0.5)

# 设置图表属性
plt.xlabel('Weight Combination Index', fontsize=14, fontweight='bold')
plt.ylabel('ROC-AUC (Top 3 Average)', fontsize=14, fontweight='bold')
plt.title('Performance Comparison Across All Datasets with Different Weight Combinations', fontsize=16, fontweight='bold')

# 设置图例
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=11, frameon=True, 
          fancybox=True, shadow=True, framealpha=0.9)

# 设置网格
plt.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)

# 设置坐标轴刻度
plt.xticks(sorted_weights, [str(i) for i in sorted_weights], fontsize=12)
plt.yticks(fontsize=12)

# 设置y轴范围
plt.ylim(0.6, 1.0)

# 设置坐标轴边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 调整布局以适应图例
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'all_datasets_prop_matrix_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n图表已生成并保存到 {save_dir} 目录:")
print("1. 每个数据集的单独图表已保存为 {dataset_name}_prop_matrix_performance.png")
print("2. 所有数据集的对比图表已保存为 all_datasets_prop_matrix_performance.png")
print("3. 所有数据集均值图表已保存为 overall_prop_matrix_performance.png")