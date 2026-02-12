import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的目录
save_dir = 'mask_ratio_analysis'
os.makedirs(save_dir, exist_ok=True)

# 日志文件列表 自定义
log_files = ['/home/hlt/WORK/HYPER/HEMM/logs/20251109_230358.log',      # 0.1
            '/home/hlt/WORK/HYPER/HEMM/logs/20251107_131952.log',       # 0.3
            '/home/hlt/WORK/HYPER/HEMM/logs/20251109_230355.log',       # 0.5
            '/home/hlt/WORK/HYPER/HEMM/logs/20251109_230353.log']       # 0.7

# mask_ratio值 自定义
mask_ratios = [0.1, 0.3, 0.5, 0.7]

# 限定只分析这8个数据集
target_datasets = {'hiv', 'bace', 'bbbp', 'tox21', 'sider', 'clintox', 'toxcast', 'muv'}

# 存储所有数据集的结果
# 数据结构: {dataset_name: {mask_ratio: (mean, std)}}
results = {}

# 从日志文件中提取mask_ratio值
def extract_mask_ratio_from_log(log_content):
    # 查找类似 "✓ 掩码比例: 0.1" 的行 或其他可能的掩码比例表示方式
    match = re.search(r'✓ 超边掩码比例: ([0-9.]+)', log_content)
    if match:
        return float(match.group(1))
    return None

# 从日志中提取数据集结果
def extract_dataset_results(log_content):
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
        best_results_pattern = r'Best 3 results: \[([0-9.,\s]+)\]'
        best_results_match = re.search(best_results_pattern, dataset_content)
        
        if best_results_match:
            # 从最佳三个结果计算平均值和标准差
            best_results_str = best_results_match.group(1)
            best_results = [float(x.strip()) for x in best_results_str.split(',')]
            mean_val = np.mean(best_results)
            std_val = np.std(best_results)
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
        
        # 提取mask_ratio（如果无法从日志中提取，则使用预设值）
        # mask_ratio = extract_mask_ratio_from_log(content)
        # if mask_ratio is None:
        mask_ratio = mask_ratios[i]
        
        # 提取数据集结果
        dataset_results = extract_dataset_results(content)
        
        # 将结果存储到总结果中
        for dataset_name, (mean_val, std_val) in dataset_results.items():
            if dataset_name not in results:
                results[dataset_name] = {}
            results[dataset_name][mask_ratio] = (mean_val, std_val)
            
    except Exception as e:
        print(f"处理文件 {log_file} 时出错: {e}")

# 打印提取的结果
print("提取的结果:")
for dataset_name, dataset_data in results.items():
    print(f"\n{dataset_name.upper()} 数据集:")
    sorted_ratios = sorted(dataset_data.keys())
    for ratio in sorted_ratios:
        mean_val, std_val = dataset_data[ratio]
        print(f"  mask_ratio={ratio}: 平均值={mean_val:.4f}, 标准差={std_val:.4f}")

# 保存提取的结果到文件
results_file = os.path.join(save_dir, 'extracted_mask_ratio_results.txt')
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("提取的掩码比例结果:\n")
    for dataset_name, dataset_data in results.items():
        f.write(f"\n{dataset_name.upper()} 数据集:\n")
        sorted_ratios = sorted(dataset_data.keys())
        for ratio in sorted_ratios:
            mean_val, std_val = dataset_data[ratio]
            f.write(f"  mask_ratio={ratio}: 平均值={mean_val:.4f}, 标准差={std_val:.4f}\n")

print(f"\n结果已保存到文件: {results_file}")

# 创建每个数据集的单独图表
for dataset_name, dataset_data in results.items():
    sorted_ratios = sorted(dataset_data.keys())
    means = [dataset_data[ratio][0] for ratio in sorted_ratios]
    stds = [dataset_data[ratio][1] for ratio in sorted_ratios]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(sorted_ratios, means, yerr=stds, marker='o', capsize=3, capthick=1, elinewidth=1, 
                markersize=6, linewidth=2, color='#1f77b4', ecolor='red', alpha=0.8)
    plt.xlabel('Mask Ratio', fontsize=12, fontweight='bold')
    plt.ylabel('ROC-AUC (Top 3 Average)', fontsize=12, fontweight='bold')
    plt.title(f'{dataset_name.upper()} Dataset Performance with Different Mask Ratios', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(sorted_ratios, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0.6, 1.0)
    
    # 添加数值标签
    for i, (ratio, mean_val) in enumerate(zip(sorted_ratios, means)):
        plt.annotate(f'{mean_val:.4f}', (ratio, mean_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_mask_ratio_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 计算所有数据集的均值和标准差
# 数据结构: {mask_ratio: (mean_of_means, std_of_means, mean_of_stds)}
overall_results = {}
sorted_ratios = sorted(mask_ratios)

for ratio in sorted_ratios:
    means_at_ratio = []
    stds_at_ratio = []
    
    for dataset_name, dataset_data in results.items():
        if ratio in dataset_data:
            means_at_ratio.append(dataset_data[ratio][0])
            stds_at_ratio.append(dataset_data[ratio][1])
    
    if means_at_ratio:
        mean_of_means = np.mean(means_at_ratio)
        std_of_means = np.std(means_at_ratio)
        mean_of_stds = np.mean(stds_at_ratio)
        overall_results[ratio] = (mean_of_means, std_of_means, mean_of_stds)

# 创建所有数据集均值的图表
plt.figure(figsize=(10, 7))
sorted_overall_ratios = sorted(overall_results.keys())
overall_means = [overall_results[ratio][0] for ratio in sorted_overall_ratios]
overall_stds_of_means = [overall_results[ratio][1] for ratio in sorted_overall_ratios]
overall_means_of_stds = [overall_results[ratio][2] for ratio in sorted_overall_ratios]

plt.errorbar(sorted_overall_ratios, overall_means, yerr=overall_stds_of_means, 
            marker='o', linewidth=2.5, markersize=8, color='#1f77b4',
            capsize=4, capthick=1.5, elinewidth=1.5, markeredgecolor='black',
            markeredgewidth=0.5, label='Mean Performance ± Std of Means')

# 添加第二组误差条表示各数据集标准差的平均值
plt.errorbar(sorted_overall_ratios, overall_means, yerr=overall_means_of_stds,
            fmt='none', ecolor='orange', capsize=3, capthick=1, elinewidth=1,
            alpha=0.7, label='Mean Performance ± Mean of Dataset Stds')

plt.xlabel('Mask Ratio', fontsize=14, fontweight='bold')
plt.ylabel('ROC-AUC (Top 3 Average)', fontsize=14, fontweight='bold')
plt.title('Overall Performance Across All Datasets with Different Mask Ratios', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)
plt.xticks(sorted_overall_ratios, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.6, 1.0)

# 添加数值标签
for i, (ratio, mean_val) in enumerate(zip(sorted_overall_ratios, overall_means)):
    plt.annotate(f'{mean_val:.4f}', (ratio, mean_val), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='black')

# 设置坐标轴边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overall_mask_ratio_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

# 创建所有数据集在同一图中的图表
plt.figure(figsize=(12, 8))
sorted_ratios = sorted(mask_ratios)

# 定义颜色列表以区分不同数据集
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# 定义不同的标记形状以更好地区分数据集
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']

# 为每个数据集绘制一条线
for idx, (dataset_name, dataset_data) in enumerate(results.items()):
    means = []
    stds = []
    available_ratios = []
    
    for ratio in sorted_ratios:
        if ratio in dataset_data:
            means.append(dataset_data[ratio][0])
            stds.append(dataset_data[ratio][1])
            available_ratios.append(ratio)
    
    if means:  # 只有当有数据时才绘制
        plt.errorbar(available_ratios, means, yerr=stds, 
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
plt.xlabel('Mask Ratio', fontsize=14, fontweight='bold')
plt.ylabel('ROC-AUC (Top 3 Average)', fontsize=14, fontweight='bold')
plt.title('Performance Comparison Across All Datasets with Different Mask Ratios', fontsize=16, fontweight='bold')

# 设置图例
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=11, frameon=True, 
          fancybox=True, shadow=True, framealpha=0.9)

# 设置网格
plt.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)

# 设置坐标轴刻度
plt.xticks(sorted_ratios, fontsize=12)
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
plt.savefig(os.path.join(save_dir, 'all_datasets_mask_ratio_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n图表已生成并保存到 {save_dir} 目录:")
print("1. 每个数据集的单独图表已保存为 {dataset_name}_mask_ratio_performance.png")
print("2. 所有数据集的对比图表已保存为 all_datasets_mask_ratio_performance.png")
print("3. 所有数据集均值图表已保存为 overall_mask_ratio_performance.png")