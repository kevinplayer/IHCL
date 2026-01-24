import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from torch_scatter import scatter_mean, scatter_std
import os
import sys

# 添加项目路径
sys.path.append('/home/hlt/WORK/HYPER/HEMM')

from data.moleculenet import get_moleculenet_loaders, HData
from tasks.trainer import Trainer

def compute_edge_importance(x, hyperedge_index, weight_group=0):
    """
    基于多种因素计算超边重要性得分
    综合考虑: 度数、特征差异、结构位置
    
    Args:
        x: 节点特征矩阵 [num_nodes, num_features]
        hyperedge_index: 超边索引 [2, num_connections]
        weight_group: 权重组索引
    
    Returns:
        torch.Tensor: 超边重要性得分
    """
    # 预定义权重组
    predefined_weights = {
        0: [0.25, 0.25, 0.25, 0.25],  # 均匀权重
        1: [1.0, 0.0, 0.0, 0.0],      # 仅度数
        2: [0.0, 1.0, 0.0, 0.0],      # 仅特征差异
        3: [0.0, 0.0, 1.0, 0.0],      # 仅节点度数
        4: [0.0, 0.0, 0.0, 1.0]       # 仅特征重要性
    }
    
    weights = predefined_weights.get(weight_group, [0.25, 0.25, 0.25, 0.25])
    
    device = x.device
    row, col = hyperedge_index
    
    # 1. 超边度数 (越大的超边越重要)
    edge_degree = torch.bincount(col).float()
    
    # 2. 节点特征差异 (节点间差异越大越重要)
    # 计算每个超边内节点特征的标准差
    edge_features = scatter_mean(x[row], col, dim=0)
    node_diff_std = scatter_std(x[row], col, dim=0)
    feature_diversity = torch.norm(node_diff_std, dim=1)
    
    # 3. 结构中心性 (连接更多不同超边的节点参与的超边更重要)
    # 计算节点度数
    node_degree = torch.bincount(row)
    # 将节点度数传播到超边
    edge_node_degree = scatter_mean(node_degree[row].float(), col, dim=0)
    
    # 4. 特征中心性 (基于节点特征的全局重要性)
    global_node_importance = torch.norm(x, dim=1)
    edge_feature_importance = scatter_mean(global_node_importance[row], col, dim=0)
    
    # 归一化各项指标
    if edge_degree.numel() > 0 and edge_degree.max() > 0:
        edge_degree_norm = edge_degree / (edge_degree.max() + 1e-8)
    else:
        edge_degree_norm = edge_degree
        
    if feature_diversity.numel() > 0 and feature_diversity.max() > 0:
        feature_diversity_norm = feature_diversity / (feature_diversity.max() + 1e-8)
    else:
        feature_diversity_norm = feature_diversity
        
    if edge_node_degree.numel() > 0 and edge_node_degree.max() > 0:
        edge_node_degree_norm = edge_node_degree / (edge_node_degree.max() + 1e-8)
    else:
        edge_node_degree_norm = edge_node_degree
        
    if edge_feature_importance.numel() > 0 and edge_feature_importance.max() > 0:
        edge_feature_importance_norm = edge_feature_importance / (edge_feature_importance.max() + 1e-8)
    else:
        edge_feature_importance_norm = edge_feature_importance
    
    # 组合多种因素
    importance_score = (
        weights[0] * edge_degree_norm + 
        weights[1] * feature_diversity_norm + 
        weights[2] * edge_node_degree_norm + 
        weights[3] * edge_feature_importance_norm
    )
    
    return importance_score

def analyze_hyperedge_importance(x, hyperedge_index, hyperedge_attr=None, weight_group=0):
    """
    分析超边重要性并返回详细的可解释性信息
    
    Args:
        x: 节点特征矩阵 [num_nodes, num_features]
        hyperedge_index: 超边索引 [2, num_connections]
        hyperedge_attr: 超边属性 [num_edges, attr_dim]
        weight_group: 权重组索引
    
    Returns:
        dict: 包含超边重要性分析结果的字典
    """
    # 计算超边重要性得分
    importance_scores = compute_edge_importance(x, hyperedge_index, weight_group)
    
    row, col = hyperedge_index
    
    # 获取每个超边包含的节点
    edge_to_nodes = {}
    for i in range(col.size(0)):
        edge_idx = col[i].item()
        node_idx = row[i].item()
        if edge_idx not in edge_to_nodes:
            edge_to_nodes[edge_idx] = []
        edge_to_nodes[edge_idx].append(node_idx)
    
    # 过滤掉节点数小于等于2的超边
    filtered_edge_to_nodes = {}
    for edge_idx, nodes in edge_to_nodes.items():
        if len(nodes) > 2:  # 只保留节点数大于2的超边
            filtered_edge_to_nodes[edge_idx] = nodes
    
    # 更新edge_to_nodes
    edge_to_nodes = filtered_edge_to_nodes
    
    # 重新计算重要性得分，只保留节点数大于2的超边
    valid_edges = list(edge_to_nodes.keys())
    if valid_edges:
        # 创建掩码来筛选有效的超边
        valid_edge_mask = torch.tensor([edge_idx in valid_edges for edge_idx in range(len(importance_scores))])
        if valid_edge_mask.sum() > 0 and valid_edge_mask.sum() <= len(importance_scores):
            importance_scores = importance_scores[valid_edge_mask]
        else:
            # 如果没有有效的超边，返回空结果
            results = {
                'importance_scores': torch.tensor([]),
                'edge_to_nodes': {},
                'edge_stats': {},
                'top_k_edges': torch.tensor([])
            }
            if hyperedge_attr is not None:
                results['hyperedge_attr'] = None
            return results
    else:
        # 如果没有有效的超边，返回空结果
        results = {
            'importance_scores': torch.tensor([]),
            'edge_to_nodes': {},
            'edge_stats': {},
            'top_k_edges': torch.tensor([])
        }
        if hyperedge_attr is not None:
            results['hyperedge_attr'] = None
        return results
    
    # 重新映射edge_to_nodes的键以匹配新的importance_scores索引
    remapped_edge_to_nodes = {}
    edge_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_edges)}
    for old_idx, nodes in edge_to_nodes.items():
        new_idx = edge_mapping[old_idx]
        remapped_edge_to_nodes[new_idx] = nodes
    edge_to_nodes = remapped_edge_to_nodes
    
    # 计算每个超边的其他统计信息
    edge_stats = {}
    edge_degree = torch.bincount(col).float()
    node_diff_std = scatter_std(x[row], col, dim=0)
    feature_diversity = torch.norm(node_diff_std, dim=1)
    
    for old_edge_idx in valid_edges:
        new_edge_idx = edge_mapping[old_edge_idx]
        if old_edge_idx < edge_degree.size(0):
            edge_stats[new_edge_idx] = {
                'degree': edge_degree[old_edge_idx].item(),
                'feature_diversity': feature_diversity[old_edge_idx].item() if old_edge_idx < feature_diversity.size(0) else 0,
                'node_count': len(edge_to_nodes[new_edge_idx]),
                'nodes': edge_to_nodes[new_edge_idx]
            }
    
    # 整理结果
    results = {
        'importance_scores': importance_scores,
        'edge_to_nodes': edge_to_nodes,
        'edge_stats': edge_stats,
        'top_k_edges': torch.topk(importance_scores, min(10, importance_scores.size(0))).indices if importance_scores.size(0) > 0 else torch.tensor([])
    }
    
    # 如果有超边属性，也包含进来
    if hyperedge_attr is not None:
        # 筛选有效的超边属性
        if valid_edges and hyperedge_attr is not None and len(hyperedge_attr) >= len(valid_edges):
            valid_hyperedge_attr = hyperedge_attr[torch.tensor(valid_edges)]
            results['hyperedge_attr'] = valid_hyperedge_attr
        else:
            results['hyperedge_attr'] = None
    
    return results

def visualize_molecule_importance(smiles, analysis_result, save_path=None):
    """
    可视化分子超边重要性
    使用RDKit的SimilarityMaps进行可视化
    
    Args:
        smiles: 分子的SMILES表示
        analysis_result: 分析结果
        save_path: 保存路径
    """
    try:
        # 创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法解析SMILES: {smiles}")
            return None
            
        # 获取重要性得分
        importance_scores = analysis_result['importance_scores']
        edge_to_nodes = analysis_result['edge_to_nodes']
        
        top_k = min(3000, len(importance_scores))
        if top_k > 0:
            top_edges = torch.topk(importance_scores, top_k).indices
        else:
            top_edges = torch.tensor([])
        
        # 为每个原子分配贡献值，未被任何超边覆盖的原子赋值为0
        num_atoms = mol.GetNumAtoms()
        atom_contribs = [0.0] * num_atoms  # 初始化所有原子贡献值为0
        
        # 将非负分数转换为有正有负的范围以增强可视化对比度
        if len(importance_scores) > 0 and (importance_scores.max() - importance_scores.min()) > 1e-8:
            # 归一化到[0,1]范围
            normalized_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
            # 将范围从[0,1]平移到[-0.5, 0.5]以获得正负值
            normalized_scores = normalized_scores - 0.5
            # 可选：放大差异以增强可视化效果
            temperature = 0.1
            normalized_scores = normalized_scores / temperature
        else:
            normalized_scores = torch.zeros_like(importance_scores) if len(importance_scores) > 0 else importance_scores

        # 遍历所有重要超边，为其中的原子分配最高重要性得分
        for i, edge_idx in enumerate(top_edges):
            edge_idx_val = edge_idx.item()
            if edge_idx_val in edge_to_nodes:
                nodes = edge_to_nodes[edge_idx_val]
                # 使用归一化后的得分作为贡献值
                normalized_intensity = normalized_scores[edge_idx_val].item()
                for node_idx in nodes:
                    # 只有当当前得分更高时才更新
                    if normalized_intensity > atom_contribs[node_idx]:
                        atom_contribs[node_idx] = normalized_intensity
        
        # 对原子贡献值进行平移，使其有正有负
        if len(atom_contribs) > 0:
            atom_contribs_array = np.array(atom_contribs)
            # 将贡献值平移到以0为中心的范围
            mean_contrib = np.mean(atom_contribs_array)
            atom_contribs_array = atom_contribs_array - mean_contrib
            # 更新atom_contribs列表
            atom_contribs = atom_contribs_array.tolist()
            
        # 输出原子贡献值
        print("原子贡献值:")
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_symbol = atom.GetSymbol()
            print(f"  {atom_symbol}{i}: {atom_contribs[i]:.4f}")
        
        # 使用RDKit的SimilarityMaps进行可视化
        from rdkit.Chem.Draw import SimilarityMaps
        
        # 创建贡献值列表，格式为[(weight1, radius1), (weight2, radius2), ...]
        # 这里我们只关心权重，半径设置为默认值
        contribs_for_similarity = [(atom_contribs[i], 0) for i in range(num_atoms)]
        
        # 生成相似性地图
        fig = SimilarityMaps.GetSimilarityMapFromWeights(
            mol, 
            atom_contribs,  # 直接传递原子贡献值列表
            contourLines=20,  # 添加等高线
            size=(600, 600),  # 图像尺寸
            alpha=0.7,  # 增加透明度使叠加效果更好
        )
        
        # 加粗分子图中的线条
        if fig is not None:
            # 获取分子绘图的轴
            ax = fig.get_axes()[0] if fig.get_axes() else None
            if ax:
                # 遍历所有线条并加粗
                for line in ax.lines:
                    line.set_linewidth(2.0)
        
        # 添加标题
        plt.title(f"Molecule Importance Analysis\n"
                 f"SMILES: {smiles[:50]+'...' if len(smiles) > 50 else smiles}", 
                 fontsize=32)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        return fig
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        return None

def extract_single_molecule_data(batch, mol_idx_in_batch):
    """
    从批次数据中提取单个分子的数据
    
    Args:
        batch: 批次数据
        mol_idx_in_batch: 分子在批次中的索引
        
    Returns:
        tuple: (hyper_x, hyperedge_index, smiles)
    """
    # 获取该分子涉及的节点和超边范围
    batch_mask = (batch.batch == mol_idx_in_batch)
    node_indices = torch.where(batch_mask)[0]
    
    if len(node_indices) == 0:
        return None, None, None
    
    # 提取超图数据
    hyper_x = batch.hyper_x
    hyperedge_index0 = batch.hyperedge_index0
    hyperedge_index1 = batch.hyperedge_index1
    smiles = batch.smi[mol_idx_in_batch] if isinstance(batch.smi, list) else batch.smi
    
    # 获取该分子的超边
    node_start, node_end = node_indices.min().item(), node_indices.max().item() + 1
    
    # 找到与这些节点相关的超边
    edge_mask = (hyperedge_index0 >= node_start) & (hyperedge_index0 < node_end)
    if edge_mask.sum() == 0:
        return None, None, None
    
    mol_hyperedge_index0 = hyperedge_index0[edge_mask] - node_start  # 重新编号
    mol_hyperedge_index1 = hyperedge_index1[edge_mask]
    
    # 获取唯一的超边索引并重新编号
    unique_edges = torch.unique(mol_hyperedge_index1)
    edge_mapping = torch.zeros(hyperedge_index1.max().item() + 1, dtype=torch.long)
    edge_mapping[unique_edges] = torch.arange(len(unique_edges))
    mol_hyperedge_index1 = edge_mapping[mol_hyperedge_index1]
    
    # 提取该分子的节点特征
    mol_hyper_x = hyper_x[node_start:node_end]
    
    # 构建超边索引
    mol_hyperedge_index = torch.stack([mol_hyperedge_index0, mol_hyperedge_index1], dim=0)
    
    return mol_hyper_x, mol_hyperedge_index, smiles

def main():
    """
    主函数：从Moleculenet数据集中加载数据并分析超边重要性
    """
    save_dir = './hyperedge_importance_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 数据集配置
    dataset_names = ['hiv', 'bace', 'bbbp', 'tox21', 'sider', 'clintox', 'toxcast', 'muv']
    data_root = './data/MoleculeNet'
    batch_size = 32
    max_molecules = 32  # 控制要处理的分子数量
    
    # 对每个数据集进行处理
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        # 为当前数据集创建子目录
        dataset_save_dir = os.path.join(save_dir, dataset_name)
        if not os.path.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        
        print(f"加载 {dataset_name} 数据集...")
        
        try:
            # 加载数据
            train_loader, val_loader, test_loader, num_features, num_tasks = get_moleculenet_loaders(
                dataset_name=dataset_name,
                root=data_root,
                batch_size=batch_size
            )
            
            print(f"数据集加载成功:")
            print(f"  节点特征数: {num_features}")
            print(f"  任务数: {num_tasks}")
            
            # 获取一个批次的数据进行分析
            print("\n正在获取测试数据...")
            # 添加计数器来跟踪已处理的分子数量
            molecules_processed = 0
            
            for batch_idx, batch in enumerate(test_loader):
                print(f"处理批次 {batch_idx}...")
                
                # 获取批次大小
                batch_size_actual = batch.batch.max().item() + 1 if len(batch.batch) > 0 else 1
                
                # 处理批次中的每个分子
                for mol_idx in range(batch_size_actual):
                    if molecules_processed >= max_molecules:
                        break
                        
                    hyper_x, hyperedge_index, smiles = extract_single_molecule_data(batch, mol_idx)
                    
                    if hyper_x is None or hyperedge_index is None:
                        print(f"无法提取批次 {batch_idx} 中分子 {mol_idx} 的数据，跳过...")
                        continue
                        
                    print(f"处理分子 {molecules_processed + 1}:")
                    print(f"  分子 SMILES: {smiles}")
                    print(f"  节点数: {hyper_x.size(0)}")
                    print(f"  超边连接数: {hyperedge_index.size(1)}")
                    print(f"  唯一超边数: {torch.unique(hyperedge_index[1]).size(0)}")
                    
                    # 分析超边重要性
                    print("\n  正在计算超边重要性...")
                    analysis_result = analyze_hyperedge_importance(hyper_x, hyperedge_index, weight_group=0)
                    
                    # 检查是否有有效的超边
                    if len(analysis_result['importance_scores']) == 0:
                        print("  没有有效的超边（节点数大于2），跳过可视化...")
                        molecules_processed += 1
                        print("-" * 50)
                        continue
                    
                    # 打印一些分析结果
                    importance_scores = analysis_result['importance_scores']
                    print(f"\n  超边重要性得分:")
                    for i in range(min(10, len(importance_scores))):  # 只打印前10个以免输出过多
                        print(f"    超边 {i}: {importance_scores[i]:.4f}")
                    
                    # 可视化结果
                    print("\n  正在生成可视化...")
                    save_path = f"hyperedge_importance_{dataset_name}_molecule_{molecules_processed}.png"
                    save_path = os.path.join(dataset_save_dir, save_path)
                    fig = visualize_molecule_importance(smiles, analysis_result, save_path=save_path)
                    
                    if fig:
                        print(f"  可视化已完成，结果保存为 {save_path}")
                    else:
                        print("  可视化失败")
                    
                    molecules_processed += 1
                    print("-" * 50)
                    
                    if molecules_processed >= max_molecules:
                        break
                
                if molecules_processed >= max_molecules:
                    break
            
            print(f"数据集 {dataset_name} 处理完成，共处理了 {molecules_processed} 个分子。")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出现错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有数据集处理完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    
    # python ./visulaze_hyperedge_importance.py > visualization_log.log 2>&1