import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_mean, scatter_std

import pdb

def drop_hyperedges(hyperedge_index, hyperedge_attr=None, n_e=None, e_order=None, drop_ratio=0.15):
    """随机丢弃若干超边，返回扰动后的 hyperedge_index 和对应的 hyperedge_attr（如果有）。"""
    # 获取输入张量的设备
    device = hyperedge_index.device
    
    # 获取总的超边数和图的数量
    E = hyperedge_index[1].max().item() + 1 if hyperedge_index.numel() > 0 else 0
    
    if E == 0:
        # 没有超边的情况
        if n_e is not None:
            new_n_e = torch.zeros_like(n_e)
        else:
            new_n_e = torch.tensor([0], device=device)
        return hyperedge_index, hyperedge_attr, new_n_e, e_order
    
    # 为每个超边生成随机丢弃标记
    edge_mask = torch.rand(E, device=device) > drop_ratio  # True表示保留
    
    # 创建连接级别的mask
    keep_connections = edge_mask[hyperedge_index[1]]
    
    # 筛选保留的连接
    new_idx = torch.where(keep_connections)[0]
    perturbed_index = hyperedge_index[:, new_idx]

    # 重新映射超边索引以保证连续性
    if perturbed_index.shape[1] > 0:
        unique_edges = torch.unique(perturbed_index[1])
        # 创建映射表
        edge_mapping = torch.zeros(E, dtype=torch.long, device=device)
        edge_mapping[unique_edges] = torch.arange(len(unique_edges), device=device)
        # 应用映射
        perturbed_index[1] = edge_mapping[perturbed_index[1]]
        
        # 更新n_e（每个图的超边数量）
        if n_e is not None:
            # 计算每个图中保留的超边数量
            new_n_e = torch.zeros_like(n_e)
            edge_count = 0
            for i in range(len(n_e)):
                original_edges_in_graph = n_e[i].item()
                if original_edges_in_graph > 0:
                    # 计算在当前图中保留的超边数
                    graph_edges_mask = (unique_edges >= edge_count) & (unique_edges < edge_count + original_edges_in_graph)
                    new_n_e[i] = graph_edges_mask.sum().item()
                    edge_count += original_edges_in_graph
                else:
                    new_n_e[i] = 0
        else:
            new_n_e = torch.tensor([len(unique_edges)], device=device)
            
        # 更新e_order
        if e_order is not None:
            new_e_order = e_order[edge_mask]
        else:
            new_e_order = None
    else:
        if n_e is not None:
            new_n_e = torch.zeros_like(n_e)
        else:
            new_n_e = torch.tensor([0], device=device)
        new_e_order = torch.tensor([], dtype=torch.long, device=device) if e_order is not None else None
    
    if hyperedge_attr is not None:
        # 直接根据保留的超边索引提取对应的属性
        perturbed_attr = hyperedge_attr[edge_mask]
        return perturbed_index, perturbed_attr, new_n_e, new_e_order
    
    return perturbed_index, None, new_n_e, new_e_order

# ---------- 按图分别mask超边 ----------
def drop_hyperedges_per_graph(hyperedge_index, hyperedge_attr=None, n_e=None, e_order=None, drop_ratio=0.15):
    """按图分别丢弃超边，每个图独立进行mask操作"""
    device = hyperedge_index.device
    
    if hyperedge_index.numel() == 0 or n_e is None or n_e.sum() == 0:
        # 没有超边或没有图信息的情况
        if n_e is not None:
            new_n_e = torch.zeros_like(n_e)
        else:
            new_n_e = torch.tensor([0], device=device)
        return hyperedge_index, hyperedge_attr, new_n_e, e_order
    
    # 为每个图分别处理
    edge_count = 0
    new_edges = []
    new_attrs = [] if hyperedge_attr is not None else None
    new_orders = [] if e_order is not None else None
    new_n_e_list = []
    
    for i, graph_edge_count in enumerate(n_e):
        graph_edge_count = graph_edge_count.item()
        if graph_edge_count ==  0:
            new_n_e_list.append(0)
            continue
            
        # 提取当前图的超边
        graph_mask = (hyperedge_index[1] >= edge_count) & (hyperedge_index[1] < edge_count + graph_edge_count)
        graph_edges = hyperedge_index[:, graph_mask]
        
        if graph_edges.shape[1] == 0:
            new_n_e_list.append(0)
            edge_count += graph_edge_count
            continue
        
        # 为当前图的超边重新编号（从0开始）
        graph_edges_local = graph_edges.clone()
        graph_edges_local[1] = graph_edges_local[1] - edge_count
        
        # 计算当前图要保留的超边数
        keep_count = max(1, int(graph_edge_count * (1 - drop_ratio)))
        
        # 随机选择要保留的超边
        if keep_count >= graph_edge_count:
            # 保留所有超边
            kept_edges = graph_edges
            kept_edge_indices = torch.arange(graph_edge_count, device=device)
        else:
            # 随机选择要保留的超边
            perm = torch.randperm(graph_edge_count, device=device)
            kept_edge_indices = perm[:keep_count].sort().values
            
            # 正确创建连接级别的mask
            edge_mask = torch.zeros(graph_edge_count, dtype=torch.bool, device=device)
            edge_mask[kept_edge_indices] = True
            
            # 使用连接级别的mask来筛选超边
            connection_mask = edge_mask[graph_edges_local[1]]
            kept_edges = graph_edges[:, connection_mask]
            
            # 重新编号保留的超边
            if kept_edges.shape[1] > 0:
                unique_edges = torch.unique(kept_edges[1])
                edge_mapping = torch.zeros(edge_count + graph_edge_count, dtype=torch.long, device=device)
                edge_mapping[unique_edges] = torch.arange(len(unique_edges), device=device)
                kept_edges[1] = edge_mapping[kept_edges[1]]
        
        # 添加偏移量以恢复全局编号
        if new_edges:
            offset = new_edges[-1][1].max().item() + 1 if new_edges[-1].numel() > 0 else 0
        else:
            offset = 0
            
        if kept_edges.numel() > 0:
            kept_edges[1] = kept_edges[1] + offset
        
        new_edges.append(kept_edges)
        new_n_e_list.append(kept_edges.shape[1] if kept_edges.numel() > 0 else 0)
        
        # 处理属性和顺序信息
        if hyperedge_attr is not None:
            # 根据保留的超边索引提取对应的属性
            if kept_edge_indices is not None and hyperedge_attr is not None:
                attr_start_idx = edge_count
                attr_end_idx = edge_count + graph_edge_count
                graph_attrs = hyperedge_attr[attr_start_idx:attr_end_idx]
                if graph_attrs.shape[0] > 0 and kept_edge_indices.shape[0] <= graph_attrs.shape[0]:
                    new_attrs.append(graph_attrs[kept_edge_indices])
                else:
                    new_attrs.append(torch.empty((0, *hyperedge_attr.shape[1:]), device=device))
            
        if e_order is not None:
            # 根据保留的超边索引提取对应的阶数
            if kept_edge_indices is not None:
                order_start_idx = edge_count
                order_end_idx = edge_count + graph_edge_count
                graph_orders = e_order[order_start_idx:order_end_idx]
                if graph_orders.shape[0] > 0 and kept_edge_indices.shape[0] <= graph_orders.shape[0]:
                    new_orders.append(graph_orders[kept_edge_indices])
                else:
                    new_orders.append(torch.empty(0, dtype=torch.long, device=device))
            
        edge_count += graph_edge_count
    
    # 合并所有图的结果
    if new_edges and any(edge.numel() > 0 for edge in new_edges):
        valid_edges = [edge for edge in new_edges if edge.numel() > 0]
        if valid_edges:
            perturbed_index = torch.cat(valid_edges, dim=1)
        else:
            perturbed_index = torch.empty((2, 0), dtype=torch.long, device=device)
        new_n_e = torch.tensor(new_n_e_list, device=device)
        
        if hyperedge_attr is not None and new_attrs:
            valid_attrs = [attr for attr in new_attrs if attr.shape[0] > 0]
            perturbed_attr = torch.cat(valid_attrs, dim=0) if valid_attrs else None
        else:
            perturbed_attr = None
            
        if e_order is not None and new_orders:
            valid_orders = [order for order in new_orders if order.shape[0] > 0]
            new_e_order = torch.cat(valid_orders, dim=0) if valid_orders else None
        else:
            new_e_order = None
    else:
        perturbed_index = torch.empty((2, 0), dtype=torch.long, device=device)
        perturbed_attr = None
        new_e_order = None
        new_n_e = torch.tensor(new_n_e_list, device=device) if new_n_e_list else torch.tensor([0], device=device)
    
    return perturbed_index, perturbed_attr, new_n_e, new_e_order



# ---------- 可学习打分器（对偶图） ----------
class EdgeGNNScore(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden)
        self.conv2 = nn.Linear(hidden, 1)

    def forward(self, x, hyperedge_index):
        row, col = hyperedge_index
        # 超边特征 = 平均所属节点特征
        edge_feat = scatter_mean(x[row], col, dim=0)   # [|E|, d]
        h = torch.relu(self.conv1(edge_feat))
        score = torch.sigmoid(self.conv2(h)).squeeze(-1)  # [|E|]
        return score


# ---------- 超边重要性评分函数 ----------
def compute_edge_importance(x, hyperedge_index, weight_group=0):
    """
    基于多种因素计算超边重要性得分
    综合考虑: 度数、特征差异、结构位置
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
    
    # 计算每个超边内节点的重要性
    node_importance_in_edge = {}
    global_node_importance = torch.norm(x, dim=1)
    
    for edge_idx, nodes in edge_to_nodes.items():
        node_importance_in_edge[edge_idx] = {}
        for node_idx in nodes:
            node_importance_in_edge[edge_idx][node_idx] = global_node_importance[node_idx].item()
    
    # 计算每个超边的其他统计信息
    edge_stats = {}
    edge_degree = torch.bincount(col).float()
    node_diff_std = scatter_std(x[row], col, dim=0)
    feature_diversity = torch.norm(node_diff_std, dim=1)
    
    for edge_idx in edge_to_nodes.keys():
        edge_stats[edge_idx] = {
            'degree': edge_degree[edge_idx].item() if edge_idx < edge_degree.size(0) else 0,
            'feature_diversity': feature_diversity[edge_idx].item() if edge_idx < feature_diversity.size(0) else 0,
            'node_count': len(edge_to_nodes[edge_idx]),
            'nodes': edge_to_nodes[edge_idx]
        }
    
    # 整理结果
    results = {
        'importance_scores': importance_scores,
        'edge_to_nodes': edge_to_nodes,
        'node_importance_in_edge': node_importance_in_edge,
        'edge_stats': edge_stats,
        'top_k_edges': torch.topk(importance_scores, min(10, importance_scores.size(0))).indices
    }
    
    # 如果有超边属性，也包含进来
    if hyperedge_attr is not None:
        results['hyperedge_attr'] = hyperedge_attr
    
    return results

# ---------- mask超边（基于重要性评分）----------
class MaskHyper:
    def __init__(self, mask_ratio=0.4, total_epochs=100, learnable=False, in_dim=None, weight_group=0):
        self.p = mask_ratio
        self.T = 50
        self.learnable = learnable
        self.scorer = EdgeGNNScore(in_dim, 64).to(self._device()) if learnable else None
        self.weight_group = weight_group

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 预定义：综合超边重要性 ----
    def _predefined_score(self, hyperedge_index, x=None):
        if hyperedge_index.numel() == 0:
            return torch.tensor([], device=hyperedge_index.device)
            
        E = hyperedge_index[1].max().item() + 1
        
        if x is not None and x.numel() > 0:
            # 使用综合重要性评分
            importance_score = compute_edge_importance(x, hyperedge_index, self.weight_group)
            # 添加一些随机性以增加探索
            noise = torch.randn_like(importance_score) * 0.1
            score = torch.clamp(importance_score + noise, 0, 1)
        else:
            # 退化到简单的度数评分
            degree = torch.bincount(hyperedge_index[1]).float()
            if degree.numel() == 0:
                return torch.tensor([], device=hyperedge_index.device)
            max_degree = degree.max() if degree.numel() > 0 else torch.tensor(1.0, device=degree.device)
            score = degree / (max_degree + 1e-8)
            
        return score

    def mask_edges(self, hyperedge_index, hyperedge_attr=None, x=None, n_e=None, e_order=None, epoch=0):
        device = hyperedge_index.device
        if hyperedge_index.numel() == 0:
            # 没有超边的情况
            if n_e is not None:
                new_n_e = torch.zeros_like(n_e)
            else:
                new_n_e = torch.tensor([0], device=device)
            new_e_order = torch.tensor([], dtype=torch.long, device=device) if e_order is not None else None
            return hyperedge_index, hyperedge_attr, new_n_e, new_e_order, torch.zeros(0, dtype=torch.bool, device=device)
                
        E = hyperedge_index[1].max().item() + 1

        # 1. 打分
        if self.learnable and self.scorer is not None:
            score = self.scorer(x, hyperedge_index)
        else:
            score = self._predefined_score(hyperedge_index, x)
            
        if score.numel() == 0:
            # 没有得分的情况
            if n_e is not None:
                new_n_e = torch.zeros_like(n_e)
            else:
                new_n_e = torch.tensor([0], device=device)
            new_e_order = torch.tensor([], dtype=torch.long, device=device) if e_order is not None else None
            return hyperedge_index, hyperedge_attr, new_n_e, new_e_order, torch.zeros(0, dtype=torch.bool, device=device)

        # 2. 当前 epoch 要 mask 的数量
        t = epoch + 1
        K = int(self.p * E * min(np.sqrt(t / self.T), 1.0))
        K = max(1, min(K, E))  # 确保K不超过总边数

        # 3. easy-to-hard：挑分数最高的 K 条掩掉 / 挑分数最低的 K 条
        if K >= score.size(0):
            topk_idx = torch.arange(score.size(0), device=device)
        else:
            _, topk_idx = torch.topk(score, k=K, largest=False, sorted=True)
            # _, topk_idx = torch.topk(score, k=K, largest=True, sorted=True) # 一般用false 1
        mask_flag = torch.zeros(E, dtype=torch.bool, device=device)
        mask_flag[topk_idx] = True

        # 4. 保留边
        edge_keep_mask = ~mask_flag  # 超边级别的保留mask
        connection_keep_mask = edge_keep_mask[hyperedge_index[1]]  # 连接级别的保留mask
        keep_idx = torch.where(connection_keep_mask)[0]
        
        perturbed_index = hyperedge_index[:, keep_idx]

        # 重新映射超边索引以保证连续性
        if perturbed_index.shape[1] > 0:
            unique_edges = torch.unique(perturbed_index[1])
            # 创建映射表
            edge_mapping = torch.zeros(E, dtype=torch.long, device=device)
            edge_mapping[unique_edges] = torch.arange(len(unique_edges), device=device)
            # 应用映射
            perturbed_index[1] = edge_mapping[perturbed_index[1]]
            
            # 更新n_e（每个图的超边数量）
            if n_e is not None:
                # 计算每个图中保留的超边数量
                new_n_e = torch.zeros_like(n_e)
                edge_count = 0
                for i in range(len(n_e)):
                    original_edges_in_graph = n_e[i].item()
                    if original_edges_in_graph > 0:
                        # 计算在当前图中保留的超边数
                        graph_edges_mask = (unique_edges >= edge_count) & (unique_edges < edge_count + original_edges_in_graph)
                        new_n_e[i] = graph_edges_mask.sum().item()
                        edge_count += original_edges_in_graph
                    else:
                        new_n_e[i] = 0
            else:
                new_n_e = torch.tensor([len(unique_edges)], device=device)
                
            # 更新e_order
            if e_order is not None:
                new_e_order = e_order[edge_keep_mask]
            else:
                new_e_order = None
        else:
            if n_e is not None:
                new_n_e = torch.zeros_like(n_e)
            else:
                new_n_e = torch.tensor([0], device=device)
            new_e_order = torch.tensor([], dtype=torch.long, device=device) if e_order is not None else None

        if hyperedge_attr is not None:
            # 直接根据保留的超边提取对应的属性
            perturbed_attr = hyperedge_attr[edge_keep_mask]
            return perturbed_index, perturbed_attr, new_n_e, new_e_order, mask_flag
            
        return perturbed_index, None, new_n_e, new_e_order, mask_flag
    
    def mask_edges_per_graph(self, hyperedge_index, hyperedge_attr=None, x=None, n_e=None, e_order=None, epoch=0):
        """按图分别mask超边，每个图独立进行重要性评估和mask操作"""
        device = hyperedge_index.device
        if hyperedge_index.numel() == 0 or n_e is None or n_e.sum() == 0:
            # 没有超边或没有图信息的情况
            if n_e is not None:
                new_n_e = torch.zeros_like(n_e)
            else:
                new_n_e = torch.tensor([0], device=device)
            new_e_order = torch.tensor([], dtype=torch.long, device=device) if e_order is not None else None
            return hyperedge_index, hyperedge_attr, new_n_e, new_e_order, torch.zeros(0, dtype=torch.bool, device=device)
        
        # 为每个图分别处理
        edge_count = 0
        new_edges = []
        new_attrs = [] if hyperedge_attr is not None else None
        new_orders = [] if e_order is not None else None
        new_n_e_list = []
        all_mask_flags = []
        
        for i, graph_edge_count in enumerate(n_e):
            graph_edge_count = graph_edge_count.item()
            if graph_edge_count == 0:
                new_n_e_list.append(0)
                all_mask_flags.append(torch.zeros(0, dtype=torch.bool, device=device))
                continue
                
            # 提取当前图的超边
            graph_mask = (hyperedge_index[1] >= edge_count) & (hyperedge_index[1] < edge_count + graph_edge_count)
            graph_edges = hyperedge_index[:, graph_mask]
            
            if graph_edges.shape[1] == 0:
                new_n_e_list.append(0)
                edge_count += graph_edge_count
                # 添加空的mask标志
                all_mask_flags.append(torch.zeros(0, dtype=torch.bool, device=device))
                continue
            
            # 为当前图的超边重新编号（从0开始）
            graph_edges_local = graph_edges.clone()
            graph_edges_local[1] = graph_edges_local[1] - edge_count
            
            # 提取当前图的节点特征（如果需要）
            if x is not None:
                # 获取当前图涉及的节点索引
                graph_nodes = torch.unique(graph_edges[0])
                graph_x = x[graph_nodes]
                # 重新编号节点索引以匹配当前图的节点特征
                node_mapping = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                node_mapping[graph_nodes] = torch.arange(len(graph_nodes), device=device)
                local_edges = graph_edges_local.clone()
                local_edges[0] = node_mapping[local_edges[0]]
            else:
                graph_x = None
                local_edges = graph_edges_local
            
            # 计算当前epoch要mask的超边数
            t = epoch + 1
            K = int(self.p * graph_edge_count * min(np.sqrt(t / self.T), 1.0))
            K = max(1, min(K, graph_edge_count))  # 确保K不超过总边数
            
            # 1. 打分
            if self.learnable and self.scorer is not None and graph_x is not None:
                score = self.scorer(graph_x, local_edges)
            else:
                score = self._predefined_score(local_edges, graph_x)
                
            if score.numel() == 0:
                # 没有得分的情况，保留所有超边
                kept_edges = graph_edges
                mask_flag = torch.zeros(graph_edge_count, dtype=torch.bool, device=device)
            else:
                # 3. easy-to-hard：挑分数最低的 K 条掩掉
                if K >= score.size(0):
                    topk_idx = torch.arange(score.size(0), device=device)
                else:
                    _, topk_idx = torch.topk(score, k=K, largest=False, sorted=True)
                mask_flag = torch.zeros(graph_edge_count, dtype=torch.bool, device=device)
                mask_flag[topk_idx] = True

                # 保留边 - 使用正确的连接级别mask
                edge_keep_mask = ~mask_flag
                connection_keep_mask = edge_keep_mask[graph_edges_local[1]]  # 连接级别的保留mask
                kept_edges = graph_edges[:, connection_keep_mask]  # 使用graph_edges而不是graph_edges_local
                
                # 重新编号保留的超边以保持连续性
                if kept_edges.shape[1] > 0:
                    unique_edges = torch.unique(kept_edges[1])
                    edge_mapping = torch.zeros(edge_count + graph_edge_count, dtype=torch.long, device=device)
                    edge_mapping[unique_edges] = torch.arange(len(unique_edges), device=device)
                    kept_edges[1] = edge_mapping[kept_edges[1]]
            
            # 添加偏移量以恢复全局编号
            if new_edges:
                offset = new_edges[-1][1].max().item() + 1 if new_edges[-1].numel() > 0 else 0
            else:
                offset = 0
            if kept_edges.numel() > 0:
                kept_edges[1] = kept_edges[1] + offset
            
            new_edges.append(kept_edges)
            new_n_e_list.append(kept_edges.shape[1] if kept_edges.numel() > 0 else 0)
            all_mask_flags.append(mask_flag)
            
            # 处理属性和顺序信息
            if hyperedge_attr is not None:
                graph_attr = hyperedge_attr[edge_count:edge_count+graph_edge_count]
                # 创建连接级别的mask来筛选属性
                if kept_edges.numel() > 0:
                    # 通过比较原始超边索引和保留的超边索引来确定哪些属性需要保留
                    original_edges_in_graph = torch.arange(edge_count, edge_count + graph_edge_count, device=device)
                    kept_global_edges = torch.unique(kept_edges[1])
                    attr_keep_mask = torch.zeros(graph_edge_count, dtype=torch.bool, device=device)
                    for global_edge_idx in kept_global_edges:
                        local_edge_idx = global_edge_idx - edge_count
                        if 0 <= local_edge_idx < graph_edge_count:
                            attr_keep_mask[local_edge_idx] = True
                    new_attrs.append(graph_attr[attr_keep_mask])
                else:
                    new_attrs.append(torch.empty((0, *graph_attr.shape[1:]), device=device))
                
            if e_order is not None:
                graph_order = e_order[edge_count:edge_count+graph_edge_count]
                # 创建连接级别的mask来筛选阶数
                if kept_edges.numel() > 0:
                    # 通过比较原始超边索引和保留的超边索引来确定哪些阶数需要保留
                    original_edges_in_graph = torch.arange(edge_count, edge_count + graph_edge_count, device=device)
                    kept_global_edges = torch.unique(kept_edges[1])
                    order_keep_mask = torch.zeros(graph_edge_count, dtype=torch.bool, device=device)
                    for global_edge_idx in kept_global_edges:
                        local_edge_idx = global_edge_idx - edge_count
                        if 0 <= local_edge_idx < graph_edge_count:
                            order_keep_mask[local_edge_idx] = True
                    new_orders.append(graph_order[order_keep_mask])
                else:
                    new_orders.append(torch.empty(0, dtype=torch.long, device=device))
                
            edge_count += graph_edge_count
        
        # 合并所有图的结果
        if new_edges and any(edge.numel() > 0 for edge in new_edges):
            valid_edges = [edge for edge in new_edges if edge.numel() > 0]
            perturbed_index = torch.cat(valid_edges, dim=1)
            new_n_e = torch.tensor(new_n_e_list, device=device)
            
            if hyperedge_attr is not None and new_attrs:
                valid_attrs = [attr for attr in new_attrs if attr.shape[0] > 0]
                perturbed_attr = torch.cat(valid_attrs, dim=0) if valid_attrs else None
            else:
                perturbed_attr = None
                
            if e_order is not None and new_orders:
                valid_orders = [order for order in new_orders if order.shape[0] > 0]
                new_e_order = torch.cat(valid_orders, dim=0) if valid_orders else None
            else:
                new_e_order = None
                
            # 合并mask标志
            mask_flag = torch.cat(all_mask_flags, dim=0) if all_mask_flags else torch.zeros(0, dtype=torch.bool, device=device)
        else:
            perturbed_index = torch.empty((2, 0), dtype=torch.long, device=device)
            perturbed_attr = None
            new_e_order = None
            new_n_e = torch.tensor(new_n_e_list, device=device) if new_n_e_list else torch.tensor([0], device=device)
            mask_flag = torch.zeros(0, dtype=torch.bool, device=device)
        
        return perturbed_index, perturbed_attr, new_n_e, new_e_order, mask_flag