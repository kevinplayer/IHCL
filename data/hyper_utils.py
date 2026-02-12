"""
超图构建模块
用于将RDKit分子对象转换为超图表示
    x: 节点特征矩阵 [num_nodes, num_node_features]
    hyperedge_index: 超边索引矩阵 [2, num_connections], 第一行: 节点索引 第二行: 超边索引
    hyperedge_attr: 超边特征矩阵 [num_connections, num_hyperedge_features] 大小 + 类型
    batch: [num_nodes] 节点所属图的批次索引
    n_e: [batch_size] 每个图中的超边数量
    e_order: [num_edges] 每条边的阶数
    可推测：
    num_nodes: 节点数量
    num_edges: 超边数量
"""

from rdkit import Chem
import torch
import numpy as np
from itertools import combinations
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector 

import pdb

def one_of_k_encoding(x, allowable_set):
    """One-hot encoding for a value in an allowable set."""
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}.")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encoding with unknown values mapped to the last element of the allowable set."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    """Enhanced atom feature extraction."""

    # One-hot encoding for atom type (symbol)
    atom_symbol = one_of_k_encoding_unk(atom.GetSymbol(),
                                        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])

    # One-hot encoding for atom degree (number of bonds)
    atom_degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # One-hot encoding for number of hydrogen atoms
    atom_num_hydrogens = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # One-hot encoding for implicit valence
    atom_implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # One-hot encoding for chirality
    atom_chirality = one_of_k_encoding(atom.GetChiralTag(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    
    # Aromaticity (True or False)
    atom_aromatic = [atom.GetIsAromatic()]
    
    # Atomic number of the atom
    atom_atomic_num = [atom.GetAtomicNum()]

    # Return combined feature vector
    return np.array(atom_symbol + atom_degree + atom_num_hydrogens + atom_implicit_valence + atom_chirality + atom_aromatic + atom_atomic_num)


def molecule_to_hypergraph(mol: Chem.Mol):
    """
    改进后的超图生成函数，支持复杂分子结构
    :param mol: RDKit分子对象
    :return: 节点特征、超边索引、超边特征、最大环大小
    """
    # 初始化
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    
    # 节点特征
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    node_features = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    node_features2 = []
    for atom in mol.GetAtoms():
        node_features2.append(atom_to_feature_vector(atom))
    node_features2 = torch.tensor(np.array(node_features2), dtype=torch.long)
    
    # 超边索引和特征列表
    hyperedge_index_list = []  # 存储超边连接的节点索引
    hyperedge_attr_list = []   # 存储超边特征
    
    # 用于检测重复超边的集合
    hyperedge_set = set()
    
    # 辅助函数：检查超边是否已存在
    def is_duplicate_hyperedge(hyperedge_nodes):
        # 将节点列表排序并转换为元组，以便比较
        sorted_nodes = tuple(sorted(hyperedge_nodes))
        if sorted_nodes in hyperedge_set:
            return True
        hyperedge_set.add(sorted_nodes)
        return False
    
    # 记录普通边的数量
    num_bond_edges = 0
    
    # 先处理普通边（类似smi2hgraph的处理步骤）
    if len(mol.GetBonds()) > 0:
        for i, bond in enumerate(mol.GetBonds()):
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            # 添加普通边作为超边（检查是否重复）
            edge_nodes = [begin_atom, end_atom]
            if not is_duplicate_hyperedge(edge_nodes):
                hyperedge_index_list.append(edge_nodes)
                num_bond_edges += 1
                
                # 根据键类型确定超边类型: 0-单键, 1-双键, 2-三键, 3-芳香键
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    hyperedge_type = 0
                elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    hyperedge_type = 1
                elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    hyperedge_type = 2
                elif bond.GetIsAromatic():
                    hyperedge_type = 3
                else:
                    hyperedge_type = 0  # 默认为单键
                
                # 超边特征：大小（度）+ 类型
                hyperedge_attr_list.append([2, hyperedge_type])  # 普通边连接2个节点
    
    # 处理超边
    rings = mol.GetRingInfo().AtomRings()
    max_ring_size = max(len(ring) for ring in rings) if rings else 0

    # === 有环分子处理 ===
    if rings:
        # 4. 环核心超边
        for ring in rings:
            if not is_duplicate_hyperedge(list(ring)):
                hyperedge_index_list.append(list(ring))
                # 环核心超边特征：大小 + 类型(4)
                hyperedge_attr_list.append([len(ring), 4])
        
        # 5. 环扩展超边（环+邻接原子）
        for ring in rings:
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in ring]
                if neighbors:
                    extended_ring = list(ring) + neighbors
                    if not is_duplicate_hyperedge(extended_ring):
                        hyperedge_index_list.append(extended_ring)
                        # 环扩展超边特征：大小 + 类型(5)
                        hyperedge_attr_list.append([len(extended_ring), 5])
        
        # 6. 桥接超边（共享原子区域）
        shared_atoms = set()
        for r1, r2 in combinations(rings, 2):
            shared = set(r1) & set(r2)
            shared_atoms.update(shared)
        for atom_idx in shared_atoms:
            neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
            bridge_hyperedge = [atom_idx] + neighbors
            if not is_duplicate_hyperedge(bridge_hyperedge):
                hyperedge_index_list.append(bridge_hyperedge)
                # 桥接超边特征：大小 + 类型(6)
                hyperedge_attr_list.append([len(bridge_hyperedge), 6])

    # === 无环分子处理 ===
    # 只有在没有环且超边数量不足时才添加额外的超边
    non_bond_hyperedges = len(hyperedge_index_list) - num_bond_edges
    if not rings and non_bond_hyperedges < 2:
        # 7. 长链超边（示例：3原子链）
        def get_chains(mol, length=3):
            chains = []
            for path in Chem.FindAllPathsOfLengthN(mol, length, useBonds=False):
                chains.append(list(path))
            return chains
        
        chains = get_chains(mol, 3)
        for chain in chains:
            if not is_duplicate_hyperedge(chain):
                hyperedge_index_list.append(chain)
                # 长链超边特征：大小 + 类型(7)
                hyperedge_attr_list.append([len(chain), 7])
        
        # 8. 官能团超边
        frags = Chem.GetMolFrags(mol, asMols=True)  # 获取分子片段
        for frag in frags:
            frag_atoms = [atom.GetIdx() for atom in frag.GetAtoms()]
            if len(frag_atoms) > 1:  # 仅保留多原子片段
                if not is_duplicate_hyperedge(frag_atoms):
                    hyperedge_index_list.append(frag_atoms)
                    # 官能团超边特征：大小 + 类型(8)
                    hyperedge_attr_list.append([len(frag_atoms), 8])
        
        # 9. 全局拓扑超边（保底）
        # 检查除了普通边之外的超边数量
        non_bond_hyperedges = len(hyperedge_index_list) - num_bond_edges
        if non_bond_hyperedges < 2:  # 如果超边数量太少
            all_atoms = list(range(mol.GetNumAtoms()))
            if not is_duplicate_hyperedge(all_atoms):
                hyperedge_index_list.append(all_atoms)
                # 全局拓扑超边特征：大小 + 类型(9)
                hyperedge_attr_list.append([len(all_atoms), 9])

    # 构建COO格式的超边索引 [2, num_connections]
    # 第一行: 节点索引 第二行: 超边索引
    node_indices = []
    hyperedge_indices = []
    for hyperedge_idx, hyperedge in enumerate(hyperedge_index_list):
        for node_idx in hyperedge:
            node_indices.append(node_idx)
            hyperedge_indices.append(hyperedge_idx)
    
    # 构建超边索引矩阵 [2, num_connections]
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
    
    # 构建超边特征矩阵 [num_hyperedges, num_hyperedge_features]
    hyperedge_attr = torch.tensor(hyperedge_attr_list, dtype=torch.long)
    
    return node_features, node_features2, hyperedge_index, hyperedge_attr, max_ring_size