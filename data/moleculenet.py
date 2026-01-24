import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from data.hyper_utils import molecule_to_hypergraph
from data.splitters import scaffold_split
# from hyper_utils import molecule_to_hypergraph
# from splitters import scaffold_split
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import os.path as osp
import os

# 禁用RDKit的所有警告
RDLogger.DisableLog('rdApp.*')

import pdb

class HData(Data):
    """ PyG data class for molecular hypergraphs
    """
    def __init__(self, hyper_x=None, edge_index=None, edge_attr=None, y=None, pos=None,
                 hyperedge_index0=None, hyperedge_index1=None, n_e=None, smi=None, **kwargs):
        super().__init__(hyper_x, edge_index, edge_attr, y, pos, **kwargs)
        self.hyperedge_index0 = hyperedge_index0
        self.hyperedge_index1 = hyperedge_index1
        self.n_e = n_e
        self.smi = smi

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'hyperedge_index0':
            return self.hyper_x.size(0)
        if key == 'hyperedge_index1':
            return self.n_e
        else:
            return super().__inc__(key, value, *args, **kwargs)

class HyperMoleculeNet(InMemoryDataset):
    """MoleculeNet数据集的超图版本"""
    
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'{self.name}_graph_hypergraph.pt']
    
    def download(self):
        pass  # 数据已经在MoleculeNet中下载
    
    def process(self):
        # 加载原始MoleculeNet数据集
        mol_dataset = MoleculeNet(root=self.root, name=self.name)
        
        # 处理所有分子为超图
        data_list = process_molecule_data(mol_dataset)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def smiles_to_mol(smiles):
    """
    将SMILES字符串转换为RDKit分子对象
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            pass
    return mol

def process_molecule_data(mol_dataset):
    """
    将SMILES列表处理为超图数据
    """
    data_list = []
    
    for i, data in enumerate(tqdm(mol_dataset, 
                                desc="Processing ...", 
                                total=len(mol_dataset))):
        try:
            # 将SMILES转换为分子对象
            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                continue

            # 使用hyper_utils中的函数将分子转换为超图
            node_features, node_features2, hyperedge_index, hyperedge_attr, max_ring_size = molecule_to_hypergraph(mol)
            
            # 确定分子中的节点数
            num_nodes = node_features.shape[0]
            
            # 获取超边数量
            num_edges = hyperedge_attr.shape[0]
            
            # 为每个分子创建batch索引
            batch = torch.full((num_nodes,), i, dtype=torch.long)
    
            # 从hyperedge_attr获取边的阶数（第一列是大小，第二列是类型）
            e_order = hyperedge_attr[:, 0].long()
            
            hyperedge_index0 = hyperedge_index[0]
            hyperedge_index1 = hyperedge_index[1]
            
            # 创建Data对象
            data = HData(
                hyper_x=node_features,
                hyper_x2=node_features2,  
                hyperedge_index0=hyperedge_index0,
                hyperedge_index1=hyperedge_index1,
                hyperedge_attr=hyperedge_attr,
                batch=batch,
                n_e=torch.tensor([num_edges]),
                e_order=e_order,
                y=data.y,
                smi=data.smiles,
                raw_x=data.x,
                raw_edge_index=data.edge_index,
                raw_edge_attr=data.edge_attr,
            )
            data_list.append(data)
            
        except Exception as e:
            pdb.set_trace()
            print(f"Error processing molecule {i}: {data.smiles}, Error: {str(e)}")
            continue
    
    return data_list

def get_moleculenet_loaders(dataset_name='HIV', root='./data/MoleculeNet', batch_size=32, num_workers=0):
    """
    加载MoleculeNet数据集并转换为超图表示
    
    Args:
        dataset_name: MoleculeNet数据集名称 (e.g., 'HIV', 'BACE', 'BBBP', 'Tox21', etc.)
        root: 数据存储路径
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
    
    Returns:
        train_loader, val_loader, test_loader, num_node_features, num_tasks
    """
    # 创建超图数据集
    hyper_dataset = HyperMoleculeNet(root=root, name=dataset_name)
    
    # 获取数据集信息
    sample_data = hyper_dataset[0]
    num_node_features = sample_data.hyper_x.size(1)
    num_tasks = sample_data.y.shape[1] if len(sample_data.y.shape) > 1 else 1
    
    # 获取SMILES列表
    smiles_list = [data.smi for data in hyper_dataset]
    
    # 使用scaffold_split划分数据集
    train_dataset, val_dataset, test_dataset = scaffold_split(
        hyper_dataset, 
        smiles_list, 
        task_idx=None, 
        null_value=0,
        frac_train=0.8, 
        frac_valid=0.1, 
        frac_test=0.1
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, num_node_features, num_tasks

# 使用示例
if __name__ == "__main__":
    # 示例：加载HIV数据集
    # train_loader, val_loader, test_loader, num_node_features, num_tasks = get_moleculenet_loaders(
    #     dataset_name='HIV', 
    #     root='./data/MoleculeNet', 
    #     batch_size=32
    # )
    
    # print(f"Dataset loaded successfully!")
    # print(f"Number of node features: {num_node_features}")
    # print(f"Number of tasks: {num_tasks}")
    # print(f"Train batches: {len(train_loader)}")
    # print(f"Validation batches: {len(val_loader)}")
    # print(f"Test batches: {len(test_loader)}")
    
    datasets = ['esol', 'freesolv', 'lipo', 'hiv', 'bace', 'bbbp', 'tox21', 'sider', 'clintox', 'toxcast', 'muv', 'pcba']
    
    for name in datasets:
        print(f"Loading {name}...")
        train_loader, val_loader, test_loader, num_node_features, num_tasks = get_moleculenet_loaders(
            dataset_name=name, 
            root='./data/MoleculeNet', 
            batch_size=32
        )
        print(f"Number of tasks: {num_tasks}")
    
    # datasets = ['esol', 'freesolv', 'lipo', 'hiv', 'bace', 'bbbp', 'tox21', 'sider', 'clintox', 'toxcast', 'muv', 'pcba']
    
    # for name in datasets:
    #     print(f"Loading {name}...")
    #     dataset = MoleculeNet(root='./data/MoleculeNet', name=name)
    #     print(dataset[0])