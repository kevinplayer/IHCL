import torch.nn as nn
from torch_geometric.nn import HypergraphConv, global_mean_pool

class HGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=128):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden)
        self.conv2 = HypergraphConv(hidden, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, hyperedge_index, batch):
        # x: [N, F], hyperedge_index: [2, E], batch: [N,]
        x = self.act(self.conv1(x, hyperedge_index))
        x = self.act(self.conv2(x, hyperedge_index))
        # 图级读出
        return global_mean_pool(x, batch)  # [B, out_channels]
    
# ①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①①
# --All_num_layers 3
# --MLP1_num_layers 2
# --MLP2_num_layers 2
# --MLP3_num_layers 2
# --MLP4_num_layers 2
# --output_num_layers 3
# --MLP_hidden 512
# --output_hidden 256
# --aggregate mean
# --lr 0.0001
# --wd 0
# --dropout 0.05
# --batch_size 256
# --epochs 400
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder

import pdb

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if x.size(0) > 1:
                x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class MHNNConv(nn.Module):
    def __init__(self, hid_dim, mlp1_layers=1, mlp2_layers=1, mlp3_layers=1,
        mlp4_layers=1, aggr='mean', dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(hid_dim*2, hid_dim, hid_dim, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = lambda X: X[..., hid_dim:]

        if mlp2_layers > 0:
            self.W2 = MLP(hid_dim*2, hid_dim, hid_dim, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., hid_dim:]

        if mlp3_layers > 0:
            self.W3 = MLP(hid_dim*2, hid_dim, hid_dim, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W3 = lambda X: X[..., hid_dim:]

        if mlp4_layers > 0:
            self.W4 = MLP(hid_dim*2, hid_dim, hid_dim, mlp4_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W4 = lambda X: X[..., hid_dim:]
        self.aggr = aggr
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W3, MLP):
            self.W3.reset_parameters()
        if isinstance(self.W4, MLP):
            self.W4.reset_parameters()

    def forward(self, X, E, vertex, edges):
        N = X.shape[-2]

        Mve = self.W1(torch.cat((X[..., vertex, :], E[..., edges, :]), -1))
        Me = scatter(Mve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        E = self.W2(torch.cat((E, Me), -1))
        # E = E*0.5 + e_in*0.5  # Residual connection.
        Mev = self.W3(torch.cat((X[..., vertex, :], E[..., edges, :]), -1))
        Mv = scatter(Mev, vertex, dim=-2, reduce=self.aggr, dim_size=N)
        X = self.W4(torch.cat((X, Mv), -1))
        # X = X*0.5 + X0*0.5  # Residual connection.

        return X, E

class MHNNSConv(nn.Module):
    def __init__(self, hid_dim, mlp1_layers=1, mlp2_layers=1, mlp3_layers=1,
                 aggr='mean', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(hid_dim, hid_dim, hid_dim, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(hid_dim*2, hid_dim, hid_dim, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., hid_dim:]

        if mlp3_layers > 0:
            self.W3 = MLP(hid_dim, hid_dim, hid_dim, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W3, MLP):
            self.W3.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :] # [nnz, C]
        Xe = scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C]
        
        Xev = Xe[..., edges, :] # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

        X = (1-self.alpha) * Xv + self.alpha * X0
        X = self.W3(X)

        return X

class BondEncoder(nn.Module):
    def __init__(self, args):
        super(BondEncoder, self).__init__()
        # 对于超边大小，使用线性层而不是嵌入层
        self.edge_size_linear = nn.Linear(1, args.MLP_hidden // 2)  
        # 键类型使用嵌入层
        self.bond_type_embedding = nn.Embedding(10, args.MLP_hidden // 2)
        
        # 合并后的MLP层
        self.mlp = nn.Sequential(
            nn.Linear(args.MLP_hidden, args.MLP_hidden),
            nn.ReLU(),
            nn.Linear(args.MLP_hidden, args.MLP_hidden)
        )
        
    def forward(self, bond_features):
        # bond_features: [batch_size, 2] 其中第二维是 [edge_size, bond_type]
        bond_type = bond_features[:, 1].long()  # 键类型
        edge_size = bond_features[:, 0].float().unsqueeze(-1)  # 超边大小，转换为浮点数并增加维度
        
        # 分别处理
        bond_embedding = self.bond_type_embedding(bond_type)
        edge_embedding = self.edge_size_linear(edge_size)
        
        # 合并特征
        combined = torch.cat([bond_embedding, edge_embedding], dim=-1)
        
        # 通过MLP
        output = self.mlp(combined)
        return output

class MHNN(nn.Module):
    def __init__(self, num_target, args):
        """ Molecular Hypergraph Neural Network (MHNN)
        (Shared parameters between all message passing layers)

        Args:
            num_target (int): number of output targets
            args (NamedTuple): global args
        """
        super().__init__()

        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = BondEncoder(args)

        self.conv = MHNNConv(args.MLP_hidden, mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers, mlp4_layers=self.mlp4_layers, aggr=args.aggregate,
            dropout=args.dropout, normalization=args.normalization)

        self.mlp_out = MLP(in_channels=args.MLP_hidden*2,
            hidden_channels=args.output_hidden*2,
            out_channels=num_target,
            num_layers=args.output_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False)

    def forward(self, data):
        V, E = data.hyperedge_index0, data.hyperedge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x.device)
        he_batch = e_batch[data.e_order > 2]

        x = self.atom_encoder(data.x2)
        e = self.bond_encoder(data.edge_attr)

        for i in range(self.nlayer):
            x, e = self.conv(x, e, V, E)
            if i == self.nlayer - 1:
                #remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_add_pool(x, data.batch)
        e = global_add_pool(e[data.e_order > 2], he_batch)
        out = self.mlp_out(torch.cat((x, e), -1))
        return out.view(-1)

class MHNNFeatureExtractor(nn.Module):
    def __init__(self, args):
        """ Molecular Hypergraph Neural Network (MHNN) Feature Extractor
        (Shared parameters between all message passing layers)
        Outputs drug features instead of final prediction

        Args:
            args (NamedTuple): global args
        """
        super().__init__()

        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = BondEncoder(args)

        self.conv = MHNNConv(args.MLP_hidden, mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers, mlp4_layers=self.mlp4_layers, aggr=args.aggregate,
            dropout=args.dropout, normalization=args.normalization)

    def forward(self, data):
        V, E = data.hyperedge_index0, data.hyperedge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.hyper_x.device)
        high_order_mask = data.e_order > 2
        he_batch = e_batch[high_order_mask]

        x = self.atom_encoder(data.hyper_x2)
        e = self.bond_encoder(data.hyperedge_attr)

        for i in range(self.nlayer):
            x, e = self.conv(x, e, V, E)
            if i == self.nlayer - 1:
                #remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_mean_pool(x, data.batch)
        
        # 处理高阶超边的池化，确保输出维度与x一致
        if high_order_mask.sum() == 0:
            # 如果没有任何高阶超边
            e_pooled = torch.zeros((x.size(0), e.size(-1)), device=x.device)
        else:
            # 如果有高阶超边，先进行池化
            e_high_order = e[high_order_mask]
            e_pooled_raw = global_mean_pool(e_high_order, he_batch)
            if e_pooled_raw.size(0) == x.size(0):
                # 如果维度匹配，直接使用
                e_pooled = e_pooled_raw
            else:
                # 创建完整batch大小的输出tensor
                e_pooled = torch.zeros((x.size(0), e.size(-1)), device=x.device)
                # 将池化结果放回对应位置
                # 注意：这里直接使用he_batch中的唯一值作为索引
                unique_batches = torch.unique(he_batch)
                # 确保维度匹配
                if e_pooled_raw.size(0) == unique_batches.size(0):
                    e_pooled[unique_batches] = e_pooled_raw
                else:
                    # 如果维度不匹配，逐个处理
                    for i, batch_idx in enumerate(unique_batches):
                        # 找到属于该batch的所有超边
                        # mask = (he_batch == batch_idx)
                        # if mask.sum() > 0:  # 确实有超边
                        #     e_pooled[batch_idx] = e[mask].sum(0)
                        
                        e_pooled[batch_idx] = e_pooled_raw[batch_idx]
            
        return torch.cat((x, e_pooled), -1)

class MHNNS(nn.Module):
    def __init__(self,  num_target, args):
        """ Molecular Hypergraph Neural Network (MHNN) simple version,
        which has similar performance with MHNN but smaller and faster.
        (Shared parameters between all message passing layers)

        Args:
            num_target (int): number of output targets
            args (NamedTuple): global args
        """
        super().__init__()

        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.nlayer = args.All_num_layers

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.conv = MHNNSConv(args.MLP_hidden, mlp1_layers=self.mlp1_layers,
            mlp2_layers=self.mlp2_layers, mlp3_layers=self.mlp3_layers,
            aggr=args.aggregate, dropout=args.dropout,
            normalization=args.normalization)

        self.mlp_out = MLP(in_channels=args.MLP_hidden,
            hidden_channels=args.output_hidden,
            out_channels=num_target,
            num_layers=args.output_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.mlp_out.reset_parameters()

    def forward(self, data):
        V, E = data.hyperedge_index0, data.hyperedge_index1
        x = self.atom_encoder(data.x2)
        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.mlp_out(x)
        return x.view(-1)

class MHNNM(nn.Module):
    def __init__(self, num_target, args):
        """ 
        Molecular Hypergraph Neural Network (MHNN)
        (Multiple message passing layers, no parameters shared between layers)

        Args:
            num_target (int): number of output targets
            args (NamedTuple): global args
        """
        super().__init__()
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = BondEncoder(args)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.nlayer):
            self.layers.append(MHNNConv(
                args.MLP_hidden,
                mlp1_layers=self.mlp1_layers,
                mlp2_layers=self.mlp2_layers,
                mlp3_layers=self.mlp3_layers,
                mlp4_layers=self.mlp4_layers,
                aggr=args.aggregate,
                dropout=args.dropout,
                normalization=args.normalization,
            ))
            self.batch_norms.append(nn.BatchNorm1d(args.MLP_hidden))

        self.mlp_out = MLP(in_channels=args.MLP_hidden,
            hidden_channels=args.output_hidden,
            out_channels=num_target,
            num_layers=args.output_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False)

    def forward(self, data):
        V, E = data.hyperedge_index0, data.hyperedge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x2.device)

        x = self.atom_encoder(data.x2)
        e = self.bond_encoder(data.edge_attr)

        for i, layer in enumerate(self.layers):
            x, e = layer(x, e, V, E)
            x = self.batch_norms[i](x)

            if i == self.nlayer - 1:
                #remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_add_pool(x, data.batch)
        out = self.mlp_out(x)
        return out.view(-1)

class MHNNMFeatureExtractor(nn.Module):
    def __init__(self, args):
        """ 
        Molecular Hypergraph Neural Network (MHNN) Feature Extractor
        (Multiple message passing layers, no parameters shared between layers)

        Args:
            args (NamedTuple): global args
        """
        super().__init__()
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = BondEncoder(args)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.nlayer):
            self.layers.append(MHNNConv(
                args.MLP_hidden,
                mlp1_layers=self.mlp1_layers,
                mlp2_layers=self.mlp2_layers,
                mlp3_layers=self.mlp3_layers,
                mlp4_layers=self.mlp4_layers,
                aggr=args.aggregate,
                dropout=args.dropout,
                normalization=args.normalization,
            ))
            self.batch_norms.append(nn.BatchNorm1d(args.MLP_hidden))

    def forward(self, data):
        V, E = data.hyperedge_index0, data.hyperedge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.hyper_x2.device)
        high_order_mask = data.e_order > 2
        he_batch = e_batch[high_order_mask]

        x = self.atom_encoder(data.hyper_x2)
        e = self.bond_encoder(data.hyperedge_attr)

        for i, layer in enumerate(self.layers):
            x, e = layer(x, e, V, E)
            if x.size(0) > 1:
                x = self.batch_norms[i](x)

            if i == self.nlayer - 1:
                #remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_mean_pool(x, data.batch)
        
        # 处理高阶超边的池化，确保输出维度与x一致
        if high_order_mask.sum() == 0:
            # 如果没有任何高阶超边
            e_pooled = torch.zeros((x.size(0), e.size(-1)), device=x.device)
        else:
            # 如果有高阶超边，先进行池化
            e_high_order = e[high_order_mask]
            e_pooled_raw = global_mean_pool(e_high_order, he_batch)
            if e_pooled_raw.size(0) == x.size(0):
                # 如果维度匹配，直接使用
                e_pooled = e_pooled_raw
            else:
                # 创建完整batch大小的输出tensor
                e_pooled = torch.zeros((x.size(0), e.size(-1)), device=x.device)
                # 将池化结果放回对应位置
                # 注意：这里直接使用he_batch中的唯一值作为索引
                unique_batches = torch.unique(he_batch)
                # 确保维度匹配
                if e_pooled_raw.size(0) == unique_batches.size(0):
                    e_pooled[unique_batches] = e_pooled_raw
                else:
                    # 如果维度不匹配，逐个处理
                    for i, batch_idx in enumerate(unique_batches):
                        # 找到属于该batch的所有超边
                        # mask = (he_batch == batch_idx)
                        # if mask.sum() > 0:  # 确实有超边
                        #     e_pooled[batch_idx] = e[mask].sum(0)
                        
                        e_pooled[batch_idx] = e_pooled_raw[batch_idx]
            
        return torch.cat((x, e), -1)


# ②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②②② 
import torch
from torch import Tensor, nn
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from typing import Optional, Callable
from torch_geometric.utils import scatter
from torch_geometric.nn.inits import zeros

class ProposedConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0, 
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False, 
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False)
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.lin_n2e.weight)
        torch.nn.init.xavier_uniform_(self.lin_e2n.weight)

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None) 
            self.register_parameter('bias_e2n', None) 
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None
    
    def forward(self, x: Tensor, hyperedge_index: Tensor, 
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]
                
            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x = self.lin_n2e(x)
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e, 
                               size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)
        
        x = self.lin_e2n(e)
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n, 
                               size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j
    

class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x, e # act, act

# TriCL Encoder
class TriCL(nn.Module):
    def __init__(self, encoder: HyperEncoder, proj_dim: int):
        super(TriCL, self).__init__()
        self.encoder = encoder

        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()
        
    def forward(self, data, num_nodes=None, num_edges=None):
        hyperedge_index = torch.stack([data.hyperedge_index0, data.hyperedge_index1], dim=0)
        x = data.hyper_x
        batch = data.batch
        if num_nodes is None:
            num_nodes = int(x.shape[0])
        if num_edges is None:
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1
            else:
                num_edges = 0

        # # 处理没有超边的特殊情况
        if num_edges == 0:
            # 只有节点特征，没有边特征
            n, e = self.encoder(x, hyperedge_index, num_nodes, num_edges + num_nodes)
            # 对节点特征进行全局池化得到图级表示
            node_graph_features = global_add_pool(n, batch[:n.size(0)] if n.size(0) > 0 else torch.zeros(1, dtype=torch.long, device=x.device))
            # 边特征为零向量
            edge_graph_features = torch.zeros((node_graph_features.size(0), e.size(-1) if e.size(0) > 0 else 64), device=x.device)
            # 拼接节点和边的图级特征
            graph_features = torch.cat([node_graph_features, edge_graph_features], dim=-1)
            return graph_features
        
        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)
        # return n, e[:num_edges]
        
        # 对节点特征和边特征进行全局池化得到图级表示
        # 获取节点和边对应的batch信息
        if batch.dim() == 1:
            # 正确地为每个节点生成batch索引
            # batch中每个元素表示对应节点属于哪个图
            node_batch = batch
        else:
            node_batch = batch
        
        # 确保node_batch的长度与n的长度一致
        if len(node_batch) != n.size(0):
            # 如果长度不一致，使用batch中每个图的节点数来正确生成batch索引
            _, counts = torch.unique(batch, return_counts=True)
            node_batch = torch.arange(batch.size(0), device=x.device).repeat_interleave(counts)
            
        # 全局池化得到图级特征
        node_graph_features = global_add_pool(n, node_batch[:n.size(0)])
        # 修复：为超边创建正确的batch索引
        # 使用scatter操作高效地为每条超边分配batch索引
        # 为超边创建正确的batch索引
        hyperedge_batch = torch.zeros(num_edges, dtype=torch.long, device=x.device)
        # 将节点的batch信息传播到对应的超边
        hyperedge_batch.scatter_(0, data.hyperedge_index1, batch[data.hyperedge_index0])
        # 进行池化
        edge_graph_features_raw = global_add_pool(e[:num_edges], hyperedge_batch)

        if edge_graph_features_raw.size(0) == node_graph_features.size(0):  
            edge_graph_features = edge_graph_features_raw
        else:  
            # 确保输出维度与节点特征一致
            edge_graph_features = torch.zeros((node_graph_features.size(0), e.size(-1)), device=x.device)
            unique_batches = torch.unique(hyperedge_batch)
            if edge_graph_features_raw.size(0) == unique_batches.size(0):
                edge_graph_features[unique_batches] = edge_graph_features_raw
            else:
                for i, batch_id in enumerate(unique_batches):
                    edge_graph_features[batch_id] = edge_graph_features_raw[batch_id]
                    
        # 拼接节点和边的图级特征
        graph_features = torch.cat([node_graph_features, edge_graph_features], dim=-1)
        
        return graph_features

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)
        
    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))
    
    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))
