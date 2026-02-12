import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
import os

from models.perturb import drop_hyperedges
from models.loss import contrastive_loss

from models.gnn import *
from models.hgnn import *
from models.perturb import drop_hyperedges, MaskHyper, drop_hyperedges_per_graph

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, in_channels, num_tasks, dataset_name, lr=1e-3, 
                 gnn_name=None, hyper_name=None, use_hyper=False,
                 ckp_dir='ckp', timestamp=None, logger=None, loss_weight=1.0,
                 mask_hyper=True, mask_ratio=0.2, total_epochs=100, learnable_score=False, perturb_hyper=True,
                 mask_per_graph=False, contrastive_on_hyper=False,
                 importance_weight_group=0):
        
        self.dataset_name = dataset_name.lower()
        self.num_tasks = num_tasks
        self.timestamp = timestamp
        self.use_hyper = use_hyper
        self.gnn_name = gnn_name
        self.hyper_name = hyper_name
        self.perturb_hyper = perturb_hyper and use_hyper  # 只有在使用超图时才扰动
        self.mask_per_graph = mask_per_graph  # 是否按图mask超边
        self.contrastive_on_hyper = contrastive_on_hyper  # 是否在超图特征上应用对比学习
        
        # 超边重要性评分权重组
        self.importance_weight_group = importance_weight_group
        
        # 根据数据集名称判断任务类型
        self.task_type = self._get_task_type()
        
        # 创建GNN模型
        self.gnn_encoder = self.create_gnn_model(gnn_name, in_channels).to(device)
        
        # 如果使用超图，创建超图模型
        if self.use_hyper:
            self.hyper_encoder = self.create_hyper_gnn_model(hyper_name, in_channels).to(device)
            # 对于多任务，输出维度需要匹配任务数
            self.pred_head = torch.nn.Sequential(
                torch.nn.Linear(256, 256),  # 256 when using hyper
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_tasks)
            ).to(device)
        else:
            # 对于多任务，输出维度需要匹配任务数
            self.pred_head = torch.nn.Sequential(
                torch.nn.Linear(128, 256),  # 128 when not using hyper
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_tasks)
            ).to(device)
                  
        self.drop_ratio = mask_ratio
        self.loss_weight = loss_weight
        
        # 创建模型保存目录
        self.ckp_dir = os.path.join(ckp_dir, self.dataset_name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        
        self.logger = logger
        
        self.mask_hyper = mask_hyper and use_hyper and perturb_hyper  # 只有在使用超图且启用扰动时才mask
        if self.mask_hyper:
            self.sgm = MaskHyper(mask_ratio=mask_ratio,
                                total_epochs=total_epochs,
                                learnable=learnable_score,
                                in_dim=in_channels,
                                weight_group=importance_weight_group)
        else:
            self.sgm = None
            
        # 初始化优化器（移到最后确保所有模型组件都已创建）
        self._init_optimizer(lr)

        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='max' if self.task_type != 'regression' else 'min', 
            factor=0.8, patience=5, verbose=True
        )
    def _init_optimizer(self, lr):
        """初始化优化器，确保包含所有需要训练的参数"""
        params = list(self.gnn_encoder.parameters()) + list(self.pred_head.parameters())
        
        if self.use_hyper:
            params += list(self.hyper_encoder.parameters())
            
        if self.mask_hyper and self.sgm is not None and self.sgm.scorer is not None:
            params += list(self.sgm.scorer.parameters())
            self.logger.info("Including scorer parameters in optimizer")
            
        self.opt = torch.optim.AdamW(params, lr=lr)
        

    def create_gnn_model(self, gnn_name, dataset_name):
        """
        根据gnn_name创建GNN模型
        """
        if gnn_name == 'GraphTrans':
            model = GraphTrans()
        # 可以添加更多GNN模型
        # elif gnn_name == '...':
        #     model = ...
        else:
            # 默认使用GraphTrans
            model = GraphTrans()
            
        return model

    def create_hyper_gnn_model(self, hyper_name, in_channels):
        """
        根据hyper_name创建超图GNN模型
        """
        from collections import namedtuple
        if hyper_name == 'HGNN':
            model = HGNNEncoder(in_channels)
        # 可以添加更多超图模型
        elif hyper_name == 'MHNN':
            Args = namedtuple('Args', [
                'activation', 'dropout', 'MLP_hidden', 'All_num_layers', 
                'MLP1_num_layers', 'MLP2_num_layers', 'MLP3_num_layers', 
                'MLP4_num_layers', 'aggregate', 'normalization'
            ])
            args = Args(
                activation='relu',
                dropout=0.00,
                MLP_hidden=64,
                All_num_layers=3,
                MLP1_num_layers=2,
                MLP2_num_layers=2,
                MLP3_num_layers=2,
                MLP4_num_layers=2,
                aggregate='mean',
                normalization='bn'
            )
            model = MHNNFeatureExtractor(args=args)
        # elif hyper_name == 'MHNNS':
        #     model = ...
        # elif hyper_name == 'MHNNM':
        #     model = ...
        elif hyper_name == 'TriCL':
            encoder_args = namedtuple('Args', [
                'in_dim', 'edge_dim', 'node_dim', 'num_layers'
            ])
            encoder_args = encoder_args(
                in_dim=in_channels,
                edge_dim=64,
                node_dim=64,
                num_layers=2
            )
            encoder = HyperEncoder(
                in_dim=encoder_args.in_dim,
                edge_dim=encoder_args.edge_dim,
                node_dim=encoder_args.node_dim,
                num_layers=encoder_args.num_layers
            )
            
            model_args = namedtuple('Args', ['proj_dim'])
            model_args = model_args(proj_dim=256)
            model = TriCL(encoder=encoder, proj_dim=model_args.proj_dim)
        else:
            # 默认使用HGNN
            model = HGNNEncoder(in_channels)
            
        return model

    def _get_task_type(self):
        """根据数据集名称判断任务类型"""
        # 回归任务数据集
        regression_datasets = ['esol', 'freesolv', 'lipo']
        # 多标签分类任务数据集
        multilabel_datasets = ['sider', 'tox21', 'toxcast', 'pcba', 'muv', 'clintox']
        # 多分类任务数据集
        multiclass_datasets = []
        # 二分类任务数据集
        binary_datasets = ['hiv', 'bace', 'bbbp']
        
        if self.dataset_name in regression_datasets:
            return 'regression'
        elif self.dataset_name in multilabel_datasets:
            return 'multilabel'
        elif self.dataset_name in multiclass_datasets:
            return 'multiclass'
        elif self.dataset_name in binary_datasets:
            return 'binary'
        else:
            return 'binary'  # 默认为二分类

    def _compute_loss(self, pred, y):
        """计算损失函数，处理NaN值"""
        if self.task_type in ['binary', 'multilabel']:
            # 对于分类任务，使用二元交叉熵，忽略NaN值
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            loss_task = F.binary_cross_entropy_with_logits(pred[mask], y[mask])
        elif self.task_type == 'multiclass':
            # 对于多分类任务，使用交叉熵
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            # 假设y是类别索引
            loss_task = F.cross_entropy(pred[mask], y[mask])
        else:  # regression
            # 对于回归任务，使用均方误差，忽略NaN值
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            loss_task = F.mse_loss(pred[mask], y[mask])
        
        return loss_task

    def _compute_metrics(self, pred, y):
        """计算评估指标"""
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            return 0.0
        
        if self.task_type in ['binary', 'multilabel']:
            # 计算ROC-AUC
            try:
                if self.num_tasks == 1:
                    auc = roc_auc_score(y[mask].cpu().numpy(), torch.sigmoid(pred[mask]).detach().cpu().numpy())
                else:
                    auc = roc_auc_score(y[mask].cpu().numpy(), torch.sigmoid(pred[mask]).detach().cpu().numpy(), average='macro')
                return auc
            except:
                return 0.0
        elif self.task_type == 'multiclass':
            # 计算准确率
            pred_labels = pred[mask].argmax(dim=1)
            accuracy = (pred_labels == y[mask].long()).float().mean().item()
            return accuracy
        else:  # regression
            # 计算RMSE
            rmse = torch.sqrt(F.mse_loss(pred[mask], y[mask])).item()
            return rmse

    def _compute_scorer_loss(self, z_clean, z_perturbed, edge_scores):
        """
        计算打分器的损失函数，鼓励保留重要超边
        :param z_clean: 未扰动数据的表示
        :param z_perturbed: 扰动数据的表示
        :param edge_scores: 超边得分 [E]
        """
        # 计算表示差异，鼓励保留重要超边
        repr_diff = torch.norm(z_clean - z_perturbed, p=2, dim=1)
        
        # 扩展repr_diff到每个超边
        repr_diff_per_edge = repr_diff.unsqueeze(1).expand(-1, edge_scores.size(0))
        
        # 损失设计：鼓励移除不重要超边（得分低）时差异小，保留重要超边（得分高）时差异大
        # 使用负相关关系：得分高的超边被移除应该导致更大的差异
        scorer_loss = torch.mean(edge_scores * repr_diff_per_edge.mean(dim=0))
        
        return scorer_loss

    def run_epoch(self, loader, train=True, epoch=0):
        if train:
            self.gnn_encoder.train()
            self.pred_head.train()
            if self.use_hyper:
                self.hyper_encoder.train()
            if self.mask_hyper and self.sgm is not None and self.sgm.scorer is not None:
                self.sgm.scorer.train()
        else:
            self.gnn_encoder.eval()
            self.pred_head.eval()
            if self.use_hyper:
                self.hyper_encoder.eval()
            if self.mask_hyper and self.sgm is not None and self.sgm.scorer is not None:
                self.sgm.scorer.eval()

        total, metric_sum, loss_sum, loss_con_sum, loss_task_sum, scorer_loss_sum = 0, 0., 0., 0., 0., 0.
        
        # 用于存储所有预测和标签以计算整体指标
        all_preds = []
        all_labels = []
        
        for batch in tqdm(loader, desc='Training' if train else 'Validation'):
            batch = batch.to(device)
            
            with torch.set_grad_enabled(train):
                # 获取GNN特征
                z_gnn = self.gnn_encoder(batch)
                pred_clean = None
                
                # 如果使用超图
                if self.use_hyper:
                    hyper_x = batch.hyper_x
                    # 构建超边索引 [2, E] 格式
                    he = torch.stack([batch.hyperedge_index0, batch.hyperedge_index1], dim=0)
                    b = batch.batch
                    
                    # 统一不同超图模型的接口
                    if hasattr(self.hyper_encoder, 'forward') and 'data' in str(self.hyper_encoder.forward.__code__.co_varnames[:self.hyper_encoder.forward.__code__.co_argcount]):
                        # 对于需要data对象的模型（如MHNN系列）
                        z_hyper = self.hyper_encoder(batch)
                    else:
                        # 对于标准接口的模型（如HGNN）
                        z_hyper = self.hyper_encoder(hyper_x, he, b)
                    
                    # 合并特征
                    z_combined_clean = torch.cat([z_gnn, z_hyper], dim=1)
                    pred_clean = self.pred_head(z_combined_clean)
                    
                    # 只有在启用扰动时才进行超边扰动
                    if self.perturb_hyper:
                        if self.mask_hyper and self.sgm is not None:
                            if self.mask_per_graph:
                                he_pert, hyperedge_attr_pert, n_e_pert, e_order_pert, mask_flag = self.sgm.mask_edges_per_graph(
                                    he, batch.hyperedge_attr, hyper_x, batch.n_e, batch.e_order, epoch)
                            else:
                                he_pert, hyperedge_attr_pert, n_e_pert, e_order_pert, mask_flag = self.sgm.mask_edges(
                                    he, batch.hyperedge_attr, hyper_x, batch.n_e, batch.e_order, epoch)
                        else:
                            if self.mask_per_graph:
                                he_pert, hyperedge_attr_pert, n_e_pert, e_order_pert = drop_hyperedges_per_graph(
                                    he, batch.hyperedge_attr, batch.n_e, batch.e_order, self.drop_ratio)
                            else:
                                he_pert, hyperedge_attr_pert, n_e_pert, e_order_pert = drop_hyperedges(
                                    he, batch.hyperedge_attr, batch.n_e, batch.e_order, self.drop_ratio)
                            
                        # 对扰动后的超边同样应用适配器逻辑
                        if hasattr(self.hyper_encoder, 'forward') and 'data' in str(self.hyper_encoder.forward.__code__.co_varnames[:self.hyper_encoder.forward.__code__.co_argcount]):
                            # 创建一个新的batch对象用于扰动情况
                            pert_batch = batch.clone()  
                            pert_batch.hyperedge_index0, pert_batch.hyperedge_index1 = he_pert[0], he_pert[1]
                            pert_batch.hyperedge_attr = hyperedge_attr_pert
                            pert_batch.n_e = n_e_pert  # 更新n_e
                            pert_batch.e_order = e_order_pert  # 更新e_order
                            z_hyper_pert = self.hyper_encoder(pert_batch)
                        else:
                            z_hyper_pert = self.hyper_encoder(hyper_x, he_pert, b)
                        
                        # 根据参数决定在哪种特征上计算对比损失
                        if self.contrastive_on_hyper:
                            # 在超图特征上计算对比损失
                            loss_con = contrastive_loss(z_hyper, z_hyper_pert)
                        else:
                            # 在融合特征上计算对比损失
                            z_combined_pert = torch.cat([z_gnn, z_hyper_pert], dim=1)
                            fused_clean = self.pred_head[:3](z_combined_clean)
                            fused_pert = self.pred_head[:3](z_combined_pert)
                            loss_con = contrastive_loss(fused_clean, fused_pert)
                        
                        # 计算打分器损失
                        scorer_loss = torch.tensor(0.0, device=z_gnn.device)
                        if self.mask_hyper and self.sgm is not None and self.sgm.scorer is not None:
                            # 获取超边得分
                            edge_scores = self.sgm.scorer(hyper_x, he)
                            scorer_loss = self._compute_scorer_loss(z_hyper if self.contrastive_on_hyper else fused_clean, 
                                                                   z_hyper_pert if self.contrastive_on_hyper else fused_pert, 
                                                                   edge_scores)
                    else:
                        # 不进行扰动
                        z_hyper_pert = z_hyper
                        loss_con = torch.tensor(0.0, device=z_gnn.device)
                        scorer_loss = torch.tensor(0.0, device=z_gnn.device)
                else:
                    # 不使用超图，直接用GNN特征
                    pred_clean = self.pred_head(z_gnn)
                    loss_con = torch.tensor(0.0, device=z_gnn.device)  # 不计算对比损失
                    scorer_loss = torch.tensor(0.0, device=z_gnn.device)  # 不计算打分器损失
                
                y = batch.y
                # 计算任务损失
                loss_task = self._compute_loss(pred_clean, y)
                
                # 总损失
                if self.use_hyper:
                    loss = loss_task + self.loss_weight * loss_con + 0.1 * scorer_loss
                else:
                    loss = loss_task

            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
 
            loss_sum += loss.item() * y.size(0)
            loss_con_sum += loss_con.item() * y.size(0)
            loss_task_sum += loss_task.item() * y.size(0)
            scorer_loss_sum += scorer_loss.item() * y.size(0)
            
            # 收集预测和标签用于整体指标计算
            mask = ~torch.isnan(y)
            if mask.sum() > 0:
                all_preds.append(pred_clean[mask].detach())
                all_labels.append(y[mask].detach())
            
            total += y.size(0)

        # 统一计算指标
        if len(all_preds) > 0 and len(all_labels) > 0:
            all_preds_tensor = torch.cat(all_preds, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)
            overall_metric = self._compute_metrics(all_preds_tensor, all_labels_tensor)
        else:
            overall_metric = 0.0

        return loss_sum/total, overall_metric, loss_con_sum/total, loss_task_sum/total, scorer_loss_sum/total
    
    def fit(self, train_l, val_l, epochs=100, patience=10):
        best_metric, patience_cnt = -float('inf'), 0
        best_epoch = 0  # 记录最佳性能出现的轮次
        if self.task_type == 'regression':
            best_metric = float('inf')  # 对于回归任务，初始最佳值设为无穷大
            
        for epoch in tqdm(range(1, epochs+1), desc='Epochs'):
            tr_loss, tr_metric, tr_loss_con, tr_loss_task, tr_scorer_loss = self.run_epoch(train_l, True, epoch)
            val_loss, val_metric, val_loss_con, val_loss_task, val_scorer_loss = self.run_epoch(val_l, False, epoch)
            
            # 使用调度器调整学习率
            self.scheduler.step(val_metric)  
            
            # 根据任务类型调整指标显示
            if self.task_type == 'regression':
                metric_name = 'RMSE'
                # 对于回归任务，指标越低越好
                improved = val_metric < best_metric
            else:
                metric_name = 'ROC-AUC' if self.task_type in ['binary', 'multilabel'] else 'Acc'
                # 对于分类任务，指标越高越好
                improved = val_metric > best_metric
            
            tr_metric_display = f'{tr_metric:.4f}'
            val_metric_display = f'{val_metric:.4f}'
            
            # 获取当前学习率的兼容方法
            current_lr = self.opt.param_groups[0]['lr']
            
            # 根据是否使用超图决定打印的内容
            if self.use_hyper:
                self.logger.info(f'Epoch {epoch:03d} | '
                    f'Train loss {tr_loss:.4f} (con: {tr_loss_con:.4f} task: {tr_loss_task:.4f} scorer: {tr_scorer_loss:.4f}) {metric_name} {tr_metric_display} | '
                    f'Val loss {val_loss:.4f} (con: {val_loss_con:.4f} task: {val_loss_task:.4f} scorer: {val_scorer_loss:.4f}) {metric_name} {val_metric_display} | '
                    f'Learning rate {current_lr:.6f}')
            else:
                self.logger.info(f'Epoch {epoch:03d} | '
                    f'Train loss {tr_loss:.4f} (task: {tr_loss_task:.4f}) {metric_name} {tr_metric_display} | '
                    f'Val loss {val_loss:.4f} (task: {val_loss_task:.4f}) {metric_name} {val_metric_display} | '
                    f'Learning rate {current_lr:.6f}')
                
            if improved:
                best_metric = val_metric
                best_epoch = epoch  # 更新最佳轮次
                # 使用时间戳作为模型文件名
                model_path = os.path.join(self.ckp_dir, f'{self.timestamp}.pt')
                
                # 根据是否使用超图保存不同的模型状态
                if self.use_hyper:
                    state_dict = {
                        'gnn': self.gnn_encoder.state_dict(),
                        'hyper': self.hyper_encoder.state_dict(),
                        'head': self.pred_head.state_dict()
                    }
                    # 如果使用了可学习打分器，也保存其状态
                    if self.mask_hyper and self.sgm is not None and self.sgm.scorer is not None:
                        state_dict['scorer'] = self.sgm.scorer.state_dict()
                    torch.save(state_dict, model_path)
                else:
                    torch.save({'gnn': self.gnn_encoder.state_dict(),
                                'head': self.pred_head.state_dict()}, model_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    self.logger.info(f'Early stop! Best performance at epoch {best_epoch}')
                    break

    def test(self, test_l):
        # 使用时间戳加载模型文件
        model_path = os.path.join(self.ckp_dir, f'{self.timestamp}.pt')
        checkpoint = torch.load(model_path, map_location=device)
        
        # 根据是否使用超图加载不同的模型状态
        self.gnn_encoder.load_state_dict(checkpoint['gnn'])
        if self.use_hyper:
            self.hyper_encoder.load_state_dict(checkpoint['hyper'])
            # 如果保存了打分器状态，则加载
            if 'scorer' in checkpoint and self.sgm is not None and self.sgm.scorer is not None:
                self.sgm.scorer.load_state_dict(checkpoint['scorer'])
        self.pred_head.load_state_dict(checkpoint['head'])
        
        test_loss, test_metric, test_loss_con, test_loss_task, test_scorer_loss = self.run_epoch(test_l, False)
        
        # 根据任务类型调整指标显示
        if self.task_type == 'regression':
            metric_name = 'RMSE'
        else:
            metric_name = 'ROC-AUC' if self.task_type in ['binary', 'multilabel'] else 'Acc'
        test_metric_display = f'{test_metric:.4f}'
        
        # 根据是否使用超图决定打印的内容
        if self.use_hyper:
            self.logger.info(f'Test loss {test_loss:.4f} (con: {test_loss_con:.4f} task: {test_loss_task:.4f} scorer: {test_scorer_loss:.4f}) | Test {metric_name} {test_metric_display}')
        else:
            self.logger.info(f'Test loss {test_loss:.4f} (task: {test_loss_task:.4f}) | Test {metric_name} {test_metric_display}')
            
        # 返回测试指标值以便主程序处理
        return test_metric