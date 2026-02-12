import argparse
import torch
import numpy as np
import random
import os
from data.moleculenet import get_moleculenet_loaders, HData
from tasks.trainer import Trainer
from utils.logger import setup_logger

import pdb

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    # Python的随机种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 更严格的确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    # 设置环境变量以确保CUDA操作的确定性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='HEMM Training')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['all', 'esol', 'freesolv', 'lipo', 'hiv', 'bace', 'bbbp', 
                                'tox21', 'sider', 'clintox', 'toxcast', 'muv', 'pcba'],
                       help='Dataset to train on')
    
    parser.add_argument('--gnn_name', type=str, default='GraphTrans',
                        choices=['GraphTrans'], help='GNN model name')
    parser.add_argument('--hyper_name', type=str, default='HGNN',
                        choices=['HGNN', 'MHNN', 'MHNNS', 'MHNNM', 'TriCL'], help='Hyper-graph model name')
    parser.add_argument('--use_hyper', action='store_true', help='Whether to use hyper-graph model')
    parser.add_argument('--mask_hyper', action='store_true', help='Whether to use mask hyper-graph model')
    parser.add_argument('--perturb_hyper', action='store_true', help='Whether to perturb hyper-edges (only effective when use_hyper is True)')
    parser.add_argument('--learnable_score', action='store_true', help='Whether to use learnable score')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of trials for each dataset')
    parser.add_argument('--top_k', type=int, default=3, help='Number of best trials to average')
    parser.add_argument('--mask_per_graph', action='store_true', help='Whether to mask hyperedges per graph instead of per batch')
    parser.add_argument('--contrastive_on_hyper', action='store_true', help='Whether to apply contrastive loss on hypergraph features directly (before fusion) instead of fused features')
    parser.add_argument('--loss_weight', type=float, default=1.0, help='Weight for the loss function')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='Mask ratio for hyperedges')
    parser.add_argument('--importance_weight_group', type=int, default=0, 
                        choices=[0, 1, 2, 3, 4],
                        help='Predefined importance weight groups: '
                             '0=[0.25,0.25,0.25,0.25], '
                             '1=[1.0,0.0,0.0,0.0], '
                             '2=[0.0,1.0,0.0,0.0], '
                             '3=[0.0,0.0,1.0,0.0], '
                             '4=[0.0,0.0,0.0,1.0]')
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)
        print(f"Using CUDA device {args.cuda}")
    else:
        print("Using CPU")
    
    set_seed()
    
    # MoleculeNet数据集列表
    all_datasets = [
        # 回归任务
        # ('esol', './data/MoleculeNet'),
        # ('freesolv', './data/MoleculeNet'), 
        # ('lipo', './data/MoleculeNet'),
        # 二分类任务
        ('hiv', './data/MoleculeNet'),
        ('bace', './data/MoleculeNet'),
        ('bbbp', './data/MoleculeNet'),
        # 多标签分类任务
        ('tox21', './data/MoleculeNet'),
        ('sider', './data/MoleculeNet'),
        ('clintox', './data/MoleculeNet'),
        ('toxcast', './data/MoleculeNet'),  
        ('muv', './data/MoleculeNet'),      
        # ('pcba', './data/MoleculeNet'),     
    ]
    
    # 如果指定了特定数据集，则只训练该数据集
    if args.dataset != 'all':
        datasets = [(args.dataset, './data/MoleculeNet')]
    else:
        datasets = all_datasets
    
    results = {}
    
    # 设置logger
    logger, timestamp = setup_logger(args.dataset)
    logger.info(f"Arguments: {args}")

    # 输出当前训练配置状况
    logger.info("=" * 60)
    logger.info("训练配置详情")
    logger.info("=" * 60)
    
    if args.use_hyper:
        logger.info(f"✓ 使用超图模型: {args.hyper_name}")
        if args.perturb_hyper:
            logger.info("✓ 启用超边扰动")
            logger.info(f"✓ 对比损失权重: {args.loss_weight}")
            if args.mask_hyper:
                logger.info("✓ 使用掩码策略: 可学习评分" if args.learnable_score else "✓ 使用掩码策略: 自身性质评分（度数等）")
                logger.info(f"✓ 超边掩码比例: {args.mask_ratio}")
                if not args.learnable_score:
                    # 预定义权重组
                    weight_group_desc = {
                        0: "均匀权重 [0.25, 0.25, 0.25, 0.25]",
                        1: "仅度数 [1.0, 0.0, 0.0, 0.0]",
                        2: "仅特征差异 [0.0, 1.0, 0.0, 0.0]",
                        3: "仅节点度数 [0.0, 0.0, 1.0, 0.0]",
                        4: "仅特征重要性 [0.0, 0.0, 0.0, 1.0]"
                    }
                    logger.info(f"✓ 预设重要权重组: {weight_group_desc.get(args.importance_weight_group, '未知')}")
                logger.info(f"✓ 超边掩码方式: {'按图掩码' if args.mask_per_graph else '按批次掩码'}")
            else:
                logger.info("✓ 使用随机扰动策略")
            logger.info(f"✓ 对比学习应用在: {'超图特征上' if args.contrastive_on_hyper else '融合特征上'}")
        else:
            logger.info("✗ 不启用超边扰动")
    else:
        logger.info("✗ 不使用超图模型")
        
    logger.info(f"✓ GNN模型: {args.gnn_name}")
    logger.info(f"✓ 数据集: {args.dataset.upper()}")
    logger.info(f"✓ CUDA设备: {'CPU' if args.cuda == -1 else f'GPU {args.cuda}'}")
    logger.info(f"✓ 重复轮次: {args.num_trials}")
    logger.info(f"✓ 最佳结果平均数: {args.top_k}")
    logger.info("=" * 60)
    
    for dataset_name, root in datasets:
        logger.info(f"Starting training on {dataset_name.upper()} dataset")

        logger.info(f"\n{'='*60}")
        logger.info(f"Training on {dataset_name.upper()} dataset")
        logger.info(f"{'='*60}")

        try:
            # 加载数据
            train_loader, val_loader, test_loader, num_features, num_tasks = get_moleculenet_loaders(
                dataset_name=dataset_name,
                root=root,
                batch_size=32
            )
            
            logger.info(f"Dataset info - Features: {num_features}, Tasks: {num_tasks}")
            logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
            
            # 存储每次试验的结果
            trial_results = []
            
            # 进行多次试验
            for trial in range(args.num_trials):
                logger.info(f"Starting trial {trial + 1}/{args.num_trials}")
                
                set_seed()
                
                train_loader, val_loader, test_loader, num_features, num_tasks = get_moleculenet_loaders(
                    dataset_name=dataset_name,
                    root=root,
                    batch_size=32
                )
                
                # 创建训练器
                trainer = Trainer(
                    in_channels=num_features,
                    num_tasks=num_tasks,
                    dataset_name=dataset_name,
                    lr=1e-3,
                    gnn_name=args.gnn_name,
                    hyper_name=args.hyper_name,
                    use_hyper=args.use_hyper,
                    timestamp=timestamp,
                    logger=logger,
                    loss_weight=args.loss_weight,
                    mask_hyper=args.mask_hyper,             # True 则用超边概率 False 则用随机遮掩
                    mask_ratio=args.mask_ratio,        
                    total_epochs=100, 
                    learnable_score=args.learnable_score,   # False 则用超边度
                    perturb_hyper=args.perturb_hyper,
                    mask_per_graph=args.mask_per_graph,     # 按图还是按批次mask
                    contrastive_on_hyper=args.contrastive_on_hyper,  # 对比学习应用在超图特征还是融合特征上
                    importance_weight_group=args.importance_weight_group
                )
                
                # 训练模型
                trainer.fit(train_loader, val_loader, epochs=100, patience=10)
                
                # 测试模型
                test_result = trainer.test(test_loader)
                trial_results.append(test_result)
                
                logger.info(f"Finished trial {trial + 1}/{args.num_trials}")
            
            # 计算最佳K次试验的平均结果
            if args.top_k > 0 and len(trial_results) >= args.top_k:
                # 根据任务类型确定如何选择最佳结果
                # 对于分类任务，ROC-AUC越高越好；对于回归任务，RMSE越低越好
                if dataset_name in ['esol', 'freesolv', 'lipo']:  # 回归任务
                    # RMSE越低越好，所以选择最小的值
                    sorted_results = sorted(trial_results)[:args.top_k]
                else:  # 分类任务
                    # ROC-AUC越高越好，所以选择最大的值
                    sorted_results = sorted(trial_results, reverse=True)[:args.top_k]
                
                logger.info(f"All results: {trial_results}")
                logger.info(f"Best {args.top_k} results: {sorted_results}")
                avg_best_result = sum(sorted_results) / len(sorted_results)
                std_best_result = np.std(sorted_results)
                logger.info(f"Average of top {args.top_k} results: {avg_best_result:.4f}")
                logger.info(f"Standard deviation of top {args.top_k} results: {std_best_result:.4f}")
            else:
                # 如果试验次数不足，计算所有试验的平均结果
                logger.info(f"All results: {trial_results}")
                logger.info(f'Best {args.top_k} results: {trial_results}')
                avg_result = sum(trial_results) / len(trial_results)
                std_result = np.std(trial_results)
                logger.info(f"Average of all {len(trial_results)} trials: {avg_result:.4f}")
                logger.info(f"Standard deviation of all {len(trial_results)} trials: {std_result:.4f}")
            
            logger.info(f"Finished training on {dataset_name.upper()} dataset successfully")
                
            results[dataset_name] = trial_results
            
        except Exception as e:
            error_msg = f"Error processing {dataset_name}: {str(e)}"
            logger.error(error_msg)
            results[dataset_name] = f"Failed: {str(e)}"
            raise e
    
    # 打印总结
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for dataset_name, result in results.items():
        logger.info(f"{dataset_name:12}: {result}")
    
    # 输出日志文件路径
    log_file_path = f"logs/{args.dataset}_{timestamp}.log"
    logger.info(f"Experiment logs saved to: {log_file_path}")

if __name__ == "__main__":
    main()
'''
# 使用默认设置（CPU，所有数据集）
python main.py

# 使用特定GPU训练所有数据集
python main.py --cuda 0

# 使用特定GPU训练特定数据集
python main.py --cuda 0 --dataset hiv

# 使用CPU训练特定数据集
python main.py --cuda -1 --dataset esol
'''