import numpy as np
import pandas as pd
import sys
import os
from random import shuffle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split  # 新增：用于拆分验证集
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from torch_geometric.loader import DataLoader


# -------------------------- 训练函数（保持核心逻辑，优化日志输出）--------------------------
def train(model, device, train_loader, optimizer, epoch):
    print(f'\n【训练轮次 {epoch}】正在训练 {len(train_loader.dataset)} 个样本...')
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

        # 简化日志：每LOG_INTERVAL批打印一次训练损失
        if batch_idx % LOG_INTERVAL == 0:
            print(
                f'训练进度: [{batch_idx * len(data.x)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t训练损失: {loss.item():.6f}')


# -------------------------- 预测函数（无修改，保证数据输出一致性）--------------------------
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'正在对 {len(loader.dataset)} 个样本进行预测...')
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# -------------------------- 命令行参数解析（保持原有逻辑，适配GPU编号）--------------------------
# 命令行参数说明：python training.py [数据集编号] [模型编号] [GPU编号]
# 数据集编号：0=davis, 1=kiba；模型编号：0=GINConvNet, 1=GATNet, 2=GAT_GCN, 3=GCNNet
datasets = [['davis', 'kiba'][int(sys.argv[1])]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

# GPU设备配置（优先使用命令行参数，默认cuda:0）
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = f"cuda:{int(sys.argv[3])}"
print(f'使用GPU设备: {cuda_name}')

# -------------------------- 超参数配置（优化后，兼顾性能和效率）--------------------------
TRAIN_BATCH_SIZE = 1024  # 增大批次（RTX 4070 8GB显存足够，加速训练）
TEST_BATCH_SIZE = 1024
LR = 0.0005  # 原学习率保持不变
LOG_INTERVAL = 50  # 减少日志打印频率，提升速度
NUM_EPOCHS = 300  # 轮次上限（比1000轮节省70%时间）
PATIENCE = 50  # 早停耐心值（连续50轮无提升则停止）
VAL_SPLIT_RATIO = 0.1  # 验证集比例（从训练集拆分10%）

# 打印超参数摘要
print(f'\n【超参数配置】')
print(f'学习率: {LR} | 训练轮次上限: {NUM_EPOCHS} | 早停耐心值: {PATIENCE}')
print(f'训练批次大小: {TRAIN_BATCH_SIZE} | 验证集比例: {VAL_SPLIT_RATIO * 100}%')

# -------------------------- 主程序（核心优化：验证集拆分+早停+学习率调度）--------------------------
for dataset in datasets:
    print(f'\n==================================================')
    print(f'正在运行模型: {model_st} | 数据集: {dataset}')
    print(f'==================================================')

    # 数据文件路径（适配用户的文件存放位置）
    data_root = r'D:\Transgelin-2\pythonProject\GraphDTA-master\data'
    processed_data_file_train = os.path.join(data_root, 'processed', f'{dataset}_train.pt')
    processed_data_file_test = os.path.join(data_root, 'processed', f'{dataset}_test.pt')

    # 检查数据文件是否存在
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print(f'错误：未找到预处理数据！请先运行 create_data.py 生成 {dataset}_train.pt 和 {dataset}_test.pt')
        sys.exit(1)  # 终止程序，避免后续报错

    # 加载数据集
    print(f'\n加载数据集...')
    train_data = TestbedDataset(root=data_root, dataset=f'{dataset}_train')
    test_data = TestbedDataset(root=data_root, dataset=f'{dataset}_test')

    # 拆分训练集为「训练集+验证集」（避免测试集泄露）
    print(f'拆分训练集为训练集({(1 - VAL_SPLIT_RATIO) * 100}%)和验证集({VAL_SPLIT_RATIO * 100}%)...')
    train_idx, val_idx = train_test_split(
        range(len(train_data)),
        test_size=VAL_SPLIT_RATIO,
        random_state=42  # 固定随机种子，保证结果可复现
    )
    val_data = [train_data[i] for i in val_idx]
    train_data = [train_data[i] for i in train_idx]

    # 构建数据加载器
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # 模型初始化
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 学习率调度：验证集MSE连续10轮无提升，学习率×0.5（加速收敛）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 以MSE最小化为目标
        patience=10,
        factor=0.5,
        verbose=True  # 打印学习率变化
    )

    # 最优指标记录
    best_val_mse = float('inf')  # 初始值设为无穷大
    best_ci = 0.0
    best_epoch = -1
    no_improvement = 0  # 连续无提升轮次计数

    # 模型和结果保存路径
    model_save_path = f'model_{model_st}_{dataset}.pth'  # 改用.pth后缀（PyTorch标准）
    result_save_path = f'result_{model_st}_{dataset}.csv'

    print(f'\n开始训练（共{NUM_EPOCHS}轮，早停耐心值{PATIENCE}）...')
    print(f'最优模型将保存至: {model_save_path}')
    print(f'测试集结果将保存至: {result_save_path}')

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        current_epoch = epoch + 1

        # 1. 训练模型
        train(model, device, train_loader, optimizer, current_epoch)

        # 2. 验证集评估（判断性能提升和早停）
        val_G, val_P = predicting(model, device, val_loader)
        val_rmse, val_mse, val_pearson, val_spearman, val_ci = rmse(val_G, val_P), mse(val_G, val_P), pearson(val_G,
                                                                                                              val_P), spearman(
            val_G, val_P), ci(val_G, val_P)

        # 3. 学习率调度（基于验证集MSE）
        scheduler.step(val_mse)

        # 4. 判断是否更新最优模型
        if val_mse < best_val_mse:
            # 性能提升：更新最优指标+保存模型+重置无提升计数
            best_val_mse = val_mse
            best_ci = val_ci
            best_epoch = current_epoch
            no_improvement = 0

            # 保存最优模型（仅保存参数，节省空间）
            torch.save(model.state_dict(), model_save_path)

            # 在测试集上评估最终性能（仅最优模型时评估，避免测试集泄露）
            test_G, test_P = predicting(model, device, test_loader)
            test_rmse, test_mse, test_pearson, test_spearman, test_ci = rmse(test_G, test_P), mse(test_G,
                                                                                                  test_P), pearson(
                test_G, test_P), spearman(test_G, test_P), ci(test_G, test_P)

            # 保存测试集结果
            with open(result_save_path, 'w', encoding='utf-8') as f:
                f.write('RMSE,MSE,Pearson,Spearman,CI\n')
                f.write(f'{test_rmse:.6f},{test_mse:.6f},{test_pearson:.6f},{test_spearman:.6f},{test_ci:.6f}')

            # 中文格式化日志（性能提升）
            print(f'\n【性能提升】第{best_epoch}轮')
            print(f'验证集指标：MSE={val_mse:.6f}, CI={val_ci:.4f}, RMSE={val_rmse:.6f}')
            print(f'测试集最优指标：MSE={test_mse:.6f}, CI={test_ci:.4f}, RMSE={test_rmse:.6f}')
            print(f'模型已保存至: {model_save_path}')

        else:
            # 无性能提升：计数+1，判断早停
            no_improvement += 1
            # 中文格式化日志（无提升）
            print(f'\n【无提升】第{current_epoch}轮（已连续{no_improvement}轮）')
            print(f'验证集指标：MSE={val_mse:.6f}, CI={val_ci:.4f}（最优MSE={best_val_mse:.6f}）')

            # 早停触发条件——改进
            if no_improvement >= PATIENCE:
                print(f'\n==================================================')
                print(f'早停触发！连续{PATIENCE}轮验证集无性能提升')
                print(f'==================================================')
                break  # 终止训练循环

    # -------------------------- 训练总结 --------------------------
    print(f'\n\n【训练完成】')
    print(f'最优轮次：第{best_epoch}轮')
    print(f'验证集最优指标：MSE={best_val_mse:.6f}, CI={best_ci:.4f}')
    print(f'测试集最终指标：RMSE={test_rmse:.6f}, MSE={test_mse:.6f}, CI={test_ci:.4f}')
    print(f'模型保存路径：{model_save_path}')
    print(f'结果保存路径：{result_save_path}')
    print(f'总训练轮次：{current_epoch}（实际训练轮次，未达上限{NUM_EPOCHS}）')

# -------------------------- 额外提醒 --------------------------
print(f'\n温馨提示：')
print(f'1. 若运行时出现FutureWarning，请修改 utils.py 第20行：')
print(f'   原代码：self.data, self.slices = torch.load(self.processed_paths[0])')
print(f'   修改为：self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)')
print(f'2. 运行命令示例（适配你的GPU编号0）：')
print(f'   python training.py 0 0 0')
print(f'   （参数说明：0=davis数据集，0=GINConvNet模型，0=GPU编号）')