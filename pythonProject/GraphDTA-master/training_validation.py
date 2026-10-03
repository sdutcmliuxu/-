import numpy as np
import pandas as pd
import sys, os
import random
import torch
import torch.nn as nn
# PyG专属工具，处理图数据核心
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from models.gcn import GCNNet
# 新增：导入相关系数/CI计算所需库
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import math

# ===================== 内置所有评估函数（彻底摆脱utils.py依赖，解决NameError） =====================
def mse(y_true, y_pred):
    """均方误差"""
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    """均方根误差"""
    return math.sqrt(mean_squared_error(y_true, y_pred))

def pearson(y_true, y_pred):
    """皮尔森相关系数（小样本判断，避免计算错误）"""
    if len(y_true) < 2:
        return 0.0
    try:
        r, _ = pearsonr(y_true, y_pred)
        return r if not np.isnan(r) else 0.0
    except:
        return 0.0

def spearman(y_true, y_pred):
    """斯皮尔曼相关系数（小样本判断）"""
    if len(y_true) < 2:
        return 0.0
    try:
        r, _ = spearmanr(y_true, y_pred)
        return r if not np.isnan(r) else 0.0
    except:
        return 0.0

def ci(y_true, y_pred):
    """置信区间CI指数（回归任务通用计算，适配小样本）"""
    if len(y_true) < 2:
        return 0.0
    try:
        n = len(y_true)
        correct = 0
        for i in range(n):
            for j in range(i+1, n):
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    correct += 1
        return correct / (n*(n-1)/2) if n*(n-1)/2 > 0 else 0.0
    except:
        return 0.0

# ===================== 训练函数（无修改，适配所有修复） =====================
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx + 1,
                len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                loss.item()
            ))

# ===================== 预测函数（无修改） =====================
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# ===================== 全局配置（适配你的环境） =====================
cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print('✅ 运行设备：', device)

# 微调核心参数（7条数据专属）
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
LR_FINETUNE = 5e-5  # 低学习率保护预训练特征
LOG_INTERVAL = 5
NUM_EPOCHS = 100
EARLY_STOP_EPOCH = 5
SEED = 42  # 固定种子可复现

# 你的文件路径（已确认，无需修改）
PRETRAIN_MODEL_PATH = "model_GCNNet_davis.pth"
DATASET_NAME = "tagln2_train"
FINETUNE_MODEL_SAVE = "model_save/tagln2_finetune_7samples_best.pth"
RESULT_FILE_NAME = "tagln2_7samples_finetune_result.csv"
PROCESSED_DATA_FILE = f'data/processed/{DATASET_NAME}.pt'

# 固定所有随机种子（彻底可复现）
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 全局损失函数
loss_fn = nn.MSELoss()

# ===================== 图数据自动修复函数（解决edge_index维度/类型错误） =====================
def fix_pyg_data(data_list):
    fixed_list = []
    for data in data_list:
        # 修复edge_index：N×2 → 2×N，强制long类型（PyG GCN要求）
        if data.edge_index.dim() == 2 and data.edge_index.shape[0] != 2:
            data.edge_index = data.edge_index.t().contiguous()
        data.edge_index = data.edge_index.long()
        # 保证特征/标签类型正确
        if data.x is not None:
            data.x = data.x.float()
        if data.y is not None:
            data.y = data.y.float()
        fixed_list.append(data)
    return fixed_list

# ===================== 主微调逻辑（所有错误全修复，直接训练出结果） =====================
if __name__ == "__main__":
    # 检查文件
    if not os.path.isfile(PROCESSED_DATA_FILE):
        print(f'❌ 未找到PT数据：{PROCESSED_DATA_FILE}')
        sys.exit(1)
    if not os.path.isfile(PRETRAIN_MODEL_PATH):
        print(f'❌ 未找到预训练权重：{PRETRAIN_MODEL_PATH}')
        sys.exit(1)
    os.makedirs(os.path.dirname(FINETUNE_MODEL_SAVE), exist_ok=True)

    # 加载并修复数据集
    print(f'\n📌 加载Transgelin-2预处理PT数据集')
    total_data = torch.load(PROCESSED_DATA_FILE, weights_only=False)
    print(f'✅ 原始数据集加载成功！共{len(total_data)}条PyG图数据')
    total_data = fix_pyg_data(total_data)
    print(f'✅ 图数据格式修复完成！校正edge_index为2×N+long类型')

    # 划分数据集：6训1验
    train_size = len(total_data) - 1
    valid_size = 1
    train_data, valid_data = torch.utils.data.random_split(
        total_data, [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f'✅ 数据集划分完成：{train_size}训 | {valid_size}验（固定种子）')

    # 构建PyG DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)

    # 加载预训练模型
    print(f'\n📌 加载davis预训练GCNNet模型')
    model = GCNNet().to(device)
    model.load_state_dict(torch.load(PRETRAIN_MODEL_PATH, map_location=device, weights_only=False))
    print('✅ 预训练权重加载完成！模型已移到', device)

    # 冻结底层，仅训顶层fc层
    print(f'\n📌 冻结底层GCN/Conv层，仅训练顶层fc*预测层...')
    for name, param in model.named_parameters():
        param.requires_grad = False if 'fc' not in name else True
    trainable_layers = [n for n,p in model.named_parameters() if p.requires_grad]
    print(f'✅ 可训练层：{trainable_layers}')

    # 初始化优化器
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_FINETUNE)
    print(f'\n✅ 优化器初始化完成：仅更新可训练层，学习率={LR_FINETUNE}')

    # 训练监控初始化
    best_mse = 1000.0
    best_epoch = -1
    no_improve_count = 0
    best_ret = [0.0, 0.0, 0.0, 0.0, 0.0]  # RMSE, MSE, Pearson, Spearman, CI

    # 启动微调训练（最终无报错版本）
    print(f'\n📌 开始Transgelin-2模型微调（学习率{LR_FINETUNE}，{EARLY_STOP_EPOCH}轮早停）')
    for epoch in range(NUM_EPOCHS):
        # 训练一轮
        train(model, device, train_loader, optimizer, epoch + 1)
        # 验证集预测
        G, P = predicting(model, device, valid_loader)
        # 计算所有评估指标（内置函数，无未定义错误）
        val_rmse = rmse(G, P)
        val_mse = mse(G, P)
        val_pearson = pearson(G, P)
        val_spearman = spearman(G, P)
        val_ci = ci(G, P)
        current_ret = [val_rmse, val_mse, val_pearson, val_spearman, val_ci]

        # 保存最优模型
        if val_mse < best_mse:
            best_mse, best_epoch, best_ret = val_mse, epoch+1, current_ret
            no_improve_count = 0
            torch.save(model.state_dict(), FINETUNE_MODEL_SAVE)
            # 保存结果到CSV
            with open(RESULT_FILE_NAME, 'w', encoding='utf-8') as f:
                f.write('RMSE,MSE,Pearson,Spearman,CI\n')
                f.write(','.join([f'{v:.6f}' for v in best_ret]))
            print(f'🎉 第{best_epoch}轮优化！RMSE={val_rmse:.6f} | MSE={val_mse:.6f}，模型已保存\n')
        else:
            no_improve_count += 1
            print(f'⚠️  无优化，当前MSE={val_mse:.6f}，最优MSE={best_mse:.6f}')
            print(f'⚠️  连续无提升{no_improve_count}/{EARLY_STOP_EPOCH}\n')
            if no_improve_count >= EARLY_STOP_EPOCH:
                print(f'\n🛑 触发早停！微调训练完成')
                break

    # 训练完成最终总结
    print(f'\n==================== Transgelin-2模型微调圆满完成 ====================')
    print(f'✅ 训练数据：{len(total_data)}条有效图数据（6训1验）| 设备：{device}')
    print(f'✅ 最优轮次：{best_epoch} | 最优RMSE：{best_ret[0]:.6f} | 最优MSE：{best_ret[1]:.6f}')
    print(f'✅ 相关系数：Pearson={best_ret[2]:.6f} | Spearman={best_ret[3]:.6f} | CI={best_ret[4]:.6f}')
    print(f'✅ 微调模型路径：{FINETUNE_MODEL_SAVE}')
    print(f'✅ 评估结果文件：{RESULT_FILE_NAME}')
    print(f'=======================================================================')