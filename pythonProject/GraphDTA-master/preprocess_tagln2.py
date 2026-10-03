# 顶部核心配置：解决Windows中文环境打印乱码+无输出问题，强制编码
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 基础模块导入
import pandas as pd
import numpy as np
import os
import json
import pickle
import traceback
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch

# 第一步强制打印：确认脚本启动，排除无输出基础问题
print("===== Transgelin-2数据集预处理脚本 启动 =====")
print(f"Python解释器路径：{sys.executable}")
print(f"当前脚本工作目录：{os.getcwd()}")

# 关键：捕获utils.py导入错误（无输出的核心原因之一）
try:
    from utils import *

    print("✅ 自定义工具类 utils.py 导入成功！")
except Exception as e:
    print(f"❌ 致命错误：utils.py 导入失败！")
    print(f"错误原因：{type(e).__name__} - {e}")
    print(f"解决方法：确保utils.py在GraphDTA-master根目录，文件名无拼写错误")
    sys.exit(1)


# ===================== 1. 核心特征提取函数（全修复：原子特征+边索引均为numpy数组） =====================
def atom_features(atom):
    """原子特征提取：匹配原GraphDTA，返回numpy数组，括号完全匹配"""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                           'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                                           'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])  # 无多余括号，语法正确


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """未知特征映射到最后一个维度，兜底处理"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """SMILES转分子图：核心修复→原子特征+边索引均为numpy数组，彻底解决TypeError"""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:  # 解析失败返回空，后续自动跳过
        return None, None, None
    c_size = mol.GetNumAtoms()
    # 提取原子特征并归一化，强制转numpy数组
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    features = np.array(features)  # 修复1：原子特征 列表→numpy数组

    # 提取化学键边，转有向图边索引
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    edge_index = np.array(edge_index)  # 修复2：边索引 列表→numpy数组（本次错误核心修复点）

    return c_size, features, edge_index  # 返回：原子数、原子特征(numpy)、边索引(numpy)


def seq_cat(prot):
    """蛋白序列编码：修复空格+非法字符兜底，解决KeyError"""
    x = np.zeros(max_seq_len)
    prot = prot.replace(" ", "")  # 强制删除所有空格
    for i, ch in enumerate(prot[:max_seq_len]):
        # 仅合法氨基酸编码，非法字符赋值0
        x[i] = seq_dict[ch] if ch in seq_dict else 0
    return x


# ===================== 2. 全局参数+绝对路径配置（固定你的环境，无需修改） =====================
print("\n===== 开始配置全局参数和绝对路径 =====")
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 氨基酸字典（无空格）
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
max_seq_len = 1000  # 蛋白序列最大长度，匹配原GraphDTA
dataset_name = 'tagln2_train'  # TestbedDataset必选参数，不可改

# 你的GraphDTA-master根目录（绝对路径，彻底解决文件找不到问题）
PROJECT_ROOT = r"D:\Transgelin-2\pythonProject\GraphDTA-master"
# 拼接绝对路径，避免工作目录错位
csv_path = os.path.join(PROJECT_ROOT, "data", "tagln2_finetune.csv")  # 你的CSV路径
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")  # PT文件保存目录
processed_data_file = os.path.join(processed_dir, "tagln2_train.pt")  # 最终PT文件绝对路径

# 自动创建目录（即使手动没建）
os.makedirs(processed_dir, exist_ok=True)
print(f"✅ 数据集保存目录：{processed_dir}")

# 关键检查：确认CSV文件存在（无输出的核心原因之二）
if os.path.exists(csv_path):
    print(f"✅ 找到CSV数据文件：{csv_path}")
else:
    print(f"❌ 致命错误：CSV文件不存在！")
    print(f"请检查路径：{csv_path}")
    print(f"解决方法：确保tagln2_finetune.csv在GraphDTA-master/data/目录下")
    sys.exit(1)

# ===================== 3. 主执行逻辑（带全局异常捕获，所有错误显式打印） =====================
if __name__ == "__main__":
    try:
        # 3.1 加载并清洗CSV数据
        print("\n===== 开始加载并清洗CSV数据 =====")
        # 读取CSV（若Excel分号分隔，取消下一行注释：df = pd.read_csv(csv_path, sep=';')）
        df = pd.read_csv(csv_path)
        # 提取核心列（和你的CSV列名严格对应，无需修改）
        drugs = list(df['compound_iso_smiles'])
        prots = list(df['target_sequence'])
        Y = list(df['affinity'])

        # 全局双重清洗：删除蛋白+SMILES中的空格（防止意外问题）
        prots = [p.replace(" ", "") for p in prots]
        drugs = [d.replace(" ", "") for d in drugs]

        # 蛋白序列编码为数值矩阵，转numpy数组
        XT = [seq_cat(t) for t in prots]
        drugs_np, prots_np, Y_np = np.asarray(drugs), np.asarray(XT), np.asarray(Y)
        print(f"✅ 数据加载完成：{len(drugs)}个小分子 | 1个Transgelin-2蛋白序列（已去空格）")

        # 3.2 构建分子图，过滤无效数据
        print("\n===== 开始构建分子图并过滤无效数据 =====")
        smile_graph = {}
        failed_smiles = []  # 记录解析失败的SMILES
        for smile in set(drugs):
            g = smile_to_graph(smile)
            if g[0] is not None:
                smile_graph[smile] = g
            else:
                print(f"⚠️  SMILES {smile} 解析失败，自动跳过")
                failed_smiles.append(smile)

        # 过滤无效数据：仅保留分子图构建成功的样本
        valid_indices = [i for i, smi in enumerate(drugs_np) if smi in smile_graph]
        drugs_np = drugs_np[valid_indices]
        prots_np = prots_np[valid_indices]
        Y_np = Y_np[valid_indices]

        # 打印分子图构建总结
        print(f"✅ 分子图构建完成：{len(smile_graph)}个合法分子 | {len(failed_smiles)}个解析失败")
        print(f"✅ 有效数据过滤完成：保留{len(drugs_np)}条数据用于模型微调")

        # 3.3 生成PyTorch Geometric格式PT文件
        print("\n===== 开始生成模型可用的PT数据集文件 =====")
        if not os.path.isfile(processed_data_file):
            # 初始化TestbedDataset（适配你的utils.py，传入所有必选参数）
            train_data = TestbedDataset(
                root=PROJECT_ROOT,  # 项目根目录绝对路径
                dataset_name=dataset_name,  # 补上必选参数（解决TypeError）
                xd=drugs_np,  # 合法SMILES数组
                xt=prots_np,  # 编码后蛋白序列(numpy)
                y=Y_np,  # 亲和力数值(numpy)
                smile_graph=smile_graph  # 预构建分子图字典（特征+边均为numpy）
            )
            # 手动强制保存PT文件到绝对路径（彻底解决文件找不到）
            torch.save(train_data, processed_data_file)
            print(f"✅ PT数据集生成成功！")
            print(f"📌 文件绝对路径：{processed_data_file}")
        else:
            print(f"✅ PT数据集已存在，无需重复生成！")
            print(f"📌 文件绝对路径：{processed_data_file}")

        # 最终完成提示
        print("\n===== 🎉 Transgelin-2数据集预处理 全部完成！🎉 =====")
        print(f"👉 下一步：修改training.py，加载{processed_data_file}进行模型微调")

    # 捕获所有可能的错误，显式打印错误类型、原因、行号
    except Exception as e:
        print("\n===== ❌ 脚本执行出错！错误详情如下 ❌ =====")
        print(f"错误类型：{type(e).__name__}")
        print(f"错误原因：{e}")
        print(f"出错位置（行号+代码）：\n{traceback.format_exc()}")
        sys.exit(1)