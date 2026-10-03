import os
import sys
import numpy as np
import torch
from rdkit import Chem
from torch_geometric import data as DATA

# 彻底屏蔽RDKit所有警告（包括废弃警告）
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# ===================== 原子特征提取（兼容新版RDKit） =====================
def atom_features(atom):
    """提取单个原子的78维特征（float32）"""
    symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'other']
    degrees = [0, 1, 2, 3, 4, 5]
    hybridization = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2,
                     'other']
    feature = []
    # 原子类型（one-hot）
    sym = atom.GetSymbol()
    feature += [1.0 if sym == s else 0.0 for s in symbols]
    # 原子度数（one-hot）
    deg = atom.GetDegree()
    feature += [1.0 if deg == d else 0.0 for d in degrees]
    # 杂化类型（one-hot）
    hyb = atom.GetHybridization()
    feature += [1.0 if hyb == h else 0.0 for h in hybridization[:-1]] + [0.0 if hyb in hybridization[:-1] else 1.0]
    # 其他特征（修复RDKit废弃API）
    feature.append(1.0 if atom.GetIsAromatic() else 0.0)
    feature.append(float(atom.GetFormalCharge()))
    feature.append(float(atom.GetNumRadicalElectrons()))
    feature.append(float(atom.GetImplicitValence()))
    feature.append(float(atom.GetExplicitValence()))
    feature.append(float(atom.GetTotalValence()))
    # 确保78维
    feature = feature[:78] + [0.0] * (78 - len(feature))
    return np.array(feature, dtype=np.float32)


# ===================== 分子图构建（源头固化类型） =====================
def smile_to_graph(smiles):
    """生成分子图数据（返回：原子数, 原子特征, 边索引）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        # 节点特征（78维 float32）
        features = [atom_features(atom) for atom in mol.GetAtoms()]
        features = np.array(features, dtype=np.float32)

        # 边索引（强制 int64）
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])  # 无向图双向边
        edge_index = np.array(edges).T.astype(np.int64) if edges else np.array([[], []], dtype=np.int64)

        return (mol.GetNumAtoms(), features, edge_index)
    except Exception as e:
        print(f"❌ smile_to_graph失败：{e}")
        return None


# ===================== 数据集类（彻底修复所有属性/类型问题） =====================
class TestbedDataset(DATA.Dataset):
    def __init__(self, root, dataset_name, xd, xt, y, smile_graph, transform=None, pre_transform=None):
        """
        :param root: 数据根目录
        :param dataset_name: 数据集名称（避免与父类属性冲突）
        :param xd: SMILES列表
        :param xt: 蛋白特征列表（氨基酸索引，int64）
        :param y: 标签列表
        :param smile_graph: 分子图字典
        """
        self.root = root
        self.dataset_name = dataset_name  # 核心：替换原self.dataset为dataset_name
        self.xd = xd
        self.xt = xt
        self.y = y
        self.smile_graph = smile_graph
        self.transform = transform
        self.pre_transform = pre_transform
        # 手动触发数据处理（摆脱父类自动触发的坑）
        self.process()
        super(TestbedDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """返回原始文件名"""
        return [f"{self.dataset_name}.csv"]

    @property
    def processed_file_names(self):
        """返回处理后文件名"""
        return [f"{self.dataset_name}.pt"]

    def process(self):
        """处理数据并保存（核心修复：target转为long）"""
        data_list = []
        for i in range(len(self.xd)):
            # 获取单条数据
            smiles = self.xd[i]
            target = self.xt[i]  # 氨基酸索引（int64）
            label = self.y[i]
            n_atoms, atom_feat, edge_index = self.smile_graph[smiles]

            # 强制类型转换（关键：target转为long）
            atom_feat = torch.from_numpy(atom_feat).float()
            edge_index = torch.from_numpy(edge_index).long()
            target = torch.from_numpy(target).long()  # 适配Embedding层
            label = torch.FloatTensor([label])

            # 构建Data对象
            data = DATA.Data(x=atom_feat, edge_index=edge_index, y=label)
            data.target = target
            data_list.append(data)

        # 确保processed目录存在
        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        # 保存处理后的数据
        torch.save(data_list, os.path.join(processed_dir, self.processed_file_names[0]))

    def len(self):
        """返回数据集长度"""
        return len(self.xd)

    def get(self, idx):
        """获取指定索引的数据（消除torch.load警告）"""
        processed_path = os.path.join(self.root, 'processed', self.processed_file_names[0])
        # 显式指定weights_only=False，兼容旧模型
        data_list = torch.load(processed_path, weights_only=False)
        return data_list[idx]