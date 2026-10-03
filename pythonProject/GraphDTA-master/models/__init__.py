# -*- coding: utf-8 -*-
"""
Models 包初始化文件
作用：
1. 将models目录标记为Python包
2. 统一导出包内的模型类，简化外部导入逻辑
3. 兼容GraphDTA项目常见的GCN/GAT等模型
"""

# 核心导出：GCNNet（预测流程必需）
from .gcn import GCNNet

# 可选导出（如项目包含其他模型，取消注释即可）
# from .gat import GATNet
# from .gat_gcn import GATGCNNet
# from .mpnn import MPNN

# 定义__all__，明确对外暴露的接口（规范导入）
__all__ = [
    "GCNNet",
    # "GATNet",
    # "GATGCNNet",
    # "MPNN"
]