import os
import sys
import torch
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from rdkit import Chem
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

# ===================== 自定义模块导入（仅保留必需） =====================
try:
    from models.gcn import GCNNet
    from utils import TestbedDataset, smile_to_graph
except ImportError as e:
    print(f"❌ 导入自定义模块失败：{e}")
    print("请确认：")
    print("  1. models/gcn.py 文件存在且包含 GCNNet 类")
    print("  2. utils.py 文件存在且包含 TestbedDataset、smile_to_graph 函数")
    sys.exit(1)

# ===================== 全局配置（仅需确认【微调模型路径】） =====================
# 数据集根目录（需与训练时完全一致）
DATA_ROOT = r'D:\Transgelin-2\pythonProject\GraphDTA-master\data'
# 🌟 核心：替换为你的Transgelin-2微调模型实际路径（必改！）
MODEL_PATH = r"D:\Transgelin-2\pythonProject\GraphDTA-master\model_save\tagln2_finetune_7samples_best.pth"
# ✅ 已替换为你指定的 Transgelin-2 蛋白序列
TARGET_PROTEIN_SEQ = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
# 蛋白序列固定长度（必须与训练时一致，不可随意修改）
MAX_PROT_LEN = 1000
# 设备配置（自动检测GPU/CPU，无需改动）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Transgelin-2 专属 pKd 结合强度分级（适配微调模型，贴合实验需求）
PKD_LEVELS = {
    "强结合": (8.0, 10.0),    # 优先实验验证，潜在候选分子
    "中等结合": (7.0, 8.0),  # 可进行结构优化，提升亲和力
    "弱结合": (5.0, 7.0),    # 亲和力较低，无直接实验价值
    "几乎不结合": (0.0, 5.0) # 无结合活性，直接排除
}
# 日志保存路径（自动生成，无需改动）
LOG_PATH = "transgelin2_finetune_predict_log.txt"


# ===================== 工具函数（无需改动） =====================
class NullIO:
    """屏蔽RDKit/PyG无关警告，避免干扰输出"""
    def write(self, msg): pass
    def flush(self): pass

def log_info(msg):
    """记录预测日志（终端+文件双保存），带时间戳"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")
    print(log_msg)
    sys.stdout.flush()


# ===================== 核心编码函数（与训练时完全一致，不可改动） =====================
def encode_protein(seq, max_len=MAX_PROT_LEN):
    """蛋白序列编码：生成氨基酸索引，与训练时Embedding层完全适配"""
    aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
               'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    # 截断/填充至训练时的固定长度
    seq = seq[:max_len] if len(seq) > max_len else seq.ljust(max_len, 'A')
    # 未知氨基酸映射为0（与训练逻辑一致）
    prot_idx = [aa_dict.get(aa, 0) for aa in seq]
    return np.array(prot_idx, dtype=np.int64)

def validate_smiles(smiles):
    """严格校验SMILES合法性，过滤无效分子"""
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except:
        return False

def get_pkd_level(pkd):
    """根据微调模型预测pKd，判断与Transgelin-2的结合强度"""
    for level, (min_val, max_val) in PKD_LEVELS.items():
        if min_val <= pkd <= max_val:
            return level
    return "异常值"


# ===================== 模型加载（适配微调模型，严格参数匹配） =====================
def load_model():
    """加载微调后的Transgelin-2模型，确保参数100%匹配"""
    null_io = NullIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = null_io, null_io

    try:
        # 校验模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"微调模型文件不存在！请检查路径：{MODEL_PATH}")
        # 加载模型（禁用宽松加载，确保微调参数完全匹配）
        model_data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        # 初始化模型并加载参数（与训练时的GCNNet完全一致）
        if isinstance(model_data, dict):
            model = GCNNet().to(DEVICE)
            model.load_state_dict(model_data, strict=True)  # 微调模型必须严格加载，不可改False
        else:
            model = model_data.to(DEVICE)
        # 切换为评估模式，禁用Dropout/BatchNorm
        model.eval()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_info(f"✅ 微调模型加载成功 | 运行设备：{DEVICE} | 蛋白序列：Transgelin-2（你的指定版本）")
        return model
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_info(f"❌ 微调模型加载失败：{str(e)[:200]}")
        log_info("💡 排查建议：1. 检查模型路径是否正确 2. 确认模型是Transgelin-2微调后的版本 3. 确保GCNNet结构与训练时一致")
        return None


# ===================== 核心预测函数（单条SMILES，微调模型专属） =====================
def predict_single_smiles(model, smiles):
    """单条SMILES预测与Transgelin-2的结合亲和力，返回(pKd, 结合强度, 错误信息)"""
    smiles = smiles.strip()
    # 第一步：校验SMILES合法性
    if not validate_smiles(smiles):
        error_msg = f"无效SMILES（无法解析为分子）：{smiles}"
        log_info(f"❌ {error_msg}")
        return None, None, error_msg

    try:
        # 1. 构建分子图（与训练时的smile_to_graph完全一致）
        log_info(f"🔧 构建分子图：{smiles}")
        graph = smile_to_graph(smiles)
        if graph is None:
            error_msg = f"分子图构建失败：{smiles}"
            log_info(f"❌ {error_msg}")
            return None, None, error_msg
        n_atoms, atom_feat, edge_index = graph
        smile_graph = {smiles: graph}

        # 2. 编码蛋白序列（使用你指定的Transgelin-2序列）
        log_info(f"🔧 编码Transgelin-2蛋白序列（长度：{len(TARGET_PROTEIN_SEQ)}）")
        prot_encoded = encode_protein(TARGET_PROTEIN_SEQ)

        # 3. 构建临时数据集（与训练时的TestbedDataset参数一致）
        temp_dataset_name = f"finetune_predict_temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        dataset = TestbedDataset(
            root=DATA_ROOT,
            dataset_name=temp_dataset_name,
            xd=[smiles],
            xt=[prot_encoded],
            y=[0.0],
            smile_graph=smile_graph
        )

        # 强制类型匹配（双重保险，与训练时张量类型一致）
        for data in dataset:
            data.edge_index = data.edge_index.long()
            data.x = data.x.float()
            data.target = data.target.long()

        # 4. 模型推理（禁用梯度，提升速度）
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        log_info(f"🔍 开始微调模型推理（设备：{DEVICE}）")
        pkd = None
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(DEVICE)
                # 最终类型校验，避免张量类型错误
                batch.edge_index = batch.edge_index.long()
                batch.target = batch.target.long()
                # 前向传播（微调模型专属推理）
                output = model(batch)
                pkd = round(output.item(), 4)  # 保留4位小数，提升精度

        if pkd is None:
            raise ValueError("微调模型推理未生成有效pKd值")

        # 5. 结合强度分级+结果记录
        level = get_pkd_level(pkd)
        log_info(f"✅ 预测完成 | SMILES：{smiles} | pKd：{pkd} | 结合强度：{level}")

        # 清理临时文件，避免冗余
        temp_file = os.path.join(DATA_ROOT, 'processed', f'{temp_dataset_name}.pt')
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return pkd, level, None
    except Exception as e:
        error_msg = f"预测失败：{str(e)[:150]}"
        log_info(f"❌ {error_msg} | SMILES：{smiles}")
        traceback.print_exc()
        return None, None, error_msg


# ===================== 批量预测函数（CSV文件，含SMILES列） =====================
def predict_batch_csv(model, input_csv, output_csv="transgelin2_finetune_batch_result.csv"):
    """批量预测CSV中的SMILES，自动生成结果文件（含pKd、结合强度、错误信息）"""
    try:
        # 读取输入CSV，必须包含"SMILES"列
        df = pd.read_csv(input_csv, encoding="utf-8")
        if "SMILES" not in df.columns:
            raise ValueError(f"输入CSV必须包含'SMILES'列！当前列：{df.columns.tolist()}")
        total = len(df)
        # 初始化结果列
        df["pKd"] = np.nan
        df["结合强度"] = ""
        df["错误信息"] = ""

        log_info(f"📦 开始批量预测 | 总分子数：{total} | 输入文件：{os.path.abspath(input_csv)}")
        # 逐行预测
        for idx, row in df.iterrows():
            smiles = str(row["SMILES"]).strip()
            pkd, level, error = predict_single_smiles(model, smiles)
            # 填充结果
            if pkd is not None:
                df.loc[idx, "pKd"] = pkd
                df.loc[idx, "结合强度"] = level
            if error is not None:
                df.loc[idx, "错误信息"] = error
            # 每10条打印一次进度
            if (idx + 1) % 10 == 0:
                progress = ((idx + 1) / total) * 100
                log_info(f"📈 进度：{idx + 1}/{total} | {progress:.1f}% | 已发现强结合分子：{df[df['结合强度']=='强结合'].shape[0]}")

        # 保存批量结果
        df.to_csv(output_csv, index=False, encoding="utf-8")
        success_count = df["pKd"].notna().sum()
        high_affinity = df[df["结合强度"] == "强结合"].shape[0]
        log_info(f"✅ 批量预测完成 | 成功：{success_count}/{total} | 强结合分子：{high_affinity}")
        log_info(f"📄 结果文件保存至：{os.path.abspath(output_csv)}")
    except Exception as e:
        log_info(f"❌ 批量预测失败：{str(e)}")
        traceback.print_exc()


# ===================== 主交互函数（可视化菜单，无需改动） =====================
def main():
    # 初始化日志文件（覆盖旧日志）
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("===== Transgelin-2 微调模型亲和力预测日志 =====\n")
        f.write(f"蛋白序列：{TARGET_PROTEIN_SEQ}\n")
        f.write(f"微调模型路径：{MODEL_PATH}\n")
        f.write(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 加载微调模型（核心步骤）
    model = load_model()
    if model is None:
        sys.exit(1)

    # 主菜单可视化
    print("\n" + "=" * 80)
    print("🔍 Transgelin-2 小分子亲和力预测工具（微调GCN模型版）")
    print(f"📌 运行配置 | 设备：{DEVICE} | 蛋白序列：你的指定版本 | 模型：微调专属")
    print("📌 功能选择：")
    print("   1. 单条SMILES预测（交互模式，适合少量分子验证）")
    print("   2. 批量CSV预测（批量模式，适合高通量筛选）")
    print("   3. 退出程序")
    print("=" * 80)

    # 交互逻辑
    while True:
        choice = input("\n请选择功能（1/2/3）：").strip()
        # 1. 单条SMILES预测
        if choice == "1":
            print("\n📝 单条预测模式 | 输入'ex'查看示例 | 输入'q'返回主菜单")
            while True:
                smiles = input("\n请输入SMILES：").strip()
                if smiles.lower() == 'q':
                    break
                if smiles.lower() == 'ex':
                    print("\n💡 示例SMILES（可直接复制测试）：")
                    print("   - 苯：C1=CC=CC=C1")
                    print("   - 乙醇：CCO")
                    print("   - 抗癌候选分子：COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1")
                    continue
                if not smiles:
                    print("⚠️  SMILES不能为空！")
                    continue
                # 执行预测并打印结果
                print(f"\n🔍 正在预测 {smiles} 与Transgelin-2的结合亲和力...")
                pkd, level, error = predict_single_smiles(model, smiles)
                print("\n" + "=" * 50)
                if error:
                    print(f"❌ 预测失败：{error}")
                else:
                    print(f"✅ Transgelin-2 亲和力预测结果")
                    print(f"   SMILES：{smiles}")
                    print(f"   预测pKd值：{pkd}")
                    print(f"   结合强度：{level}")
                    print(f"   实验建议：{'👉 优先进行湿实验验证！' if level == '强结合' else '👉 无实验价值/需结构优化'}")
                print("=" * 50)
        # 2. 批量CSV预测
        elif choice == "2":
            input_csv = input("\n请输入CSV文件路径（必须包含'SMILES'列）：").strip()
            if not os.path.exists(input_csv):
                print(f"❌ 文件不存在：{input_csv}")
                continue
            output_csv = input("请输入输出CSV路径（默认：transgelin2_batch_result.csv）：").strip()
            if not output_csv:
                output_csv = "transgelin2_batch_result.csv"
            # 执行批量预测
            predict_batch_csv(model, input_csv, output_csv)
        # 3. 退出程序
        elif choice == "3":
            log_info("👋 程序正常退出 | 所有预测日志已保存至：transgelin2_finetune_predict_log.txt")
            print("\n👋 程序已退出！预测日志已保存至当前目录的 transgelin2_finetune_predict_log.txt")
            break
        # 无效选择
        else:
            print("⚠️  无效选择，请输入1/2/3！")


# ===================== 程序入口（无需改动） =====================
if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs(DATA_ROOT, exist_ok=True)
    # 禁用PyTorch冗余警告
    torch.set_warn_always(False)
    # 启动预测工具
    main()