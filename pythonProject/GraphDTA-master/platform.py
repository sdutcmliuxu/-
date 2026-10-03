import sys
import os
import csv
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton,
    QGridLayout, QTextEdit, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from rdkit import Chem
from torch_geometric import data as DATA

from models.gcn import GCNNet

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ===================== 工具函数 =====================
def atom_features(atom):
    symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'other']
    degrees = [0, 1, 2, 3, 4, 5]
    hybridization = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]

    feature = []

    sym = atom.GetSymbol()
    if sym not in symbols[:-1]:
        sym = 'other'
    feature += [1.0 if sym == s else 0.0 for s in symbols]

    deg = atom.GetDegree()
    feature += [1.0 if deg == d else 0.0 for d in degrees]

    hyb = atom.GetHybridization()
    feature += [1.0 if hyb == h else 0.0 for h in hybridization[:-1]]
    feature += [0.0 if hyb in hybridization[:-1] else 1.0]

    feature.append(1.0 if atom.GetIsAromatic() else 0.0)
    feature.append(float(atom.GetFormalCharge()))
    feature.append(float(atom.GetNumRadicalElectrons()))
    feature.append(float(atom.GetImplicitValence()))
    feature.append(float(atom.GetExplicitValence()))
    feature.append(float(atom.GetTotalValence()))

    feature = feature[:78] + [0.0] * (78 - len(feature))
    return np.array(feature, dtype=np.float32)


def smile_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        features = [atom_features(atom) for atom in mol.GetAtoms()]
        features = np.array(features, dtype=np.float32)

        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])

        edge_index = np.array(edges).T.astype(np.int64) if edges else np.array([[], []], dtype=np.int64)
        return mol.GetNumAtoms(), features, edge_index
    except Exception:
        return None


def protein_seq_to_index(seq):
    aa_dict = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20
    }

    clean_seq = ''.join(seq.split()).upper()
    index_seq = [aa_dict.get(aa, 20) for aa in clean_seq]

    max_len = 1000
    if len(index_seq) > max_len:
        index_seq = index_seq[:max_len]
    else:
        index_seq += [20] * (max_len - len(index_seq))

    return np.array(index_seq, dtype=np.int64)


def parse_smiles_text_lines(lines):
    """
    支持：
    1. 每行一个 SMILES
    2. 每行: SMILES 名称
    """
    records = []

    for idx, raw_line in enumerate(lines, start=1):
        if raw_line is None:
            continue

        line = raw_line.replace('\ufeff', '').strip()
        if not line:
            continue

        if line.startswith('#'):
            continue

        parts = line.split()
        if not parts:
            continue

        smiles = parts[0].strip()
        name = " ".join(parts[1:]).strip() if len(parts) > 1 else f"molecule_{idx}"

        records.append({
            "line_no": idx,
            "smiles": smiles,
            "name": name
        })

    return records


def parse_smiles_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if os.path.getsize(file_path) == 0:
        return []

    records = []

    if ext == ".csv":
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []

            fieldnames = [name.strip() for name in reader.fieldnames]

            smiles_key = None
            name_key = None

            for col in fieldnames:
                low = col.lower()
                if low in ["smiles", "smile", "canonical_smiles"]:
                    smiles_key = col
                if low in ["name", "compound", "molecule", "id"]:
                    name_key = col

            if smiles_key is None:
                raise ValueError("CSV 文件中未找到 smiles 列，请使用 smiles / smile / canonical_smiles")

            for idx, row in enumerate(reader, start=2):
                smiles = str(row.get(smiles_key, "")).replace('\ufeff', '').strip()
                if not smiles:
                    continue

                name = str(row.get(name_key, f"molecule_{idx}")).strip() if name_key else f"molecule_{idx}"

                records.append({
                    "line_no": idx,
                    "smiles": smiles,
                    "name": name
                })
    else:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        records = parse_smiles_text_lines(lines)

    return records


def summarize_smiles_records(records):
    valid_count = 0
    invalid_count = 0

    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is not None:
            valid_count += 1
        else:
            invalid_count += 1

    return valid_count, invalid_count


# ===================== 预测线程 =====================
class PredictThread(QThread):
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)

    def __init__(self, dataset, drug_records, target_seq, model_path, device):
        super().__init__()
        self.dataset = dataset
        self.drug_records = drug_records
        self.target_seq = target_seq
        self.model_path = model_path
        self.device = device

    def run(self):
        results = []
        try:
            if not self.drug_records:
                self.error_signal.emit("药物分子列表为空！")
                self.finished_signal.emit(results)
                return

            if not self.target_seq.strip():
                self.error_signal.emit("蛋白序列不能为空！")
                self.finished_signal.emit(results)
                return

            self.result_signal.emit("🔍 正在加载微调模型...")
            model = GCNNet().to(self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
            model.eval()

            self.result_signal.emit("🔧 正在编码蛋白序列...")
            target_index = protein_seq_to_index(self.target_seq)
            target_tensor = torch.from_numpy(target_index).long().to(self.device)

            total = len(self.drug_records)
            self.result_signal.emit(f"📦 已载入 {total} 条药物记录，开始预测...\n")

            for i, record in enumerate(self.drug_records, start=1):
                smiles = record["smiles"]
                name = record["name"]
                line_no = record["line_no"]

                try:
                    graph = smile_to_graph(smiles)
                    if graph is None:
                        results.append({
                            "line_no": line_no,
                            "name": name,
                            "smiles": smiles,
                            "status": "failed",
                            "error": "无效SMILES",
                            "pkd": "",
                            "kd": "",
                            "strength": "",
                            "suggestion": ""
                        })

                        self.result_signal.emit(
                            f"❌ 第 {i}/{total} 条失败\n"
                            f"名称：{name}\n"
                            f"SMILES：{smiles}\n"
                            f"原因：无效 SMILES\n"
                            f"{'=' * 60}"
                        )
                        continue

                    _, atom_feat, edge_index = graph

                    atom_feat_tensor = torch.from_numpy(atom_feat).float()
                    edge_index_tensor = torch.from_numpy(edge_index).long()
                    label_tensor = torch.FloatTensor([0.0])

                    data = DATA.Data(x=atom_feat_tensor, edge_index=edge_index_tensor, y=label_tensor)
                    data.target = target_tensor
                    data = data.to(self.device)

                    with torch.no_grad():
                        output = model(data)
                        pkd = float(output.cpu().numpy().flatten()[0])

                    #if pkd >= 8.0:
                        strength = "强结合"
                        suggestion = "有实验价值，建议进一步验证"
                    #elif pkd >= 6.0:
                        strength = "中等结合"
                        suggestion = "需进一步结构优化"
                    #else:
                        strength = "弱结合"
                        suggestion = "不建议深入研究"

                    pkd = round(pkd, 4)
                    kd = 10 ** (-pkd)

                    results.append({
                        "line_no": line_no,
                        "name": name,
                        "smiles": smiles,
                        "status": "success",
                        "error": "",
                        "pkd": pkd,
                        "kd": f"{kd:.3e}",
                        "strength": strength,
                     #  "suggestion": suggestion
                    })

                    self.result_signal.emit(
                        f"✅ 第 {i}/{total} 条完成\n"
                        f"名称：{name}\n"
                        f"SMILES：{smiles}\n"
                        f"预测 pKd：{pkd}\n"
                        f"预测 Kd：{kd:.3e} M\n"
                        f"结合强度：{strength}\n"
                      #  f"实验建议：{suggestion}\n"
                        f"{'=' * 60}"
                    )

                except Exception as e:
                    results.append({
                        "line_no": line_no,
                        "name": name,
                        "smiles": smiles,
                        "status": "failed",
                        "error": str(e),
                        "pkd": "",
                        "kd": "",
                        "strength": "",
                     #   "suggestion": ""
                    })

                    self.result_signal.emit(
                        f"❌ 第 {i}/{total} 条异常\n"
                        f"名称：{name}\n"
                        f"SMILES：{smiles}\n"
                        f"原因：{str(e)}\n"
                        f"{'=' * 60}"
                    )

            success_count = sum(1 for r in results if r["status"] == "success")
            fail_count = len(results) - success_count

            self.result_signal.emit(
                "\n"
                f"✅ 批量预测完成！\n"
                f"数据集：{self.dataset}\n"
                f"蛋白序列长度：{len(''.join(self.target_seq.split()))}\n"
                f"总数：{len(results)}\n"
                f"成功：{success_count}\n"
                f"失败：{fail_count}\n"
            )

            self.finished_signal.emit(results)

        except Exception as e:
            self.error_signal.emit(f"批量预测出错：{str(e)}")
            self.finished_signal.emit(results)


# ===================== 主界面 =====================
class DrugTargetAffinityUI(QWidget):
    def __init__(self):
        super().__init__()

        self.PROJECT_ROOT = r"D:\Transgelin-2\pythonProject\GraphDTA-master"
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        self.DEFAULT_DRUG_FILE = os.path.join(self.DATA_DIR, "Examples.txt")

        self.MODEL_PATH = r"D:\Transgelin-2\pythonProject\GraphDTA-master\model_save\tagln2_finetune_7samples_best.pth"
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.drug_records = []
        self.predict_results = []
        self.current_drug_file_path = ""

        self.initUI()

    def initUI(self):
        self.setWindowTitle('一个判断药靶相互作用和结合亲和力的预测系统')
        self.setGeometry(100, 100, 1080, 760)

        title_font = QFont("微软雅黑", 14, QFont.Bold)
        label_font = QFont("微软雅黑", 11)
        button_font = QFont("微软雅黑", 10)

        self.title_label = QLabel('一个判断药靶相互作用和结合亲和力的预测系统')
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)

        self.dataset_label = QLabel('请选择数据集')
        self.dataset_label.setFont(label_font)
        self.dataset_label.setStyleSheet('color: red;')
        self.dataset_label.setAlignment(Qt.AlignCenter)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['请选择', 'Davis', 'KIBA', 'Human', 'C. elegans'])
        self.dataset_combo.setFont(label_font)

        self.drug_input_label = QLabel('输入药物分子')
        self.drug_input_label.setFont(label_font)

        self.drug_file_btn = QPushButton('选择药物分子文件')
        self.drug_file_btn.setFont(button_font)
        self.drug_file_btn.setStyleSheet('background-color: #00b359; color: white;')
        self.drug_file_btn.clicked.connect(self.select_drug_file)

        self.default_drug_btn = QPushButton('加载默认 Examples.txt')
        self.default_drug_btn.setFont(button_font)
        self.default_drug_btn.setStyleSheet('background-color: #16a085; color: white;')
        self.default_drug_btn.clicked.connect(self.load_default_drug_file)

        self.drug_seq_text = QTextEdit()
        self.drug_seq_text.setPlaceholderText('请输入药物分子 SMILES，支持一行一个，或导入 txt/smi/csv 文件')
        self.drug_seq_text.setFont(label_font)

        self.target_input_label = QLabel('输入蛋白分子')
        self.target_input_label.setFont(label_font)

        self.target_file_btn = QPushButton('选择蛋白分子文件')
        self.target_file_btn.setFont(button_font)
        self.target_file_btn.setStyleSheet('background-color: #00b359; color: white;')
        self.target_file_btn.clicked.connect(self.select_target_file)

        self.target_seq_text = QTextEdit()
        self.target_seq_text.setPlaceholderText('请输入蛋白氨基酸序列')
        self.target_seq_text.setFont(label_font)

        default_seq = (
            "MAAAKGRRGAILSRVQKIEKQYADLEQILIYWITTQCRKDVGRPQPGREFQNTQVQATQGVKGLQTLNVDLTKVVGDNTLSVELKELQAEAERYSVKEVETRLKQKLEEIQAKLDGADLSS"
        )
        self.target_seq_text.setText(default_seq)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText('实验结果将显示在这里...')
        self.result_text.setFont(label_font)
        self.result_text.setReadOnly(True)

        self.cancel_btn = QPushButton('取消')
        self.cancel_btn.setFont(button_font)
        self.cancel_btn.setStyleSheet('background-color: #3498db; color: white;')
        self.cancel_btn.clicked.connect(self.close)

        self.export_btn = QPushButton('导出结果')
        self.export_btn.setFont(button_font)
        self.export_btn.setStyleSheet('background-color: #27ae60; color: white;')
        self.export_btn.clicked.connect(self.export_results)

        self.predict_btn = QPushButton('预测')
        self.predict_btn.setFont(button_font)
        self.predict_btn.setStyleSheet('background-color: #e74c3c; color: white;')
        self.predict_btn.clicked.connect(self.start_predict)

        grid = QGridLayout()
        grid.addWidget(self.title_label, 0, 0, 1, 4)

        grid.addWidget(self.dataset_label, 1, 1, 1, 1)
        grid.addWidget(self.dataset_combo, 1, 2, 1, 1)

        grid.addWidget(self.drug_input_label, 2, 0, 1, 1)
        grid.addWidget(self.drug_file_btn, 2, 2, 1, 2)
        grid.addWidget(self.default_drug_btn, 3, 2, 1, 2)

        grid.addWidget(self.target_input_label, 4, 0, 1, 1)
        grid.addWidget(self.target_file_btn, 4, 2, 1, 2)

        grid.addWidget(self.drug_seq_text, 5, 0, 2, 2)
        grid.addWidget(self.target_seq_text, 5, 2, 2, 2)

        grid.addWidget(self.result_text, 7, 0, 2, 4)

        grid.addWidget(self.cancel_btn, 9, 0, 1, 1)
        grid.addWidget(self.export_btn, 9, 1, 1, 1)
        grid.addWidget(self.predict_btn, 9, 2, 1, 2)

        self.setLayout(grid)

    def load_drug_file(self, file_path):
        try:
            self.current_drug_file_path = file_path

            if not os.path.exists(file_path):
                QMessageBox.warning(self, '错误', f'文件不存在：\n{file_path}')
                self.drug_records = []
                self.drug_seq_text.clear()
                return

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(
                    self,
                    '错误',
                    f'文件为空，未读取到任何药物分子记录！\n\n文件路径：\n{file_path}'
                )
                self.drug_records = []
                self.drug_seq_text.clear()
                return

            records = parse_smiles_file(file_path)

            if not records:
                QMessageBox.warning(
                    self,
                    '错误',
                    f'文件中没有读到有效记录！\n\n文件路径：\n{file_path}\n文件大小：{file_size} 字节'
                )
                self.drug_records = []
                self.drug_seq_text.clear()
                return

            self.drug_records = records

            valid_count, invalid_count = summarize_smiles_records(self.drug_records)

            preview_lines = []
            for record in self.drug_records[:12]:
                smiles = record["smiles"]
                name = record["name"]
                ok = "合法" if Chem.MolFromSmiles(smiles) is not None else "非法"
                preview_lines.append(f"{name}\t{smiles}\t[{ok}]")

            preview_text = "\n".join(preview_lines)
            if len(self.drug_records) > 12:
                preview_text += f"\n...（其余 {len(self.drug_records) - 12} 条未显示）"

            self.drug_seq_text.setText(preview_text)

            self.result_text.append(f"已导入药物文件：{file_path}")
            self.result_text.append(f"文件大小：{file_size} 字节")
            self.result_text.append(f"记录数：{len(self.drug_records)}")
            self.result_text.append("=" * 60)

            QMessageBox.information(
                self,
                '成功',
                f'药物文件加载完成！\n'
                f'文件路径：{file_path}\n'
                f'文件大小：{file_size} 字节\n'
                f'总记录数：{len(self.drug_records)}\n'
                f'可解析 SMILES：{valid_count}\n'
                f'不可解析 SMILES：{invalid_count}'
            )

        except Exception as e:
            self.current_drug_file_path = ""
            self.drug_records = []
            self.drug_seq_text.clear()
            QMessageBox.warning(self, '错误', f'文件读取失败：{str(e)}')

    def select_drug_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择药物分子文件",
            self.DATA_DIR,
            "支持文件 (*.txt *.smi *.csv);;文本文件 (*.txt *.smi);;CSV文件 (*.csv);;所有文件 (*.*)"
        )

        if not file_path:
            return

        self.load_drug_file(file_path)

    def load_default_drug_file(self):
        self.load_drug_file(self.DEFAULT_DRUG_FILE)

    def select_target_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择蛋白分子文件",
            "",
            "文本文件 (*.txt *.fasta *.fa);;所有文件 (*.*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('>')]
                seq = ''.join(lines)

            if not seq:
                QMessageBox.warning(self, '错误', '未读取到有效蛋白序列！')
                return

            self.target_seq_text.setText(seq)
            QMessageBox.information(self, '成功', '蛋白序列加载完成！')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'文件读取失败：{str(e)}')

    def _build_records_from_manual_input(self):
        manual_text = self.drug_seq_text.toPlainText().strip()
        if not manual_text:
            return []

        lines = manual_text.splitlines()
        return parse_smiles_text_lines(lines)

    def start_predict(self):
        dataset = self.dataset_combo.currentText()
        target_seq = self.target_seq_text.toPlainText().strip()

        if dataset == '请选择':
            QMessageBox.warning(self, '警告', '请先选择数据集！')
            return

        if not target_seq:
            QMessageBox.warning(self, '警告', '请输入蛋白序列！')
            return

        if not os.path.exists(self.MODEL_PATH):
            QMessageBox.critical(self, '错误', f'模型文件不存在：{self.MODEL_PATH}')
            return

        if self.drug_records:
            records_to_use = self.drug_records
            input_mode = "文件导入模式"
        else:
            records_to_use = self._build_records_from_manual_input()
            input_mode = "手动输入模式"

        if not records_to_use:
            QMessageBox.warning(self, '警告', '请先导入药物文件，或手动输入至少一条 SMILES！')
            return

        self.result_text.clear()
        self.result_text.append("📌 开始预测")
        self.result_text.append(f"输入方式：{input_mode}")
        if self.current_drug_file_path:
            self.result_text.append(f"当前药物文件：{self.current_drug_file_path}")
        self.result_text.append(f"记录数：{len(records_to_use)}")
        self.result_text.append("=" * 60)

        self.predict_btn.setEnabled(False)
        self.predict_results = []

        self.predict_thread = PredictThread(
            dataset=dataset,
            drug_records=records_to_use,
            target_seq=target_seq,
            model_path=self.MODEL_PATH,
            device=self.DEVICE
        )

        self.predict_thread.result_signal.connect(self.show_result)
        self.predict_thread.error_signal.connect(self.show_error)
        self.predict_thread.finished_signal.connect(self.save_predict_results)
        self.predict_thread.finished.connect(self.reset_predict_btn)
        self.predict_thread.start()

    def save_predict_results(self, results):
        self.predict_results = results

    def show_result(self, result):
        self.result_text.append(result)

    def show_error(self, error):
        self.result_text.append(f"❌ {error}")

    def reset_predict_btn(self):
        self.predict_btn.setEnabled(True)

    def export_results(self):
        if not self.predict_results:
            QMessageBox.warning(self, '警告', '当前没有可导出的预测结果，请先完成预测！')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出预测结果",
            "predict_results.csv",
            "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
        )

        if not file_path:
            return

        try:
            fieldnames = ["line_no", "name", "smiles", "status", "error", "pkd", "kd", "strength", "suggestion"]

            if file_path.endswith(".csv"):
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.predict_results)

            elif file_path.endswith(".xlsx"):
                if not HAS_PANDAS:
                    QMessageBox.warning(self, '错误', '未安装 pandas/openpyxl，无法导出 xlsx。\n请先安装：pip install pandas openpyxl')
                    return
                df = pd.DataFrame(self.predict_results)
                df.to_excel(file_path, index=False)

            else:
                file_path += ".csv"
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.predict_results)

            QMessageBox.information(self, '成功', f'预测结果已导出到：\n{file_path}')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'导出失败：{str(e)}')


if __name__ == '__main__':
    os.environ["QT_IM_MODULE"] = "qtvirtualkeyboard"
    app = QApplication(sys.argv)
    ui = DrugTargetAffinityUI()
    ui.show()
    sys.exit(app.exec_())
