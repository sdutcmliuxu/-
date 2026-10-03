# Transgelin-2

基于 GraphDTA 的药物–靶点亲和力预测项目，包含 Transgelin-2（TAGLN2）数据预处理、模型微调和预测脚本。

## 项目位置

代码和数据位于 [`pythonProject/GraphDTA-master`](pythonProject/GraphDTA-master)。原项目的环境安装、数据准备和模型训练说明见该目录下的 [README](pythonProject/GraphDTA-master/README.md)。

主要脚本：

- `preprocess_tagln2.py`：TAGLN2 数据预处理。
- `finetune_tagln2.py`：TAGLN2 模型微调。
- `predict.py`：亲和力预测。
- `training.py`、`training_validation.py`：GraphDTA 模型训练。

仓库保留原目录结构、模型权重、TAGLN2 数据和结果文件。按照项目原有 `.gitignore`，Davis/KIBA 生成的训练与测试 CSV、预处理张量不会上传，可通过原项目的数据准备步骤生成。IDE 设置、Python 缓存和临时预测张量也不会上传。

`data.zip` 使用 Git LFS 保存。下载完整数据压缩包时，请安装 Git LFS，并在克隆后运行 `git lfs pull`。

## 上传记录

最近上传完成时间：**2026-10-04 00:20:06（北京时间，Asia/Shanghai，UTC+08:00）**。
