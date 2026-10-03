# 检查CUDA设备
import torch
print("CUDA是否可用:", torch.cuda.is_available())
print("可用GPU数量:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("GPU编号及名称:", [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])
#D:\anaconda\python.exe D:\Transgelin - 2\pythonProject\GraphDTA - master\training.py 0 3 0