"""
本文件用于查看模型中间输入
"""
import numpy as np
import torch
from examples.resnet20.gen_aespa_weights.HerPN import get_resnet18_HerPN, change_all_HerPN_by_PAF_MutalChannel
from examples.resnet20.src.resnet18_aespa import read_image

def main():
    # 准备明文模型，测速时可以删除
    model = get_resnet18_HerPN(num_classes=10)
    device = torch.device("cuda:0")
    model.to(device)
    model_path = '/home/yhfan/PNP/GPU-FHE/examples/resnet20/Aespa/ResNet18_Aespa.pth'
    stict = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(stict, strict=False)
    model.eval()
    model = change_all_HerPN_by_PAF_MutalChannel(model)

    image_vector, label, index = read_image(2)
    image_vector = torch.tensor(np.array(image_vector), device="cuda")
    # 明文模型输出
    input = torch.tensor(image_vector, device="cuda", dtype=torch.float32)
    input = torch.stack([input[i * 1024: (i + 1) * 1024].view(32, 32) for i in range(3)], dim=0)
    x, fea = model(input, fea_out=True)


if __name__ == '__main__':
    main()
