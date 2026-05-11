import easyfhe as torch

pth_path = './ResNet18_Aespa.pth'         # 你的 .pth 文件路径
state_dict = torch.load(pth_path, map_location="cpu")  # dict: {name: tensor}

total_params = 0
total_bytes  = 0

print(f"{'Parameter':40s} {'shape':20s} {'#params':>10s} {'MB':>10s}")
print("-" * 80)

for name, param in state_dict.items():
    numel  = param.numel()            # 元素个数
    nbytes = numel * param.element_size()   # 占用字节
    total_params += numel
    total_bytes  += nbytes
    mb = nbytes / 1024**2
    print(f"{name:40s} {str(tuple(param.shape)):20s} {numel:10d} {mb:10.2f}")

print("-" * 80)
print(f"Total parameters: {total_params:,}")
print(f"Total size      : {total_bytes / 1024**2:.2f} MB")
