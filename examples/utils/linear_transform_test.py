import time
import warnings

warnings.filterwarnings("ignore")
import math
import sys, os

sys.path.append("/".join(os.getcwd().split("/")[:-5]))
sys.path.append("/".join(os.getcwd().split("/")[:-4]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch.fhe as fhe
import numpy as np
import torch
from linear_transform import eval_linear_transform as eval_linear_transform



maxLevelsRemaining = 3
logBsSlots_list = []
logN = 16
dnum = 1
dcrtBits = 52
firstMod = 55
levelBudget_list = []
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO"  "FIXEDMANUAL"  "FIXEDAUTO"
path = os.environ["DATA_DIR"]
secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
device = "cpu"  # "cuda"  "cpu"
config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True)
rot_list = [i for i in range(1, 33)]
cryptoContext, openfhe_context = fhe.utils.try_load_context(
    int(maxLevelsRemaining),
    rot_list,
    logBsSlots_list,
    int(logN),
    int(dnum),
    int(dcrtBits),
    int(firstMod),
    levelBudget_list,
    secretKeyDist,
    rescaleTech,
    device,
    save_dir=path,
    config=config,
)

slots = 32
M = np.random.rand(slots, slots)
ptx = np.random.rand(slots)

bStep = 2
gStep = 16


def get_bsgs_diagonals(M, bStep, gStep):
    n = M.shape[0]
    base_diags = []
    for i in range(n):
        diag = [(M[k, (i + k) % n]) for k in range(n)]
        base_diags.append(diag)
    diagonals = []
    for j in range(gStep):
        for i in range(bStep):
            rot = (j * bStep) % n
            row = j * bStep + i
            rotated = base_diags[row][-rot:] + base_diags[row][:-rot] if rot != 0 else base_diags[row].copy()
            diagonals.append(rotated)

    return np.array(diagonals)


Mdiag_raw = get_bsgs_diagonals(M, bStep, gStep)
Mdiag = [fhe.encode(diag, "name", 0, slots, True, cryptoContext) for i, diag in enumerate(Mdiag_raw)]

ct = openfhe_context.encrypt(ptx, cryptoContext.device, 1, 0, slots)
res = eval_linear_transform(Mdiag, ct, cryptoContext, bStep, gStep)

ctx_gpu = cryptoContext.cuda()
ct_gpu = ct.cuda()
Mdiag_gpu = [diag.cuda() for diag in Mdiag]
res_gpu = eval_linear_transform(Mdiag_gpu, ct_gpu, ctx_gpu, bStep, gStep)

ctx_cpu = cryptoContext.cpu()
ct_cpu = ct.cpu()
Mdiag_cpu = [diag.cpu() for diag in Mdiag]
res_cpu = eval_linear_transform(Mdiag_cpu, ct_cpu, ctx_cpu, bStep, gStep)

fhe_bsgs_out = openfhe_context.decrypt(res_cpu)
print("FHE BSGS output:", fhe_bsgs_out)
fhe_bsgs_out_gpu = openfhe_context.decrypt(res_gpu)
print("FHE BSGS output gpu:", fhe_bsgs_out_gpu)


plain_out = M @ ptx
print("Plain matrix output:", plain_out)
