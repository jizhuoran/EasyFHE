import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils

logN = 14
logSlots = 6
maxLevelsRemaining = 3
levelBudget0 = 4
levelBudget1 = 4
dnum = 3
dcrtBits = 59
firstMod = 60
approxModDepth = 9
rescaleTech = "FIXEDMANUAL"
path = "data/"

cryptoContext, openfhe_context = utils.try_load_context(
    int(logN),
    int(logSlots),
    int(maxLevelsRemaining),
    [int(levelBudget0), int(levelBudget1)],
    int(dnum),
    int(dcrtBits),
    int(firstMod),
    int(approxModDepth),
    "UNIFORM_TERNARY",
    rescaleTech,
    save_dir=path)

# Test the correctness of the bootstrapping
values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
x = np.array([values[i % len(values)] for i in range((1<<logSlots))])
x = torch.tensor(x, device="cuda")
cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1)

result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots), cryptoContext=cryptoContext)
openfhe_result = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)
is_equal = utils.compare_bs_ct_with_openfhe(result, openfhe_result)
if is_equal:
    print("Test passed!")
else:
    print("Test failed!")
    print("result", result.cv[0].cpu().numpy()[0][:10])
    print("data", data.reshape(-1)[:10])
