import sys, os, time
import numpy as np

sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils

DATA_DIR = os.environ["DATA_DIR"]

maxLevelsRemaining = 3
logBsSlots_list = [12]
logN = 14
dnum = 3
dcrtBits = 59
firstMod = 60
levelBudget_list = [[4, 4]]
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
path = DATA_DIR
secretKeyDist = "UNIFORM_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
device = "cuda"  # "cuda" # "cpu"

maxLevelsRemaining = 11
logBsSlots_list = [14]
logN = 16
dnum = 6
dcrtBits = 56
firstMod = 60
levelBudget_list = [[4,4]]
rescaleTech = "FIXEDMANUAL"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
path = DATA_DIR
secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
device = "cuda"  # "cuda" # "cpu"
config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, COUNT_OPS=True)
rot_list = []
cryptoContext, openfhe_context = utils.try_load_context(
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
print(f"length of rot list: {len(rot_list)}")
print(f"level budget: {levelBudget_list}")
print("cryptoContext: ", cryptoContext)

encode_slots = (1<<(logN-1))

# Test the correctness of the bootstrapping
values = [
    0.111111,
    0.222222,
    0.333333,
    0.444444,
    0.555555,
    0.666666,
    0.777777,
    0.888888,
]
x = np.array([values[i % len(values)] for i in range((1 << encode_slots))])
cipher = openfhe_context.encrypt(
    x, device, 1, openfhe_context.depth - 1, encode_slots
)  # specify the slots value explicitly

repeat = 2
for j in range(len(logBsSlots_list)):
    result = BS.eval_bootstrap(
        cipher, cryptoContext.L-6, logBsSlots_list[j], levelBudget_list[0], cryptoContext
    )
    start_time = time.time()
    for i in range(repeat):
        result = BS.eval_bootstrap(
            cipher, cryptoContext.L-6, logBsSlots_list[j], levelBudget_list[0], cryptoContext
        )
    end_time = time.time()
    print(f"Time taken for bootstrapping slots {logBsSlots_list[j]}: ", (end_time - start_time)/repeat)


clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
clear_result = clear_result.cpu().numpy().reshape(-1)
print("HE decryption result: ", clear_result[:10])
