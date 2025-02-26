import sys, os, time
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.bs_compilered as COMPILE
import torch.fhe.utils as utils

maxLevelsRemaining = 3
logBsSlots_list = [8]
logN = 14
dnum = 3
dcrtBits = 59
firstMod = 60
levelBudget_list = [[4, 4]]
rescaleTech = "FLEXIBLEAUTO"
path = "data"

secretKeyDist = "UNIFORM_TERNARY" # "SPARSE_TERNARY"  "UNIFORM_TERNARY"

cryptoContext, openfhe_context, openfhe_boot_contexts = (
    utils.try_load_context(int(maxLevelsRemaining), [], logBsSlots_list, int(logN), int(dnum), int(dcrtBits),
                           int(firstMod), levelBudget_list, secretKeyDist, rescaleTech, save_dir=path,
                           autoLoadAndSetConfig=False, mode="debug"))

logBsSlots = logBsSlots_list[0]

# Test the correctness of the bootstrapping
values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
x = np.array([values[i % len(values)] for i in range((1<<logBsSlots))])
x = torch.tensor(x, device="cuda")
cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, (1<<logBsSlots)) #specify the slots value explicitly

cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
cryptoContext.BsContext.to_cuda()
utils.load_rotation_keys(logBsSlots, cryptoContext)

# with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(
#             "/home/zrji/log"
#         ),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#     ) as profiler:
#         # Start profiling specific functions with torch.profiler.record_function()
#         result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots, cryptoContext=cryptoContext)
#         profiler.step()

# # Get the profiling results
# profiler_results = profiler.key_averages()

# # Print the profiling summary in a table format
# print(profiler_results.table(sort_by="self_cuda_time_total"))

result1 = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots, cryptoContext=cryptoContext)
print("=======================")
print("=======================")
print("=======================")
result2 = COMPILE.eval_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots, cryptoContext=cryptoContext)

print("result1", result1.cv[0].cpu().numpy()[0][:10])
print("result2", result2.cv[0].cpu().numpy()[0][:10])

if np.array_equal(result1.cv[0].cpu().numpy(), result2.cv[0].cpu().numpy()):
    print("Test passed!")
    print("Test passed!")
    print("Test passed!")
else:
    print("Test failed!")
    print("Test failed!")
    print("Test failed!")




start_time = time.time()
result = COMPILE.eval_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots, cryptoContext=cryptoContext)
print("Time taken for bootstrapping:", time.time() - start_time)
openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots)]
openfhe_result = openfhe_boot_context.cc.EvalBootstrap(cipher_openfhe)
data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)
is_equal = utils.compare_bs_ct_with_openfhe(result, openfhe_result)
if is_equal:
    print("Test passed!")
else:
    print("Test failed!")
    print("result", result.cv[0].cpu().numpy()[0][:10])
    print("data", data.reshape(-1)[:10])

