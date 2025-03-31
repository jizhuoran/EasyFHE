import pickle, sys, os, time
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils

logN = 16
logSlots_list = [12]
maxLevelsRemaining = 11
levelBudget_list = [[4, 4]]
dnum = 3
dcrtBits = 59
firstMod = 60
approxModDepth = 9
rescaleTech = "FLEXIBLEAUTO"
path = "data"

secretKeyDist = "UNIFORM_TERNARY" # "SPARSE_TERNARY"  "UNIFORM_TERNARY"

# logN = 15
# logSlots_list = [12]
# maxLevelsRemaining = 3
# levelBudget_list = [[4, 4]]
# dnum = 1
# dcrtBits = 59
# firstMod = 60

logN = 14
logSlots_list = [4]
maxLevelsRemaining = 3
levelBudget_list = [[4, 4]]
dnum = 1
dcrtBits = 59
firstMod = 60

# logN = 17
# logSlots_list = [12, 13, 14]
# levelBudget_list = [[4, 4], [4, 4], [4, 4]]
# dnum = 3
# dcrtBits = 59
# firstMod = 60
# max_relu_degree = 59
# secretKeyDist = "UNIFORM_TERNARY"
# rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"

cryptoContext, openfhe_contexts = utils.try_load_context(
    int(logN),
    logSlots_list,
    int(maxLevelsRemaining),
    levelBudget_list,
    int(dnum),
    int(dcrtBits),
    int(firstMod),
    int(approxModDepth),
    [],
    secretKeyDist,
    rescaleTech,
    save_dir=path,
    mode = "debug")

# Though looks stupid, the Context will always be loaded to GPU first...
logSlots = logSlots_list[0]
openfhe_context = openfhe_contexts[str(logSlots)]
values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
x_ = np.array([values[i % len(values)] for i in range((1<<logSlots))])
cryptoContext.BsContext = cryptoContext.BsContext_map[str(logSlots)]
cryptoContext.BsContext.to_cuda()

cryptoContext.cuda()
x = torch.tensor(x_, device="cuda")
cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, (1<<logSlots)) #specify the slots value explicitly




cryptoContext.load_rotation_keys(logSlots)
cryptoContext.BsContext.cuda()
#result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)

cipher.cv = [cv.cpu() for cv in cipher.cv]
cryptoContext.cpu()
cryptoContext.load_rotation_keys(logSlots)
cryptoContext.BsContext.cpu()


# result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)

start_time = time.time()
result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)
print("Time taken for bootstrapping:", time.time() - start_time)
start_time1 = time.time()
openfhe_result = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
print("Time taken for openfhe bootstrapping:", time.time() - start_time1)
data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)
is_equal = utils.compare_bs_ct_with_openfhe(result, openfhe_result)
if is_equal:
    print("Test passed!")
else:
    print("Test failed!")
    print("result", result.cv[0].cpu().numpy()[0][:10])
    print("data", data.reshape(-1)[:10])

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
#         result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)
#         profiler.step()

# # Get the profiling results
# profiler_results = profiler.key_averages()

# # Print the profiling summary in a table format
# print(profiler_results.table(sort_by="self_cuda_time_total"))