import sys, os,warnings
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import time, subprocess

warnings.warn("This script has not been tested and may not function as intended. Please remove this message once verified.")

with open("result.txt", "w") as f:
    print("BEGIN", file=f)

start_time = time.time()
#find all context in the directory
path = "/mnt/public_data/data/"
for context_file in os.listdir(path):
    if context_file.endswith(".pkl") and context_file.startswith("GPU-FHE-CONTEXT"):
        print("Testing", context_file)
        context_file = context_file.replace("_UNIFORM_TERNARY_", "_")
        logN, logBsSlots_str, maxLevelsRemaining, levelBudgets_str, dnum, dcrtBits, firstMod, rescaleTech = context_file[:-4].split("_")[1:]
        try:
            logBsSlots_list = [int(logBsSlots) for logBsSlots in logBsSlots_str.split("-")]
            levelBudgets_list = []
            for levelBudgets in range(len(levelBudgets_str.split("-")) // 2):
                levelBudgets_list.append([int(levelBudgets_str.split("-")[2 * levelBudgets]), int(levelBudgets_str.split("-")[2 * levelBudgets + 1])])
            code_string = """
import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils
import time
context_file = "{}"
maxLevelsRemaining = int({})
logBsSlots_list = {}
logN = int({})
dnum = int({})
dcrtBits = int({})
firstMod = int({})
levelBudgets_list = {}
rescaleTech = "{}"
path = "{}"
cryptoContext, openfhe_context, openfhe_boot_contexts = utils.try_load_context(
    int(maxLevelsRemaining),
    [],
    logBsSlots_list,
    int(logN),
    int(dnum),
    int(dcrtBits),
    int(firstMod),
    levelBudgets_list,
    "UNIFORM_TERNARY",
    rescaleTech,
    save_dir=path,
    XXXX")

cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots_list[0])]
cryptoContext.BsContext.to_cuda()

with open("result.txt", "a") as f:
    print(context_file, file=f)

# Test the correctness of the bootstrapping
values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
x = np.array([values[i % len(values)] for i in range((1<<logBsSlots_list[0]))])
x = torch.tensor(x, device="cuda")
cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1)
result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots_list[0], cryptoContext=cryptoContext)
start_time = time.time()
result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots_list[0], cryptoContext=cryptoContext)
end_time = time.time()
openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots)]
openfhe_result = openfhe_boot_context.cc.EvalBootstrap(cipher_openfhe)

data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)
with open("result.txt", "a") as f:
    print("Time taken:", end_time - start_time, file=f)
    if np.equal(np.concatenate([result.cv[0].cpu().numpy(), result.cv[1].cpu().numpy()]).reshape(-1), data.reshape(-1)).all():
        print("Test passed!", file=f)
    else:
        print("Test failed!", file=f)
        print("result", result.cv[0].cpu().numpy()[0][:10], file=f)
        print("data", data.reshape(-1)[:10], file=f)
""".format(context_file, logN, logBsSlots_list, maxLevelsRemaining, levelBudgets_list, dnum, dcrtBits, firstMod, rescaleTech, path)

            # Create a temporary file to store the code
            with open("temp_file.py", "w") as temp_file:
                print(code_string, file=temp_file)

            try:
                # Execute the temporary file as a separate process
                command = "bash -c 'source ~/.bashrc && python3 temp_file.py'"
                process = subprocess.Popen(command, shell=True)
                process.wait()

                print("Process finished with exit code:", process.returncode)
            finally:
                # Clean up the temporary file
                os.remove("temp_file.py")


        except Exception as e:
            print(e)
            continue


print("Time taken:", time.time() - start_time)