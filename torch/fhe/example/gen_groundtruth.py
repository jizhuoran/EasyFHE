import itertools, subprocess, os, sys
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch.fhe.bootstrapping as bstest

logN_cand = list(range(14, 17))
logSlots_cand = list(range(5, 14, 3)) + list(range(14, 17))
maxLevelsRemaining_cand = [3, 5, 7]
levelBudget_cand = [[2, 2], [4, 4]]
dnum_cand = [1, 3, 5]
rescaleTech_cand = ["FLEXIBLEAUTO", "FIXEDMANUAL"]

i = 0
for logN, logSlots, maxLevelsRemaining, levelBudget, dnum, rescaleTech in itertools.product(logN_cand, logSlots_cand, maxLevelsRemaining_cand, levelBudget_cand, dnum_cand, rescaleTech_cand):
    if logSlots >= logN - 1:
        continue
    try:
        print(i, ": ", logN, logSlots, maxLevelsRemaining, levelBudget, dnum)
        i += 1
        code_string = """
import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch
import torch.fhe.bootstrapping as BS
logN = {0}
logSlots = {1}
maxLevelsRemaining = {2}
levelBudget = [{3}, {4}]
dnum = {5}
rescaleTech = "{6}"
path_ctx = "data/context_{0}_{1}_{2}_{3}_{4}_{5}_{6}.pkl"
path_io = "data/groundtruth_{0}_{1}_{2}_{3}_{4}_{5}_{6}.pkl"
openfhe_context, cryptoContext = BS.client.gen_contexts(
    logN=logN,
    logSlots=logSlots,
    maxLevelsRemaining=maxLevelsRemaining,
    levelBudget=levelBudget,
    dnum=dnum,
    dcrtBits=59,
    firstMod=60,
    approxModDepth=9,
    rotate_index=[],
    secretKeyDist="UNIFORM_TERNARY",
    rescaleTech=rescaleTech,
)

save_path=path_ctx
BS.utils.save_context(cryptoContext, openfhe_context, save_path)
cryptoContext, _ = BS.utils.load_context(save_path)

dim1 = [0, 0]
cryptoContext.BsContext = BS.BsContext(
    cryptoContext,
    cryptoContext.levelBudget,
    dim1,
    cryptoContext.slots,
    0,
    cryptoContext.rescaleTech,
    cryptoContext.secretKeyDist,
)

BS.eval_bootstrap_setup(
    cryptoContext, cryptoContext.levelBudget, dim1, cryptoContext.slots, 0
)

# Test the correctness of the bootstrapping
x = np.array([0.111111111 * (i & 0xFF) for i in range(cryptoContext.slots)])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(x)
cipher.cv[0] = cipher.cv[0][:2]
cipher.cv[1] = cipher.cv[1][:2]
cipher.cur_limbs = 2

result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, slots=cryptoContext.slots, cryptoContext=cryptoContext)
after_boot = openfhe_context.decrypt(result)
after_boot = after_boot.cpu().numpy().reshape(-1)
print(after_boot[:10])
x = x.cpu().numpy().reshape(-1)
if(np.any(np.abs(after_boot - x) > 3e-2)):
    print("Error is too large!")
    print("Error is too large!")
    print("Error is too large!")
else:
    print("BootstrapTest_N65536L26lB44: Test passed!")
    print("BootstrapTest_N65536L26lB44: Test passed!")
    print("BootstrapTest_N65536L26lB44: Test passed!")

cipher.cv = [cipher.cv[0].tolist(), cipher.cv[1].tolist()]
result.cv = [result.cv[0].tolist(), result.cv[1].tolist()]
with open(path_io, "wb") as file:
    pickle.dump((cipher, result), file)
""".format(logN, logSlots, maxLevelsRemaining, levelBudget[0], levelBudget[1], dnum, rescaleTech)
        
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