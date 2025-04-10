import itertools, subprocess, os, sys
import warnings

warnings.warn("This script has not been tested and may not function as intended. Please remove this message once verified.")
logN_cand = list(range(15, 16))
logBsSlots_cand = list(range(5, 14, 3)) + list(range(14, 17))
maxLevelsRemaining_cand = [3, 6]
levelBudget_cand = [[3, 3],[4, 4]]
dnum_cand = [1, 3, 4]
rescaleTech_cand = ["FLEXIBLEAUTO", "FIXEDMANUAL"]
# rescaleTech_cand = ["FIXEDMANUAL"]
path = "/mnt/public_data/data/"
i = 0
for logN, logBsSlots, maxLevelsRemaining, levelBudget, dnum, rescaleTech in itertools.product(logN_cand, logBsSlots_cand, maxLevelsRemaining_cand, levelBudget_cand, dnum_cand, rescaleTech_cand):
    if logN == 16 and logBsSlots > 12:
        continue
    if logBsSlots > logN - 1:
        continue
    try:
        print(i, ": ", logN, logBsSlots, maxLevelsRemaining, levelBudget, dnum)
        i += 1

        save_path_meta = "_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            logN,
            "-".join(map(str, [logBsSlots])),
            maxLevelsRemaining,
            "-".join("-".join(map(str, levelBudget)) for levelBudget in [levelBudget]),
            dnum,
            59,
            60,
            "UNIFORM_TERNARY",
            rescaleTech,
        )

        GPUFHE_path = path + "/GPU-FHE-CONTEXT" + save_path_meta
        if os.path.exists(GPUFHE_path):
            print("Context already exists")
            continue
        else:
            print("Context does not exist, generating...")

        code_string = """
import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
from fhe.client.gen_context import gen_contexts

maxLevelsRemaining = {0}
rotIndex_list = []  # List of rotation indices
logBsSlots_list = [{1}]  # List of possible slots value of runtime ciphertext
logN = {2}
dnum = {3}
dcrtBits=59
firstMod=60
levelBudget_list = [[{4}, {5}]]
secretKeyDist = "UNIFORM_TERNARY"
rescaleTech = "{6}"
save_dir = "{7}"
dim1 = [0, 0]  # Default value for dim1

gen_contexts(
    maxLevelsRemaining=maxLevelsRemaining,
    rotIndex_list=[],  # List of rotation indices
    logBsSlots_list=logBsSlots_list,  # List of possible slots value of runtime ciphertext
    logN=logN,
    dnum=dnum,
    dcrtBits=dcrtBits,
    firstMod=firstMod,
    levelBudget_list=levelBudget_list,
    secretKeyDist="UNIFORM_TERNARY",
    rescaleTech=rescaleTech,
    save_dir="{7}",
    dim1=[0, 0],  # Default value for dim1
)
""".format(maxLevelsRemaining, logBsSlots, logN, dnum, levelBudget[0], levelBudget[1], rescaleTech, path)
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
