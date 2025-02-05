import itertools, subprocess, os, sys

logN_cand = list(range(15, 16))
logSlots_cand = list(range(5, 14, 3)) + list(range(14, 17))
maxLevelsRemaining_cand = [3, 6]
levelBudget_cand = [[3, 3],[4, 4]]
dnum_cand = [1, 3, 4]
rescaleTech_cand = ["FLEXIBLEAUTO", "FIXEDMANUAL"]
# rescaleTech_cand = ["FIXEDMANUAL"]
path = "/mnt/public_data/data/"
i = 0
for logN, logSlots, maxLevelsRemaining, levelBudget, dnum, rescaleTech in itertools.product(logN_cand, logSlots_cand, maxLevelsRemaining_cand, levelBudget_cand, dnum_cand, rescaleTech_cand):
    if logN == 16 and logSlots > 12:
        continue
    if logSlots > logN - 1:
        continue
    try:
        print(i, ": ", logN, logSlots, maxLevelsRemaining, levelBudget, dnum)
        i += 1

        save_path_meta = "_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            logN,
            "-".join(map(str, [logSlots])),
            maxLevelsRemaining,
            "-".join("-".join(map(str, levelBudget)) for levelBudget in [levelBudget]),
            dnum,
            59,
            60,
            9,
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
logN = {0}
logSlots_list = [{1}]
maxLevelsRemaining = {2}
levelBudget_list = [[{3}, {4}]]
dnum = {5}
rescaleTech = "{6}"
dcrtBits=59
firstMod=60
approxModDepth=9

gen_contexts(
    logN=logN,
    logSlots_list=logSlots_list, # possible slots value of runtime ciphertext #todo: should be a list?
    maxLevelsRemaining=maxLevelsRemaining,
    levelBudget_list=levelBudget_list,
    dnum=dnum,
    dcrtBits=dcrtBits,
    firstMod=firstMod,
    approxModDepth=approxModDepth,
    rotate_index=[],
    secretKeyDist="UNIFORM_TERNARY",
    rescaleTech=rescaleTech,
    save_dir="{7}",
    mode = "debug"
)
""".format(logN, logSlots, maxLevelsRemaining, levelBudget[0], levelBudget[1], dnum, rescaleTech, path)
        
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
