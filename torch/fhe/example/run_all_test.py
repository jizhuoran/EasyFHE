import pickle, sys, os
import numpy as np
import csv
import time

sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils

# find all context in the directory
all_correct = True
path = "data/"
f = open('./test_res.csv', 'w', encoding='utf-8', newline="")
csv_write = csv.writer(f)
csv_write.writerow(
    ['logN', 'logSlots', 'maxLevelsRemaining', 'levelBudget', 'dnum', 'dcrtBits', 'firstMod', 'approxModDepth',
     'rescaleTech', 'res', 'exec_time'])
for context_file in os.listdir(path):
    if context_file.endswith(".pkl") and "GPU-FHE-CONTEXT" in context_file:
        context_file = context_file.replace("_UNIFORM_TERNARY_", "_")
        logN, logSlots, maxLevelsRemaining, levelBudget0, levelBudget1, dnum, dcrtBits, firstMod, approxModDepth, rescaleTech = context_file[:-4].split("_")[1:]

        logN = int(logN)
        logSlots = int(logSlots)
        maxLevelsRemaining = int(maxLevelsRemaining)
        levelBudget0 = int(levelBudget0)
        levelBudget1 = int(levelBudget1)
        dnum = int(dnum)
        dcrtBits = int(dcrtBits)
        firstMod = int(firstMod)
        approxModDepth = int(approxModDepth)

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
        x = np.array([values[i % len(values)] for i in range((1 << logSlots))])
        x = torch.tensor(x, device="cuda")
        cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1)
        try:
            start = time.time()
            result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1 << logSlots), cryptoContext=cryptoContext)
            end = time.time()
            openfhe_result = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
            # data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)
            is_equal = utils.compare_bs_ct_with_openfhe(result, openfhe_result)
            if is_equal:
                csv_write.writerow([logN,
                                    logSlots,
                                    maxLevelsRemaining,
                                    [levelBudget0, levelBudget1],
                                    dnum,
                                    dcrtBits,
                                    firstMod,
                                    approxModDepth,
                                    rescaleTech, "pass", end - start])
                print("Test passed!")
            else:
                csv_write.writerow([logN,
                                    logSlots,
                                    maxLevelsRemaining,
                                    [levelBudget0, levelBudget1],
                                    dnum,
                                    dcrtBits,
                                    firstMod,
                                    approxModDepth,
                                    rescaleTech, "fail", end - start])
                print("Test failed!")
                all_correct = False
        except:
            csv_write.writerow([logN,
                                logSlots,
                                maxLevelsRemaining,
                                [levelBudget0, levelBudget1],
                                dnum,
                                dcrtBits,
                                firstMod,
                                approxModDepth,
                                rescaleTech, "exception", 0])
            print("Something wrong !!!")
            all_correct = False
            f.close()
            break

f.close()
if all_correct:
    print("All test cases passed!")
