import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils

#find all context in the directory
all_correct = True
path = "data/"
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
        x = np.array([values[i % len(values)] for i in range((1<<logSlots))])
        x = torch.tensor(x, device="cuda")
        cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1)

        result = BS.eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots), cryptoContext=cryptoContext)
        openfhe_result = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
        data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)
        if np.equal(np.concatenate([result.cv[0].cpu().numpy(), result.cv[1].cpu().numpy()]).reshape(-1), data.reshape(-1)).all():
            print("Test passed!")
        else:
            print("Test failed!")
            print("result", result.cv[0].cpu().numpy()[0][:10])
            print("data", data.reshape(-1)[:10])
            all_correct = False

if all_correct:
    print("All test cases passed!")