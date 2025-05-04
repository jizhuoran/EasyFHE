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

compare = False

if compare == True:
    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, COMPARE_WITH_OPENFHE=compare)

    cryptoContext, openfhe_context, openfhe_boot_contexts = utils.try_load_context(
        int(maxLevelsRemaining),
        [],
        logBsSlots_list,
        int(logN),
        int(dnum),
        int(dcrtBits),
        int(firstMod),
        levelBudget_list,
        secretKeyDist,
        rescaleTech,
        save_dir=path,
        config=config,
    )

    logBsSlots = logBsSlots_list[0]

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
    x = np.array([values[i % len(values)] for i in range((1 << logBsSlots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(
        x, 1, openfhe_context.depth - 1, (1 << logBsSlots)
    )  # specify the slots value explicitly


    start_time = time.time()
    result = BS.eval_bootstrap(
        cipher, cryptoContext.L, logBsSlots, levelBudget_list[0], cryptoContext
    )
    print("Time taken for bootstrapping:", time.time() - start_time)
    openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots)]
    openfhe_result = openfhe_boot_context.cc.EvalBootstrap(cipher_openfhe)
    data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)

    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[:10])

    is_equal = utils.compare_gpufhe_ct_with_openfhe(result, openfhe_result)
    if is_equal:
        print("Test passed!")
    else:
        print("Test failed!")
        print("result", result.cv[0].cpu().numpy()[0][:10])
        print("data", data.reshape(-1)[:10])
else:

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, COMPARE_WITH_OPENFHE=compare)

    cryptoContext, openfhe_context = utils.try_load_context(
        int(maxLevelsRemaining),
        [],
        logBsSlots_list,
        int(logN),
        int(dnum),
        int(dcrtBits),
        int(firstMod),
        levelBudget_list,
        secretKeyDist,
        rescaleTech,
        save_dir=path,
        config=config,
    )

    logBsSlots = logBsSlots_list[0]

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
    x = np.array([values[i % len(values)] for i in range((1 << logBsSlots))])
    x = torch.tensor(x, device="cuda")
    cipher = openfhe_context.encrypt(
        x, 1, openfhe_context.depth - 1, (1 << logBsSlots)
    )  # specify the slots value explicitly


    start_time = time.time()
    result = BS.eval_bootstrap(
        cipher, cryptoContext.L, logBsSlots, levelBudget_list[0], cryptoContext
    )
    print("Time taken for bootstrapping:", time.time() - start_time)

    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[:10])
