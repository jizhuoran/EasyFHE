import sys, os, warnings
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import easyfhe as torch
import easyfhe.fhe.utils as utils
import numpy as np
from termcolor import colored

DATA_DIR = os.environ["DATA_DIR"]

def print_failed(message):
    print(colored(message, "red"))


def evalpolyps_debug(
        maxLevelsRemaining=10,
        appRotIndex_list = [1],
        logBsSlots_list=[10],
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=[[4,4]],
        scale_mode = "fixed",
        rescale_policy = "manual",
        device = "cuda",
        save_dir=DATA_DIR
):

    config = torch.fhe.config.Config(CHECK_CIPHER=False, PTX_TWIN=False, AUTO_LOAD_KEYS=False, COMPARE_WITH_OPENFHE=True) #eval_bootstrap and PTX_TWIN cannot pass CHECK_CIPHER
    cryptoContext, openfhe_context, openfhe_boot_contexts = (
        utils.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", scale_mode, rescale_policy, device, save_dir=save_dir,
                               config=config))

    encode_slots = (1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    cipher, cipher_openfhe = openfhe_context.encrypt(x, cryptoContext.device, 1, 0, encode_slots)

    import examples.utils.approx as approx
    res = approx.eval_poly_ps(cipher, [1, 1, 1 / (2.0), 1 / (6.0), 1 / (24.0), 1 / (120.0), 1 / (720.0)], cryptoContext)
    res_openfhe = openfhe_context.cc.EvalPoly(cipher_openfhe, [1, 1, 1 / (2.0), 1 / (6.0), 1 / (24.0), 1 / (120.0), 1 / (720.0)])
    is_euqal = utils.compare_gpufhe_ct_with_openfhe(res, res_openfhe)

    if is_euqal:
        print("eval_poly_ps: Test passed!")
    else:
        print_failed("eval_poly_ps: Test failed!")


##############
## run tests #
##############

if __name__ == "__main__":
    evalpolyps_debug()
