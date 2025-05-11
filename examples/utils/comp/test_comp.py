import time
from sealFunc import *
from program import *
import torch
import torch.fhe as fhe

import os
DATA_DIR = os.environ["DATA_DIR"]



def test_minimax_relu():
    # Parameter setting
    alpha = 13
    comp_no = 3
    degs = [15, 15, 27]
    scaled_val = 1.7
    eval_type = EvalType.ODDBABY
    # scalingfactor_log2 = 45 # todo: corresponding to our dcrtBits
    # scale = 2 ** scalingfactor_log2 # from seal

    maxLevelsRemaining = 14 # level = 14
    rotate_index_list = [-1, 2]
    logBsSlots_list = []
    logN = 16
    dnum = 3
    dcrtBits = 52
    firstMod = 56
    levelBudget_list = []
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"


    print("==> Generating evaluation tree...")
    trees = []
    for i in range(comp_no):
        tr = Tree()
        if eval_type == EvalType.ODDBABY:
            upgrade_oddbaby(degs[i], tr)
        else:
            raise ValueError("Unsupported evaltype")
        tr.print()
        trees.append(tr)



    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, CHECK_CIPHER=False, SAVE_MIDDLE=False,
                                     PTX_TWIN=False)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, save_dir=DATA_DIR,
                             config=config))

    slots = cryptoContext.N//2
    cryptoContext.openfhe_context = openfhe_context
    cryptoContext.cnt=0

    print("==> Generating input vector...")
    input_vec = np.array([-1.0 + 2.0 * i / (slots - 1) for i in range(slots)], dtype=np.float64)

    print("==> Encrypting input...")
    # Generate encrypted zero ciphertext
    Nh = cryptoContext.N//2
    zeros_vec = np.zeros(Nh, dtype=np.float64)
    ctxt_zero = cryptoContext.openfhe_context.encrypt(zeros_vec, 1, 0, Nh)
    cryptoContext.zeros_Nh = ctxt_zero

    ones_vec = np.ones(Nh, dtype=np.float64)
    ctxt_1 = cryptoContext.openfhe_context.encrypt(ones_vec, 1, 0, Nh)
    cryptoContext.ones_Nh = ctxt_1

    half_vec = np.full(Nh, 0.5, dtype=np.float64)
    cipher_half = cryptoContext.openfhe_context.encrypt(half_vec, 1, 0, Nh)     # fixme: should check if slots here cant be hardcoded to Nh
    cryptoContext.cipher_half = cipher_half


    cipher_x = openfhe_context.encrypt(input_vec, 1, 0, slots)

    print("==> Starting Minimax ReLU evaluation...")
    t0 = time.time()
    result = minimax_relu(comp_no, degs, alpha, trees, scaled_val, cipher_x, cryptoContext)
    t1 = time.time()
    print("execution time:", t1 - t0)

    print("==> Decrypting partial result...")
    decrypted = openfhe_context.decrypt(result).cpu().numpy()
    print(f"Partial decrypted output (first 5 values): {decrypted[:5]}")

    print("==> Validating ReLU approximation accuracy...")
    failures = show_failure_relu(result, input_vec, alpha, cryptoContext)

    print(f"ReLU function time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Failure count = {failures} / {slots}")

if __name__ == "__main__":
    test_minimax_relu()
