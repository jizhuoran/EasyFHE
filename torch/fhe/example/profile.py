import sys, os,warnings
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from functools import partial
import time, os
from torch.fhe.ciphertext import Cipher, Plaintext
from torch.fhe.bs_context import *
from torch.fhe import homo_ops, hybrid_keyswitch
from torch.fhe  import utils
from collections import defaultdict
import numpy as np


perf_result = defaultdict(dict)

def perf_single_operation(func, name, limb):
    func()
    torch.cuda.synchronize()
    torch.cpu.synchronize()
    start_time = time.time()
    for _ in range(100):
        func()
    torch.cuda.synchronize()
    torch.cpu.synchronize()
    end_time = time.time()

    exec_time_repeat = end_time - start_time
    perf_result[name][limb] = exec_time_repeat * 10
    

def print_profiling_result():
    for name, record in perf_result.items():
        print("============{}================".format(name))
        for i in range(len(record)):
            print("limb {}: {:.5f} ms, {:.2f}".format(i + 1, record[i + 1], record[i + 1] / record[1]))
        print("==============================")


def profiling_single_op(
        maxLevelsRemaining=20,
        appRotIndex_list = [-1, 2],
        logBsSlots_list=[12],
        logN=14,
        dnum=3,
        dcrtBits=59,
        firstMod=60,
        levelBudget_list=[[4, 4]],
        rescaleTech="FIXEDMANUAL",  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="data",
        mode="release"  # "debug" or "release"
):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    cryptoContext, openfhe_context = (
        utils.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               autoLoadAndSetConfig=True, mode=mode))

    encode_slots = (1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    cipher = openfhe_context.encrypt(x, 1, 0, encode_slots, mode)
    cipher_rescale = openfhe_context.encrypt(x, 2, 0, encode_slots, mode)
    plaintext = openfhe_context.encode(values, 1, 0, encode_slots)
    

    for limb in range(1, cipher.cur_limbs + 1):
        tmp_ct = homo_ops.drop_last_elements(cipher, cipher.cur_limbs - limb)
        tmp_ct_rescale = homo_ops.drop_last_elements(cipher_rescale, cipher_rescale.cur_limbs - max(limb, 2))
        tmp_cv = homo_ops.extract_cv(tmp_ct, 0)
        tmp_cv_ext = hybrid_keyswitch.modup_to_ext(tmp_cv, cryptoContext)
        tmp_pt = homo_ops.drop_last_elements(plaintext, plaintext.cur_limbs - limb)

        perf_single_operation(partial(homo_ops.encode, x, 1, limb, encode_slots, True, cryptoContext), "encode", limb)
        perf_single_operation(partial(homo_ops.homo_add, tmp_ct, tmp_ct, cryptoContext), "homo_add", limb)
        perf_single_operation(partial(homo_ops.homo_sub, tmp_ct, tmp_ct, cryptoContext), "homo_sub", limb)
        perf_single_operation(partial(homo_ops.homo_mul, tmp_ct, tmp_ct, cryptoContext), "homo_mul", limb)
        perf_single_operation(partial(homo_ops.homo_square, tmp_ct, cryptoContext), "homo_square", limb)
        perf_single_operation(partial(homo_ops.homo_rescale_internal, tmp_ct_rescale, 1, cryptoContext), "homo_rescale", limb)
        perf_single_operation(partial(homo_ops.homo_add_scalar_double, tmp_ct, 1.0, cryptoContext), "homo_add_scalar_double", limb)
        perf_single_operation(partial(homo_ops.homo_add_scalar_int, tmp_ct, 1, cryptoContext), "homo_add_scalar_int", limb)
        perf_single_operation(partial(homo_ops.homo_mul_scalar_double, tmp_ct, 1.0, cryptoContext), "homo_mul_scalar_double", limb)
        perf_single_operation(partial(homo_ops.homo_mul_scalar_int, tmp_ct, 1, cryptoContext), "homo_mul_scalar_int", limb)
        perf_single_operation(partial(homo_ops.homo_rotate, tmp_ct, 2, cryptoContext), "homo_rotate", limb)
        perf_single_operation(partial(homo_ops.homo_mul_pt, tmp_ct, tmp_pt, cryptoContext), "homo_mul_pt", limb)
        perf_single_operation(partial(homo_ops.homo_add_pt, tmp_ct, tmp_pt, cryptoContext), "homo_add_pt", limb)

        perf_single_operation(partial(homo_ops.eval_fast_rotate, tmp_cv_ext, tmp_cv, 2, True, False, cryptoContext), "eval_fast_rotate", limb)
        perf_single_operation(partial(hybrid_keyswitch.modup_to_ext, tmp_cv, cryptoContext), "modup_to_ext", limb)
        perf_single_operation(partial(hybrid_keyswitch.moddown_from_ext, tmp_cv_ext, cryptoContext), "moddown_from_ext", limb)
        perf_single_operation(partial(hybrid_keyswitch.key_switch_P_ext, tmp_cv, cryptoContext), "key_switch_P_ext", limb)
        perf_single_operation(partial(hybrid_keyswitch.mult_rot_key_and_sum_ext, tmp_cv_ext, 2, cryptoContext), "mult_rot_key_and_sum_ext", limb)



    print_profiling_result()

if __name__ == "__main__":
    profiling_single_op()