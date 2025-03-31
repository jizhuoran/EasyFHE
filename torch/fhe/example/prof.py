import sys, os, time
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from functools import partial
from collections import defaultdict
import numpy as np
import torch
from torch.fhe.bs_context  import * 
from torch.fhe import homo_ops, hoisting_keyswitch, bootstrapping,ciphertext
from torch.fhe import utils

perf_result = defaultdict(dict)

def perf(func, name, limb, repeat_time=100):
    assert repeat_time < 1000
    func()
    torch.cuda.synchronize()
    torch.cpu.synchronize()
    start_time = time.time()
    for _ in range(repeat_time):
        func()
    torch.cuda.synchronize()
    torch.cpu.synchronize()
    end_time = time.time()

    exec_time_repeat = end_time - start_time
    perf_result[name][limb] = exec_time_repeat * (1000 / repeat_time)


def print_profiling_result():
    for name, record in perf_result.items():
        print("============{}================".format(name))
        for i in range(len(record)):
            print(
                "limb {}: {:.5f} ms, {:.2f}".format(
                    i + 1, record[i + 1], record[i + 1] / record[1]
                )
            )
        print("==============================")


def profiling_single_op(
    maxLevelsRemaining=3,
    appRotIndex_list=[-1, 2],
    logBsSlots_list=[4],
    logN=14,
    dnum=1,
    dcrtBits=59,
    firstMod=60,
    levelBudget_list=[[4, 4]],
    rescaleTech="FLEXIBLEAUTO",  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    save_dir="data",
    mode="debug",  # "debug" or "release"
):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")
    
    # cryptoContext, openfhe_context = utils.try_load_context(
    #     logN,
    #     logBsSlots_list,       
    #     maxLevelsRemaining,
    #     levelBudget_list,
    #     dnum,
    #     dcrtBits,
    #     firstMod,
    #     9,
    #     [],
    #     "UNIFORM_TERNARY",
    #     rescaleTech,
    #     save_dir=save_dir,
    #     mode=mode,
    # )
    cryptoContext, openfhe_context = utils.try_load_context(
        logN,
        [12],       
        20,
        levelBudget_list,
        3,
        dcrtBits,
        firstMod,
        9,
        [],
        "UNIFORM_TERNARY",
        rescaleTech,
        save_dir=save_dir,
        mode=mode,
    )
    log_encode_slot = 12
    encode_slots = 1 << log_encode_slot
    openfhe_context = openfhe_context[str(log_encode_slot)]
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(log_encode_slot)]
    cryptoContext.BsContext.to_cuda()
    cryptoContext.load_rotation_keys(log_encode_slot)

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
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, 6, (1<<log_encode_slot))
    cipher_rescale,cipher_openfhe_rescale = openfhe_context.encrypt(x, 2, 6, (1<<log_encode_slot))
    print('openfhe_context.depth is ',openfhe_context.depth - 1)
    for limb in range(1, cipher.cur_limbs + 1):
        tmp_ct=cipher.deep_copy()
        tmp_ct.drop_last_elements(tmp_ct.cur_limbs - limb)
        tmp_ct_rescale=cipher_rescale.deep_copy()
        tmp_ct_rescale.drop_last_elements( tmp_ct_rescale.cur_limbs - max(limb, 2))
        tmp_cv = tmp_ct.cipher_like([tmp_ct.cv[0], torch.zeros_like(tmp_ct.cv[0])])# ciphertext.py 的 cipher_like
        tmp_ct.cipher_like([tmp_ct.cv[0], torch.zeros_like(tmp_ct.cv[0])])
        tmp_cv_ext = hoisting_keyswitch.modup_to_ext(tmp_cv, cryptoContext)
        
        perf(partial(homo_ops.homo_add, tmp_ct, tmp_ct, cryptoContext), "homo_add", limb)
        perf(partial(homo_ops.homo_sub, tmp_ct, tmp_ct, cryptoContext), "homo_sub", limb)
        perf(partial(homo_ops.homo_mul, tmp_ct, tmp_ct, cryptoContext), "homo_mul", limb)
        perf(partial(homo_ops.homo_square, tmp_ct, cryptoContext), "homo_square", limb)
        perf(partial(homo_ops.homo_rescale, tmp_ct_rescale, 1, cryptoContext), "homo_rescale", limb)
        perf(partial(homo_ops.homo_add_scalar_double, tmp_ct, 1.0, cryptoContext), "homo_add_scalar_double", limb)
        perf(partial(homo_ops.homo_add_scalar_int, tmp_ct, 1, cryptoContext), "homo_add_scalar_int", limb)
        perf(partial(homo_ops.homo_mul_scalar_double, tmp_ct, 1.0, cryptoContext), "homo_mul_scalar_double", limb)
        perf(partial(homo_ops.homo_mul_scalar_int, tmp_ct, 1, cryptoContext), "homo_mul_scalar_int", limb)
        perf(partial(homo_ops.homo_rotate, tmp_ct, -1, cryptoContext), "homo_rotate", limb)


        # perf(partial(homo_ops.eval_fast_rotate, tmp_cv_ext, tmp_cv, 2, True, False, cryptoContext), "eval_fast_rotate", limb)
        perf(partial(hoisting_keyswitch.modup_to_ext, tmp_cv, cryptoContext), "modup_to_ext", limb)
        perf(partial(hoisting_keyswitch.moddown_from_ext, tmp_cv_ext, cryptoContext), "moddown_from_ext", limb)
        perf(partial(hoisting_keyswitch.key_switch_ext, tmp_cv, cryptoContext), "key_switch_P_ext", limb)
        perf(partial(hoisting_keyswitch.mult_rot_key_and_sum_ext, tmp_cv_ext, -1, cryptoContext), "mult_rot_key_and_sum_ext", limb) #偷一个
    cipher_last=cipher.deep_copy()
    cipher_last.drop_last_elements(cipher_last.cur_limbs - 2)
    perf( partial(bootstrapping.eval_bootstrap, cipher_last, cryptoContext.L, log_encode_slot, cryptoContext),"eval_bootstrap",1, repeat_time=3,)

    print_profiling_result()

if __name__ == "__main__":
    profiling_single_op()
