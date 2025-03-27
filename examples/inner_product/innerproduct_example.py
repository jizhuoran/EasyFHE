##########################
#### example for app #####
##########################
import sys, os, time
import numpy as np

sys.path.append("/".join(os.getcwd().split("/")[:-2]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))

import torch.fhe as fhe
import numpy as np
import torch
import os
import math

import torch.fhe.homo_ops as homo_ops
import torch.fhe.functional as F


@fhe.utils.profile_python_function
def homo_inner_product(cipher_A, cipher_B, n, cryptoContext):
    assert (n & (n - 1)) == 0, "n must be a power of two"
    cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)
    log_n = int(math.log2(n))
    shifts = [2**i for i in range(log_n)] 
    for shift in shifts:
        rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)
        cipher_product = fhe.homo_add(cipher_product, rotated, cryptoContext)
    
    return cipher_product

@fhe.utils.profile_python_function
def homo_inner_product_hoisting(cipher_A, cipher_B, n, cryptoContext):
    assert (n & (n - 1)) == 0, "n must be a power of two"
    cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)
    log_n = int(math.log2(n))
    shifts = [2 ** i for i in range(log_n)]


    bxExt = fhe.key_switch_P_ext(fhe.extract_cv(cipher_product,0, cryptoContext), cryptoContext)
    ax = fhe.extract_cv(cipher_product,1, cryptoContext)
    for shift in shifts:
        # ver1
        # rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)

        # ver2
        # rotated_modup = fhe.modup_to_ext(fhe.extract_cv(cipher_product, 1, cryptoContext), cryptoContext)
        # rotated_innerp = fhe.mult_rot_key_and_sum_ext(rotated_modup, shift, cryptoContext)
        # rotated = fhe.moddown_from_ext(rotated_innerp, cryptoContext)
        # rotated.cv[0] = F.cv_add(rotated.cv[0], cipher_product.cv[0], cryptoContext.moduliQ, rotated.cur_limbs)
        # rotated = fhe.cipher_automorphism(rotated, cryptoContext.norm_rot_index(shift), cryptoContext, printInfo=False)

        # ver3
        # rotated_modup = fhe.modup_to_ext(ax, cryptoContext)
        # norm_index = cryptoContext.norm_rot_index(shift)
        # rotated_innerp = fhe.mult_rot_key_and_sum_ext(rotated_modup, norm_index, cryptoContext)
        #
        # tmp_bxExt = fhe.homo_add(fhe.extract_cv(rotated_innerp,0, cryptoContext), bxExt, cryptoContext)
        # tmp_ax    = fhe.moddown_from_ext(fhe.extract_cv(rotated_innerp,1, cryptoContext), cryptoContext)
        #
        # tmp_bxExt = fhe.cipher_automorphism(tmp_bxExt, norm_index, cryptoContext)
        # tmp_ax  = fhe.cipher_automorphism(tmp_ax, norm_index, cryptoContext)
        #
        # bxExt = fhe.homo_add(tmp_bxExt, bxExt, cryptoContext)
        # ax    = fhe.homo_add(tmp_ax, ax, cryptoContext)

        # ver4
        rotated_modup = fhe.modup_to_ext(ax, cryptoContext)
        tmp = fhe.eval_fast_rotate(rotated_modup, bxExt, shift, True, False, cryptoContext)
        tmp_bxExt=fhe.extract_cv(tmp,0, cryptoContext)
        tmp_ax = fhe.moddown_from_ext(fhe.extract_cv(tmp,1, cryptoContext), cryptoContext)

        bxExt = fhe.homo_add(tmp_bxExt, bxExt, cryptoContext)
        ax = fhe.homo_add(tmp_ax, ax, cryptoContext)

    bx = fhe.moddown_from_ext(bxExt, cryptoContext)
    cipher_product = cipher_product.cipher_like([bx.cv[0], ax.cv[0]])

    return cipher_product


def homo_inner_product_hoisting_mult_down(cipher_A, cipher_B, n, cryptoContext):
    assert (n & (n - 1)) == 0, "n must be a power of two"
    assert cryptoContext.rescaleTech != "FIXEDMANUAL", "cannot deal with manual rescale"
    log_n = int(math.log2(n))

    # ver1
    # cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    # cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)

    # ver2
    # in0=cipher_A
    # in1=cipher_B
    # in0, in1 = homo_ops._adjust_for_mult(in0, in1, cryptoContext)
    # res = homo_ops._cipher_mul(in0, in1, cryptoContext)
    # digits = fhe.modup_to_ext(in0.cipher_like([res.cv[2]]), cryptoContext)

    # swk = [cryptoContext.swk_bx, cryptoContext.swk_ax,]
    # sum_mult = digits.cipher_like(F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
    #                                      swk_bx=swk[0], swk_ax=swk[1]), is_ext=True)
    # tmp = fhe.moddown_from_ext(sum_mult, cryptoContext)
    # res.cv = res.cv[:2]
    # cipher_product = homo_ops._cipher_add(res, tmp, cryptoContext)

    # ver3
    # in0=cipher_A
    # in1=cipher_B
    # in0, in1 = homo_ops._adjust_for_mult(in0, in1, cryptoContext)
    # res = homo_ops._cipher_mul(in0, in1, cryptoContext)
    # digits = fhe.modup_to_ext(in0.cipher_like([res.cv[2]]), cryptoContext)
    # swk = [cryptoContext.swk_bx, cryptoContext.swk_ax,]
    # sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
    #                                                 swk_bx=swk[0], swk_ax=swk[1])
    # tmp0 = F.cv_moddown(sum_mult[0], cipher.cur_limbs, cryptoContext)
    # tmp1 = F.cv_moddown(sum_mult[1], cipher.cur_limbs, cryptoContext)
    # tmp0 = F.cv_add(tmp0, res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)
    # tmp1 = F.cv_add(tmp1, res.cv[1], cryptoContext.moduliQ, in0.cur_limbs)
    # cipher_product = res.cipher_like([tmp0, tmp1])
    #
    # bxExt = fhe.key_switch_P_ext(fhe.extract_cv(cipher_product,0, cryptoContext), cryptoContext)
    # ax = fhe.extract_cv(cipher_product,1, cryptoContext)

    # ver4 correct
    in0=cipher_A
    in1=cipher_B
    in0, in1 = homo_ops._adjust_for_mult(in0, in1, cryptoContext)
    res = homo_ops._cipher_mul(in0, in1, cryptoContext)
    digits = fhe.modup_to_ext(res.cipher_like([res.cv[2]]), cryptoContext)
    swk = [cryptoContext.swk_bx, cryptoContext.swk_ax,]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
                                 swk_bx=swk[0], swk_ax=swk[1])


    tmp1 = F.cv_moddown(sum_mult[1], cipher.cur_limbs, cryptoContext)
    tmp1 = F.cv_add(tmp1, res.cv[1], cryptoContext.moduliQ, in0.cur_limbs)
    ax = res.cipher_like([tmp1])


    # try: correct
    # tmp0 = F.cv_moddown(sum_mult[0], cipher.cur_limbs, cryptoContext)
    # tmp0 = F.cv_add(tmp0, res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)
    # bxExt = fhe.key_switch_P_ext(res.cipher_like([tmp0]), cryptoContext)


    # try: correct
    #
    # # bxExt = fhe.key_switch_P_ext(fhe.extract_cv(res,0, cryptoContext), cryptoContext)
    # # cv = [F.cv_add(sum_mult[0], bxExt.cv[0], cryptoContext.QplusP_map[in0.cur_limbs], in0.cur_limbs + cryptoContext.K,)]
    # bxExt_cv0 =  torch.cat((
    #     F.cv_mul_scalar(res.cv[0], cryptoContext.PModq, cryptoContext.moduliQ, cryptoContext.q_mu, cipher.cur_limbs),
    #     torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    # ), dim=0)
    # cv = [F.cv_add(sum_mult[0], bxExt_cv0, cryptoContext.QplusP_map[in0.cur_limbs], in0.cur_limbs + cryptoContext.K,)]
    # bxExt = sum_mult_ct.cipher_like(cv)
    #
    # tmp0=fhe.moddown_from_ext(bxExt, cryptoContext)
    #
    # cipher_product = res.cipher_like([tmp0.cv[0], tmp1])
    # shifts = [2**i for i in range(log_n)]
    # for shift in shifts:
    #     rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)
    #     cipher_product = fhe.homo_add(cipher_product, rotated, cryptoContext)
    #
    # return cipher_product


    # try: correct
    # # bxExt = fhe.key_switch_P_ext(fhe.extract_cv(res,0, cryptoContext), cryptoContext)
    # # cv = [F.cv_add(sum_mult[0], bxExt.cv[0], cryptoContext.QplusP_map[in0.cur_limbs], in0.cur_limbs + cryptoContext.K,)]
    # bxExt_cv0 =  torch.cat((
    #     F.cv_mul_scalar(res.cv[0], cryptoContext.PModq, cryptoContext.moduliQ, cryptoContext.q_mu, cipher.cur_limbs),
    #     torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    # ), dim=0)
    # cv = [F.cv_add(sum_mult[0], bxExt_cv0, cryptoContext.QplusP_map[in0.cur_limbs], in0.cur_limbs + cryptoContext.K,)]
    # bxExt = sum_mult_ct.cipher_like(cv)
    #
    # tmp0=fhe.moddown_from_ext(bxExt, cryptoContext)
    #
    # cipher_product = res.cipher_like([tmp0.cv[0], tmp1])
    # bxExt=fhe.key_switch_P_ext(fhe.extract_cv(cipher_product,0, cryptoContext),cryptoContext)

    # try: correct
    bxExt_cv0 =  torch.cat((
        F.cv_mul_scalar(res.cv[0], cryptoContext.PModq, cryptoContext.moduliQ, cryptoContext.q_mu, cipher.cur_limbs),
        torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    ), dim=0)
    cv = [F.cv_add(sum_mult[0], bxExt_cv0, cryptoContext.QplusP_map[in0.cur_limbs], in0.cur_limbs + cryptoContext.K,)]
    bxExt = res.cipher_like(cv)
    bxExt.is_ext = True


    shifts = [2 ** i for i in range(log_n)]

    for shift in shifts:
        # ver1
        # rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)

        # ver2
        # rotated_modup = fhe.modup_to_ext(fhe.extract_cv(cipher_product, 1, cryptoContext), cryptoContext)
        # rotated_innerp = fhe.mult_rot_key_and_sum_ext(rotated_modup, shift, cryptoContext)
        # rotated = fhe.moddown_from_ext(rotated_innerp, cryptoContext)
        # rotated.cv[0] = F.cv_add(rotated.cv[0], cipher_product.cv[0], cryptoContext.moduliQ, rotated.cur_limbs)
        # rotated = fhe.cipher_automorphism(rotated, cryptoContext.norm_rot_index(shift), cryptoContext, printInfo=False)

        # ver3
        rotated_modup = fhe.modup_to_ext(ax, cryptoContext)
        norm_index = cryptoContext.norm_rot_index(shift)
        rotated_innerp = fhe.mult_rot_key_and_sum_ext(rotated_modup, norm_index, cryptoContext)

        tmp_bxExt = fhe.homo_add(fhe.extract_cv(rotated_innerp,0, cryptoContext), bxExt, cryptoContext)
        tmp_ax    = fhe.moddown_from_ext(fhe.extract_cv(rotated_innerp,1, cryptoContext), cryptoContext)

        tmp_bxExt = fhe.cipher_automorphism(tmp_bxExt, norm_index, cryptoContext)
        tmp_ax  = fhe.cipher_automorphism(tmp_ax, norm_index, cryptoContext)

        bxExt = fhe.homo_add(tmp_bxExt, bxExt, cryptoContext)
        ax    = fhe.homo_add(tmp_ax, ax, cryptoContext)

    bx = fhe.moddown_from_ext(bxExt, cryptoContext)
    cipher_product = res.cipher_like([bx.cv[0], ax.cv[0]])

    return cipher_product


@fhe.utils.profile_python_function
def homo_inner_product_hybrid(cipher_A, cipher_B, n, cryptoContext):
    assert (n & (n - 1)) == 0, "n must be a power of two"
    assert n in [8, 32, 128, 512], "Currently only supports n in [8, 32, 128, 512]"
    cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)


    digits_ext = fhe.modup_to_ext(
        fhe.extract_cv(cipher_product, 1, cryptoContext), cryptoContext
    )

    fast_rotation_ext=[]
    fast_rotation_ext.append(fhe.key_switch_P_ext(cipher_product, cryptoContext))
    for i in range(1, 8):
        fast_rotation_ext.append(
            fhe.eval_fast_rotate(
                digits_ext, cipher_product, i, True, False, cryptoContext
            )
        )

    # sum up
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0],fast_rotation_ext[1],cryptoContext)
    fast_rotation_ext[2] = fhe.homo_add(fast_rotation_ext[2],fast_rotation_ext[3],cryptoContext)
    fast_rotation_ext[4] = fhe.homo_add(fast_rotation_ext[4],fast_rotation_ext[5],cryptoContext)
    fast_rotation_ext[6] = fhe.homo_add(fast_rotation_ext[6],fast_rotation_ext[7],cryptoContext)
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0],fast_rotation_ext[2],cryptoContext)
    fast_rotation_ext[4] = fhe.homo_add(fast_rotation_ext[4],fast_rotation_ext[6],cryptoContext)
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0],fast_rotation_ext[4],cryptoContext)

    cipher_product = fast_rotation_ext[0]

    if (n<=8):
        return fhe.moddown_from_ext(cipher_product, cryptoContext)

    tmp_bxExt = fhe.extract_cv(cipher_product, 0, cryptoContext)
    ax = fhe.moddown_from_ext(fhe.extract_cv(cipher_product, 1, cryptoContext), cryptoContext)

    digits_ext = fhe.modup_to_ext(ax, cryptoContext)

    fast_rotation_ext = []
    start_idx = 8
    for i in range(start_idx, start_idx*4, start_idx):
        fast_rotation_ext.append(
            fhe.eval_fast_rotate(
                digits_ext, cipher_product, i, True, False, cryptoContext
            )
        )

    # sum up
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0], fast_rotation_ext[1], cryptoContext)
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0], fast_rotation_ext[2], cryptoContext)

    cipher_product = fast_rotation_ext[0]
    tmp_bxExt = fhe.homo_add(fhe.extract_cv(cipher_product, 0, cryptoContext), tmp_bxExt, cryptoContext)
    ax = fhe.homo_add(fhe.moddown_from_ext(fhe.extract_cv(cipher_product, 1, cryptoContext), cryptoContext),ax,cryptoContext)

    if n<=32:
        bx = fhe.moddown_from_ext(tmp_bxExt, cryptoContext)
        cipher_product = cipher_product.cipher_like([bx.cv[0], ax.cv[0]])
        cipher_product.is_ext = False
        return cipher_product

    digits_ext = fhe.modup_to_ext(ax, cryptoContext)

    fast_rotation_ext = []
    start_idx = 32
    for i in range(start_idx, start_idx * 4, start_idx):
        fast_rotation_ext.append(
            fhe.eval_fast_rotate(
                digits_ext, tmp_bxExt, i, True, False, cryptoContext
            )
        )

    # sum up
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0], fast_rotation_ext[1], cryptoContext)
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0], fast_rotation_ext[2], cryptoContext)


    tmp_bxExt = fhe.homo_add(fhe.extract_cv(fast_rotation_ext[0], 0, cryptoContext), tmp_bxExt, cryptoContext)
    ax = fhe.homo_add(fhe.moddown_from_ext(fhe.extract_cv(fast_rotation_ext[0], 1, cryptoContext), cryptoContext),ax,cryptoContext)

    if n<=128:
        bx = fhe.moddown_from_ext(tmp_bxExt, cryptoContext)
        cipher_product = cipher_product.cipher_like([bx.cv[0], ax.cv[0]])
        return cipher_product


    digits_ext = fhe.modup_to_ext(ax, cryptoContext)

    fast_rotation_ext = []
    start_idx = 128
    for i in range(start_idx, start_idx * 4, start_idx):
        fast_rotation_ext.append(
            fhe.eval_fast_rotate(
                digits_ext, tmp_bxExt, i, True, False, cryptoContext
            )
        )

    # sum up
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0], fast_rotation_ext[1], cryptoContext)
    fast_rotation_ext[0] = fhe.homo_add(fast_rotation_ext[0], fast_rotation_ext[2], cryptoContext)

    tmp_bxExt = fhe.homo_add(fhe.extract_cv(fast_rotation_ext[0], 0, cryptoContext), tmp_bxExt, cryptoContext)
    ax = fhe.homo_add(fhe.moddown_from_ext(fhe.extract_cv(fast_rotation_ext[0], 1, cryptoContext), cryptoContext),ax,cryptoContext)

    if n<=512:
        bx = fhe.moddown_from_ext(tmp_bxExt, cryptoContext)
        cipher_product = cipher_product.cipher_like([bx.cv[0], ax.cv[0]])
        return cipher_product



def plain_inner_product(x, x1):
    return torch.dot(x, x1)

encode_slots = (1 << 9)
print("encode_slots:", encode_slots)
maxLevelsRemaining = 15
# appRotIndex_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
appRotIndex_list = [1, 2, 3, 4, 5, 6, 7,
                    8, 16, 24, 32,
                    64, 96, 128,
                    256, 384, 512,
                    1024
                    ]


logBsSlots_list = []
logN = 16
dnum = 3
dcrtBits = 58
firstMod = 60
levelBudget_list = []
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
mode = "release"  # "debug" or "release"
autoLoadAndSetConfig = True # note: currently only support True

DATA_DIR = os.environ["DATA_DIR"]

config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True)
cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                         levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=DATA_DIR, config=config))


# values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
# values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

x = np.array([values[i % len(values)] for i in range(encode_slots)])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

values1 = [0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
x1 = torch.tensor(x1, device="cuda")
ptx = fhe.encode(x1, 1, 0, encode_slots, False, cryptoContext)
cipher1 = openfhe_context.encrypt(x1, 1, openfhe_context.depth - 1, encode_slots)

for n in [8, 32, 128, 512]:
    print("vector length for inner product operations", n, "\n")

    cipher_inner_product = homo_inner_product(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    start_time = time.time()
    cipher_inner_product = homo_inner_product(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    print("time: ", time.time() - start_time)
    print("homo_inner_product done!")
    clear_result = openfhe_context.decrypt(cipher_inner_product)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[0], "\n\n")


    cipher_inner_product_hoisting = homo_inner_product_hoisting(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    start_time = time.time()
    cipher_inner_product_hoisting = homo_inner_product_hoisting(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    print("time: ", time.time() - start_time)
    print("homo_inner_product_hoisting done!")
    clear_result = openfhe_context.decrypt(cipher_inner_product_hoisting)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[0], "\n\n")

    cipher_inner_product_hoisting_mult_down = homo_inner_product_hoisting_mult_down(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    start_time = time.time()
    cipher_inner_product_hoisting_mult_down = homo_inner_product_hoisting_mult_down(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    print("time: ", time.time() - start_time)
    print("homo_inner_product_hoisting_mult_down done!")
    clear_result = openfhe_context.decrypt(cipher_inner_product_hoisting_mult_down)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[0], "\n\n")


    cipher_inner_product_hybrid = homo_inner_product_hybrid(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    start_time = time.time()
    cipher_inner_product_hybrid = homo_inner_product_hybrid(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    print("time: ", time.time() - start_time)
    print("homo_inner_product_hybrid done!")
    clear_result = openfhe_context.decrypt(cipher_inner_product_hybrid)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[0], "\n\n")

    print("plain",plain_inner_product(x, x1))

    print("\n\n")