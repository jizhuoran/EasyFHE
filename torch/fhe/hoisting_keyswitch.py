import numpy as np
from .ciphertext import Cipher
from . import functional as F

# @profile_python_function
def eval_fast_rotation_precompute(input, curr_limbs, cryptoContext):
    res = F.cv_modup(input, curr_limbs, cryptoContext)
    return res.clone()

# @profile_python_function
def eval_fast_key_switch_core_ext(d2Tilde, auto_index, beta, curr_limbs, cryptoContext):
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    swk_bx = swk[0][:beta, :, :]
    swk_ax = swk[1][:beta, :, :]

    res = F.cv_innerproduct(
        d2Tilde.reshape(-1),
        curr_limbs=curr_limbs,
        context=cryptoContext,
        swk_bx=swk_bx,
        swk_ax=swk_ax
    )
    return res[0], res[1]


# @profile_python_function
#todo: use ct instead of bx
def eval_fast_rotation_ext(bx, digits, curr_limbs, scaling_factor, noise_deg, slots, index, add_first, cryptoContext):
    alpha = cryptoContext.K
    K = cryptoContext.K
    beta = int(np.ceil(curr_limbs / alpha))  # Calculate beta as per the original C++ code
    expand_limbs = curr_limbs + K

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)

    # Inner Product
    sumbxmult, sumaxmult = eval_fast_key_switch_core_ext(digits, auto_index, beta, curr_limbs, cryptoContext)

    if (add_first):
        cMult = F.cv_mul_scalar(bx, cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                cryptoContext.q_mu_cuda, curr_limbs)
        sumbxmult = F.cv_add(sumbxmult, cMult, cryptoContext.moduliQ_cuda, curr_limbs, inplace=True)

    cv0 = F.cv_automorphism_transform(sumbxmult, expand_limbs, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(sumaxmult, expand_limbs, auto_index, cryptoContext)
    return Cipher([cv0, cv1], curr_limbs, scaling_factor, noise_deg, slots)

# @profile_python_function
def key_switch_down(sumbxmult, sumaxmult, curr_limbs, scaling_factor, noise_deg, slots, cryptoContext):
    res_bx = F.cv_moddown(sumbxmult, curr_limbs, cryptoContext)
    res_ax = F.cv_moddown(sumaxmult, curr_limbs, cryptoContext)
    return Cipher([res_bx, res_ax], curr_limbs, scaling_factor, noise_deg, slots)

def eval_fast_rotation(ciphertext, index, digits, cryptoContext):
    if index == 0:
        return ciphertext.clone()

    cur_limbs = ciphertext.cur_limbs
    beta = int(np.ceil(cur_limbs / cryptoContext.K))  # Calculate beta as per the original C++ code

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)

    # EvalFastKeySwitchCore = InnerProduct + ModDown
    sumbxmult, sumaxmult = eval_fast_key_switch_core_ext(digits, auto_index, beta, cur_limbs, cryptoContext)
    result = key_switch_down(sumbxmult, sumaxmult,
                             cur_limbs, ciphertext.scaling_factor, ciphertext.noise_deg, ciphertext.slots,
                             cryptoContext)
    # post add after ks
    result.cv[0] = F.cv_add(ciphertext.cv[0], result.cv[0], cryptoContext.moduliQ_cuda, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    result.cv[0] = F.cv_automorphism_transform(result.cv[0], cur_limbs, auto_index, cryptoContext)
    result.cv[1] = F.cv_automorphism_transform(result.cv[1], cur_limbs, auto_index, cryptoContext)

    return result