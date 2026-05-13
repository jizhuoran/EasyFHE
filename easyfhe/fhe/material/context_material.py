from dataclasses import dataclass
from typing import Any

import easyfhe as torch
import numpy as np

from ..ciphertext import Cipher, Plaintext, PreparedPlaintext


@dataclass
class RuntimeContextMaterial:
    L: Any
    dnum: Any
    alpha: Any
    K: Any
    M: Any
    N: Any
    Nh: Any
    approxSF: Any
    h: Any
    levelBudget: Any
    logN: Any
    logNh: Any
    logBsSlots_list: Any
    auxModSize: Any
    rescaleTech: Any
    dcrtBits: Any
    max_num_moduli: Any
    secretKeyDist: Any
    sigma: Any
    primes: Any
    barret_k: Any
    barret_ratio: Any
    q_mu: Any
    moduliP_scalar: Any
    moduliQ_scalar: Any
    moduliQ: Any
    scalingFactorsReal: Any
    scalingFactorsRealBig: Any
    PModq: Any
    max_int_diffs: Any
    QmuplusPmu_map: Any
    QplusP_map: Any
    automorphism_transform_out: Any
    inner_out: Any
    moddown_out_ax: Any
    moddown_out_bx: Any
    modup_out: Any
    rescale_out: Any
    mod_raise_out: Any
    hat_inverse_vec_moddown: Any
    hat_inverse_vec_shoup_moddown: Any
    prod_inv_moddown: Any
    prod_inv_shoup_moddown: Any
    prod_q_i_mod_q_j_moddown: Any
    hat_inverse_vec_modup: Any
    hat_inverse_vec_shoup_modup: Any
    prod_q_i_mod_q_j_modup: Any
    inner_workspace: Any
    mult_swk_ax: Any
    mult_swk_bx: Any
    inverse_power_of_roots_div_two: Any
    inverse_scaled_power_of_roots_div_two: Any
    power_of_roots: Any
    power_of_roots_shoup: Any
    left_rot_key_map: Any
    precompute_auto_map: Any
    q_inv_mod_q: Any
    q_inv_mod_q_shoup: Any
    qlql_inv_mod_ql_div_ql_mod_q: Any
    qlql_inv_mod_ql_div_ql_mod_q_shoup: Any
    QmaxdiffplusPmaxdiff_map: Any
    encode_params_ksiPows: Any
    encode_params_rotGroup: Any
    encode_bitrev_indices: Any
    encode_values: Any
    QbarretKplusPbarretK_map: Any
    QbarretRatioplusPbarretRatio_map: Any
    inBS: bool = False


def _as_tensor(value, *, dtype):
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.dtype != dtype else value
    return torch.as_tensor(value, dtype=dtype)


def _builder_value(builder, name):
    return getattr(builder, name, None)


def runtime_material_from_builder(builder):
    L = _builder_value(builder, "L")
    dnum = _builder_value(builder, "dnum")
    alpha = _builder_value(builder, "alpha")
    K = _builder_value(builder, "K")
    M = _builder_value(builder, "M")
    N = _builder_value(builder, "N")
    Nh = _builder_value(builder, "Nh")
    approxSF = _builder_value(builder, "approxSF")
    h = _builder_value(builder, "h")
    levelBudget = _builder_value(builder, "levelBudget")
    logN = _builder_value(builder, "logN")
    logNh = _builder_value(builder, "logNh")
    logBsSlots_list = _builder_value(builder, "logBsSlots_list")
    specialMod = _builder_value(builder, "specialMod")
    rescaleTech = _builder_value(builder, "rescaleTech")
    dcrtBits = _builder_value(builder, "dcrtBits")
    max_num_moduli = _builder_value(builder, "max_num_moduli")
    secretKeyDist = _builder_value(builder, "secretKeyDist")
    sigma = _builder_value(builder, "sigma")
    primes = _builder_value(builder, "primes")
    barret_k = _builder_value(builder, "barret_k")
    barret_ratio = _builder_value(builder, "barret_ratio")
    q_mu = _builder_value(builder, "q_mu")
    moduliP_scalar = _builder_value(builder, "moduliP_scalar")
    moduliQ_scalar = _builder_value(builder, "moduliQ_scalar")
    moduliQ = _builder_value(builder, "moduliQ")
    scalingFactorsReal = _builder_value(builder, "scalingFactorsReal")
    scalingFactorsRealBig = _builder_value(builder, "scalingFactorsRealBig")
    PModq = _builder_value(builder, "PModq")
    QmuplusPmu_map = _builder_value(builder, "QmuplusPmu_map")
    QplusP_map = _builder_value(builder, "QplusP_map")
    automorphism_transform_out = _builder_value(builder, "automorphism_transform_out")
    inner_out = _builder_value(builder, "inner_out")
    moddown_out_ax = _builder_value(builder, "moddown_out_ax")
    moddown_out_bx = _builder_value(builder, "moddown_out_bx")
    modup_out = _builder_value(builder, "modup_out")
    rescale_out = _builder_value(builder, "rescale_out")
    mod_raise_out = _builder_value(builder, "mod_raise_out")
    hat_inverse_vec_moddown = _builder_value(builder, "hat_inverse_vec_moddown")
    hat_inverse_vec_shoup_moddown = _builder_value(builder, "hat_inverse_vec_shoup_moddown")
    prod_inv_moddown = _builder_value(builder, "prod_inv_moddown")
    prod_inv_shoup_moddown = _builder_value(builder, "prod_inv_shoup_moddown")
    prod_q_i_mod_q_j_moddown = _builder_value(builder, "prod_q_i_mod_q_j_moddown")
    hat_inverse_vec_modup = _builder_value(builder, "hat_inverse_vec_modup")
    hat_inverse_vec_shoup_modup = _builder_value(builder, "hat_inverse_vec_shoup_modup")
    prod_q_i_mod_q_j_modup = _builder_value(builder, "prod_q_i_mod_q_j_modup")
    inner_workspace = _builder_value(builder, "inner_workspace")
    mult_swk_ax = _builder_value(builder, "mult_swk_ax")
    mult_swk_bx = _builder_value(builder, "mult_swk_bx")
    inverse_power_of_roots_div_two = _builder_value(builder, "inverse_power_of_roots_div_two")
    inverse_scaled_power_of_roots_div_two = _builder_value(builder, "inverse_scaled_power_of_roots_div_two")
    power_of_roots = _builder_value(builder, "power_of_roots")
    power_of_roots_shoup = _builder_value(builder, "power_of_roots_shoup")
    total_left_rot_key_map = _builder_value(builder, "total_left_rot_key_map")
    total_precompute_auto_map = _builder_value(builder, "total_precompute_auto_map")
    q_inv_mod_q = _builder_value(builder, "q_inv_mod_q")
    q_inv_mod_q_shoup = _builder_value(builder, "q_inv_mod_q_shoup")
    qlql_inv_mod_ql_div_ql_mod_q = _builder_value(builder, "qlql_inv_mod_ql_div_ql_mod_q")
    qlql_inv_mod_ql_div_ql_mod_q_shoup = _builder_value(builder, "qlql_inv_mod_ql_div_ql_mod_q_shoup")
    QmaxdiffplusPmaxdiff_map = _builder_value(builder, "QmaxdiffplusPmaxdiff_map")
    encode_values = _builder_value(builder, "encode_values")
    QbarretKplusPbarretK_map = _builder_value(builder, "QbarretKplusPbarretK_map")
    QbarretRatioplusPbarretRatio_map = _builder_value(builder, "QbarretRatioplusPbarretRatio_map")

    encode_params_ksiPows = _builder_value(builder, "encode_params_ksiPows")
    encode_params_rotGroup = _builder_value(builder, "encode_params_rotGroup")
    encode_bitrev_indices = _builder_value(builder, "encode_bitrev_indices")

    q_mu = _as_tensor(q_mu, dtype=torch.uint64)
    moduliQ = _as_tensor(moduliQ, dtype=torch.uint64)
    primes = _as_tensor(primes, dtype=torch.uint64)
    power_of_roots = _as_tensor(power_of_roots, dtype=torch.uint64)
    power_of_roots_shoup = _as_tensor(power_of_roots_shoup, dtype=torch.uint64)
    inverse_power_of_roots_div_two = _as_tensor(inverse_power_of_roots_div_two, dtype=torch.uint64)
    inverse_scaled_power_of_roots_div_two = _as_tensor(inverse_scaled_power_of_roots_div_two, dtype=torch.uint64)
    barret_k = _as_tensor(barret_k, dtype=torch.uint64)
    barret_ratio = _as_tensor(barret_ratio, dtype=torch.uint64)
    hat_inverse_vec_modup = _as_tensor(hat_inverse_vec_modup, dtype=torch.uint64)
    hat_inverse_vec_shoup_modup = _as_tensor(hat_inverse_vec_shoup_modup, dtype=torch.uint64)
    prod_q_i_mod_q_j_modup = _as_tensor(prod_q_i_mod_q_j_modup, dtype=torch.uint64)
    hat_inverse_vec_moddown = _as_tensor(hat_inverse_vec_moddown, dtype=torch.uint64)
    hat_inverse_vec_shoup_moddown = _as_tensor(hat_inverse_vec_shoup_moddown, dtype=torch.uint64)
    prod_q_i_mod_q_j_moddown = _as_tensor(prod_q_i_mod_q_j_moddown, dtype=torch.uint64)
    prod_inv_moddown = _as_tensor(prod_inv_moddown, dtype=torch.uint64)
    prod_inv_shoup_moddown = _as_tensor(prod_inv_shoup_moddown, dtype=torch.uint64)
    qlql_inv_mod_ql_div_ql_mod_q = _as_tensor(qlql_inv_mod_ql_div_ql_mod_q, dtype=torch.uint64)
    qlql_inv_mod_ql_div_ql_mod_q_shoup = _as_tensor(qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype=torch.uint64)
    q_inv_mod_q = _as_tensor(q_inv_mod_q, dtype=torch.uint64)
    q_inv_mod_q_shoup = _as_tensor(q_inv_mod_q_shoup, dtype=torch.uint64)
    mult_swk_bx = _as_tensor(mult_swk_bx, dtype=torch.uint64)
    mult_swk_ax = _as_tensor(mult_swk_ax, dtype=torch.uint64)
    inner_workspace = _as_tensor(inner_workspace, dtype=torch.uint64)
    inner_out = _as_tensor(inner_out, dtype=torch.uint64)
    moddown_out_ax = _as_tensor(moddown_out_ax, dtype=torch.uint64)
    moddown_out_bx = _as_tensor(moddown_out_bx, dtype=torch.uint64)
    modup_out = _as_tensor(modup_out, dtype=torch.uint64)
    rescale_out = _as_tensor(rescale_out, dtype=torch.uint64)
    automorphism_transform_out = _as_tensor(automorphism_transform_out, dtype=torch.uint64)
    mod_raise_out = _as_tensor(mod_raise_out, dtype=torch.uint64)
    PModq = _as_tensor(PModq, dtype=torch.uint64)
    max_int_diffs = _as_tensor([(9223372036854775295 - prime) % prime for prime in primes.tolist()], dtype=torch.uint64)

    encode_params_rotGroup = _as_tensor(encode_params_rotGroup, dtype=torch.uint32)
    encode_params_ksiPows = _as_tensor(encode_params_ksiPows, dtype=torch.complex128)
    for key, value in encode_bitrev_indices.items():
        encode_bitrev_indices[key] = _as_tensor(value, dtype=torch.uint32)


    for key, value in QplusP_map.items():
        QplusP_map[key] = _as_tensor(value, dtype=torch.uint64)
    for key, value in QmuplusPmu_map.items():
        QmuplusPmu_map[key] = _as_tensor(value, dtype=torch.uint64)
    for key, value in QbarretKplusPbarretK_map.items():
        QbarretKplusPbarretK_map[key] = _as_tensor(value, dtype=torch.uint64)
    for key, value in QbarretRatioplusPbarretRatio_map.items():
        QbarretRatioplusPbarretRatio_map[key] = _as_tensor(value, dtype=torch.uint64)
    for key, value in QmaxdiffplusPmaxdiff_map.items():
        QmaxdiffplusPmaxdiff_map[key] = _as_tensor(value, dtype=torch.uint64)

    left_rot_key_map = {
        int(rotIdx): [
            _as_tensor(key_pair[0], dtype=torch.uint64),
            _as_tensor(key_pair[1], dtype=torch.uint64),
        ]
        for rotIdx, key_pair in total_left_rot_key_map.items()
    }

    precompute_auto_map = {
        int(rotIdx): _as_tensor(auto_map, dtype=torch.int32)
        for rotIdx, auto_map in total_precompute_auto_map.items()
    }

    for key, value in encode_values.items():
        if isinstance(value, Plaintext):
            encode_values[key].cv = [_as_tensor(value.cv, dtype=torch.uint64)]
            Cipher._id_counter = max(Cipher._id_counter, value.cipher_id)
        elif isinstance(value, PreparedPlaintext):
            encode_values[key].encoded_values = _as_tensor(value.encoded_values, dtype=torch.float64)


    return RuntimeContextMaterial(
        L=L,
        dnum=dnum,
        alpha=alpha,
        K=K,
        M=M,
        N=N,
        Nh=Nh,
        approxSF=approxSF,
        h=h,
        levelBudget=levelBudget,
        logN=logN,
        logNh=logNh,
        logBsSlots_list=logBsSlots_list,
        auxModSize=specialMod,
        rescaleTech=rescaleTech,
        dcrtBits=dcrtBits,
        max_num_moduli=max_num_moduli,
        secretKeyDist=secretKeyDist,
        sigma=sigma,
        primes=primes,
        barret_k=barret_k,
        barret_ratio=barret_ratio,
        q_mu=q_mu,
        moduliP_scalar=moduliP_scalar,
        moduliQ_scalar=moduliQ_scalar,
        moduliQ=moduliQ,
        scalingFactorsReal=scalingFactorsReal,
        scalingFactorsRealBig=scalingFactorsRealBig,
        PModq=PModq,
        max_int_diffs=max_int_diffs,
        QmuplusPmu_map=QmuplusPmu_map,
        QplusP_map=QplusP_map,
        automorphism_transform_out=automorphism_transform_out,
        inner_out=inner_out,
        moddown_out_ax=moddown_out_ax,
        moddown_out_bx=moddown_out_bx,
        modup_out=modup_out,
        rescale_out=rescale_out,
        mod_raise_out=mod_raise_out,
        hat_inverse_vec_moddown=hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown=hat_inverse_vec_shoup_moddown,
        prod_inv_moddown=prod_inv_moddown,
        prod_inv_shoup_moddown=prod_inv_shoup_moddown,
        prod_q_i_mod_q_j_moddown=prod_q_i_mod_q_j_moddown,
        hat_inverse_vec_modup=hat_inverse_vec_modup,
        hat_inverse_vec_shoup_modup=hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j_modup=prod_q_i_mod_q_j_modup,
        inner_workspace=inner_workspace,
        mult_swk_ax=mult_swk_ax,
        mult_swk_bx=mult_swk_bx,
        inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        power_of_roots=power_of_roots,
        power_of_roots_shoup=power_of_roots_shoup,
        left_rot_key_map=left_rot_key_map,
        precompute_auto_map=precompute_auto_map,
        q_inv_mod_q=q_inv_mod_q,
        q_inv_mod_q_shoup=q_inv_mod_q_shoup,
        qlql_inv_mod_ql_div_ql_mod_q=qlql_inv_mod_ql_div_ql_mod_q,
        qlql_inv_mod_ql_div_ql_mod_q_shoup=qlql_inv_mod_ql_div_ql_mod_q_shoup,
        QmaxdiffplusPmaxdiff_map=QmaxdiffplusPmaxdiff_map,
        encode_params_ksiPows=encode_params_ksiPows,
        encode_params_rotGroup=encode_params_rotGroup,
        encode_bitrev_indices=encode_bitrev_indices,
        encode_values=encode_values,
        QbarretKplusPbarretK_map=QbarretKplusPbarretK_map,
        QbarretRatioplusPbarretRatio_map=QbarretRatioplusPbarretRatio_map,
    )
