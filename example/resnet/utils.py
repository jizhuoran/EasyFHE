import torch
import pickle
from torch.fhe import homo_ops

normalized_deltas = [
    [0.30245313974658655, 0, 0, 0, 0, 0],
    [
        0.25771464233502284,
        0.17572235969058683,
        0.26867995906162545,
        0.16879219146810473,
        0.32389941065236755,
        0.16670296717723732,
    ],
    [
        0.29577777852997955,
        0.20468562391210693,
        0.45305236761033496,
        0.1940840042412194,
        0.3655523676384972,
        0.13282571451191513,
    ],
    [
        0.3620743161940029,
        0.2372317323595584,
        0.32624424495604537,
        0.13859561075656615,
        0.34910082672803205,
        0.053238969339825734,
    ],
]


def log2_int(x):
    import math

    return int(math.log2(x))


def SerializeToFile(file_path, obj):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def DeserializeFromFile(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def set_bootstrapping_keys(specify_slots, cryptoContext):
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(log2_int(specify_slots))]


def read_values_from_file(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "{}_{}_{}_{}".format(filename, level, scale_deg, slots)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "{}_{}_{}_{}".format(filename, level, scale_deg, slots)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def read_fc_weight(cryptoContext, level, scale_deg, slots):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded["fc_{}_{}_{}".format(level, scale_deg, slots)]
    else:
        ptx = cryptoContext.pre_encoded[
            "fc_{}_{}_{}".format(level, scale_deg, slots)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_mod(n, cur_limbs, custom_val, he_res20_ctx, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_scecond_n(n, cur_limbs, he_res20_ctx, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_first_n(n, cur_limbs, he_res20_ctx, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_from_to(from_, to, cur_limbs, he_res20_ctx, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_from_to_{}_{}_{}_{}".format(
                from_, to, cur_limbs, he_res20_ctx.cur_num_slots
            )
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_from_to_{}_{}_{}_{}".format(
                from_, to, cur_limbs, he_res20_ctx.cur_num_slots
            )
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def gen_mask(n, cur_limbs, he_res20_ctx, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_first_n_mod(n, padding, pos, cur_limbs, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_first_n_mod2(n, padding, pos, cur_limbs, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_channel(n, cur_limbs, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384 * 2)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384 * 2)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def mask_channel2(n, cur_limbs, cryptoContext):
    if cryptoContext.PRELOAD_ALL:
        return cryptoContext.pre_encoded[
            "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192 * 2)
        ]
    else:
        ptx = cryptoContext.pre_encoded[
            "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192 * 2)
        ].shallow_copy()
        ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
        return ptx


def rotsum(input, slots, cryptoContext):
    result = input.deep_copy()
    for i in range(log2_int(slots)):
        result = homo_ops.homo_add(
            result,
            homo_ops.homo_rotate(result, pow(2, i), cryptoContext),
            cryptoContext,
        )
    return result


def rotsum_padded(input, slots, cryptoContext):
    result = input.deep_copy()
    for i in range(log2_int(slots)):
        result = homo_ops.homo_add(
            result,
            homo_ops.homo_rotate(result, slots * pow(2, i), cryptoContext),
            cryptoContext,
        )
    return result


def repeat(input, slots, cryptoContext):
    return homo_ops.homo_rotate(
        rotsum(input, slots, cryptoContext), -slots + 1, cryptoContext
    )
