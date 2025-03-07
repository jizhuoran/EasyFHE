import os
import pickle
import numpy as np
from .client import client as client
from .client.gen_context import gen_contexts
from .context import *
import torch


def round_half_away_from_zero(number, ndigits=0):
    multiplier = 10**ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    elif number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    else:
        return 0.0


def try_load_context(
    maxLevelsRemaining,
    rotIndex_list,
    logBsSlots_list,
    logN,
    levelBudget_list,
    save_dir,
    autoLoadAndSetConfig,
):

    NO_BS = False
    if logBsSlots_list is None or logBsSlots_list == []:
        assert (logBsSlots_list is None or logBsSlots_list == []) == (
            levelBudget_list is None or levelBudget_list == []
        ), "ERROR: logBsSlots_list and levelBudget_list must be both None or both not None!"
        logBsSlots_list = [0]
        levelBudget_list = [[0, 0]]
        NO_BS = True
    else:
        sorted_pairs = sorted(
            zip(logBsSlots_list, levelBudget_list), key=lambda x: x[0]
        )
        logBsSlots_list, levelBudget_list = zip(*sorted_pairs)
        logBsSlots_list = list(logBsSlots_list)
        levelBudget_list = list(levelBudget_list)

    load_path = save_dir + "/GPU-FHE-CONTEXT_{}_{}_{}_{}.pkl".format(
        maxLevelsRemaining,
        "-".join(map(str, logBsSlots_list)),
        "-".join("-".join(map(str, levelBudget)) for levelBudget in levelBudget_list),
        logN,
    )

    if not os.path.exists(load_path):
        gen_contexts(
            maxLevelsRemaining=maxLevelsRemaining,
            rotIndex_list=rotIndex_list,
            logBsSlots_list=logBsSlots_list,
            logN=logN,
            levelBudget_list=levelBudget_list,
            save_dir=save_dir,
        )

    with open(load_path, "rb") as file:
        gpufheMembers, openfheMembers, BsContextMembers = pickle.load(file)

    cryptoContext = Context(BsContextMembers, gpufheMembers, autoLoadAndSetConfig)
    openfhe_context = client.OpenFHEContext(openfheMembers)
    if cryptoContext.autoLoadAndSetConfig:
        if rotIndex_list is not None and rotIndex_list != []:
            load_rotation_keys("app", cryptoContext)
        if NO_BS == False:
            for logBsSlots in logBsSlots_list:
                cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
                cryptoContext.BsContext.to_cuda()
                load_rotation_keys(logBsSlots, cryptoContext)

    return cryptoContext, openfhe_context


def compare_bs_ct_with_openfhe(bs_cipher, openfhe_cipher):
    gpu_bootstrapping_res = np.array(
        [bs_cipher.cv[0].cpu().numpy(), bs_cipher.cv[1].cpu().numpy()]
    ).reshape(-1)
    openfhe_bootstrapping_res = np.array(openfhe_cipher.GetVectorOfData()).reshape(-1)
    return np.array_equal(gpu_bootstrapping_res, openfhe_bootstrapping_res)


def load_rotation_keys(key_name, cryptoContext):
    if (str(key_name) not in cryptoContext.slots_left_rot_key_map) or (
        not cryptoContext.slots_left_rot_key_map[str(key_name)]
    ):
        print("Warning: slots_left_rot_key_map[", key_name, "] is None")
        return
    for key, value in cryptoContext.slots_left_rot_key_map[str(key_name)].items():
        cryptoContext.left_rot_key_map[key] = [
            torch.tensor(v, dtype=torch.uint64, device="cuda") for v in value
        ]
    for key, value in cryptoContext.slots_precompute_auto_map[str(key_name)].items():
        cryptoContext.precompute_auto_map[key] = torch.tensor(
            value, dtype=torch.int32, device="cuda"
        )


def load_bootstrapping_context(logBsSlots, cryptoContext):
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
    cryptoContext.BsContext.to_cuda()
    load_rotation_keys(logBsSlots, cryptoContext)
