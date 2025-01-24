from os import utime
from . import openfhe as openfhe
from . import context as Context
import pickle, time
import numpy as np
import psutil, os


def gen_contexts(
    logN,
    logSlots_list,
    maxLevelsRemaining,
    levelBudget_list,
    dnum,
    dcrtBits,
    firstMod,
    approxModDepth,
    rotate_index,
    secretKeyDist,
    rescaleTech,
    save_dir,
    mode,
    dim1=[0, 0],
):

    print("Generating context")

    save_path_meta = "_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
        logN,
        "-".join(map(str, logSlots_list)),
        maxLevelsRemaining,
        "-".join("-".join(map(str, levelBudget)) for levelBudget in levelBudget_list),
        dnum,
        dcrtBits,
        firstMod,
        approxModDepth,
        secretKeyDist,
        rescaleTech,
    )

    GPUFHE_path = save_dir + "/GPU-FHE-CONTEXT" + save_path_meta
    debug_save_path = save_dir + "/DEBUG-GPU-FHE-CONTEXT" + save_path_meta
    OPENFHE_path = save_dir + "/OPEN-FHE-CONTEXT" + save_path_meta


    SecretKeyDist_MAP = {
        "GAUSSIAN": openfhe.SecretKeyDist.GAUSSIAN,
        "UNIFORM_TERNARY": openfhe.SecretKeyDist.UNIFORM_TERNARY,
        "SPARSE_TERNARY": openfhe.SecretKeyDist.SPARSE_TERNARY,
    }

    ScalingTechnique_MAP = {
        "FIXEDMANUAL": openfhe.ScalingTechnique.FIXEDMANUAL,
        "FIXEDAUTO": openfhe.ScalingTechnique.FIXEDAUTO,
        "FLEXIBLEAUTO": openfhe.ScalingTechnique.FLEXIBLEAUTO,
        "FLEXIBLEAUTOEXT": openfhe.ScalingTechnique.FLEXIBLEAUTOEXT,
        "NORESCALE": openfhe.ScalingTechnique.NORESCALE,
    }

    N = int(2**logN)
    max_level_budget = max(
        levelBudget_list, key=lambda level_budget: level_budget[0] + level_budget[1]
    )

    openfhe_secretKeyDist = SecretKeyDist_MAP[secretKeyDist]
    openfhe_rescaleTech = ScalingTechnique_MAP[rescaleTech]

    depth = maxLevelsRemaining + openfhe.FHECKKSRNS.GetBootstrapDepth(
        approxModDepth, max_level_budget, openfhe_secretKeyDist
    )

    L = depth + 1  # GPUFHE: L
    K = (L + dnum - 1) // dnum  # GPUFHE: K = ceil(L/dnum)

    parameters = openfhe.CCParamsCKKSRNS()

    parameters.SetMultiplicativeDepth(depth)
    parameters.SetScalingModSize(dcrtBits)  #  dcrtBits GPU-FHE
    parameters.SetFirstModSize(firstMod)  # firstMod GPU-FHE
    # parameters.SetAuxModSize(AuxModSize) #  auxModSize (yhh added) GPU-FHE
    parameters.SetScalingTechnique(openfhe_rescaleTech)
    parameters.SetSecretKeyDist(openfhe_secretKeyDist)
    parameters.SetNumLargeDigits(dnum)  # dnum GPU-FHE
    parameters.SetRingDim(N)
    parameters.SetSecurityLevel(openfhe.SecurityLevel.HEStd_NotSet)
    parameters.SetKeySwitchTechnique(openfhe.KeySwitchTechnique.HYBRID)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.FHE)
    cc.Enable(openfhe.PKESchemeFeature.PRE)

    keys = cc.KeyGen()
    evalKey = cc.ReKeyGen(keys.secretKey, keys.publicKey)
    cc.EvalMultKeyGen(keys.secretKey)
    moduliQ, rootsQ, moduliP, rootsP = cc.GetPQ()
    rot_swk_map = {}

    MULT_SWK = np.array(cc.GetEvalMultKey(), dtype=np.uint64)
    if rotate_index:
        cc.EvalRotateKeyGen(keys.secretKey, rotate_index)
        APP_ROT_SWK = cc.GetEvalRotateKey()
        rot_swk_map["app"] = APP_ROT_SWK

    boot_gen_time = 0
    rot_get_time = 0
    for logslots, level_budget in zip(logSlots_list, levelBudget_list):
        timei1 = time.time()
        cc.EvalBootstrapSetup(level_budget, [0, 0], 1 << logslots)
        cc.EvalBootstrapKeyGen(keys.secretKey, 1 << logslots)
        timei2 = time.time()
        ROT_SWK = cc.GetEvalRotateKey()
        rot_swk_map[str(logslots)] = ROT_SWK
        timei3 = time.time()
        boot_gen_time += timei2 - timei1
        rot_get_time += timei3 - timei2

    BOOT_KEY = cc.GetEvalBootstrapKey()

    openfheMembers = {}
    openfheMembers["cc"] = openfhe.Serialize(cc, openfhe.BINARY)
    openfheMembers["publicKey"] = openfhe.Serialize(keys.publicKey, openfhe.BINARY)
    openfheMembers["secretKey"] = openfhe.Serialize(keys.secretKey, openfhe.BINARY)
    openfheMembers["depth"] = depth
    with open(OPENFHE_path, "wb") as file:
        pickle.dump(openfheMembers, file)
    del openfheMembers


    if mode == "debug":
        debugKeys = {}
        debugKeys["eval_key"] = openfhe.Serialize(evalKey, openfhe.BINARY)
        debugKeys["mul_key"] = openfhe.SerializeEvalMultKeyString(openfhe.BINARY)
        debugKeys["rot_key"] = openfhe.SerializeEvalAutomorphismKeyString(
            openfhe.BINARY
        )
        with open(debug_save_path, "wb") as file:
            pickle.dump(debugKeys, file)
        del debugKeys

    openfhe.ClearEvalMultKeys()
    cc.ClearEvalAutomorphismKeys()
    openfhe.ReleaseAllContexts()

    boot_key_map = {}

    for idx, logslots in enumerate(logSlots_list):
        slot, C2S_dim, C2S_limbs, C2S_FC, C2S, S2C_dim, S2C_limbs, S2C_FC, S2C = BOOT_KEY[idx]
        assert slot == 1 << logslots
        boot_key = {
            "C2S": C2S,
            "S2C": S2C,
            "C2S_dim": C2S_dim,
            "S2C_dim": S2C_dim,
            "C2S_limbs": C2S_limbs,
            "S2C_limbs": S2C_limbs,
            "U0hatTPreFFTScalingFactor": C2S_FC,
            "U0PreFFTScalingFactor": S2C_FC,
        }
        boot_key_map[str(logslots)] = boot_key

    gpufhe_context = Context.__FOR_SAVE_ONLY_Context(
        logN,
        logSlots_list,
        firstMod,
        dcrtBits,
        60,  # auxModSize of openfhe is 60 bits in default
        L,
        K,
        levelBudget_list,
        moduliQ,
        moduliP,
        rootsQ,
        rootsP,
        MULT_SWK,
        rot_swk_map,
        boot_key_map,
        secretKeyDist,
        rescaleTech,
        dim1,
    )

    for logslots, level_budget in zip(logSlots_list, levelBudget_list):
        print("BsContext_map: ", logslots)
        gpufhe_context.BsContext_map[str(logslots)].eval_bootstrap_setup(
            gpufhe_context, level_budget, dim1, (1 << logslots), 0
        )

    gpufheMembers = {}
    for item in dir(gpufhe_context):
        if (
            (not callable(getattr(gpufhe_context, item)))
            and (not item.startswith("__"))
            and (not item.startswith("BsContext"))
        ):
            gpufheMembers[item] = getattr(gpufhe_context, item)

    BsContextMembers_dict = {}
    for logSlots in logSlots_list:
        BsContextMembers = {}
        for item in dir(gpufhe_context.BsContext_map[str(logSlots)]):
            if (
                not callable(getattr(gpufhe_context.BsContext_map[str(logSlots)], item))
            ) and not item.startswith("__"):
                BsContextMembers[item] = getattr(
                    gpufhe_context.BsContext_map[str(logSlots)], item
                )
        BsContextMembers_dict[str(logSlots)] = BsContextMembers

    with open(OPENFHE_path, "rb") as file:
        openfheMembers = pickle.load(file)
    with open(GPUFHE_path, "wb") as file:
        pickle.dump((gpufheMembers, openfheMembers, BsContextMembers_dict), file)
