from os import utime
from . import openfhe as openfhe
from . import context as Context
import pickle
import numpy as np


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
    dim1=[0, 0],
):

    print("Generating context")

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
    # slots_list = [int(2**logSlots) for logSlots in logSlots_list]
    max_level_budget = max(levelBudget_list, key=lambda level_budget: level_budget[0] + level_budget[1])
    # for level_budget in levelBudget_list:
    #      if ((level_budget[0] + level_budget[1]) > (max_level_budget[0] + max_level_budget[0])) :
    #          max_level_budget = level_budget

    openfhe_secretKeyDist = SecretKeyDist_MAP[secretKeyDist]
    openfhe_rescaleTech = ScalingTechnique_MAP[rescaleTech]

    depth = maxLevelsRemaining + openfhe.FHECKKSRNS.GetBootstrapDepth(
        approxModDepth, max_level_budget, openfhe_secretKeyDist
    )

    L = depth + 1  # GPUFHE: L
    K = (L + dnum - 1) // dnum  # GPUFHE: K = ceil(L/dnum)
    # specify_slots = logSlots_list[0] #todo: to be removed?

    parameters = openfhe.CCParamsCKKSRNS()

    parameters.SetMultiplicativeDepth(depth)
    parameters.SetScalingModSize(dcrtBits)  #  dcrtBits GPU-FHE
    parameters.SetFirstModSize(firstMod)  # logq0 GPU-FHE
    # parameters.SetAuxModSize(AuxModSize) #  auxModSize (yhh added) GPU-FHE
    parameters.SetScalingTechnique(openfhe_rescaleTech)
    parameters.SetSecretKeyDist(openfhe_secretKeyDist)
    parameters.SetNumLargeDigits(dnum)  # dnum GPU-FHE
    parameters.SetRingDim(N)
    # parameters.SetBatchSize(slots)  # ZRJI: slots #todo: to be removed
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
    MULT_SWK = np.array(cc.GetEvalMultKey(), dtype=np.uint64)
    moduliQ, rootsQ, moduliP, rootsP = cc.GetPQ()
    boot_key_map = {}
    rot_swk_map = {}
    for logslots, level_budget in zip(logSlots_list, levelBudget_list):
        cc.EvalBootstrapSetup(level_budget, [0, 0], 1<<logslots)
        cc.EvalBootstrapKeyGen(keys.secretKey, 1<<logslots)
        cc.EvalRotateKeyGen(keys.secretKey, rotate_index)

    BOOT_KEY = cc.GetEvalBootstrapKey()
    for idx, logslots in enumerate(logSlots_list):
        C2S, S2C = [], []
        C2S_dim, S2C_dim = [], []
        C2S_limbs, S2C_limbs = [], []
        for slot, C2S_arr, S2C_arr, scfactor_U0hatTPreFFT, scfactor_U0PreFFT in [BOOT_KEY[idx]]:
            if(slot != 1<<logslots):
                print("Error: BOOT_KEY order not match logSlots_list order")
                break
            U0hatTPreFFTScalingFactor = scfactor_U0hatTPreFFT
            U0PreFFTScalingFactor = scfactor_U0PreFFT
            for i in range(len(C2S_arr)):
                C2S_dim.append(len(C2S_arr[i]))
                for j in range(len(C2S_arr[i])):
                    if j == 0:
                        C2S_limbs.append(len(C2S_arr[i][j]))
                    for k in range(len(C2S_arr[i][j])):
                        C2S += C2S_arr[i][j][k]
            for i in range(len(S2C_arr)):
                S2C_dim.append(len(S2C_arr[i]))
                for j in range(len(S2C_arr[i])):
                    if j == 0:
                        S2C_limbs.append(len(S2C_arr[i][j]))
                    for k in range(len(S2C_arr[i][j])):
                        S2C += S2C_arr[i][j][k]
        boot_key = {
            "C2S": C2S,
            "S2C": S2C,
            "C2S_dim": C2S_dim,
            "S2C_dim": S2C_dim,
            "C2S_limbs": C2S_limbs,
            "S2C_limbs": S2C_limbs,
            "U0hatTPreFFTScalingFactor": U0hatTPreFFTScalingFactor,
            "U0PreFFTScalingFactor": U0PreFFTScalingFactor,
        }
        ROT_SWK = cc.GetEvalRotateKey()
        boot_key_map[str(logslots)] = boot_key
        rot_swk_map[str(logslots)] = ROT_SWK

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
        gpufhe_context.BsContext_map[str(logslots)].eval_bootstrap_setup(
            gpufhe_context, level_budget, dim1, (1<<logslots), 0
        )

    save_path = (
        save_dir
        + "/GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            logN,
            logSlots_list,
            maxLevelsRemaining,
            levelBudget_list,
            dnum,
            dcrtBits,
            firstMod,
            approxModDepth,
            secretKeyDist,
            rescaleTech,
        )
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
                BsContextMembers[item] = getattr(gpufhe_context.BsContext_map[str(logSlots)], item)
        BsContextMembers_dict[str(logSlots)] = BsContextMembers


    openfheMembers = {}
    openfheMembers["cc"] = openfhe.Serialize(cc, openfhe.BINARY)
    openfheMembers["eval_key"] = openfhe.Serialize(evalKey, openfhe.BINARY)
    openfheMembers["mul_key"] = openfhe.SerializeEvalMultKeyString(openfhe.BINARY)
    openfheMembers["rot_key"] = openfhe.SerializeEvalAutomorphismKeyString(openfhe.BINARY)
    openfheMembers["publicKey"] = openfhe.Serialize(keys.publicKey, openfhe.BINARY)
    openfheMembers["secretKey"] = openfhe.Serialize(keys.secretKey, openfhe.BINARY)
    openfheMembers["depth"] = depth
    # openfheMembers["slots"] = 1<<specify_slots #todo: to be removed?
    # openfheMembers["level_budget"] = levelBudget

    with open(save_path, "wb") as file:
        pickle.dump(
            (gpufheMembers, openfheMembers, BsContextMembers_dict), file
        )


