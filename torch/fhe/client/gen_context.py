from . import openfhe as openfhe
from . import context as Context
import pickle
import numpy as np


def gen_contexts(
    maxLevelsRemaining,
    rotIndex_list,
    logBsSlots_list,
    logN,
    dnum,
    dcrtBits,
    firstMod,
    levelBudget_list,
    secretKeyDist,
    rescaleTech,
    save_dir,
    config,
    dim1=[0, 0],
):

    print("Generating context")

    save_path_meta = "_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
        maxLevelsRemaining,
        '-'.join(map(str, logBsSlots_list)),
        '-'.join('-'.join(map(str, levelBudget)) for levelBudget in levelBudget_list),
        logN,
        dnum,
        dcrtBits,
        firstMod,
        secretKeyDist,
        rescaleTech,
    )

    GPUFHE_path = save_dir + "/GPU-FHE-CONTEXT" + save_path_meta
    DEBUG_save_path = save_dir + "/DEBUG-GPU-FHE-CONTEXT" + save_path_meta
    # OPENFHE_path = save_dir + "/OPEN-FHE-CONTEXT" + save_path_meta


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

    openfhe_secretKeyDist = SecretKeyDist_MAP[secretKeyDist]
    openfhe_rescaleTech = ScalingTechnique_MAP[rescaleTech]

    NO_BS = False
    if logBsSlots_list[0] == 0 and levelBudget_list == [[0, 0]]:
        NO_BS = True

    if NO_BS == True:
        depth = maxLevelsRemaining
    else:
        max_level_budget = max(
            levelBudget_list, key=lambda level_budget: level_budget[0] + level_budget[1]
        )
        approxModDepth = 9 # 9 is the default value of approxModDepth in openfhe
        depth = maxLevelsRemaining + openfhe.FHECKKSRNS.GetBootstrapDepth(
            approxModDepth, max_level_budget, openfhe_secretKeyDist
        )

    parameters = openfhe.CCParamsCKKSRNS()

    parameters.SetMultiplicativeDepth(depth)
    parameters.SetScalingModSize(dcrtBits)  #  dcrtBits GPU-FHE
    parameters.SetFirstModSize(firstMod)  # firstMod GPU-FHE
    # parameters.SetAuxModSize(AuxModSize) #  auxModSize (yhh added) GPU-FHE
    parameters.SetScalingTechnique(openfhe_rescaleTech)
    parameters.SetSecretKeyDist(openfhe_secretKeyDist)
    parameters.SetNumLargeDigits(dnum)  # dnum GPU-FHE
    parameters.SetRingDim(int(2**logN))
    parameters.SetSecurityLevel(openfhe.SecurityLevel.HEStd_NotSet)
    parameters.SetKeySwitchTechnique(openfhe.KeySwitchTechnique.HYBRID)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.FHE)
    cc.Enable(openfhe.PKESchemeFeature.PRE)

    openfhe.ClearEvalMultKeys()
    cc.ClearEvalAutomorphismKeys()

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    moduliQ, rootsQ, moduliP, rootsP = cc.GetPQ()
    rot_swk_map = {}
    autoIdx2rotIdx_map = {}
    MULT_SWK = np.array(cc.GetEvalMultKey(), dtype=np.uint64)
    if rotIndex_list is not None and rotIndex_list != []: # deal with "app" rot Index
        cc.EvalRotateKeyGen(keys.secretKey, rotIndex_list)
        APP_ROT_SWK = cc.GetEvalRotateKey()
        rot_swk_map["app"] = APP_ROT_SWK
        rotIndex_list_int_32t = [rotIndex & 0xFFFFFFFF if rotIndex < 0 else rotIndex for rotIndex in rotIndex_list]
        autoIdx_list = cc.FindAutomorphismIndices(rotIndex_list_int_32t)
        autoIdx2rotIdx_map.update(dict(zip(autoIdx_list, rotIndex_list)))

    openfheMembers = {}
    openfheMembers["cc"] = openfhe.Serialize(cc, openfhe.BINARY)
    openfheMembers["publicKey"] = openfhe.Serialize(keys.publicKey, openfhe.BINARY)
    openfheMembers["secretKey"] = openfhe.Serialize(keys.secretKey, openfhe.BINARY)
    openfheMembers["depth"] = depth
    openfheMembers["app_rot_key"] = openfhe.SerializeEvalAutomorphismKeyString(
        openfhe.BINARY
    )
    # with open(OPENFHE_path, "wb") as file:
    #     pickle.dump(openfheMembers, file)
    # del openfheMembers

    boot_cnst_map = {}
    if NO_BS == False: # need to do BS
        for logBsSlots, level_budget in zip(logBsSlots_list, levelBudget_list):
            cc.EvalBootstrapSetup(level_budget, [0, 0], 1 << logBsSlots)
            cc.EvalBootstrapKeyGen(keys.secretKey, 1 << logBsSlots)
            ROT_SWK = cc.GetEvalRotateKey()
            AUTOIDX_TO_ROTIDX = cc.GetEvalBootstrapAutoIdx2RotIdxMap(logBsSlots)
            rot_swk_map[str(logBsSlots)] = ROT_SWK
            autoIdx2rotIdx_map.update(AUTOIDX_TO_ROTIDX)
        N = int(2 ** logN)
        autoIdx2rotIdx_map[N * 2 - 1] = N * 2 - 1  # add conjugation automorphism index

        BOOT_KEY = cc.GetEvalBootstrapContext() # get matirx saved in boot_key
        for idx, logBsSlots in enumerate(logBsSlots_list):
            slot, C2S_dim, C2S_limbs, C2S_FC, C2S, S2C_dim, S2C_limbs, S2C_FC, S2C = BOOT_KEY[idx]
            assert slot == 1 << logBsSlots
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
            boot_cnst_map[str(logBsSlots)] = boot_key

    if config.COMPARE_WITH_OPENFHE:
        debugKeys = {}
        debugKeys["mul_key"] = openfhe.SerializeEvalMultKeyString(openfhe.BINARY)
        debugKeys["rot_key"] = openfhe.SerializeEvalAutomorphismKeyString(
            openfhe.BINARY
        )
        with open(DEBUG_save_path, "wb") as file:
            pickle.dump(debugKeys, file)
        del debugKeys

    openfhe.ClearEvalMultKeys()
    cc.ClearEvalAutomorphismKeys()
    openfhe.ReleaseAllContexts()

    gpufhe_context = Context.__FOR_SAVE_ONLY_Context(
        logN,
        logBsSlots_list,
        firstMod,
        dcrtBits,
        60,  # auxModSize of openfhe is 60 bits in default
        dnum,
        levelBudget_list,
        moduliQ,
        moduliP,
        rootsQ,
        rootsP,
        MULT_SWK,
        rot_swk_map,
        autoIdx2rotIdx_map,
        boot_cnst_map,
        secretKeyDist,
        rescaleTech,
        dim1,
    )

    BsContextMembers_dict = {}
    if NO_BS == False:
        for logBsSlots, level_budget in zip(logBsSlots_list, levelBudget_list):
            print("BsContext_map: ", logBsSlots)
            gpufhe_context.BsContext_map[str(logBsSlots)].eval_bootstrap_setup(
                gpufhe_context, level_budget, dim1, (1 << logBsSlots), 0
            )

        for logBsSlots in logBsSlots_list:
            BsContextMembers = {}
            for item in dir(gpufhe_context.BsContext_map[str(logBsSlots)]):
                if (
                    not callable(getattr(gpufhe_context.BsContext_map[str(logBsSlots)], item))
                ) and not item.startswith("__"):
                    BsContextMembers[item] = getattr(
                        gpufhe_context.BsContext_map[str(logBsSlots)], item
                    )
            BsContextMembers_dict[str(logBsSlots)] = BsContextMembers

    gpufheMembers = {}
    for item in dir(gpufhe_context):
        if (
            (not callable(getattr(gpufhe_context, item)))
            and (not item.startswith("__"))
            and (not item.startswith("BsContext"))
        ):
            gpufheMembers[item] = getattr(gpufhe_context, item)

    # with open(OPENFHE_path, "rb") as file:
    #     openfheMembers = pickle.load(file)
    with open(GPUFHE_path, "wb") as file:
        pickle.dump((gpufheMembers, openfheMembers, BsContextMembers_dict), file)
