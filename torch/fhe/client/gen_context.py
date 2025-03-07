import numpy as np
import pickle
from . import context as Context
from . import openfhe


def gen_contexts(
    maxLevelsRemaining,
    rotIndex_list,
    logBsSlots_list,
    logN,
    levelBudget_list,
    save_dir,
    dim1=[0, 0],
):

    print("Generating context, very very slow!")

    save_path_meta = "_{}_{}_{}_{}.pkl".format(
        maxLevelsRemaining,
        "-".join(map(str, logBsSlots_list)),
        "-".join("-".join(map(str, levelBudget)) for levelBudget in levelBudget_list),
        logN,
    )

    GPUFHE_path = save_dir + "/GPU-FHE-CONTEXT" + save_path_meta
    OPENFHE_path = save_dir + "/OPEN-FHE-CONTEXT" + save_path_meta

    NO_BS = False
    if logBsSlots_list[0] == 0 and levelBudget_list == [[0, 0]]:
        NO_BS = True

    if NO_BS == True:
        depth = maxLevelsRemaining
    else:
        max_level_budget = max(
            levelBudget_list, key=lambda level_budget: level_budget[0] + level_budget[1]
        )
        depth = maxLevelsRemaining + 9 + max_level_budget[0] + max_level_budget[1]

    parameters = openfhe.CCParamsCKKSRNS()
    parameters.Init()
    parameters.SetMultiplicativeDepth(depth)
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
    if rotIndex_list is not None and rotIndex_list != []:
        cc.EvalRotateKeyGen(keys.secretKey, rotIndex_list)
        rot_swk_map["app"] = cc.GetEvalRotateKey()
        rotIndex_list_int_32t = [
            rotIndex & 0xFFFFFFFF if rotIndex < 0 else rotIndex
            for rotIndex in rotIndex_list
        ]
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
    with open(OPENFHE_path, "wb") as file:
        pickle.dump(openfheMembers, file)
    del openfheMembers

    boot_cnst_map = {}
    if NO_BS == False:
        for logBsSlots, level_budget in zip(logBsSlots_list, levelBudget_list):
            cc.EvalBootstrapSetup(level_budget, [0, 0], 1 << logBsSlots)
            cc.EvalBootstrapKeyGen(keys.secretKey, 1 << logBsSlots)
            ROT_SWK = cc.GetEvalRotateKey()
            AUTOIDX_TO_ROTIDX = cc.GetEvalBootstrapAutoIdx2RotIdxMap(logBsSlots)
            rot_swk_map[str(logBsSlots)] = ROT_SWK
            autoIdx2rotIdx_map.update(AUTOIDX_TO_ROTIDX)
        N = int(2**logN)
        autoIdx2rotIdx_map[N * 2 - 1] = N * 2 - 1

        BOOT_KEY = cc.GetEvalBootstrapContext()
        for idx, logBsSlots in enumerate(logBsSlots_list):
            slot, C2S_dim, C2S_limbs, _, C2S, S2C_dim, S2C_limbs, _, S2C = BOOT_KEY[idx]
            assert slot == 1 << logBsSlots
            boot_key = {
                "C2S": C2S,
                "S2C": S2C,
                "C2S_dim": C2S_dim,
                "S2C_dim": S2C_dim,
                "C2S_limbs": C2S_limbs,
                "S2C_limbs": S2C_limbs,
            }
            boot_cnst_map[str(logBsSlots)] = boot_key

    openfhe.ClearEvalMultKeys()
    cc.ClearEvalAutomorphismKeys()
    openfhe.ReleaseAllContexts()

    gpufhe_context = Context.__FOR_SAVE_ONLY_Context(
        logN,
        logBsSlots_list,
        60,
        levelBudget_list,
        moduliQ,
        moduliP,
        rootsQ,
        rootsP,
        MULT_SWK,
        rot_swk_map,
        autoIdx2rotIdx_map,
        boot_cnst_map,
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
                    not callable(
                        getattr(gpufhe_context.BsContext_map[str(logBsSlots)], item)
                    )
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

    with open(OPENFHE_path, "rb") as file:
        openfheMembers = pickle.load(file)
    with open(GPUFHE_path, "wb") as file:
        pickle.dump((gpufheMembers, openfheMembers, BsContextMembers_dict), file)
