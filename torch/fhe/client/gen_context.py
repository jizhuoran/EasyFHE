from . import openfhe as openfhe
from . import context as Context
import pickle
import numpy as np


def gen_contexts(
    logN,
    logSlots,
    maxLevelsRemaining,
    levelBudget,
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
    slots = int(2**logSlots)

    openfhe_secretKeyDist = SecretKeyDist_MAP[secretKeyDist]
    openfhe_rescaleTech = ScalingTechnique_MAP[rescaleTech]

    depth = maxLevelsRemaining + openfhe.FHECKKSRNS.GetBootstrapDepth(
        approxModDepth, levelBudget, openfhe_secretKeyDist
    )

    L = depth + 1  # GPUFHE: L
    K = (L + dnum - 1) // dnum  # GPUFHE: K = ceil(L/dnum)

    parameters = openfhe.CCParamsCKKSRNS()

    parameters.SetMultiplicativeDepth(depth)
    parameters.SetScalingModSize(dcrtBits)  #  logqi GPU-FHE
    parameters.SetFirstModSize(firstMod)  # logq0 GPU-FHE
    # parameters.SetAuxModSize(AuxModSize) #  logp (yhh added) GPU-FHE
    parameters.SetScalingTechnique(openfhe_rescaleTech)
    parameters.SetSecretKeyDist(openfhe_secretKeyDist)
    parameters.SetNumLargeDigits(dnum)  # dnum GPU-FHE
    parameters.SetRingDim(N)
    parameters.SetBatchSize(slots)  # ZRJI: slots
    parameters.SetSecurityLevel(openfhe.SecurityLevel.HEStd_NotSet)
    parameters.SetKeySwitchTechnique(openfhe.KeySwitchTechnique.HYBRID)


    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.FHE)
    cc.Enable(openfhe.PKESchemeFeature.PRE)

    cc.EvalBootstrapSetup(levelBudget, [0, 0], slots)
    keys = cc.KeyGen()

    evalKey = cc.ReKeyGen(keys.secretKey, keys.publicKey)
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    cc.EvalRotateKeyGen(keys.secretKey, rotate_index)

    moduliQ, rootsQ, moduliP, rootsP = cc.GetPQ()
    MULT_SWK = np.array(cc.GetEvalMultKey(), dtype=np.uint64)
    BOOT_KEY = cc.GetEvalBootstrapKey()
    C2S, S2C = [], []
    C2S_dim, S2C_dim = [], []
    C2S_limbs, S2C_limbs = [], []
    for slot, C2S_arr, S2C_arr, scfactor_U0hatTPreFFT, scfactor_U0PreFFT in BOOT_KEY:
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
    BOOT_KEY = {
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

    gpufhe_context = Context.__FOR_SAVE_ONLY_Context(
        logN,
        logSlots,
        firstMod,
        dcrtBits,
        59,  # openfhe is 59 bits
        L,
        K,
        levelBudget,
        moduliQ,
        moduliP,
        rootsQ,
        rootsP,
        MULT_SWK,
        ROT_SWK,
        BOOT_KEY,
        secretKeyDist,
        rescaleTech,
        dim1,
    )

    Context.eval_bootstrap_setup(
        gpufhe_context, gpufhe_context.levelBudget, dim1, (1<<logSlots), 0
    )

    save_path = (
        save_dir
        + "/GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            logN,
            logSlots,
            maxLevelsRemaining,
            levelBudget[0],
            levelBudget[1],
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

    BsContextMembers = {}
    for item in dir(gpufhe_context.BsContext):
        if (
            not callable(getattr(gpufhe_context.BsContext, item))
        ) and not item.startswith("__"):
            BsContextMembers[item] = getattr(gpufhe_context.BsContext, item)


    openfheMembers = {}
    openfheMembers["cc"] = openfhe.Serialize(cc, openfhe.BINARY)
    openfheMembers["eval_key"] = openfhe.Serialize(evalKey, openfhe.BINARY)
    openfheMembers["mul_key"] = openfhe.SerializeEvalMultKeyString(openfhe.BINARY)
    openfheMembers["rot_key"] = openfhe.SerializeEvalAutomorphismKeyString(openfhe.BINARY)
    openfheMembers["publicKey"] = openfhe.Serialize(keys.publicKey, openfhe.BINARY)
    openfheMembers["secretKey"] = openfhe.Serialize(keys.secretKey, openfhe.BINARY)
    openfheMembers["depth"] = depth
    openfheMembers["slots"] = slots
    openfheMembers["level_budget"] = levelBudget

    with open(save_path, "wb") as file:
        pickle.dump(
            (gpufheMembers, openfheMembers, BsContextMembers), file
        )


