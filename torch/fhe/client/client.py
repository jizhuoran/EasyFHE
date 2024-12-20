from . import openfhe as openfhe
import torch
from .. import Ciphertext as Cipher
from .. import context as Context

import numpy as np
import time

R_UNIFORM = 6


def gen_crypto_context(
    encodedLevel=0,
    logN=16,
    logSlots=14,
    maxLevelsRemaining=1,
    levelEnc=4,
    levelDec=4,
    dnum=3,
    dcrtBits=59,
    firstMod=60,
    AuxModSize=60,
    approxModDepth=9,
    rotate_index=[1, 2, 4],
    secretKeyDist=openfhe.UNIFORM_TERNARY,
    rescaleTech=openfhe.ScalingTechnique.FIXEDMANUAL,
):

    start = time.time()

    AuxModSize = 60  # OpenFHE is 60bits

    N = int(2**logN)
    slots = int(2**logSlots)

    depth = maxLevelsRemaining + openfhe.FHECKKSRNS.GetBootstrapDepth(
        approxModDepth, [levelEnc, levelDec], secretKeyDist
    )

    L = depth + 1  # GPUFHE: L
    K = (L + dnum - 1) // dnum  # GPUFHE: K = ceil(L/dnum)

    parameters = openfhe.CCParamsCKKSRNS()

    parameters.SetMultiplicativeDepth(depth)
    parameters.SetScalingModSize(dcrtBits)  #  logqi GPU-FHE
    parameters.SetFirstModSize(firstMod)  # logq0 GPU-FHE
    # parameters.SetAuxModSize(AuxModSize) #  logp (yhh added) GPU-FHE
    parameters.SetScalingTechnique(rescaleTech)
    parameters.SetSecretKeyDist(secretKeyDist)
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

    cc.EvalBootstrapSetup([levelEnc, levelDec], [0, 0], slots)
    keys = cc.KeyGen()

    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    cc.EvalRotateKeyGen(keys.secretKey, rotate_index)


    stop1 = time.time()
    print("Time to generate keys: ", stop1 - start)

    moduliQ, rootsQ, moduliP, rootsP = cc.GetPQ()
    stop2 = time.time()
    print("Time to get moduli: ", stop2 - stop1)
    MULT_SWK = np.array(cc.GetEvalMultKey(), dtype=np.uint64)
    stop3 = time.time()
    print("Time to get mult key: ", stop3 - stop2)
    BOOT_KEY = cc.GetEvalBootstrapKey()
    C2S, S2C = [], []
    C2S_dim, S2C_dim = [], []
    C2S_limbs, S2C_limbs = [], []
    for slot, C2S_arr, S2C_arr in BOOT_KEY:
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
    }
    stop4 = time.time()
    print("Time to get boot key: ", stop4 - stop3)
    ROT_SWK = cc.GetEvalRotateKey()
    stop5 = time.time()
    print("Time to get rotate key: ", stop5 - stop4)
    
    return parameters, cc, keys, moduliQ, moduliP, rootsQ, rootsP, MULT_SWK, ROT_SWK, BOOT_KEY

    # GPUFHE_Context = Context.Context(
    #     logN,
    #     firstMod,
    #     dcrtBits,
    #     AuxModSize,
    #     L,
    #     K,
    #     moduliQ,
    #     moduliP,
    #     rootsQ,
    #     rootsP,
    #     MULT_SWK,
    #     ROT_SWK,
    #     BOOT_KEY
    # )

    # return parameters, cc, keys, GPUFHE_Context


def gen_rotate_keys(cc, keys, rot):
    return cc.EvalRotateKeyGen(keys.secretKey, rot)


def encrypt(x, cc, keys):
    ptx = cc.MakeCKKSPackedPlaintext(x.tolist())
    cipher = cc.Encrypt(keys.publicKey, ptx)
    data = cipher.GetVectorOfData()
    cv = [torch.tensor(elem, device=x.device, dtype=torch.uint64) for elem in data]
    return Cipher.Cipher(cv, cv[0].shape[0])


def decrypt(x, param, cc, keys):
    assert len(x.cv) == 2
    ptx = cc.MakeCKKSPackedPlaintext([0.0])
    ctx = cc.Encrypt(keys.publicKey, ptx)
    for _ in range(param.GetMultiplicativeDepth() + 1 - x.cur_limbs):
        ctx = cc.EvalMult(ctx, ctx)
        ctx = cc.Rescale(ctx)
    data = [cv.tolist() for cv in x.cv]
    ctx.SetVectorOfData(data, x.cur_limbs)
    ptx = cc.Decrypt(ctx, keys.secretKey)
    return torch.tensor(
        ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64
    )
