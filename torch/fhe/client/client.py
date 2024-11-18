from . import openfhe as openfhe
import torch
from .. import Ciphertext as Cipher
from .. import context as Context

import numpy as np

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

    moduliQ, rootsQ, moduliP, rootsP = cc.GetPQ()

    cc.EvalMultKeyGen(keys.secretKey)
    MULT_SWK = np.array(cc.GetEvalMultKey(), dtype=np.uint64)

    cc.EvalRotateKeyGen(keys.secretKey, rotate_index)
    ROT_SWK = cc.GetEvalRotateKey()

    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    BOOT_KEY = cc.GetEvalBootstrapKey()

    GPUFHE_Context = Context.Context(
        logN,
        firstMod,
        dcrtBits,
        AuxModSize,
        L,
        K,
        moduliQ,
        moduliP,
        rootsQ,
        rootsP,
        MULT_SWK,
        ROT_SWK,
        BOOT_KEY
    )

    return parameters, cc, keys, GPUFHE_Context


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
