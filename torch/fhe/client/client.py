from . import openfhe as openfhe
import torch
from .. import ciphertext as Cipher
from .. import context as Context
import pickle
import numpy as np

class OpenFHEContext:
    def __init__(self, cc, pk, sk, depth):
        self.cc = cc
        self.publicKey = pk
        self.secretKey = sk
        self.depth = depth

    def Serialize(self):
        cc_bytes = openfhe.Serialize(self.cc, openfhe.BINARY)
        mul_key_bytes = openfhe.SerializeEvalMultKeyString(openfhe.BINARY)
        public_key_bytes = openfhe.Serialize(self.publicKey, openfhe.BINARY)
        secret_key_bytes = openfhe.Serialize(self.secretKey, openfhe.BINARY)
        depth = self.depth
        return pickle.dumps((cc_bytes, mul_key_bytes, public_key_bytes, secret_key_bytes, depth))

    def Deserialize(ser):
        openfhe.ClearEvalMultKeys()
        openfhe.ReleaseAllContexts()

        cc_bytes, mul_key_bytes, public_key_bytes, secret_key_bytes, depth = pickle.loads(ser)
        cc = openfhe.DeserializeCryptoContextString(cc_bytes, openfhe.BINARY)
        pk = openfhe.DeserializePublicKeyString(public_key_bytes, openfhe.BINARY)
        sk = openfhe.DeserializePrivateKeyString(secret_key_bytes, openfhe.BINARY)
        openfhe.DeserializeEvalMultKeyString(mul_key_bytes, openfhe.BINARY)
        return OpenFHEContext(cc, pk, sk, depth)

    def encode(self, x):
        ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
        ptx.Encode()
        return np.array(ptx.GetVectorOfData(), dtype=np.uint64)

    def encrypt(self, x):
        ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        # sc_Factor = cipher.GetScalingFactor()
        # noise_deg = cipher.GetNoiseDeg()
        data = cipher.GetVectorOfData()
        cv = [torch.tensor(elem, device=x.device, dtype=torch.uint64) for elem in data]
        # return Cipher.Cipher(cv, cv[0].shape[0], sc_Factor, noise_deg) # todo:set scaling factor and noise deg here?
        return Cipher.Cipher(cv, cv[0].shape[0], 0.0, 1)

    def decrypt(self, x):
        assert len(x.cv) == 2
        ptx = self.cc.MakeCKKSPackedPlaintext([0.0])
        ctx = self.cc.Encrypt(self.publicKey, ptx)
        for _ in range(self.depth + 1 - x.cur_limbs):
            ctx = self.cc.EvalMult(ctx, ctx)
            ctx = self.cc.Rescale(ctx)
        data = [cv.tolist() for cv in x.cv]
        ctx.SetVectorOfData(data, x.cur_limbs)
        ptx = self.cc.Decrypt(ctx, self.secretKey)
        return torch.tensor(
            ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64
        )


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
    rescaleTech
):

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

    cc.EvalBootstrapSetup(levelBudget, [0, 0], slots)
    keys = cc.KeyGen()

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

    openfhe_context = OpenFHEContext(cc, keys.publicKey, keys.secretKey, depth)
    gpufhe_context = Context.Context(
        logN,
        logSlots,
        firstMod,
        dcrtBits,
        60,  # openfhe is 60 bits
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
        rescaleTech
    )

    return openfhe_context, gpufhe_context
