from . import openfhe as openfhe
import torch
from .. import Ciphertext as Cipher

def gen_crypto_context(mult_depth, scale_mod_size, batch_size):
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(scale_mod_size)
    parameters.SetBatchSize(batch_size)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()

    return parameters, cc, keys

def gen_mult_keys(cc, keys):
    return cc.EvalMultKeyGen(keys.secretKey)

def gen_rotate_keys(cc, keys, rot):
    return cc.EvalRotateKeyGen(keys.secretKey, rot)

def encrypt(x, cc, keys):
    ptx = cc.MakeCKKSPackedPlaintext(x.tolist())
    cipher = cc.Encrypt(keys.publicKey, ptx)
    data = cipher.GetVectorOfData()
    cv0 = torch.tensor(data[:len(data)//2], device=x.device, dtype=torch.uint64)
    cv1 = torch.tensor(data[len(data)//2:], device=x.device, dtype=torch.uint64)
    return Cipher.Cipher([cv0, cv1], len(data) // 2 // cc.GetRingDimension())

def decrypt(x, param, cc, keys):
    assert len(x.cv) == 2
    ptx = cc.MakeCKKSPackedPlaintext([.0] * param.GetBatchSize())
    ctx = cc.Encrypt(keys.publicKey, ptx)
    data = []
    for cv in x.cv:
        data.extend(cv.tolist())
    ctx.SetVectorOfData(data, len(x.cv), x.cur_limbs, cc.GetRingDimension())
    ptx = cc.Decrypt(ctx, keys.secretKey)
    return torch.tensor(ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64)
