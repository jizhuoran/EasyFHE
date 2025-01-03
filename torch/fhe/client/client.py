from . import openfhe as openfhe
import torch
from .. import ciphertext as Cipher
import numpy as np

class OpenFHEContext:
    def __init__(self, content_map):
        openfhe.ClearEvalMultKeys()
        openfhe.ReleaseAllContexts()

        self.cc = openfhe.DeserializeCryptoContextString(content_map["cc"], openfhe.BINARY)
        self.publicKey = openfhe.DeserializePublicKeyString(content_map["publicKey"], openfhe.BINARY)
        self.secretKey = openfhe.DeserializePrivateKeyString(content_map["secretKey"], openfhe.BINARY)
        self.depth = content_map["depth"]
        self.slots = content_map["slots"]
        openfhe.DeserializeEvalMultKeyString(content_map["mul_key"], openfhe.BINARY)

    def encode(self, x):
        ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
        ptx.Encode()
        return np.array(ptx.GetVectorOfData(), dtype=np.uint64)

    def encrypt(self, x):
        ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        data = cipher.GetVectorOfData()
        cv = [torch.tensor(elem, device=x.device, dtype=torch.uint64) for elem in data]
        return Cipher.Cipher(cv, cv[0].shape[0], cipher.GetScalingFactor(), cipher.GetNoiseScaleDeg(), cipher.GetSlots())
    
    def decrypt(self, x):
        assert len(x.cv) == 2
        ptx = self.cc.MakeCKKSPackedPlaintext([0.0])
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        cipher.SetNoiseScaleDeg(x.noise_deg)
        cipher.SetLevel(self.depth + 1 - x.cur_limbs)
        cipher.SetScalingFactor(x.scaling_factor)
        cipher.SetSlots(self.slots)

        data = [cv.tolist() for cv in x.cv]
        cipher.SetVectorOfData(data, x.cur_limbs)
        ptx = self.cc.Decrypt(cipher, self.secretKey)

        return torch.tensor(
            ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64
        )
