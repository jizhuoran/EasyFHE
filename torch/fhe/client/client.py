from . import openfhe as openfhe
import torch
from .. import ciphertext as Cipher
import numpy as np

from ..ciphertext import Plaintext


class OpenFHEContext:
    def __init__(self, content_map):
        openfhe.ClearEvalMultKeys()
        openfhe.ReleaseAllContexts()

        self.cc = openfhe.DeserializeCryptoContextString(content_map["cc"], openfhe.BINARY)
        self.publicKey = openfhe.DeserializePublicKeyString(content_map["publicKey"], openfhe.BINARY)
        self.secretKey = openfhe.DeserializePrivateKeyString(content_map["secretKey"], openfhe.BINARY)
        self.depth = content_map["depth"]
        self.slots = content_map["slots"]
        self.cc.EvalBootstrapSetup(content_map["level_budget"], [0, 0], self.slots)
        openfhe.DeserializeEvalKeyString(content_map["eval_key"], openfhe.BINARY)
        openfhe.DeserializeEvalMultKeyString(content_map["mul_key"], openfhe.BINARY)
        openfhe.DeserializeEvalAutomorphismKeyString(content_map["rot_key"], openfhe.BINARY)

    def encode(self, x, level=None, scale_deg=None, slots=None):
        if not ((scale_deg is None and level is None and slots is None) or
                (scale_deg is not None and level is not None and slots is not None)):
            # 输出警告
            print("Warning: scale_deg, level, and slots must either all be None or all not None.")

        if level is None and scale_deg is None and slots is None:
            ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
            ptx.Encode()
            return np.array(ptx.GetVectorOfData(), dtype=np.uint64)
        else:
            if slots is None:
                slots = len(x)
            ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist(), scale_deg, level, None, slots)
            data = ptx.GetVectorOfData()
            cv = [torch.tensor(elem, device=torch.device('cuda'), dtype=torch.uint64) for elem in data] #todo: do we need to use "device=x.device" instead
            # return Plaintext(cv, cv[0].shape[0], ptx.GetScalingFactor(), ptx.GetNoiseScaleDeg, ptx.GetSlots()) #todo: can be used after refactor Plaintext in ciphertext.py
            return Plaintext(cv, cv[0].shape[1], ptx.GetSlots(), cv[0].shape[0], ptx.GetScalingFactor(), ptx.GetNoiseScaleDeg)


    def encrypt(self, x, scale_deg = 1, level = 0, slots= None):
        if slots is None:
            slots = len(x)
        ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist(), scale_deg, level, None, slots)
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        data = cipher.GetVectorOfData()
        cv = [torch.tensor(elem, device=x.device, dtype=torch.uint64) for elem in data]
        return Cipher.Cipher(cv, cv[0].shape[0], cipher.GetScalingFactor(), cipher.GetNoiseScaleDeg(), cipher.GetSlots()), cipher
    
    def decrypt(self, x):
        assert len(x.cv) == 2
        ptx = self.cc.MakeCKKSPackedPlaintext([0.0])
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        # for _ in range(self.depth + 1 - x.cur_limbs):
        #     cipher = self.cc.EvalMult(cipher, cipher)
        #     cipher = self.cc.Rescale(cipher)
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
