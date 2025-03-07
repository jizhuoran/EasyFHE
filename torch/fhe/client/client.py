import sys

sys.path.append("/usr/local")
import openfhe as openfhe
import torch
import numpy as np
from .. import ciphertext as Cipher, Plaintext


class OpenFHEContext:
    def __init__(self, content_map):
        openfhe.ClearEvalMultKeys()
        openfhe.ReleaseAllContexts()

        self.cc = openfhe.DeserializeCryptoContextString(
            content_map["cc"], openfhe.BINARY
        )
        self.publicKey = openfhe.DeserializePublicKeyString(
            content_map["publicKey"], openfhe.BINARY
        )
        self.secretKey = openfhe.DeserializePrivateKeyString(
            content_map["secretKey"], openfhe.BINARY
        )
        openfhe.DeserializeEvalAutomorphismKeyString(
            content_map["app_rot_key"], openfhe.BINARY
        )
        self.depth = content_map["depth"]

    def encode(self, x, scale_deg=None, level=None, slots=None):
        if not (
            (scale_deg is None and level is None and slots is None)
            or (scale_deg is not None and level is not None and slots is not None)
        ):
            raise ValueError(
                "Error: check if scale_deg, level, and slots are set correctly."
            )

        if level is None and scale_deg is None and slots is None:
            ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
            ptx.Encode()
            data = ptx.GetVectorOfData()
            mv = [torch.tensor(data, device="cuda", dtype=torch.uint64)]
            return Plaintext(
                mv, mv[0].shape[0], ptx.GetNoiseScaleDeg(), ptx.GetSlots(), False
            )
        else:
            if isinstance(x, (np.ndarray, torch.Tensor)):
                ptx = self.cc.MakeCKKSPackedPlaintext(
                    x.tolist(), scale_deg, level, None, slots
                )
            else:
                ptx = self.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
            ptx.Encode()
            data = ptx.GetVectorOfData()
            mv = [torch.tensor(data, device="cuda", dtype=torch.uint64)]
            return Plaintext(
                mv, mv[0].shape[0], ptx.GetNoiseScaleDeg(), ptx.GetSlots(), False
            )

    def encrypt(self, x, scale_deg=1, level=0, slots=None):
        if slots is None:
            slots = len(x)
        if isinstance(x, (np.ndarray, torch.Tensor)):
            ptx = self.cc.MakeCKKSPackedPlaintext(
                x.tolist(), scale_deg, level, None, slots
            )
        else:
            ptx = self.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        data = cipher.GetVectorOfData()
        cv = [torch.tensor(elem, device="cuda", dtype=torch.uint64) for elem in data]
        return Cipher.Cipher(
            cv,
            cv[0].shape[0],
            cipher.GetNoiseScaleDeg(),
            cipher.GetSlots(),
            is_ext=False,
        )

    def decrypt(self, x):
        assert len(x.cv) == 2
        ptx = self.cc.MakeCKKSPackedPlaintext([0.0])
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        cipher.SetNoiseScaleDeg(x.noise_deg)
        cipher.SetLevel(self.depth + 1 - x.cur_limbs)
        cipher.SetSlots(x.slots)

        data = [cv.tolist() for cv in x.cv]
        cipher.SetVectorOfData(data, x.cur_limbs)
        ptx = self.cc.Decrypt(cipher, self.secretKey)

        return torch.tensor(
            ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64
        )
