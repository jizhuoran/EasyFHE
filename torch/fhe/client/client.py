import time, atexit
from . import openfhe as openfhe
import torch
from .. import ciphertext as Cipher
import numpy as np

from ..ciphertext import Plaintext

execution_times = {}

def profile_python_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate the execution time for this call
        exec_time = end_time - start_time

        # Update the global dictionary with the accumulated time for this function
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = 0
        execution_times[func.__name__] += exec_time

        # print(f"Function {func.__name__} executed in {exec_time:.6f} seconds")
        return result

    return wrapper

@atexit.register
def print_execution_times():
    print("\nFunction Execution Times:")
    for func_name, exec_time in execution_times.items():
        print(f"Function '{func_name}' took {exec_time:.6f} seconds to finish.")

class OpenFHEContext:
    def __init__(self, content_map):
        openfhe.ClearEvalMultKeys()
        openfhe.ReleaseAllContexts()

        self.cc = openfhe.DeserializeCryptoContextString(content_map["cc"], openfhe.BINARY)
        self.publicKey = openfhe.DeserializePublicKeyString(content_map["publicKey"], openfhe.BINARY)
        self.secretKey = openfhe.DeserializePrivateKeyString(content_map["secretKey"], openfhe.BINARY)
        self.depth = content_map["depth"]
        # self.slots = content_map["slots"]
        # self.slots = slots
        # self.cc.EvalBootstrapSetup(level_budget, [0, 0], self.slots)
        # openfhe.DeserializeEvalKeyString(content_map["eval_key"], openfhe.BINARY)
        # openfhe.DeserializeEvalMultKeyString(content_map["mul_key"], openfhe.BINARY)
        # openfhe.DeserializeEvalAutomorphismKeyString(content_map["rot_key"], openfhe.BINARY)

    def setup_for_debug(self, debug_keys, slots, level_budget):
        self.cc.EvalBootstrapSetup(level_budget, [0, 0], slots)
        openfhe.DeserializeEvalKeyString(debug_keys["eval_key"], openfhe.BINARY)
        openfhe.DeserializeEvalMultKeyString(debug_keys["mul_key"], openfhe.BINARY)
        openfhe.DeserializeEvalAutomorphismKeyString(debug_keys["rot_key"], openfhe.BINARY)

    def encode(self, x, level=None, scale_deg=None, slots=None): # todo: align the input order wtih the encrypt function
        # print("encode", "level", level, "scale_deg", scale_deg, "slots", slots)
        if not ((scale_deg is None and level is None and slots is None) or
                (scale_deg is not None and level is not None and slots is not None)):
            # 输出警告
            print("Warning: scale_deg, level, and slots must either all be None or all not None.")

        if level is None and scale_deg is None and slots is None:
            ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist())
            ptx.Encode()
            return np.array(ptx.GetVectorOfData(), dtype=np.uint64)
        else:
            cur_limbs = self.depth
            if slots is None:
                slots = len(x)
            if isinstance(x, (np.ndarray, torch.Tensor)):
                ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist(), scale_deg, level, None, slots)
            else:
                ptx = self.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
            ptx.Encode()
            data = ptx.GetVectorOfData()
            cv = [torch.tensor(data, device="cuda", dtype=torch.uint64)] #fixme: shall we set device = "cuda" directly?
            return Plaintext(cv, cv[0].shape[0], ptx.GetScalingFactor(), ptx.GetNoiseScaleDeg(), ptx.GetSlots(),False)

    def encrypt(self, x, scale_deg = 1, level = 0, slots= None):
        if slots is None:
            slots = len(x)
        if isinstance(x, (np.ndarray, torch.Tensor)):
            ptx = self.cc.MakeCKKSPackedPlaintext(x.tolist(), scale_deg, level, None, slots)
        else:
            ptx = self.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        data = cipher.GetVectorOfData()
        cv = [torch.tensor(elem, device="cuda", dtype=torch.uint64) for elem in data] #fixme: shall we set device = "cuda" directly?
        return Cipher.Cipher(cv, cv[0].shape[0], cipher.GetScalingFactor(), cipher.GetNoiseScaleDeg(), cipher.GetSlots(), is_ext=False), cipher
    
    def decrypt(self, x):
        assert len(x.cv) == 2
        ptx = self.cc.MakeCKKSPackedPlaintext([0.0])
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        cipher.SetNoiseScaleDeg(x.noise_deg)
        cipher.SetLevel(self.depth + 1 - x.cur_limbs)
        cipher.SetScalingFactor(x.scaling_factor)
        cipher.SetSlots(x.slots)

        data = [cv.tolist() for cv in x.cv]
        cipher.SetVectorOfData(data, x.cur_limbs)
        ptx = self.cc.Decrypt(cipher, self.secretKey)

        return torch.tensor(
            ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64
        )
