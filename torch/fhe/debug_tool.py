import functools
import torch
import numpy as np


def auto_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        if func.__name__ == "eval_bootstrap":
            cryptoContext.inBS = True
        if cryptoContext.inBS:
            result = func(*args, **kwargs)
        else:
            torch.cpu.synchronize()
            torch.cuda.synchronize()
            result = func(*args, **kwargs)
            torch.cpu.synchronize()
            torch.cuda.synchronize()
        if func.__name__ == "eval_bootstrap":
            cryptoContext.inBS = False
        return result

    return wrapper


def check_meta_equal(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        in0, in1 = args[0], args[1]
        assert len(in0.cv) == len(in1.cv)
        assert in0.is_ext == in1.is_ext
        assert in0.slots == in1.slots

        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            assert in0.cur_limbs == in1.cur_limbs
            assert in0.scaling_factor == in1.scaling_factor
            assert in0.noise_deg == in1.noise_deg

        return func(*args, **kwargs)

    return wrapper


def plaintext_twin(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        if cryptoContext.config.PTX_TWIN:
            if func.__name__ == "eval_bootstrap":
                cryptoContext.inBS = True
                res = func(*args, **kwargs)
                res.ptx_twin = args[0].ptx_twin
                cryptoContext.inBS = False
                return res
            elif not cryptoContext.inBS and func.__name__ in [
                "homo_add",
                "homo_sub",
                "homo_square",
                "homo_mul",
                "homo_mul_pt",
                "homo_add_pt",
                "homo_mul_scalar_int",
                "homo_mul_scalar_double",
                "homo_add_scalar_double",
                "homo_rotate",
                "cipher_automorphism",
                "mult_rot_key_and_sum_ext"
            ]:
                res = func(*args, **kwargs)
                if func.__name__ == "homo_add" or func.__name__ == "homo_add_pt":
                    res.ptx_twin = args[0].ptx_twin + args[1].ptx_twin
                elif func.__name__ == "homo_sub":
                    res.ptx_twin = args[0].ptx_twin - args[1].ptx_twin
                elif func.__name__ == "homo_sub":
                    res.ptx_twin = args[0].ptx_twin ** 2
                elif func.__name__ == "homo_mul" or func.__name__ == "homo_mul_pt":
                    res.ptx_twin = args[0].ptx_twin * args[1].ptx_twin
                elif func.__name__ == "homo_mul_scalar_int" or func.__name__ == "homo_mul_scalar_double":
                    res.ptx_twin = args[0].ptx_twin * args[1]
                elif func.__name__ == "homo_add_scalar_double":
                    res.ptx_twin = args[0].ptx_twin + args[1]
                elif func.__name__ == "homo_rotate" or func.__name__ == "cipher_automorphism" or func.__name__ == "mult_rot_key_and_sum_ext":
                    res.ptx_twin = np.array(args[0].ptx_twin[args[1]:].tolist() + args[0].ptx_twin[:args[1]].tolist())
                return res
            else:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper
