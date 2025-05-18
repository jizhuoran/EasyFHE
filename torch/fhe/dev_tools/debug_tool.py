import functools
import torch
import numpy as np
from termcolor import colored

def print_failed(message):
    print(colored(message, "red"))


def pass_checker(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # print("pass_checker", func.__name__)
        result = func(*args, **kwargs)
        return result

    return wrapper


def auto_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cpu.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cpu.synchronize()
        return result

    return wrapper


def check_meta_equal(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        if func.__name__ in ["homo_add", "homo_sub", "homo_mul", "homo_add_pt", "homo_mul_pt"]:
            in0, in1 = args[0], args[1]
            assert in0.is_ext == in1.is_ext, \
                f"Assertion failed: in0.is_ext = {in0.is_ext}, in1.is_ext = {in1.is_ext}. " \
                "in0.is_ext should be equal to in1.is_ext."

            assert in0.slots == in1.slots, \
                f"Assertion failed: in0.slots = {in0.slots}, in1.slots = {in1.slots}. " \
                "in0.slots should be equal to in1.slots."

            if cryptoContext.rescaleTech == "FIXEDMANUAL":
                if not in0.cur_limbs == in1.cur_limbs:
                    print("cur_limbs not equal! ", in0.cur_limbs, in1.cur_limbs)

                if not in0.noise_deg == in1.noise_deg:
                    print("noise_deg not equal! ", in0.noise_deg, in1.noise_deg)
                if in0.noise_deg > 2:
                    print("noise_deg should not > 2", in0.noise_deg)

                assert in0.noise_deg <= 2, \
                    f"Assertion failed: in0.noise_deg = {in0.noise_deg}. in0.noise_deg should be less than or equal to 2."

                assert in0.noise_deg == in1.noise_deg, \
                    f"Assertion failed: in0.noise_deg = {in0.noise_deg}, in1.noise_deg = {in1.noise_deg}. " \
                    "in0.noise_deg should be equal to in1.noise_deg."

                assert in0.cur_limbs == in1.cur_limbs, \
                    f"Assertion failed: in0.cur_limbs = {in0.cur_limbs}, in1.cur_limbs = {in1.cur_limbs}. " \
                    "in0.cur_limbs should be equal to in1.cur_limbs."

                # assert in0.cur_limbs == in1.cur_limbs
                # assert in0.scaling_factor == in1.scaling_factor
                # assert in0.noise_deg == in1.noise_deg

        return func(*args, **kwargs)

    return wrapper


def plaintext_twin(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if func.__name__ in [
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
            "eval_fast_rotate",
            "cipher_automorphism",
            "mult_rot_key_and_sum_ext",
        ]:
            if func.__name__ == "homo_add" or func.__name__ == "homo_add_pt":
                res.ptx_twin = args[0].ptx_twin + args[1].ptx_twin
            elif func.__name__ == "homo_sub":
                res.ptx_twin = args[0].ptx_twin - args[1].ptx_twin
            elif func.__name__ == "homo_square":
                res.ptx_twin = args[0].ptx_twin ** 2
            elif func.__name__ == "homo_mul" or func.__name__ == "homo_mul_pt":
                res.ptx_twin = args[0].ptx_twin * args[1].ptx_twin
            elif func.__name__ == "homo_mul_scalar_int" or func.__name__ == "homo_mul_scalar_double":
                res.ptx_twin = args[0].ptx_twin * args[1]
            elif func.__name__ == "homo_add_scalar_double":
                res.ptx_twin = args[0].ptx_twin + args[1]
            elif func.__name__ in ["homo_rotate", "cipher_automorphism", "mult_rot_key_and_sum_ext"]:
                res.ptx_twin = np.array(args[0].ptx_twin[args[1] :].tolist() + args[0].ptx_twin[: args[1]].tolist())
            elif func.__name__ == "eval_fast_rotate":
                res.ptx_twin = np.array(args[0].ptx_twin[args[2] :].tolist() + args[0].ptx_twin[: args[2]].tolist())
        
        #check
        cryptoContext = args[-1]
        if cryptoContext.in_check_period == True:
            if res.is_ext == False and len(res.cv) == 2 and func.__name__ not in ["moddown_from_ext"]:
                decrypted_result = cryptoContext.openfhe_context.decrypt(res)
                decrypted_result = decrypted_result.cpu().numpy().reshape(-1)[:len(res.ptx_twin)]
                if np.allclose(decrypted_result, res.ptx_twin, rtol=0.1, atol=0.1):
                    print("{} passed!".format(func.__name__))
                else:
                    mask = np.isclose(decrypted_result, res.ptx_twin, rtol=0.1, atol=0.1)
                    # Invert the mask to get indices where the arrays differ.
                    diff_indices = np.where(~mask)
                    print_failed("{} failed!".format(func.__name__))
                    print("diff indices", diff_indices)
                    print("diff values at ciphertext", decrypted_result[diff_indices])
                    print("diff values at plaintext twin", res.ptx_twin[diff_indices])
        
        return res

    return wrapper
