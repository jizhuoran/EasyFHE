import functools
import easyfhe as torch
import numpy as np
from termcolor import colored

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import traceback, sys

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
            "fast_rotate",
            "mult_rot_key_and_sum_ext",
            "slot_resize",
            "fused_pairwise_mac",
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
            elif func.__name__ in ["homo_rotate", "mult_rot_key_and_sum_ext"]:
                res.ptx_twin = np.array(args[0].ptx_twin[args[1] :].tolist() + args[0].ptx_twin[: args[1]].tolist())
            elif func.__name__ == "fast_rotate":
                for item, offset in zip(res, args[1]):
                    item.ptx_twin = np.array(args[0].ptx_twin[offset :].tolist() + args[0].ptx_twin[: offset].tolist())
            elif func.__name__ == "slot_resize":
                if args[0].slots >= args[1]:
                    res.ptx_twin = args[0].ptx_twin[:args[0].slots]
                else:
                    assert args[1] // args[0].slots == (args[1] + args[0].slots - 1) // args[0].slots # only support slots be power of 2
                    repeat_times = args[1] // args[0].slots
                    res.ptx_twin = np.tile(args[0].ptx_twin[:args[0].slots], repeat_times)
            elif func.__name__ == "fused_pairwise_mac":
                cryptoContext = args[-1]
                decrypted_result = cryptoContext.openfhe_context.decrypt(res)
                decrypted_result = decrypted_result.cpu().numpy().reshape(-1)[:len(res.ptx_twin)]
                res.ptx_twin = decrypted_result

        #check
        cryptoContext = args[-1]
        if cryptoContext.in_check_period == True:
            results = res if isinstance(res, list) else [res]
            for item in results:
                if item.is_ext == False and len(item.cv) == 2 and func.__name__ not in ["moddown_from_ext"]:
                    decrypted_result = cryptoContext.openfhe_context.decrypt(item)
                    decrypted_result = decrypted_result.cpu().numpy().reshape(-1)[:len(item.ptx_twin)]
                    if np.allclose(decrypted_result, item.ptx_twin, rtol=0.1, atol=0.1):
                        continue
                    mask = np.isclose(decrypted_result, item.ptx_twin, rtol=0.1, atol=0.1)
                    # Invert the mask to get indices where the arrays differ.
                    diff_indices = np.where(~mask)
                    print_failed("{} failed!".format(func.__name__))
                    print("diff indices", diff_indices)
                    print("diff values at ciphertext", decrypted_result[diff_indices])
                    print("diff values at plaintext twin", item.ptx_twin[diff_indices])

                    # print call stack and end the program
                    traceback.print_stack()
                    sys.exit(1)
            if len(results) > 0:
                print("{} passed!".format(func.__name__))
        
        return res

    return wrapper
