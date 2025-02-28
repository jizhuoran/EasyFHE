from datetime import datetime
import time, os, pickle
import numpy as np
import functools
import atexit
from .client import client as client
from .client.gen_context import gen_contexts
from .context import *
from .ciphertext import Cipher
import torch

unary_op = {
    "homo_square": "homo_ops.",
    "drop_last_elements_": "homo_ops.",
    "key_switch_P_ext": "hybrid_keyswitch.",
    "modup_to_ext": "hybrid_keyswitch.",
    "moddown_from_ext": "hybrid_keyswitch.",
}

unary_cnst_op = {
    "homo_rescale": "homo_ops.",
    "homo_mul_scalar_double": "homo_ops.",
    "mod_raise": "",
    "mult_by_monomial_inplace": "",
    "homo_rotate": "homo_ops.",
    "homo_mul_scalar_int": "homo_ops.",
    "homo_add_scalar_double": "homo_ops.",
    "_cipher_automorphism": "homo_ops.",
    "mult_rot_key_and_sum_ext": "hybrid_keyswitch.",
}

binary_op = {
    "homo_add": "homo_ops.",
    "homo_sub": "homo_ops.",
    "homo_mul": "homo_ops.",
    "homo_mul_pt": "homo_ops.",
}


def printFrontend(func):

    if func.__name__ in unary_cnst_op:

        def wrapper(*args, **kwargs):
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                return func(*args)
            ct, val, _ = args
            in_node_id = ct.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args)
            res.cipher_id = out_node_id
            print(
                "NODE{} = {}{}(NODE{}, {}, cryptoContext) #out: limb={}, noise={}, in0: limb={}, noise={}".format(
                    out_node_id,
                    unary_cnst_op[func.__name__],
                    func.__name__,
                    in_node_id,
                    repr(val),
                    res.cur_limbs,
                    res.noise_deg,
                    ct.cur_limbs,
                    ct.noise_deg,
                )
            )

            return res

        return wrapper

    if func.__name__ in binary_op:

        def wrapper(*args, **kwargs):
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                return func(*args)
            in0, in1, _ = args
            in0_node_id = in0.cipher_id
            in1_node_id = in1.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
                "NODE{} = {}{}(NODE{}, NODE{}, cryptoContext) #out: limb={}, noise={}, in0: limb={}, noise={}, in1: limb={}, noise={}".format(
                    out_node_id,
                    binary_op[func.__name__],
                    func.__name__,
                    in0_node_id,
                    in1_node_id,
                    res.cur_limbs,
                    res.noise_deg,
                    in0.cur_limbs,
                    in0.noise_deg,
                    in1.cur_limbs,
                    in1.noise_deg,
                )
            )

            return res

        return wrapper

    if func.__name__ in unary_op:

        def wrapper(*args, **kwargs):
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                return func(*args)
            in0, _ = args
            in0_node_id = in0.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
                "NODE{} = {}{}(NODE{}, cryptoContext) #out: limb={}, noise={}, in0: limb={}, noise={}".format(
                    out_node_id,
                    unary_op[func.__name__],
                    func.__name__,
                    in0_node_id,
                    res.cur_limbs,
                    res.noise_deg,
                    in0.cur_limbs,
                    in0.noise_deg,
                )
            )
            return res

        return wrapper

    if func.__name__ == "eval_fast_rotate":

        def wrapper(*args, **kwargs):
            digits, cipher, index, need_KS_add, need_moddown, cryptoContext = args
            digits_node_id = digits.cipher_id
            cipher_node_name = (
                "NODE{}".format(cipher.cipher_id) if cipher is not None else "None"
            )
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
                "NODE{} = homo_ops.eval_fast_rotate(NODE{}, {}, {}, {}, {}, cryptoContext) #out: limb={}, noise={}, in0: limb={}, noise={}".format(
                    out_node_id,
                    digits_node_id,
                    cipher_node_name,
                    index,
                    need_KS_add,
                    need_moddown,
                    res.cur_limbs,
                    res.noise_deg,
                    digits.cur_limbs,
                    digits.noise_deg,
                )
                + (""
                if cipher is None
                else "in1: limb={}, noise={}".format(cipher.cur_limbs, cipher.noise_deg))
            )
            return res

        return wrapper

    if func.__name__ == "extract_cv":

        def wrapper(*args, **kwargs):
            in0, index = args
            in0_node_id = in0.cipher_id
            out_node_id = Cipher.get_next_id()
            if "append_zeros" in kwargs:
                append_zeros = ", append_zeros = " + str(kwargs["append_zeros"])
            else:
                append_zeros = ""
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
                "NODE{} = homo_ops.extract_cv(NODE{}, {}{}) #out: limb={}, noise={}, in0: limb={}, noise={}".format(
                    out_node_id,
                    in0_node_id,
                    index,
                    append_zeros,
                    res.cur_limbs,
                    res.noise_deg,
                    in0.cur_limbs,
                    in0.noise_deg,
                )
            )

            return res

        return wrapper

    if func.__name__ == "adjust_levels_and_depth":

        def wrapper(*args, **kwargs):
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                return func(*args)
            ct1, ct2, cryptoContext = args
            ct1_node_id = ct1.cipher_id
            ct2_node_id = ct2.cipher_id
            out1_node_id = Cipher.get_next_id()
            out2_node_id = Cipher.get_next_id()
            out1, out2 = func(*args)
            out1.cipher_id = out1_node_id
            out2.cipher_id = out2_node_id
            print(
                "NODE{}, NODE{} = homo_ops.adjust_levels_and_depth(NODE{}, NODE{}, cryptoContext) #out0: limb={}, noise={}, #out1: limb={}, noise={}, in0: limb={}, noise={}, in1: limb={}, noise={}".format(
                    out1_node_id,
                    out2_node_id,
                    ct1_node_id,
                    ct2_node_id,
                    out1.cur_limbs,
                    out1.noise_deg,
                    out2.cur_limbs,
                    out2.noise_deg,
                    ct1.cur_limbs,
                    ct1.noise_deg,
                    ct2.cur_limbs,
                    ct2.noise_deg,
                )
            )


            return out1, out2

        return wrapper


# Global dictionary to accumulate execution time for each function
execution_times = {}

# Registry to keep track of function call counts
call_registry = {}


def call_counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1  # Increment the call count
        return func(*args, **kwargs)

    wrapper.count = 0  # Initialize the call count
    call_registry[func.__name__] = wrapper  # Register the function
    return wrapper


# @atexit.register
def print_call_counts():
    print("\nFunction Call Counts:")
    for func_name, wrapper in call_registry.items():
        print(f"Function '{func_name}' was called {wrapper.count} times.")


# @atexit.register
def print_execution_times():
    print("\nExecution Times:")
    for func_name, exec_time in execution_times.items():
        print(f"Function '{func_name}' executed in {exec_time:.6f} seconds.")


def check_meta_equal(func):
    def wrapper(*args, **kwargs):
        in0, in1 = args[0], args[1]
        # assert len(in0.cv) == len(in1.cv)
        # assert in0.cur_limbs == in1.cur_limbs
        # assert in0.scaling_factor == in1.scaling_factor
        # assert in0.noise_deg == in1.noise_deg
        # assert in0.is_ext == in1.is_ext
        # assert in0.slots == in1.slots
        return func(*args, **kwargs)

    return wrapper


def check_cipher_len(func):
    def wrapper(*args, **kwargs):
        assert len(args[0].cv) == 2
        return func(*args, **kwargs)

    return wrapper


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


def profile_pytorch_function(func):
    def wrapper(*args, **kwargs):
        # Set up the profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/zrji/log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            result = func(*args, **kwargs)
            profiler.step()

        profiler_results = profiler.key_averages()
        print(profiler_results.table(sort_by="self_cuda_time_total"))
        print(profiler_results.table(sort_by="self_cpu_time_total"))

        return result

    return wrapper


def round_half_away_from_zero(number, ndigits=0):
    multiplier = 10**ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    elif number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    else:
        return 0.0


def try_load_context(
    maxLevelsRemaining,
    rotIndex_list,
    logBsSlots_list,
    logN,
    dnum,
    dcrtBits,
    firstMod,
    levelBudget_list,
    secretKeyDist,
    rescaleTech,
    save_dir,
    autoLoadAndSetConfig,
    mode,
):

    NO_BS = False
    if logBsSlots_list is None:
        assert (logBsSlots_list is None) == (
            levelBudget_list is None
        ), "ERROR: logBsSlots_list and levelBudget_list must be both None or both not None!"
        logBsSlots_list = [0]
        levelBudget_list = [[0, 0]]
        NO_BS = True
    else:
        sorted_pairs = sorted(
            zip(logBsSlots_list, levelBudget_list), key=lambda x: x[0]
        )
        logBsSlots_list, levelBudget_list = zip(*sorted_pairs)
        logBsSlots_list = list(logBsSlots_list)
        levelBudget_list = list(levelBudget_list)

    load_path = save_dir + "/GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
        maxLevelsRemaining,
        "-".join(map(str, logBsSlots_list)),
        "-".join("-".join(map(str, levelBudget)) for levelBudget in levelBudget_list),
        logN,
        dnum,
        dcrtBits,
        firstMod,
        secretKeyDist,
        rescaleTech,
    )

    debug_load_path = (
        save_dir
        + "/DEBUG-GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            maxLevelsRemaining,
            "-".join(map(str, logBsSlots_list)),
            "-".join(
                "-".join(map(str, levelBudget)) for levelBudget in levelBudget_list
            ),
            logN,
            dnum,
            dcrtBits,
            firstMod,
            secretKeyDist,
            rescaleTech,
        )
    )

    if (not os.path.exists(load_path)) or (
        not os.path.exists(debug_load_path) and mode == "debug"
    ):
        gen_contexts(
            maxLevelsRemaining=maxLevelsRemaining,
            rotIndex_list=rotIndex_list,
            logBsSlots_list=logBsSlots_list,
            logN=logN,
            dnum=dnum,
            dcrtBits=dcrtBits,
            firstMod=firstMod,
            levelBudget_list=levelBudget_list,
            secretKeyDist=secretKeyDist,
            rescaleTech=rescaleTech,
            save_dir=save_dir,
            mode=mode,
        )

    with open(load_path, "rb") as file:
        gpufheMembers, openfheMembers, BsContextMembers = pickle.load(file)

    if mode == "debug":
        if not os.path.exists(debug_load_path):
            print("ERROR: There is no debug context file! Please regenerate context!")
        with open(debug_load_path, "rb") as file:
            debug_keys = pickle.load(file)

    cryptoContext = Context(BsContextMembers, gpufheMembers, autoLoadAndSetConfig)
    openfhe_context = client.OpenFHEContext(openfheMembers)
    if cryptoContext.autoLoadAndSetConfig:
        if rotIndex_list is not None:
            load_rotation_keys("app", cryptoContext)
        if NO_BS == False:
            for logBsSlots in logBsSlots_list:
                cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
                cryptoContext.BsContext.to_cuda()
                load_rotation_keys(logBsSlots, cryptoContext)

    if mode == "debug":
        openfhe_boot_contexts = {}
        if NO_BS == False:
            for logBsSlots, level_budget in zip(logBsSlots_list, levelBudget_list):
                openfhe_boot_contexts[str(logBsSlots)] = client.OpenFHEContext(
                    openfheMembers
                )
                openfhe_boot_contexts[str(logBsSlots)].setup_for_debug(
                    debug_keys, 1 << logBsSlots, level_budget
                )
        return cryptoContext, openfhe_context, openfhe_boot_contexts
    else:
        return cryptoContext, openfhe_context


def compare_bs_ct_with_openfhe(bs_cipher, openfhe_cipher):
    gpu_bootstrapping_res = np.array(
        [bs_cipher.cv[0].cpu().numpy(), bs_cipher.cv[1].cpu().numpy()]
    ).reshape(-1)
    openfhe_bootstrapping_res = np.array(openfhe_cipher.GetVectorOfData()).reshape(-1)
    return np.array_equal(gpu_bootstrapping_res, openfhe_bootstrapping_res)


def load_rotation_keys(key_name, cryptoContext):
    if (str(key_name) not in cryptoContext.slots_left_rot_key_map) or (
        not cryptoContext.slots_left_rot_key_map[str(key_name)]
    ):
        print("Warning: slots_left_rot_key_map[", key_name, "] is None")
        return
    for key, value in cryptoContext.slots_left_rot_key_map[str(key_name)].items():
        cryptoContext.left_rot_key_map[key] = [
            torch.tensor(v, dtype=torch.uint64, device="cuda") for v in value
        ]
    for key, value in cryptoContext.slots_precompute_auto_map[str(key_name)].items():
        cryptoContext.precompute_auto_map[key] = torch.tensor(
            value, dtype=torch.int32, device="cuda"
        )


def load_bootstrapping_context(logBsSlots, cryptoContext):
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
    cryptoContext.BsContext.to_cuda()
    load_rotation_keys(logBsSlots, cryptoContext)
