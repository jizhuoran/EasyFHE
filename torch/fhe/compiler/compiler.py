import functools, os
from ..ciphertext import Cipher
import atexit

unary_op = {
    "homo_square": "homo_ops.",
    "key_switch_P_ext": "hybrid_keyswitch.",
    "modup_to_ext": "hybrid_keyswitch.",
    "moddown_from_ext": "hybrid_keyswitch.",
}

unary_cnst_op = {
    "drop_last_elements": "homo_ops.",
    "homo_rescale": "homo_ops.",
    "force_rescale": "homo_ops.",
    "homo_mul_scalar_double": "homo_ops.",
    "mod_raise": "",
    "assign_scaling_factor": "",
    "mult_by_monomial_inplace": "",
    "homo_rotate": "homo_ops.",
    "homo_mul_scalar_int": "homo_ops.",
    "homo_add_scalar_double": "homo_ops.",
    "cipher_automorphism": "homo_ops.",
    "mult_rot_key_and_sum_ext": "hybrid_keyswitch.",
}

binary_op = {
    "homo_add": "homo_ops.",
    "homo_sub": "homo_ops.",
    "homo_mul": "homo_ops.",
    "homo_mul_pt": "homo_ops.",
}

compiled_code = []

@atexit.register
def print_call_counts():
    if len(compiled_code) > 0:
        DATA_DIR = os.environ["DATA_DIR"]
        with open(DATA_DIR + "/compiled_code.txt", 'w') as f:
            for line in compiled_code:
                print(line, file=f)

def frontend(func):

    if func.__name__ in unary_cnst_op:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ct, val, cryptoContext = args
            in_node_id = ct.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            compiled_code.append(
                "NODE{} = {}{}(NODE{}, {}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}".format(
                    out_node_id,
                    unary_cnst_op[func.__name__],
                    func.__name__,
                    in_node_id,
                    repr(val),
                    res.cur_limbs,
                    res.noise_deg,
                    res.scaling_factor,
                    ct.cur_limbs,
                    ct.noise_deg,
                    ct.scaling_factor,
                )
            )

            return res

        return wrapper

    if func.__name__ in binary_op:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            in0, in1, cryptoContext = args
            in0_node_id = in0.cipher_id
            in1_node_id = in1.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            compiled_code.append(
                "NODE{} = {}{}(NODE{}, NODE{}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}, in1: limb={}, noise={}, sf={}".format(
                    out_node_id,
                    binary_op[func.__name__],
                    func.__name__,
                    in0_node_id,
                    in1_node_id,
                    res.cur_limbs,
                    res.noise_deg,
                    res.scaling_factor,
                    in0.cur_limbs,
                    in0.noise_deg,
                    in0.scaling_factor,
                    in1.cur_limbs,
                    in1.noise_deg,
                    in1.scaling_factor,
                )
            )

            return res

        return wrapper

    if func.__name__ in unary_op:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            in0, cryptoContext = args
            in0_node_id = in0.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            compiled_code.append(
                "NODE{} = {}{}(NODE{}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}".format(
                    out_node_id,
                    unary_op[func.__name__],
                    func.__name__,
                    in0_node_id,
                    res.cur_limbs,
                    res.noise_deg,
                    res.scaling_factor,
                    in0.cur_limbs,
                    in0.noise_deg,
                    in0.scaling_factor,
                )
            )
            return res

        return wrapper

    if func.__name__ == "eval_fast_rotate":

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            digits, cipher, index, need_KS_add, need_moddown, cryptoContext = args
            digits_node_id = digits.cipher_id
            cipher_node_name = (
                "NODE{}".format(cipher.cipher_id) if cipher is not None else "None"
            )
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            compiled_code.append(
                "NODE{} = homo_ops.eval_fast_rotate(NODE{}, {}, {}, {}, {}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}".format(
                    out_node_id,
                    digits_node_id,
                    cipher_node_name,
                    index,
                    need_KS_add,
                    need_moddown,
                    res.cur_limbs,
                    res.noise_deg,
                    res.scaling_factor,
                    digits.cur_limbs,
                    digits.noise_deg,
                    digits.scaling_factor,
                )
                + (
                    ""
                    if cipher is None
                    else "in1: limb={}, noise={}, sf={}".format(
                        cipher.cur_limbs, cipher.noise_deg, cipher.scaling_factor
                    )
                )
            )
            return res

        return wrapper

    if func.__name__ == "extract_cv":

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            in0, index, cryptoContext = args
            in0_node_id = in0.cipher_id
            out_node_id = Cipher.get_next_id()
            if "append_zeros" in kwargs:
                append_zeros = ", append_zeros = " + str(kwargs["append_zeros"])
            else:
                append_zeros = ""
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            compiled_code.append(
                "NODE{} = homo_ops.extract_cv(NODE{}, {}{}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}".format(
                    out_node_id,
                    in0_node_id,
                    index,
                    append_zeros,
                    res.cur_limbs,
                    res.noise_deg,
                    res.scaling_factor,
                    in0.cur_limbs,
                    in0.noise_deg,
                    in0.scaling_factor,
                )
            )

            return res

        return wrapper

    if func.__name__ == "adjust_to":

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ct1, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext = args
            ct1_node_id = ct1.cipher_id
            out1_node_id = Cipher.get_next_id()
            out1 = func(*args)
            out1.cipher_id = out1_node_id
            if ct1.cur_limbs == target_limbs and ct1.noise_deg == target_noise_deg:
                compiled_code.append(
                    "NODE{} = NODE{} # limb={}, noise={}, sf={}".format(
                        out1_node_id,
                        ct1_node_id,
                        ct1.cur_limbs,
                        ct1.noise_deg,
                        ct1.scaling_factor,
                    )
                )
            compiled_code.append(
                "NODE{} = homo_ops.adjust_to(NODE{}, {}, {}, {}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}".format(
                    out1_node_id,
                    ct1_node_id,
                    repr(target_limbs),
                    repr(target_noise_deg),
                    repr(target_scaling_factor),
                    out1.cur_limbs,
                    out1.noise_deg,
                    out1.scaling_factor,
                    ct1.cur_limbs,
                    ct1.noise_deg,
                    ct1.scaling_factor
                )
            )

            return out1

        return wrapper

    if func.__name__ == "eval_bootstrap":

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cryptoContext = args[-1]
            m_U0hatTPreFFT = cryptoContext.BsContext.m_U0hatTPreFFT
            m_U0PreFFT = cryptoContext.BsContext.m_U0PreFFT
            for i in range(len(m_U0hatTPreFFT)):
                for j in range(len(m_U0hatTPreFFT[i])):
                    compiled_code.append(
                        "NODE{} = cryptoContext.BsContext.m_U0hatTPreFFT[{}][{}] # limb={}, noise={}, sf={}".format(
                            m_U0hatTPreFFT[i][j].cipher_id,
                            i,
                            j,
                            m_U0hatTPreFFT[i][j].cur_limbs,
                            m_U0hatTPreFFT[i][j].noise_deg,
                            m_U0hatTPreFFT[i][j].scaling_factor,
                        )
                    )

            for i in range(len(m_U0PreFFT)):
                for j in range(len(m_U0PreFFT[i])):
                    compiled_code.append(
                        "NODE{} = cryptoContext.BsContext.m_U0PreFFT[{}][{}] # limb={}, noise={}, sf={}".format(
                            m_U0PreFFT[i][j].cipher_id,
                            i,
                            j,
                            m_U0PreFFT[i][j].cur_limbs,
                            m_U0PreFFT[i][j].noise_deg,
                            m_U0PreFFT[i][j].scaling_factor,
                        )
                    )
            compiled_code.append("NODE{} = NODE_IN".format(Cipher._id_counter))
            res = func(*args, **kwargs)

            return res

        return wrapper
