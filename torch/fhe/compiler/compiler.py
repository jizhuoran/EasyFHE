from ..ciphertext import Cipher

unary_op = {
    "homo_square": "homo_ops.",
    "drop_last_elements_": "homo_ops.",
    "key_switch_P_ext": "hybrid_keyswitch.",
    "modup_to_ext": "hybrid_keyswitch.",
    "moddown_from_ext": "hybrid_keyswitch.",
}

unary_cnst_op = {
    "homo_rescale": "homo_ops.",
    "homo_rescale_internal": "homo_ops.",
    "homo_mul_scalar_double": "homo_ops.",
    "mod_raise": "",
    "assign_scaling_factor": "",
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

COMPILE = "OFF"


def omitFrontend(func):
    def wrapper(*args, **kwargs):
        if "printInfo" in kwargs:
            del kwargs["printInfo"]
        return func(*args, **kwargs)

    return wrapper


def compilerFrontend(func):

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
                + (
                    ""
                    if cipher is None
                    else "in1: limb={}, noise={}".format(
                        cipher.cur_limbs, cipher.noise_deg
                    )
                )
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

    if func.__name__ == "eval_bootstrap":

        def wrapper(*args, **kwargs):
            if 'ciphertext' in kwargs:
                ciphertext = kwargs['kwargs']
            else:
                ciphertext = args[0]
            if 'cryptoContext' in kwargs:
                cryptoContext = kwargs['cryptoContext']
            else:
                cryptoContext = args[-1]
            m_U0hatTPreFFT = cryptoContext.BsContext.m_U0hatTPreFFT
            m_U0PreFFT = cryptoContext.BsContext.m_U0PreFFT
            for i in range(len(m_U0hatTPreFFT)):
                for j in range(len(m_U0hatTPreFFT[i])):
                    print(
                        "NODE{} = cryptoContext.BsContext.m_U0hatTPreFFT[{}][{}] # limb={}, noise={}".format(
                            m_U0hatTPreFFT[i][j].cipher_id,
                            i,
                            j,
                            m_U0hatTPreFFT[i][j].cur_limbs,
                            m_U0hatTPreFFT[i][j].noise_deg,
                        )
                    )

            for i in range(len(m_U0PreFFT)):
                for j in range(len(m_U0PreFFT[i])):
                    print(
                        "NODE{} = cryptoContext.BsContext.m_U0PreFFT[{}][{}] # limb={}, noise={}".format(
                            m_U0PreFFT[i][j].cipher_id,
                            i,
                            j,
                            m_U0PreFFT[i][j].cur_limbs,
                            m_U0PreFFT[i][j].noise_deg,
                        )
                    )

            print("NODE{} = IN_NODE".format(ciphertext.cipher_id))
            res = func(*args, **kwargs)
            return res

        return wrapper


frontend = compilerFrontend if COMPILE == "ON" else omitFrontend
