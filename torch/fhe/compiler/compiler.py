from ..ciphertext import Cipher

unary_op = {
    "homo_square": "homo_ops.",
    "key_switch_P_ext": "hybrid_keyswitch.",
    "modup_to_ext": "hybrid_keyswitch.",
    "moddown_from_ext": "hybrid_keyswitch.",
}

unary_cnst_op = {
    "drop_last_elements": "homo_ops.",
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
            ct, val, cryptoContext = args
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                del kwargs["printInfo"]
                return func(*args, **kwargs)
            if cryptoContext.inBS == True:
                return func(*args, **kwargs)
            in_node_id = ct.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
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

        def wrapper(*args, **kwargs):
            in0, in1, cryptoContext = args
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                del kwargs["printInfo"]
                return func(*args, **kwargs)
            if cryptoContext.inBS == True:
                return func(*args, **kwargs)
            in0_node_id = in0.cipher_id
            in1_node_id = in1.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
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

        def wrapper(*args, **kwargs):
            in0, cryptoContext = args
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                del kwargs["printInfo"]
                return func(*args, **kwargs)
            if cryptoContext.inBS == True:
                return func(*args, **kwargs)
            in0_node_id = in0.cipher_id
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
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

        def wrapper(*args, **kwargs):
            digits, cipher, index, need_KS_add, need_moddown, cryptoContext = args
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                del kwargs["printInfo"]
                return func(*args, **kwargs)
            if cryptoContext.inBS == True:
                return func(*args, **kwargs)
            digits_node_id = digits.cipher_id
            cipher_node_name = (
                "NODE{}".format(cipher.cipher_id) if cipher is not None else "None"
            )
            out_node_id = Cipher.get_next_id()
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
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

        def wrapper(*args, **kwargs):
            in0, index, cryptoContext = args
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                del kwargs["printInfo"]
                return func(*args, **kwargs)
            if cryptoContext.inBS == True:
                return func(*args, **kwargs)
            in0_node_id = in0.cipher_id
            out_node_id = Cipher.get_next_id()
            if "append_zeros" in kwargs:
                append_zeros = ", append_zeros = " + str(kwargs["append_zeros"])
            else:
                append_zeros = ""
            res = func(*args, **kwargs)
            res.cipher_id = out_node_id
            print(
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

        def wrapper(*args, **kwargs):
            ct1, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext = args
            if "printInfo" in kwargs and kwargs["printInfo"] == False:
                del kwargs["printInfo"]
                return func(*args, **kwargs)
            if cryptoContext.inBS == True:
                return func(*args, **kwargs)
            ct1_node_id = ct1.cipher_id
            out1_node_id = Cipher.get_next_id()
            out1 = func(*args)
            out1.cipher_id = out1_node_id
            print(
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

        def wrapper(*args, **kwargs):
            ciphertext, L0, logBsSlots, cryptoContext = args
            if True: #in application
                m_U0hatTPreFFT = cryptoContext.BsContext.m_U0hatTPreFFT
                m_U0PreFFT = cryptoContext.BsContext.m_U0PreFFT
                for i in range(len(m_U0hatTPreFFT)):
                    for j in range(len(m_U0hatTPreFFT[i])):
                        print(
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
                        print(
                            "NODE{} = cryptoContext.BsContext.m_U0PreFFT[{}][{}] # limb={}, noise={}, sf={}".format(
                                m_U0PreFFT[i][j].cipher_id,
                                i,
                                j,
                                m_U0PreFFT[i][j].cur_limbs,
                                m_U0PreFFT[i][j].noise_deg,
                                m_U0PreFFT[i][j].scaling_factor,
                            )
                        )
                print("NODE{} = NODE_IN".format(Cipher._id_counter))
                res = func(*args, **kwargs)
            else:
                cryptoContext.inBS = True
                in0_node_id = ciphertext.cipher_id
                out_node_id = Cipher.get_next_id()
                res = func(*args, **kwargs)
                res.cipher_id = out_node_id
                print(
                    "NODE{} = eval_bootstrap(NODE{}, {}, {}, cryptoContext) #out: limb={}, noise={}, sf={}, in0: limb={}, noise={}, sf={}".format(
                        out_node_id,
                        in0_node_id,
                        L0,
                        logBsSlots,
                        res.cur_limbs,
                        res.noise_deg,
                        res.scaling_factor,
                        ciphertext.cur_limbs,
                        ciphertext.noise_deg,
                        ciphertext.scaling_factor,
                    )
                )
                res = func(*args, **kwargs)
                cryptoContext.inBS = False
            return res

        return wrapper


frontend = compilerFrontend if COMPILE == "ON" else omitFrontend
