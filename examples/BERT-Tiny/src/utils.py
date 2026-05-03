import torch
import numpy as np
import pickle
import torch.fhe as fhe
import examples.utils.approx as approx
import os, csv
import warnings
from pathlib import Path




DATA_DIR = os.environ["DATA_DIR"]
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR.parent / "weights-sst2"

DIRECT_LOAD = False

if DIRECT_LOAD:
    def load_weight(encode_weight_path, cryptoContext):
        with open(encode_weight_path, 'rb') as f:
            pre_encoded = pickle.load(f)
        for key, _ in pre_encoded.items():
            if cryptoContext.pre_encode_type == "middle":
                pre_encoded[key].encoded_values = torch.tensor(pre_encoded[key].encoded_values, device="cuda")
            elif cryptoContext.pre_encode_type == "end":
                pre_encoded[key].cv = [torch.tensor(pre_encoded[key].cv[0], dtype=torch.uint64, device="cuda")]
        cryptoContext.pre_encoded = pre_encoded


    def read_values_from_file(filename, scale=1):
        values = []
        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                for value in row:
                    try:
                        num = float(value)
                        values.append(num * scale)
                    except ValueError:
                        print(f"Can not convert: {value}")
        return values

    def read_expanded_input(cryptoContext, openfhe_context,filename,level, scale_deg, slots, scale=1):
        input=read_values_from_file(filename)
        repeated=[]
        for j in range(128):
            for i in range(128):
                repeated.append(input[j])
        size=len(repeated)
        if scale!=1:
            for i in range(size):
                repeated[i]=repeated[i]*scale
        return openfhe_context.encrypt(repeated, cryptoContext.device, scale_deg, level, slots)


    def read_plain_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        full_name = "{}_{}_{}_{}_{}".format(filename, level, scale_deg, slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format(filename, cnt)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)


    def read_plain_repeated_input(cryptoContext, filename, level, scale_deg, slots, scale):
        # Assumption: inputs have 128 values
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        full_name = "{}_{}_{}_{}_{}".format(filename, level, scale_deg, slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format(filename, cnt)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)


    def read_plain_expanded_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0,num_inputs=None):
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        full_name = "{}_{}_{}_{}_{}".format(filename, level, scale_deg, slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format(filename, cnt)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)


    def mask_block(c, fro, to, mask_value, cryptoContext):
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        full_name = "{}_{}_{}_{}_{}".format("mask_block", cryptoContext.L-c.cur_limbs, c.noise_deg, c.slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("mask_block", cnt)
        else:
            name = full_name
        ptx =  fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L-c.cur_limbs, c.slots, False, cryptoContext)
        return fhe.homo_mul_pt(c, ptx, cryptoContext)


    def mask_heads(c, mask_value, cryptoContext):
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        full_name = "{}_{}_{}_{}_{}".format("mask_heads", cryptoContext.L-c.cur_limbs, c.noise_deg, c.slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("mask_heads", cnt)
        else:
            name = full_name
        ptx =  fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L-c.cur_limbs, c.slots, False, cryptoContext)
        return fhe.homo_mul_pt(c, ptx, cryptoContext)


    def mask_mod_n(c, n, padding, cryptoContext):
        cnt = cryptoContext.cnt
        cryptoContext.cnt += 1
        full_name = "{}_{}_{}_{}_{}".format("mask_mod_n", cryptoContext.L - c.cur_limbs, c.noise_deg, c.slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("mask_mod_n", cnt)
        else:
            name = full_name
        ptx = fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - c.cur_limbs, c.slots, False,
                         cryptoContext)
        return fhe.homo_mul_pt(c, ptx, cryptoContext)


    def mask_first_n(c, n, mask_value, cryptoContext):
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        full_name = "{}_{}_{}_{}_{}".format("mask_first_n", cryptoContext.L-c.cur_limbs, c.noise_deg, c.slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("mask_first_n", cnt)
        else:
            name = full_name
        ptx =  fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L-c.cur_limbs, c.slots, False, cryptoContext)
        return fhe.homo_mul_pt(c, ptx, cryptoContext)


    def eval_exp(c, inputs_number, cryptoContext):
        res = approx.eval_poly_ps(c, [1, 1, 1/(2.0), 1/(6.0), 1/(24.0), 1/(120.0), 1/(720.0)], cryptoContext)

        if cryptoContext.L - res.cur_limbs + 4 > cryptoContext.L:
            res = fhe.homo_bootstrap(res, cryptoContext.L, 14, #todo: should use logBsSlots list
                                     cryptoContext)
        res = eval_mult_many([res, res, res, res, res, res, res, res], cryptoContext)
        cnt = cryptoContext.cnt
        cryptoContext.cnt += 1
        full_name = "{}_{}_{}_{}_{}".format("eval_exp", cryptoContext.L - c.cur_limbs, c.noise_deg, c.slots, cnt)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("eval_exp", cnt)
        else:
            name = full_name
        encoded = fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - c.cur_limbs, c.slots, False,
                         cryptoContext)
        temp = fhe.homo_add_pt(res, encoded, cryptoContext)
        return temp

else:
    def load_weight(encode_weight_path, cryptoContext):
        pass

    # raise ValueError("deprecated branch")
    def read_values_from_file(filename, scale=1):
        values = []
        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                for value in row:
                    try:
                        num = float(value)
                        values.append(num * scale)
                    except ValueError:
                        print(f"Can not convert: {value}")
        return values


    def read_expanded_input(cryptoContext, openfhe_context,filename, level, scale_deg, slots,scale=1):
        input=read_values_from_file(filename)
        assert len(input)==128, f"len of input {len(input)} is not equal to 128"
        repeated=[]
        for j in range(128):
            for i in range(128):
                repeated.append(input[j])
        size=len(repeated)
        if scale!=1:
            for i in range(size):
                repeated[i]=repeated[i]*scale
        return openfhe_context.encrypt(repeated, cryptoContext.device, scale_deg, level, slots)



    def read_plain_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        values = []
        val_name = filename
        filename = WEIGHTS_DIR / f"{val_name}.txt"
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values
        input = read_values_from_file(filename)
        size = len(input)
        assert size<=slots, "The num of value in filename : {} is :{} is more than slots: {}".format(filename, size, slots)
        if scale != 1:
            for i in range(size):
                input[i] = input[i] * scale

        x=np.array(input, dtype=np.double)
        x = np.pad(x, (0, slots - len(x)), mode='constant')

        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format(val_name, cnt)
        print(name)
        encoded=fhe.encode(x, name, level, slots, False, cryptoContext)
        return  encoded


    def read_plain_repeated_input(cryptoContext, filename, level, scale_deg, slots, scale):
        values = []
        val_name = filename
        filename = WEIGHTS_DIR / f"{val_name}.txt"
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values

        input = read_values_from_file(filename)
        assert len(input) == 128, f"len of input {len(input)} is not equal to 128"
        repeated = []
        for j in range(128):
            for i in range(128):
                repeated.append(input[i])
        size = len(repeated)
        if scale != 1:
            for i in range(size):
                repeated[i] = repeated[i] * scale
        x=np.array(repeated, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format(val_name, cnt)
        print(name)
        encoded=fhe.encode(x, name, level, slots, False, cryptoContext)
        return  encoded


    def read_plain_expanded_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0,num_inputs=None):
        values = []
        val_name = filename
        filename = WEIGHTS_DIR / f"{val_name}.txt"
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values

        input_values = read_values_from_file(filename)
        repeated = []
        if len(input_values) < 128:
            warnings.warn(
                f"The num of value in filename : {filename} is :{len(input_values)} is less than 128",
                Warning,
            )
            for j in range(128):
                if j < len(input_values):
                    if num_inputs is None:
                        for i in range(128):
                            repeated.append(input_values[j])
                    else:
                        for i in range(num_inputs):
                            repeated.append(input_values[j])
                        for i in range(128 - num_inputs):
                            repeated.append(0)
                else:
                    for i in range(128):
                        repeated.append(0)
        else:
            assert len(input_values) == 128,f"The num of value in filename : {filename} is :{len(input_values)} is more than 128"
            for j in range(128):
                if num_inputs is None:
                    for i in range(128):
                        repeated.append(input_values[j])
                else:
                    for i in range(num_inputs):
                        repeated.append(input_values[j])
                    for i in range(128 - num_inputs):
                        repeated.append(0)

        if scale != 1:
            repeated = [x * scale for x in repeated]
        x = np.array(repeated, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format(val_name, cnt)
        print(name)
        encoded=fhe.encode(x, name, level, slots, False, cryptoContext)
        return encoded


    def mask_block(c, fro, to, mask_value, cryptoContext):
        mask = []
        for i in range(c.slots):
            if i >= fro and i < to:
                mask.append(mask_value)
            else:
                mask.append(0)
        x = np.array(mask, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format("mask_block", cnt)
        print(name)
        encoded=fhe.encode(x, name, cryptoContext.L - c.cur_limbs, c.slots, False, cryptoContext)
        temp=fhe.homo_mul_pt(c, encoded, cryptoContext)
        return temp




    def mask_heads(c, mask_value, cryptoContext):
        mask = []
        for i in range(c.slots):
            if i % 64 == 0:
                mask.append(mask_value)
            else:
                mask.append(0)
        x = np.array(mask, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format("mask_heads", cnt)
        print(name)
        encoded=fhe.encode(x, name, cryptoContext.L-c.cur_limbs, c.slots, False, cryptoContext)
        temp=fhe.homo_mul_pt(c, encoded, cryptoContext)
        return temp


    def mask_mod_n(c, n, padding, cryptoContext):
        num_slots = c.slots # todo: check if they are equal for the original italian global num_slots
        mask = []
        for i in range(num_slots):
            if i % n == padding:
                mask.append(1)
            else:
                mask.append(0)
        x = np.array(mask, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format("mask_mod_n", cnt)
        print(name)
        encoded=fhe.encode(x, name, cryptoContext.L-c.cur_limbs, c.slots, False, cryptoContext)
        temp = fhe.homo_mul_pt(c, encoded, cryptoContext)
        return temp


    def mask_first_n(c, n, mask_value, cryptoContext):
        mask = []
        for i in range(c.slots):
            if i < n:
                mask.append(mask_value)
            else:
                mask.append(0)
        x = np.array(mask, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format("mask_first_n", cnt)
        print(name)
        encoded=fhe.encode(x, name, cryptoContext.L-c.cur_limbs, c.slots, False, cryptoContext)
        temp = fhe.homo_mul_pt(c, encoded, cryptoContext)
        return temp


    def eval_exp(c, inputs_number, cryptoContext):
        res = approx.eval_poly_ps(c, [1, 1, 1/(2.0), 1/(6.0), 1/(24.0), 1/(120.0), 1/(720.0)], cryptoContext)

        if cryptoContext.L - res.cur_limbs + 4 > cryptoContext.L:
            res = fhe.homo_bootstrap(res, cryptoContext.L, 14, #todo: should use logBsSlots list
                                     cryptoContext)
        res = eval_mult_many([res, res, res, res, res, res, res, res], cryptoContext)
        mask = []
        num_slots = (1<<14) # fixme: should not hardcode here, introduce global management?
        for i in range(num_slots):
            if i % 64 < inputs_number and i < (128 * inputs_number):
                mask.append(0)
            else:
                mask.append(-1)

        x = np.array(mask, dtype=np.double)
        cnt = cryptoContext.cnt
        cryptoContext.cnt+=1
        name = "{}_{}".format("eval_exp", cnt)
        print(name)
        encoded=fhe.encode(x, name, cryptoContext.L-res.cur_limbs, num_slots, False, cryptoContext)
        temp = fhe.homo_add_pt(res, encoded, cryptoContext)
        return temp

def eval_add_many(ciphertexts, cryptoContext):
    # plain implementation of EvalAddMany
    inSize = len(ciphertexts)
    if inSize < 1:
        raise ValueError("Input ciphertext vector size should be 1 or more")
    sum = ciphertexts[0].deep_copy()
    for i in range(1,inSize):
        sum = fhe.homo_add(sum, ciphertexts[i], cryptoContext)

    return sum


def eval_mult_many(ciphertexts, cryptoContext):
    # plain implementation of EvalAddMany
    inSize = len(ciphertexts)
    if inSize < 1:
        raise ValueError("Input ciphertext vector size should be 1 or more")
    sum = ciphertexts[0].deep_copy()
    for i in range(1, inSize):
        sum = fhe.homo_mul(sum, ciphertexts[i], cryptoContext)

    return sum
