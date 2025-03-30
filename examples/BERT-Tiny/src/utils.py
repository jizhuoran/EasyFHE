import torch
import pickle
import torch.fhe as fhe
import atexit, os,csv

DATA_DIR = os.environ["DATA_DIR"]
global_circuit_depth=0
encoded_weight = {}
NEW_VERSION = False

if NEW_VERSION:



    def load_weight(encode_weight_path, cryptoContext):
        with open(encode_weight_path, 'rb') as f:
            pre_encoded = pickle.load(f)
        if cryptoContext.PRELOAD_ALL:
            for key, _ in pre_encoded.items():
                pre_encoded[key].cv = [torch.tensor(pre_encoded[key].cv[0], dtype=torch.uint64, device="cuda")]
                # print("NODE{} = pre_encoded[{}]".format(pre_encoded[key].node_id, key))
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

    def read_expanded_input(openfhe_context,filename,slots,scale=1):
        input=read_values_from_file(filename)
        repeated=[]
        for j in range(128):
            for i in range(128):
                repeated.append(input[j])
        size=len(repeated)
        if scale!=1:
            for i in range(size):
                repeated[i]=repeated[i]*scale
        x = torch.tensor(repeated, device="cuda")
        return openfhe_context.encrypt(x, 0, 1, slots, "release")


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

    def read_plain_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        val_name = filename
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                 "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
            ]

        else:
            ptx = cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx


    def read_plain_repeated_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0):

        val_name = filename
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)            ]

        else:
            ptx = cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx


    def read_plain_expanded_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0,num_inputs=None):
        val_name = filename
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)            ]

        else:
            ptx = cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx


    def mask_block(c, fro, to, mask_value,level, cryptoContext):
        if cryptoContext.PRELOAD_ALL:

            encoded=cryptoContext.pre_encoded[
                "mask_block_{}_{}_{}_{}".format( level, fro,to, c.cur_num_slots)]
            temp = fhe.homo_mul_pt(c, encoded, cryptoContext)
            return temp
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_block_{}_{}_{}_{}".format(level, fro, to, c.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return fhe.homo_mul_pt(c, ptx, cryptoContext)

    def mask_heads(c, mask_value, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            encoded=cryptoContext.pre_encoded[
                "mask_heads_{}_{}_{}".format(cryptoContext.L - c.cur_limb, mask_value, c.cur_num_slots)
            ]
            temp = fhe.homo_mul_pt(c, encoded, cryptoContext)
            return temp

        else:
            ptx = cryptoContext.pre_encoded[
                "mask_heads_{}_{}_{}".format(cryptoContext.L - c.cur_limb, mask_value, c.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return fhe.homo_mul_pt(c, ptx, cryptoContext)

    def eval_mult_many(ciphertexts, cryptoContext):
        # plain implementation of EvalAddMany
        inSize = len(ciphertexts)
        if inSize < 1:
            raise ValueError("Input ciphertext vector size should be 1 or more")
        sum = ciphertexts[0].deep_copy()
        for i in range(1, inSize):
            sum = fhe.homo_mul(sum, ciphertexts[i], cryptoContext)

        return sum



    def eval_exp(c, inputs_number, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            encoded=cryptoContext.pre_encoded[
                "eval_exp_{}_{}_{}".format(cryptoContext.L - c.cur_limb, inputs_number, c.cur_num_slots)
            ]
            temp = fhe.homo_add_pt(c, encoded, cryptoContext)
            return temp
        else:
            ptx = cryptoContext.pre_encoded[
                "eval_exp_{}_{}_{}".format(cryptoContext.L - c.cur_limb, inputs_number, c.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return fhe.homo_add_pt(c, ptx, cryptoContext)

    def mask_mod_n(c, n, padding, max_slots, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            encoded=cryptoContext.pre_encoded[
                "mask_mod_n_{}_{}_{}".format(cryptoContext.L - c.cur_limb, padding, c.cur_num_slots)
            ]
            temp=fhe.homo_add_pt(c, encoded, cryptoContext)
            return temp

        else:
            ptx = cryptoContext.pre_encoded[
                "mask_mod_n_{}_{}_{}".format(cryptoContext.L - c.cur_limb, padding, c.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return fhe.homo_add_pt(c, ptx, cryptoContext)

    def mask_first_n(c, n, mask_value, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            encoded=cryptoContext.pre_encoded[
                "mask_first_n_{}_{}_{}".format(cryptoContext.L - c.cur_limb, mask_value, c.cur_num_slots)
            ]
            temp=fhe.homo_add_pt(c, encoded, cryptoContext)
            return temp
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_first_n_{}_{}_{}".format(cryptoContext.L - c.cur_limb, mask_value, c.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return fhe.homo_add_pt(c, ptx, cryptoContext)


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


    def read_plain_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        values = []
        val_name = filename
        filename = DATA_DIR  + filename
        if not os.path.isfile(filename):
            print(f"无法打开文件: {filename}")
            return values
        input = read_values_from_file(filename)
        size = len(input)
        if scale != 1:
            for i in range(size):
                input[i] = input[i] * scale
        x = torch.tensor(input, dtype=torch.float64).cuda()
        encoded=fhe.encode(x, scale_deg, level, slots, False, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
        encoded_weight[key] = encoded
        return  encoded


    def read_plain_repeated_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        values = []
        val_name = filename
        filename = DATA_DIR  + filename
        if not os.path.isfile(filename):
            print(f"无法打开文件: {filename}")
            return values

        input = read_values_from_file(filename)
        repeated = []
        for j in range(128):
            for i in range(128):
                repeated.append(input[i])
        size = len(repeated)
        if scale != 1:
            for i in range(size):
                repeated[i] = repeated[i] * scale
        x = torch.tensor(repeated, dtype=torch.float64).cuda()
        encoded=fhe.encode(x, scale_deg, level, slots, False, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
        encoded_weight[key] = encoded
        return  encoded


    def read_plain_expanded_input(cryptoContext, filename, level, scale_deg, slots, scale=1.0,num_inputs=None):

        values = []
        val_name = filename
        filename = DATA_DIR + filename
        if not os.path.isfile(filename):
            print(f"无法打开文件: {filename}")
            return values

        input_values = read_values_from_file(filename)
        repeated = []

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

        x = torch.tensor(repeated, dtype=torch.float64).cuda()
        encoded=fhe.encode(x, scale_deg, level, slots, False, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
        encoded_weight[key] = encoded
        return  encoded


    def mask_block(c, fro, to, mask_value,level, cryptoContext):
        mask = []
        for i in range(c.cur_num_slots):
            if i >= fro and i < to:
                mask.append(mask_value)
            else:
                mask.append(0)
        x = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded=fhe.encode(x, 1, level, c.cur_num_slots, False, cryptoContext)
        temp=fhe.homo_mul_pt(c, encoded, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "mask_block_{}_{}_{}_{}".format( level, fro,to, c.cur_num_slots)
        encoded_weight[key] = encoded

        return temp




    def mask_heads(c, mask_value, cryptoContext):
        mask = []
        for i in range(c.cur_num_slots):
            if i % 64 == 0:
                mask.append(mask_value)
            else:
                mask.append(0)
        x = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(x, 1, cryptoContext.L - c.cur_limb, c.cur_num_slots, False, cryptoContext)
        temp = fhe.homo_mul_pt(c, encoded, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "mask_heads_{}_{}_{}".format(cryptoContext.L - c.cur_limb, mask_value, c.cur_num_slots)
        encoded_weight[key] = encoded
        return temp

    def read_expanded_input(openfhe_context,filename,slots,scale=1):
        input=read_values_from_file(filename)
        repeated=[]
        for j in range(128):
            for i in range(128):
                repeated.append(input[j])
        size=len(repeated)
        if scale!=1:
            for i in range(size):
                repeated[i]=repeated[i]*scale
        x = torch.tensor(repeated, device="cuda")
        return openfhe_context.encrypt(x, 0, 1, slots, "release")



    def eval_mult_many(ciphertexts, cryptoContext):
        # plain implementation of EvalAddMany
        inSize = len(ciphertexts)
        if inSize < 1:
            raise ValueError("Input ciphertext vector size should be 1 or more")
        sum = ciphertexts[0].deep_copy()
        for i in range(1, inSize):
            sum = fhe.homo_mul(sum, ciphertexts[i], cryptoContext)

        return sum
    def eval_exp(c, inputs_number, cryptoContext):
        # Todo:    Ctxt res = context->EvalPoly(c, {1, 1, 1/(2.0), 1/(6.0), 1/(24.0), 1/(120.0), 1/(720.0)});
        res = c

        if cryptoContext.L - res.cur_limb + 4 > global_circuit_depth:
            res = fhe.homo_bootstrap(res, L0=cryptoContext.L, logBsSlots=14,
                                     cryptoContext=cryptoContext)
        res = eval_mult_many([res, res, res, res, res, res, res, res], cryptoContext)
        mask = []
        for i in range(c.cur_num_slots):
            if i % 64 < inputs_number and i < (128 * inputs_number):
                mask.append(0)
            else:
                mask.append(1)

        x = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(x, 1, cryptoContext.L - c.cur_limb, c.cur_num_slots, False, cryptoContext)
        temp = fhe.homo_add_pt(res, encoded, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "eval_exp_{}_{}_{}".format(cryptoContext.L - c.cur_limb, inputs_number, c.cur_num_slots)
        encoded_weight[key] = encoded
        return temp


    def mask_mod_n(c, n, padding, max_slots, cryptoContext):
        mask = []
        for i in range(c.cur_num_slots):
            if i % n == padding:
                mask.append(1)
            else:
                mask.append(0)
        x = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(x, 1, cryptoContext.L - c.cur_limb, c.cur_num_slots, False, cryptoContext)
        temp = fhe.homo_add_pt(c, encoded, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "mask_mod_n_{}_{}_{}".format(cryptoContext.L - c.cur_limb, padding, c.cur_num_slots)
        encoded_weight[key] = encoded
        return temp


    def mask_first_n(c, n, mask_value, cryptoContext):
        mask = []
        for i in range(c.cur_num_slots):
            if i < n:
                mask.append(mask_value)
            else:
                mask.append(0)
        x = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(x, 1, cryptoContext.L - c.cur_limb, c.cur_num_slots, False, cryptoContext)
        temp = fhe.homo_add_pt(c, encoded, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "mask_first_n_{}_{}_{}".format(cryptoContext.L - c.cur_limb, mask_value, c.cur_num_slots)
        encoded_weight[key] = encoded
        return temp

@atexit.register
def save_encoded_weight():
    if NEW_VERSION == False:
        for key, value in encoded_weight.items():
            encoded_weight[key].cv = [value.cv[0].cpu().numpy()]
        with open(DATA_DIR + "/weight.pkl", "wb") as f:
            pickle.dump(encoded_weight, f)