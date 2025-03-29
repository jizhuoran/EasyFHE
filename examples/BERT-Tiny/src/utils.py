import torch
import pickle
import torch.fhe as fhe
import atexit, os

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


    def read_values_from_file(
        cryptoContext, filename, level, scale_deg, slots, scale=1.0
    ):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(filename, level, scale_deg, slots)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "{}_{}_{}_{}".format(filename, level, scale_deg, slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def read_fc_weight(cryptoContext, level, scale_deg, slots):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "fc_{}_{}_{}".format(level, scale_deg, slots)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "fc_{}_{}_{}".format(level, scale_deg, slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_mod(n, cur_limbs, custom_val, he_res20_ctx, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_scecond_n(n, cur_limbs, he_res20_ctx, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_scecond_n_{}_{}_{}".format(
                    n, cur_limbs, he_res20_ctx.cur_num_slots
                )
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_scecond_n_{}_{}_{}".format(
                    n, cur_limbs, he_res20_ctx.cur_num_slots
                )
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_first_n(n, cur_limbs, he_res20_ctx, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_from_to(from_, to, cur_limbs, he_res20_ctx, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_from_to_{}_{}_{}_{}".format(
                    from_, to, cur_limbs, he_res20_ctx.cur_num_slots
                )
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_from_to_{}_{}_{}_{}".format(
                    from_, to, cur_limbs, he_res20_ctx.cur_num_slots
                )
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def gen_mask(n, cur_limbs, he_res20_ctx, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_first_n_mod(n, padding, pos, cur_limbs, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_first_n_mod2(n, padding, pos, cur_limbs, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_channel(n, cur_limbs, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384 * 2)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384 * 2)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx

    def mask_channel2(n, cur_limbs, cryptoContext):
        if cryptoContext.PRELOAD_ALL:
            return cryptoContext.pre_encoded[
                "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192 * 2)
            ]
        else:
            ptx = cryptoContext.pre_encoded[
                "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192 * 2)
            ].shallow_copy()
            ptx.cv = [torch.tensor(ptx.cv[0], dtype=torch.uint64, device="cuda")]
            return ptx


else:
    def load_weight(encode_weight_path, cryptoContext):
        pass

    # raise ValueError("deprecated branch")
    def read_values_from_file(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        # print("read_values_from_file", filename, "level", level, "scale_deg", scale_deg, "slots", slots, "scale", scale)
        values = []
        val_name = filename
        filename = DATA_DIR + '/weights/' + filename + '.bin'
        if not os.path.isfile(filename):
            print(f"无法打开文件: {filename}")
            return values

        try:
            # 打开文件并逐行读取
            with open(filename, 'r') as file:
                for row in file:
                    # 按行解析
                    for value in row.strip().split(','):
                        try:
                            num = float(value)
                            values.append(num * scale)
                        except ValueError:
                            print(f"unconvert:: {value}")
        except IOError as e:
            print(f"error: {e}")

        values = torch.tensor(values, dtype=torch.float64).cuda()
        encoded = fhe.encode(values, scale_deg, level, slots, False, cryptoContext)
        encoded.cv[0].cpu().numpy()
        key = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)

        return encoded


    def read_fc_weight(cryptoContext, level, scale_deg, slots):
        # print("read_values_from_file", "fc", "level", level, "scale_deg", scale_deg, "slots", slots, "scale", 1)
        values = []
        filename = DATA_DIR + '/weights/fc.bin'
        if not os.path.isfile(filename):
            print(f"无法打开文件: {filename}")
            return values

        try:
            # 打开文件并逐行读取
            with open(filename, 'r') as file:
                for row in file:
                    # 按行解析
                    for value in row.strip().split(','):
                        try:
                            num = float(value)
                            values.append(num)
                        except ValueError:
                            print(f"unconvert:: {value}")
        except IOError as e:
            print(f"error: {e}")

        weight = values

        weight_corrected=[]
        for i in range(64):
            for j in range(10):
                weight_corrected.append(weight[(10 * i) + j])
            for j in range(64 - 10):
                weight_corrected.append(0)
        weight_corrected = torch.tensor(weight_corrected, dtype=torch.float64).cuda()
        encoded = fhe.encode(weight_corrected, scale_deg, level, slots, False, cryptoContext)
        key = "fc_{}_{}_{}".format(level, scale_deg, slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded


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


    def mask_mod(n,cur_limbs,custom_val, he_res0_ctx, cryptoContext):
        # print("mask_mod", "n", n, "cur_limbs", cur_limbs, "custom_val", custom_val, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        level = cryptoContext.L-cur_limbs
        vec=[]
        for i in range(he_res20_ctx.cur_num_slots):
            if i%n==0:
                vec.append(custom_val)
            else:
                vec.append(0)
        vec = torch.tensor(vec, dtype=torch.float64).cuda()
        encoded = fhe.encode(vec,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        key = "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_scecond_n(n, cur_limbs, he_res20_ctx, cryptoContext):
        # print("mask_scecond_n", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        mask=[]
        level=cryptoContext.L-cur_limbs
        for i in range(he_res20_ctx.cur_num_slots):
            if i >=n :
                mask.append(1)
            else:
                mask.append(0)
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        key = "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_first_n(n, cur_limbs, he_res20_ctx, cryptoContext):
        # print("mask_first_n", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        mask=[]
        level=cryptoContext.L-cur_limbs
        for i in range(he_res20_ctx.cur_num_slots):
            if i < n:
                mask.append(1)
            else:
                mask.append(0)
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask, 1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        key = "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_from_to(from_, to, cur_limbs, he_res20_ctx, cryptoContext):
        # print("mask_from_to", "from_", from_, "to", to, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        vec=[]
        level=cryptoContext.L-cur_limbs
        for i in range(he_res20_ctx.cur_num_slots):
            if i>=from_ and i<to:
                vec.append(1)
            else:
                vec.append(0)
        vec = torch.tensor(vec, dtype=torch.float64).cuda()
        encoded = fhe.encode(vec,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        key = "mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, he_res20_ctx.cur_num_slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def gen_mask(n,cur_limbs, he_res20_ctx, cryptoContext):
        # print("gen_mask", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        level = cryptoContext.L - cur_limbs
        mask=[]
        copy_interval=n
        for i in range(he_res20_ctx.cur_num_slots):
            if copy_interval>0:
                mask.append(1)
            else:
                mask.append(0)
            copy_interval-=1
            if copy_interval<= -n:
                copy_interval=n
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        key = "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_first_n_mod(n,padding,pos,cur_limbs, cryptoContext):
        # print("mask_first_n_mod", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(32):
            for j in range(pos*n):
                mask.append(0)
            for j in range(n):
                mask.append(1)
            for j in range(padding-n-(pos*n)):
                mask.append(0)
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask, 1, level, 16384*2, False, cryptoContext)
        key = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_first_n_mod2(n,padding,pos,cur_limbs, cryptoContext):
        # print("mask_first_n_mod2", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(64):
            for j in range(pos*n):
                mask.append(0)
            for j in range(n):
                mask.append(1)
            for j in range(padding-n-(pos*n)):
                mask.append(0)
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask, 1, level, 8192*2, False, cryptoContext)
        key = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_channel(n,cur_limbs,cryptoContext):
        # print("mask_channel", "n", n, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(n):
            for j in range(1024):
                mask.append(0)

        for i in range(256):
            mask.append(1)

        for i in range(1024-256):
            mask.append(0)
        for i in range(31-n):
            for j in range(1024):
                mask.append(0)
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask, 1, level, 16384*2, False, cryptoContext)
        key = "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384*2)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_channel2(n,cur_limbs,cryptoContext):
        # print("mask_channel2", "n", n, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(n):
            for j in range(256):
                mask.append(0)

        for i in range(64):
            mask.append(1)

        for i in range(256-64):
            mask.append(0)
        for i in range(63-n):
            for j in range(256):
                mask.append(0)
        mask = torch.tensor(mask, dtype=torch.float64).cuda()
        encoded = fhe.encode(mask, 1, level, 8192*2, False, cryptoContext)
        key = "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192*2)
        encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

@atexit.register
def save_encoded_weight():
    if NEW_VERSION == False:
        for key, value in encoded_weight.items():
            encoded_weight[key].cv = [value.cv[0].cpu().numpy()]
        with open(DATA_DIR + "/weight.pkl", "wb") as f:
            pickle.dump(encoded_weight, f)