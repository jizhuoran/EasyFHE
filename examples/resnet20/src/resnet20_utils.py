import torch
import pickle
import torch.fhe as fhe
import atexit, os
import numpy as np

DATA_DIR = os.environ["DATA_DIR"]

encoded_weight = {}

# Normalized deltas (trained by CIFAR-10 test data)
normalized_deltas = [
    [0.30245313974658655, 0, 0, 0, 0, 0],
    [
        0.25771464233502284,
        0.17572235969058683,
        0.26867995906162545,
        0.16879219146810473,
        0.32389941065236755,
        0.16670296717723732,
    ],
    [
        0.29577777852997955,
        0.20468562391210693,
        0.45305236761033496,
        0.1940840042412194,
        0.3655523676384972,
        0.13282571451191513,
    ],
    [
        0.3620743161940029,
        0.2372317323595584,
        0.32624424495604537,
        0.13859561075656615,
        0.34910082672803205,
        0.053238969339825734,
    ],
]


def log2_int(x):
    import math

    return int(math.log2(x))


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

    def read_values_from_file(cryptoContext, val_name, level, scale_deg, slots, scale=1.0):
        full_name = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format(val_name, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)

    def read_fc_weight(cryptoContext, level, scale_deg, slots):
        full_name = "fc_{}_{}_{}".format(level, scale_deg, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("fc", slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)

    def mask_mod(n, cur_limbs, custom_val, he_res20_ctx, cryptoContext):
        full_name = "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_mod_{}_{}_{}".format(n, custom_val, he_res20_ctx.cur_num_slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, he_res20_ctx.cur_num_slots, False, cryptoContext)

    def mask_scecond_n(n, cur_limbs, he_res20_ctx, cryptoContext):
        full_name = "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_scecond_n_{}_{}".format(n, he_res20_ctx.cur_num_slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, he_res20_ctx.cur_num_slots, False, cryptoContext)

    def mask_first_n(n, cur_limbs, he_res20_ctx, cryptoContext):
        full_name = "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_{}_{}".format(n, he_res20_ctx.cur_num_slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, he_res20_ctx.cur_num_slots, False, cryptoContext)

    def mask_from_to(from_, to, cur_limbs, he_res20_ctx, cryptoContext):
        full_name = "mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, he_res20_ctx.cur_num_slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_from_to_{}_{}_{}".format(from_, to, he_res20_ctx.cur_num_slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, he_res20_ctx.cur_num_slots, False, cryptoContext)

    def gen_mask(n, cur_limbs, he_res20_ctx, cryptoContext):
        full_name = "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "gen_mask_{}_{}".format(n, he_res20_ctx.cur_num_slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, he_res20_ctx.cur_num_slots, False, cryptoContext)

    def mask_first_n_mod(n, padding, pos, cur_limbs, cryptoContext):
        full_name = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, 16384*2)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, 16384*2, False, cryptoContext)

    def mask_first_n_mod2(n, padding, pos, cur_limbs, cryptoContext):
        full_name = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, 8192*2)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, 8192*2, False, cryptoContext)

    def mask_channel(n, cur_limbs, cryptoContext):
        full_name = "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384*2)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_channel_{}_{}".format(n, 16384*2)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, 16384*2, False, cryptoContext)

    def mask_channel2(n, cur_limbs, cryptoContext):
        full_name = "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192*2)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_channel2_{}_{}".format(n, 8192*2)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, 8192*2, False, cryptoContext)


else:

    def load_weight(encode_weight_path, cryptoContext):
        pass

    # raise ValueError("deprecated branch")
    def read_values_from_file(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        # print("read_values_from_file", filename, "level", level, "scale_deg", scale_deg, "slots", slots, "scale", scale)
        values = []
        val_name = filename
        filename = cryptoContext.weight_path + filename + '.bin'
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
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

        values = np.array(values, dtype=np.double)
        name = "{}_{}".format(val_name, slots)
        print(name)
        encoded = fhe.encode(values, name, level, slots, False, cryptoContext)
        # encoded.cv[0].cpu().numpy()
        # key = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
        # encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)

        return encoded

    def read_fc_weight(cryptoContext, level, scale_deg, slots):
        # print("read_values_from_file", "fc", "level", level, "scale_deg", scale_deg, "slots", slots, "scale", 1)
        values = []
        filename = cryptoContext.weight_path + 'fc.bin'
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
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
        weight_corrected = np.array(weight_corrected, dtype=np.double)
        name = "{}_{}".format("fc", slots)
        print(name)
        encoded = fhe.encode(weight_corrected, name, level, slots, False, cryptoContext)
        # key = "fc_{}_{}_{}".format(level, scale_deg, slots)
        # encoded_weight[key] = encoded
        # ptx = cryptoContext.pre_encoded[key]
        # check_encoded_equal(encoded, ptx, key)
        return encoded

    def mask_mod(n,cur_limbs,custom_val, he_res20_ctx, cryptoContext):
        # print("mask_mod", "n", n, "cur_limbs", cur_limbs, "custom_val", custom_val, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        level = cryptoContext.L-cur_limbs
        vec=[]
        for i in range(he_res20_ctx.cur_num_slots):
            if i%n==0:
                vec.append(custom_val)
            else:
                vec.append(0)
        vec = np.array(vec, dtype=np.double)
        name = "mask_mod_{}_{}_{}".format(n, custom_val, he_res20_ctx.cur_num_slots)
        print(name)
        encoded = fhe.encode(vec, name, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        # key = "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "mask_scecond_n_{}_{}".format(n, he_res20_ctx.cur_num_slots)
        print(name)
        encoded = fhe.encode(mask, name, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        # key = "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_{}_{}".format(n, he_res20_ctx.cur_num_slots)
        print(name)
        encoded = fhe.encode(mask, name, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        # key = "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        # encoded_weight[key] = encoded
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
        vec = np.array(vec, dtype=np.double)
        name = "mask_from_to_{}_{}_{}".format(from_, to, he_res20_ctx.cur_num_slots)
        print(name)
        encoded = fhe.encode(vec, name, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        # key = "mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, he_res20_ctx.cur_num_slots)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "gen_mask_{}_{}".format(n, he_res20_ctx.cur_num_slots)
        print(name)
        encoded = fhe.encode(mask, name, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
        # key = "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, 16384*2)
        print(name)
        encoded = fhe.encode(mask, name, level, 16384*2, False, cryptoContext)
        # key = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, 8192*2)
        print(name)
        encoded = fhe.encode(mask, name, level, 8192*2, False, cryptoContext)
        # key = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "mask_channel_{}_{}".format(n, 16384*2)
        print(name)
        encoded = fhe.encode(mask, name, level, 16384*2,False, cryptoContext)
        # key = "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384*2)
        # encoded_weight[key] = encoded
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
        mask = np.array(mask, dtype=np.double)
        name = "mask_channel2_{}_{}".format(n, 8192*2)
        print(name)
        encoded = fhe.encode(mask, name, level, 8192*2, False, cryptoContext)
        # key = "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192*2)
        # encoded_weight[key] = encoded
        return encoded


# else:
#     def load_weight(encode_weight_path, cryptoContext):
#         pass
#
#     # raise ValueError("deprecated branch")
#     def read_values_from_file(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
#         # print("read_values_from_file", filename, "level", level, "scale_deg", scale_deg, "slots", slots, "scale", scale)
#         values = []
#         val_name = filename
#         filename = DATA_DIR + '/weights/' + filename + '.bin'
#         if not os.path.isfile(filename):
#             print(f"Failed to open file: {filename}")
#             return values
#
#         try:
#             # 打开文件并逐行读取
#             with open(filename, 'r') as file:
#                 for row in file:
#                     # 按行解析
#                     for value in row.strip().split(','):
#                         try:
#                             num = float(value)
#                             values.append(num * scale)
#                         except ValueError:
#                             print(f"unconvert:: {value}")
#         except IOError as e:
#             print(f"error: {e}")
#
#         values = torch.tensor(values, dtype=torch.float64).cuda()
#         key = "{}_{}_{}_{}".format(val_name, level, scale_deg, slots)
#         encoded = fhe.encode(values, key, level, slots, False, cryptoContext)
#         encoded.cv[0].cpu().numpy()
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#
#         return encoded
#
#     def read_fc_weight(cryptoContext, level, scale_deg, slots):
#         # print("read_values_from_file", "fc", "level", level, "scale_deg", scale_deg, "slots", slots, "scale", 1)
#         values = []
#         filename = DATA_DIR + '/weights/fc.bin'
#         if not os.path.isfile(filename):
#             print(f"Failed to open file: {filename}")
#             return values
#
#         try:
#             # 打开文件并逐行读取
#             with open(filename, 'r') as file:
#                 for row in file:
#                     # 按行解析
#                     for value in row.strip().split(','):
#                         try:
#                             num = float(value)
#                             values.append(num)
#                         except ValueError:
#                             print(f"unconvert:: {value}")
#         except IOError as e:
#             print(f"error: {e}")
#
#         weight = values
#
#         weight_corrected=[]
#         for i in range(64):
#             for j in range(10):
#                 weight_corrected.append(weight[(10 * i) + j])
#             for j in range(64 - 10):
#                 weight_corrected.append(0)
#         weight_corrected = torch.tensor(weight_corrected, dtype=torch.float64).cuda()
#         encoded = fhe.encode(weight_corrected, scale_deg, level, slots, False, cryptoContext)
#         key = "fc_{}_{}_{}".format(level, scale_deg, slots)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_mod(n,cur_limbs,custom_val, he_res20_ctx, cryptoContext):
#         # print("mask_mod", "n", n, "cur_limbs", cur_limbs, "custom_val", custom_val, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
#         level = cryptoContext.L-cur_limbs
#         vec=[]
#         for i in range(he_res20_ctx.cur_num_slots):
#             if i%n==0:
#                 vec.append(custom_val)
#             else:
#                 vec.append(0)
#         vec = torch.tensor(vec, dtype=torch.float64).cuda()
#         encoded = fhe.encode(vec,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
#         key = "mask_mod_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_scecond_n(n, cur_limbs, he_res20_ctx, cryptoContext):
#         # print("mask_scecond_n", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
#         mask=[]
#         level=cryptoContext.L-cur_limbs
#         for i in range(he_res20_ctx.cur_num_slots):
#             if i >=n :
#                 mask.append(1)
#             else:
#                 mask.append(0)
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
#         key = "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_first_n(n, cur_limbs, he_res20_ctx, cryptoContext):
#         # print("mask_first_n", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
#         mask=[]
#         level=cryptoContext.L-cur_limbs
#         for i in range(he_res20_ctx.cur_num_slots):
#             if i < n:
#                 mask.append(1)
#             else:
#                 mask.append(0)
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask, 1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
#         key = "mask_first_n_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_from_to(from_, to, cur_limbs, he_res20_ctx, cryptoContext):
#         # print("mask_from_to", "from_", from_, "to", to, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
#         vec=[]
#         level=cryptoContext.L-cur_limbs
#         for i in range(he_res20_ctx.cur_num_slots):
#             if i>=from_ and i<to:
#                 vec.append(1)
#             else:
#                 vec.append(0)
#         vec = torch.tensor(vec, dtype=torch.float64).cuda()
#         encoded = fhe.encode(vec,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
#         key = "mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, he_res20_ctx.cur_num_slots)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def gen_mask(n,cur_limbs, he_res20_ctx, cryptoContext):
#         # print("gen_mask", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
#         level = cryptoContext.L - cur_limbs
#         mask=[]
#         copy_interval=n
#         for i in range(he_res20_ctx.cur_num_slots):
#             if copy_interval>0:
#                 mask.append(1)
#             else:
#                 mask.append(0)
#             copy_interval-=1
#             if copy_interval<= -n:
#                 copy_interval=n
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask,1, level, he_res20_ctx.cur_num_slots, False, cryptoContext)
#         key = "gen_mask_{}_{}_{}".format(n, cur_limbs, he_res20_ctx.cur_num_slots)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_first_n_mod(n,padding,pos,cur_limbs, cryptoContext):
#         # print("mask_first_n_mod", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
#         mask=[]
#         level = cryptoContext.L - cur_limbs
#         for i in range(32):
#             for j in range(pos*n):
#                 mask.append(0)
#             for j in range(n):
#                 mask.append(1)
#             for j in range(padding-n-(pos*n)):
#                 mask.append(0)
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask, 1, level, 16384*2, False, cryptoContext)
#         key = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_first_n_mod2(n,padding,pos,cur_limbs, cryptoContext):
#         # print("mask_first_n_mod2", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
#         mask=[]
#         level = cryptoContext.L - cur_limbs
#         for i in range(64):
#             for j in range(pos*n):
#                 mask.append(0)
#             for j in range(n):
#                 mask.append(1)
#             for j in range(padding-n-(pos*n)):
#                 mask.append(0)
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask, 1, level, 8192*2, False, cryptoContext)
#         key = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_channel(n,cur_limbs,cryptoContext):
#         # print("mask_channel", "n", n, "cur_limbs", cur_limbs)
#         mask=[]
#         level = cryptoContext.L - cur_limbs
#         for i in range(n):
#             for j in range(1024):
#                 mask.append(0)
#
#         for i in range(256):
#             mask.append(1)
#
#         for i in range(1024-256):
#             mask.append(0)
#         for i in range(31-n):
#             for j in range(1024):
#                 mask.append(0)
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask, 1, level, 16384*2, False, cryptoContext)
#         key = "mask_channel_{}_{}_{}".format(n, cur_limbs, 16384*2)
#         encoded_weight[key] = encoded
#         # ptx = cryptoContext.pre_encoded[key]
#         # check_encoded_equal(encoded, ptx, key)
#         return encoded
#
#     def mask_channel2(n,cur_limbs,cryptoContext):
#         # print("mask_channel2", "n", n, "cur_limbs", cur_limbs)
#         mask=[]
#         level = cryptoContext.L - cur_limbs
#         for i in range(n):
#             for j in range(256):
#                 mask.append(0)
#
#         for i in range(64):
#             mask.append(1)
#
#         for i in range(256-64):
#             mask.append(0)
#         for i in range(63-n):
#             for j in range(256):
#                 mask.append(0)
#         mask = torch.tensor(mask, dtype=torch.float64).cuda()
#         encoded = fhe.encode(mask, 1, level, 8192*2, False, cryptoContext)
#         key = "mask_channel2_{}_{}_{}".format(n, cur_limbs, 8192*2)
#         encoded_weight[key] = encoded
#         return encoded
#
# @atexit.register
# def save_encoded_weight():
#     if DIRECT_LOAD == False:
#         for key, value in encoded_weight.items():
#             encoded_weight[key].cv = [value.cv[0].cpu().numpy()]
#         with open(DATA_DIR + "/weight.pkl", "wb") as f:
#             pickle.dump(encoded_weight, f)

def rotsum(input,slots,cryptoContext):
    result=input.deep_copy()
    for i in range(log2_int(slots)):
        result=fhe.homo_add(result,fhe.homo_rotate(result,pow(2,i),cryptoContext),cryptoContext)
    return result


def rotsum_padded(input,slots,num_channel,cryptoContext):
    result=input.deep_copy()
    for i in range(log2_int(num_channel)):
        result=fhe.homo_add(result,fhe.homo_rotate(result,slots*pow(2,i),cryptoContext),cryptoContext)
    return result


def repeat(input,slots,cryptoContext):
    return fhe.homo_rotate(rotsum(input,slots,cryptoContext),-slots+1,cryptoContext)
