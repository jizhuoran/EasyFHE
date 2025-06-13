import torch
import pickle
import torch.fhe as fhe
import os
import numpy as np

DATA_DIR = os.environ["DATA_DIR"]

encoded_weight = {}

# for original res20 only
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


DIRECT_LOAD = True


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

    def read_fc_weight(num_channel,spatial_size, cryptoContext, level, scale_deg, slots):
        full_name = "fc_{}_{}_{}".format(level, scale_deg, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("fc", slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)

    def read_fc_bias(num_channel, spatial_size, cryptoContext, level, scale_deg, slots):
        full_name = "{}_{}_{}_{}".format("bias", level, scale_deg, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "{}_{}".format("bias", slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, level, slots, False, cryptoContext)

    def mask_mod(n, cur_limbs, custom_val, slots, cryptoContext):
        full_name = "mask_mod_{}_{}_{}".format(n, cur_limbs, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_mod_{}_{}_{}".format(n, custom_val, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_scecond_n(n, cur_limbs, slots, cryptoContext):
        full_name = "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_scecond_n_{}_{}".format(n, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_first_n(n, cur_limbs, slots, cryptoContext):
        full_name = "mask_first_n_{}_{}_{}".format(n, cur_limbs, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_{}_{}".format(n, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_from_to(from_, to, cur_limbs, slots, cryptoContext):
        full_name = "mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_from_to_{}_{}_{}".format(from_, to, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def gen_mask(n, cur_limbs, slots, cryptoContext):
        full_name = "gen_mask_{}_{}_{}".format(n, cur_limbs, slots)
        if cryptoContext.pre_encode_type == "middle":
            name = "gen_mask_{}_{}".format(n, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_first_n_mod(n, padding, pos, length, cur_limbs, slots, cryptoContext):
        full_name = "mask_first_n_mod_{}_{}_{}_{}_{}".format(n, padding, pos, slots,cur_limbs)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_first_n_mod2(n, padding, pos, length, cur_limbs, slots, cryptoContext):
        full_name = "mask_first_n_mod2_{}_{}_{}_{}_{}".format(n, padding, pos, slots,cur_limbs)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_first_n_mod3(n, padding, pos, length, cur_limbs, slots, cryptoContext):
        full_name = "mask_first_n_mod3_{}_{}_{}_{}".format(n, padding, pos, slots,cur_limbs)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_first_n_mod3_{}_{}_{}_{}".format(n, padding, pos, slots)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, slots, False, cryptoContext)

    def mask_channel(n, num_channel, spatial_size, num_cipher, cur_limbs, cryptoContext):
        channel_per_cipher = num_channel // num_cipher
        full_name = "mask_channel_{}_{}_{}_{}".format(n, channel_per_cipher, spatial_size, cur_limbs)
        if cryptoContext.pre_encode_type == "middle":
            name = "mask_channel_{}_{}_{}".format(n, channel_per_cipher, spatial_size)
        else:
            name = full_name
        return fhe.encode(cryptoContext.pre_encoded[name], full_name, cryptoContext.L - cur_limbs, 2 * channel_per_cipher*spatial_size, False, cryptoContext)
else:

    def load_weight(encode_weight_path, cryptoContext):
        pass


    def read_values_from_file(cryptoContext, filename, level, scale_deg, slots, scale=1.0):
        # print("read_values_from_file", filename, "level", level, "scale_deg", scale_deg, "slots", slots, "scale", scale)
        values = []
        val_name = filename
        filename = cryptoContext.weight_path + filename + '.bin'
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values

        try:
            with open(filename, 'r') as file:
                for row in file:
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
        return encoded

    def read_fc_weight(num_channel,spatial_size, cryptoContext, level, scale_deg, slots):
        # print("read_values_from_file", "fc", "level", level, "scale_deg", scale_deg, "slots", slots, "scale", 1)
        values = []
        filename = cryptoContext.weight_path + 'fc.bin'
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values

        try:
            with open(filename, 'r') as file:
                for row in file:
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
        #  i is channels number
        for i in range(num_channel):
            # First j is output channels,in cifar10 j is 10,in cifar100 j is 100
            for j in range(10):
                weight_corrected.append(weight[(10 * i) + j])
            # This j is wide * high - out_channel ,so resnet18 is 4*4-10
            for j in range(spatial_size - 10):
                weight_corrected.append(0)
        weight_corrected = np.array(weight_corrected, dtype=np.double)
        name = "{}_{}".format("fc", slots)
        print(name)
        encoded = fhe.encode(weight_corrected, name, level, slots, False, cryptoContext)
        return encoded


    def read_fc_bias(num_channel, spatial_size, cryptoContext, level, scale_deg, slots):
        values = []
        filename = cryptoContext.weight_path + 'bias.bin'
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values
        try:
            with open(filename, 'r') as file:
                for row in file:
                    for value in row.strip().split(','):
                        try:
                            num = float(value)
                            values.append(num)
                        except ValueError:
                            print(f"unconvert:: {value}")
        except IOError as e:
            print(f"error: {e}")

        bias = values

        bias_corrected = []
        #  i is channels number
        for i in range(num_channel):
            # First j is output channels,in cifar10 j is 10,in cifar100 j is 100
            for j in range(10):
                bias_corrected.append(bias[j])
            # This j is wide * high - out_channel ,so resnet18 is 4*4-10
            for j in range(spatial_size - 10):
                bias_corrected.append(0)
        bias_corrected = np.array(bias_corrected, dtype=np.double)
        name = "{}_{}".format("bias", slots)
        print(name)
        encoded = fhe.encode(bias_corrected, name, level, slots, False, cryptoContext)
        return encoded

    def mask_mod(n,cur_limbs,custom_val, slots, cryptoContext):
        # print("mask_mod", "n", n, "cur_limbs", cur_limbs, "custom_val", custom_val, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        level = cryptoContext.L-cur_limbs
        vec=[]
        for i in range(slots):
            if i%n==0:
                vec.append(custom_val)
            else:
                vec.append(0)
        vec = np.array(vec, dtype=np.double)
        name = "mask_mod_{}_{}_{}".format(n, custom_val, slots)
        print(name)
        encoded = fhe.encode(vec, name, level, slots, False, cryptoContext)
        return encoded

    def mask_scecond_n(n, cur_limbs, slots, cryptoContext):
        # print("mask_scecond_n", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        mask=[]
        level=cryptoContext.L-cur_limbs
        for i in range(slots):
            if i >=n :
                mask.append(1)
            else:
                mask.append(0)
        mask = np.array(mask, dtype=np.double)
        name = "mask_scecond_n_{}_{}".format(n, slots)
        print(name)
        encoded = fhe.encode(mask, name, level, slots, False, cryptoContext)
        return encoded

    def mask_first_n(n, cur_limbs, slots, cryptoContext):
        # print("mask_first_n", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        mask=[]
        level=cryptoContext.L-cur_limbs
        for i in range(slots):
            if i < n:
                mask.append(1)
            else:
                mask.append(0)
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_{}_{}".format(n, slots)
        print(name)
        encoded = fhe.encode(mask, name, level, slots, False, cryptoContext)
        return encoded

    def mask_from_to(from_, to, cur_limbs, slots, cryptoContext):
        # print("mask_from_to", "from_", from_, "to", to, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        vec=[]
        level=cryptoContext.L-cur_limbs
        for i in range(slots):
            if i>=from_ and i<to:
                vec.append(1)
            else:
                vec.append(0)
        vec = np.array(vec, dtype=np.double)
        name = "mask_from_to_{}_{}_{}".format(from_, to, slots)
        print(name)
        encoded = fhe.encode(vec, name, level, slots, False, cryptoContext)
        return encoded

    def gen_mask(n,cur_limbs, slots, cryptoContext):
        # print("gen_mask", "n", n, "cur_limbs", cur_limbs, "he_res20_ctx.cur_num_slots", he_res20_ctx.cur_num_slots)
        level = cryptoContext.L - cur_limbs
        mask=[]
        copy_interval=n
        for i in range(slots):
            if copy_interval>0:
                mask.append(1)
            else:
                mask.append(0)
            copy_interval-=1
            if copy_interval<= -n:
                copy_interval=n
        mask = np.array(mask, dtype=np.double)
        name = "gen_mask_{}_{}".format(n, slots)
        print(name)
        encoded = fhe.encode(mask, name, level, slots, False, cryptoContext)
        return encoded

    def mask_first_n_mod(n,padding,pos,length,cur_limbs, slots, cryptoContext):
        # print("mask_first_n_mod", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(length):
            for j in range(pos*n):
                mask.append(0)
            for j in range(n):
                mask.append(1)
            for j in range(padding-n-(pos*n)):
                mask.append(0)
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, slots)
        print(name)
        encoded = fhe.encode(mask, name, level, slots, False, cryptoContext)
        return encoded



    def mask_first_n_mod2(n,padding,pos,length, cur_limbs, slots, cryptoContext):
        # print("mask_first_n_mod2", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(length):
            for j in range(pos*n):
                mask.append(0)
            for j in range(n):
                mask.append(1)
            for j in range(padding-n-(pos*n)):
                mask.append(0)
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, slots)
        print(name)
        encoded = fhe.encode(mask, name, level, slots, False, cryptoContext)
        return encoded

    def mask_first_n_mod3(n,padding,pos,length, cur_limbs, slots, cryptoContext):
        # print("mask_first_n_mod3", "n", n, "padding", padding, "pos", pos, "cur_limbs", cur_limbs)
        mask=[]
        level = cryptoContext.L - cur_limbs
        for i in range(length):
            for j in range(pos*n):
                mask.append(0)
            for j in range(n):
                mask.append(1)
            for j in range(padding-n-(pos*n)):
                mask.append(0)
        mask = np.array(mask, dtype=np.double)
        name = "mask_first_n_mod3_{}_{}_{}_{}".format(n, padding, pos, slots)
        print(name)
        encoded = fhe.encode(mask, name, level, slots, False, cryptoContext)
        return encoded


    def mask_channel(n, num_channel, spatial_size, num_cipher, cur_limbs, cryptoContext):
        # print("mask_channel", "n", n, "cur_limbs", cur_limbs)
        # num_cipher means the total number of cipher involved in downsample computation, e.g. split data of size 2^17 into two 2^16 leads to two ciphers.
        mask = []
        level = cryptoContext.L - cur_limbs

        channel_per_cipher = num_channel // num_cipher
        for i in range(n):
            for j in range(spatial_size):
                mask.append(0)

        for i in range(spatial_size // 4):
            mask.append(1)

        for i in range(spatial_size - spatial_size // 4):
            mask.append(0)
        for i in range(2 * channel_per_cipher  - 1 - n):
            for j in range(spatial_size):
                mask.append(0)
        mask = np.array(mask, dtype=np.double)
        name = "mask_channel_{}_{}_{}".format(n, channel_per_cipher, spatial_size)
        print(name)
        encoded = fhe.encode(mask, name, level, 2 * channel_per_cipher * spatial_size, False, cryptoContext)
        return encoded


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
