import os, sys
import numpy as np
import pickle

sys.path.append("/".join(os.getcwd().split("/")[:-3]))

from torch.fhe import utils
from torch.fhe.ciphertext import Plaintext


def read_values_from_file(filename):
    values = []

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

    return values

def read_fc_weight(filename):
    weight=read_values_from_file(filename)
    weight_corrected=[]
    for i in range(64):
        for j in range(10):
            weight_corrected.append(weight[(10 * i) + j])
        for j in range(64 - 10):
            weight_corrected.append(0)
    return weight_corrected


def mask_mod(n,cur_limbs,custom_val, cur_num_slots, cryptoContext, openfhe_context):
    level = cryptoContext.L-cur_limbs
    vec=[]
    for i in range(cur_num_slots):
        if i%n==0:
            vec.append(custom_val)
        else:
            vec.append(0)

    return vec


def mask_scecond_n(n, cur_limbs, cur_num_slots, cryptoContext, openfhe_context):
    mask=[]
    level=cryptoContext.L-cur_limbs
    for i in range(cur_num_slots):
        if i >=n :
            mask.append(1)
        else:
            mask.append(0)
    return mask


def mask_first_n(n, cur_limbs, cur_num_slots, cryptoContext, openfhe_context):
    mask=[]
    level=cryptoContext.L-cur_limbs
    for i in range(cur_num_slots):
        if i < n:
            mask.append(1)
        else:
            mask.append(0)

    return mask


def mask_from_to(from_, to, cur_limbs, cur_num_slots, cryptoContext, openfhe_context):
    vec=[]
    level=cryptoContext.L-cur_limbs
    for i in range(cur_num_slots):
        if i>=from_ and i<to:
            vec.append(1)
        else:
            vec.append(0)

    return vec


def gen_mask(n,cur_limbs, cur_num_slots, cryptoContext, openfhe_context):
    level = cryptoContext.L - cur_limbs
    mask=[]
    copy_interval=n
    for i in range(cur_num_slots):
        if copy_interval>0:
            mask.append(1)
        else:
            mask.append(0)
        copy_interval-=1
        if copy_interval<= -n:
            copy_interval=n
    return mask


def mask_first_n_mod(n,padding,pos,cur_limbs, cryptoContext, openfhe_context):
    mask=[]
    level = cryptoContext.L - cur_limbs
    for i in range(32):
        for j in range(pos*n):
            mask.append(0)
        for j in range(n):
            mask.append(1)
        for j in range(padding-n-(pos*n)):
            mask.append(0)

    return mask


def mask_first_n_mod2(n,padding,pos,cur_limbs, cryptoContext, openfhe_context):
    mask=[]
    level = cryptoContext.L - cur_limbs
    for i in range(64):
        for j in range(pos*n):
            mask.append(0)
        for j in range(n):
            mask.append(1)
        for j in range(padding-n-(pos*n)):
            mask.append(0)
    return mask


def mask_channel(n,cur_limbs,cryptoContext, openfhe_context):
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
    return mask


def mask_channel2(n,cur_limbs,cryptoContext, openfhe_context):
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
    return mask


def get_relu_depth(degree):
    ranges = [
        (1, 5, 3),
        (6, 13, 4),
        (14, 27, 5),
        (28, 59, 6),
        (60, 119, 7),
        (120, 247, 8),
        (248, 495, 9),
        (496, 1007, 10),
        (1008, 2031, 11)
    ]

    for lower, upper, depth in ranges:
        if lower <= degree <= upper:
            return depth

    raise ValueError("Set a valid degree for ReLU")


def pre_encode(val, openfhe_context, level, scale_deg, slots):
    encode_val = openfhe_context.encode(val, level, scale_deg, slots)
    assert isinstance(encode_val, Plaintext)
    encode_val.mx = [encode_val.mx[0].cpu().numpy()]
    return encode_val

#glob file in weights

def gen_pre_encode_file(cryptoContext, openfhe_context):

    if cryptoContext is None and openfhe_context is None:

        logN = 16
        logSlots_list = [12, 13, 14]
        levelBudget_list = [[4, 4], [4, 4], [4, 4]]
        dnum = 3
        dcrtBits = 59
        firstMod = 60
        max_relu_degree = 59
        secretKeyDist = "UNIFORM_TERNARY"
        rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir = "/data/yhh/data/"

            # generate context
        approxModDepth = 9
        maxLevelsRemaining = get_relu_depth(max_relu_degree) + 3
        if max_relu_degree < 59:
            diff = get_relu_depth(59)-get_relu_depth(max_relu_degree)
            maxLevelsRemaining +=diff


        rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                                1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]
        
        cryptoContext, openfhe_context_dict = (
            utils.try_load_context(logN,
                                    logSlots_list,
                                    maxLevelsRemaining,
                                    levelBudget_list,
                                    dnum,
                                    dcrtBits,
                                    firstMod,
                                    approxModDepth,
                                    rotate_index_list,
                                    secretKeyDist,
                                    rescaleTech,
                                    save_dir=save_dir))


        openfhe_context = openfhe_context_dict[str(14)]
        cryptoContext.weight_dir = "/data/yhh/data/"


    weight_map = {}
    path = cryptoContext.weight_dir + "/weights/"
    for weight_file in os.listdir(path):
        assert weight_file.endswith(".bin")
        weight_name = weight_file.split('.')[0]
        if weight_name == "fc":
            value = read_fc_weight(path + weight_file)
        else:
            value = read_values_from_file(path + weight_file)
        weight_map[weight_name] = np.array(value)

    encode_val = {}

    with open(cryptoContext.weight_dir + "/yhh_exec_log.txt", 'r') as f:
        commands = f.readlines()

    print("exec_log loaded")

    for i in range(0, len(commands), 2):
        command = commands[i][:-1]
        encode_command = commands[i+1][:-1]
        # print(command)
        # print(encode_command)

        if "mask_mod" in command:
            n = int(command.split(" ")[2])
            cur_limbs = int(command.split(" ")[4])
            custom_val = float(command.split(" ")[6])
            slots = int(encode_command.split(" ")[-1])
            val = mask_mod(n, cur_limbs, custom_val, slots, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_mod_{}_{}_{}".format(n, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_mod_{}_{}_{}".format(n, cur_limbs, slots)] = encoded

        elif "mask_scecond_n" in command:
            n = int(command.split(" ")[2])
            cur_limbs = int(command.split(" ")[4])
            slots = int(encode_command.split(" ")[-1])
            val = mask_scecond_n(n, cur_limbs, slots, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_scecond_n_{}_{}_{}".format(n, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_scecond_n_{}_{}_{}".format(n, cur_limbs, slots)] = encoded

        elif "mask_first_n n" in command:
            n = int(command.split(" ")[2])
            cur_limbs = int(command.split(" ")[4])
            slots = int(encode_command.split(" ")[-1])
            val = mask_first_n(n, cur_limbs, slots, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_first_n_{}_{}_{}".format(n, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_first_n_{}_{}_{}".format(n, cur_limbs, slots)] = encoded
        
        elif "mask_from_to" in command:
            from_ = int(command.split(" ")[2])
            to = int(command.split(" ")[4])
            cur_limbs = int(command.split(" ")[6])
            slots = int(encode_command.split(" ")[-1])
            val = mask_from_to(from_, to, cur_limbs, slots, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_from_to_{}_{}_{}_{}".format(from_, to, cur_limbs, slots)] = encoded
        
        elif "gen_mask" in command:
            n = int(command.split(" ")[2])
            cur_limbs = int(command.split(" ")[4])
            slots = int(encode_command.split(" ")[-1])
            val = gen_mask(n, cur_limbs, slots, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "gen_mask_{}_{}_{}".format(n, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["gen_mask_{}_{}_{}".format(n, cur_limbs, slots)] = encoded
        
        elif "mask_first_n_mod n" in command:
            n = int(command.split(" ")[2])
            padding = int(command.split(" ")[4])
            pos = int(command.split(" ")[6])
            cur_limbs = int(command.split(" ")[8])
            slots = int(encode_command.split(" ")[-1])
            val = mask_first_n_mod(n, padding, pos, cur_limbs, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_first_n_mod_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)] = encoded
        
        elif "mask_first_n_mod2 n" in command:
            n = int(command.split(" ")[2])
            padding = int(command.split(" ")[4])
            pos = int(command.split(" ")[6])
            cur_limbs = int(command.split(" ")[8])
            slots = int(encode_command.split(" ")[-1])
            val = mask_first_n_mod2(n, padding, pos, cur_limbs, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_first_n_mod2_{}_{}_{}_{}".format(n, padding, pos, cur_limbs)] = encoded
        
        elif "mask_channel n" in command:
            n = int(command.split(" ")[2])
            cur_limbs = int(command.split(" ")[4])
            slots = int(encode_command.split(" ")[-1])
            val = mask_channel(n, cur_limbs, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_channel_{}_{}_{}".format(n, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_channel_{}_{}_{}".format(n, cur_limbs, slots)] = encoded
        
        elif "mask_channel2 n" in command:
            n = int(command.split(" ")[2])
            cur_limbs = int(command.split(" ")[4])
            slots = int(encode_command.split(" ")[-1])
            val = mask_channel2(n, cur_limbs, cryptoContext, openfhe_context)
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            # key = "mask_channel2_{}_{}_{}".format(n, cur_limbs, slots)
            # if key in encode_val:
                # print("Already encoded")
                # print(key)
                # print(command)
                # print(encode_command)
                # continue
            encode_val["mask_channel2_{}_{}_{}".format(n, cur_limbs, slots)] = encoded
        
        elif "read_values_from_file" in command:
            val = weight_map[command.split(" ")[1]]
            if command.split(" ")[1] == "fc":
                print("I am fc")
                print("I am fc")
                print("I am fc")
                val = val
            else:
                scale = float(command.split(" ")[-1])
                val = val * scale
            level = int(encode_command.split(" ")[2])
            scale_deg = int(encode_command.split(" ")[4])
            slots = int(encode_command.split(" ")[-1])
            encoded = pre_encode(val, openfhe_context, level, scale_deg, slots)
            encode_val[command.split(" ")[1]+"_{}_{}_{}".format(level, scale_deg, slots)] = encoded
        
        else:
            print("Invalid command")
            break
            
            
        
    with open(cryptoContext.weight_dir + "/encode_val.pkl", "wb") as f:
        pickle.dump(encode_val, f)


    # if weight_file.endswith(".bin") and "GPU-FHE-CONTEXT" in weight_file:
    #     print("Testing", weight_file)
    #     weight_file = weight_file.replace("_UNIFORM_TERNARY_", "_")
    #     logN, logSlots_str, maxLevelsRemaining, levelBudgets_str, dnum, dcrtBits, firstMod, approxModDepth, rescaleTech = weight_file[:-4].split("_")[1:]
    #     try:
    #         logSlots_list = [int(logSlots) for logSlots in logSlots_str.split("-")]
    #         levelBudgets_list = []
    #         for levelBudgets in range(len(levelBudgets_str.split("-")) // 2):
    #             levelBudgets_list.append([int(levelBudgets_str.split("-")[2 * levelBudgets]), int(levelBudgets_str.split("-")[2 * levelBudgets + 1])])
    #         code_string = """


if __name__ == "__main__":
    gen_pre_encode_file(None, None)
