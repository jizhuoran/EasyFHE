import datetime
import os
import sys
import time

sys.path.append("/".join(os.getcwd().split("/")[:-5]))
sys.path.append("/".join(os.getcwd().split("/")[:-4]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
from termcolor import colored
import torch.fhe as fhe
from examples.resnet.src.convs import *
from examples.utils.utils import *
from examples.utils import approx

DATA_DIR = os.environ["DATA_DIR"]

# # config1
# total=1000
# SAVE_END=False
# SAVE_MIDDLE=False

# # config2
total = 10
SAVE_END = False
SAVE_MIDDLE = False

# # config3
# total=1
# SAVE_END=False
# SAVE_MIDDLE=True

#######################
#######################
weight_dir = "../weights/"

rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                     1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]

maxLevelsRemaining = 14
logBsSlots_list = [14]  # [14, 13, 12]
logN = 16
dnum = 3
dcrtBits = 59
firstMod = 60
levelBudget_list = [[4, 4]]  # [[4, 4], [4, 4], [4, 4]]
secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
device = "cuda"

relu_degree = 59

print("rotate_index_list: ", rotate_index_list)
print("maxLevelsRemaining: ", maxLevelsRemaining)
print("logBsSlots_list: ", logBsSlots_list)
print("logN: ", logN)
print("dnum: ", dnum)
print("dcrtBits: ", dcrtBits)
print("firstMod: ", firstMod)
print("levelBudget_list: ", levelBudget_list)
print("secretKeyDist: ", secretKeyDist)
print("rescaleTech: ", rescaleTech)
print("\n\n")


def homo_relu(ciphertext, scale, degree, cryptoContext):
    def scaled_relu_function(x):
        return 0 if x < 0 else (1 / scale) * x

    result = approx.eval_chebyshev_function(scaled_relu_function, ciphertext, -1, 1, degree, cryptoContext)
    return result


def initial_layer(input, cryptoContext):
    scale = normalized_deltas[0][0]
    res = conv_initial(input, 32, 1, 16, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, "conv1bn1-bias", cryptoContext.L - res.cur_limbs, 1, res.slots, scale)
    res = fhe.homo_add_pt(res, bias, cryptoContext)
    res = homo_relu(res, scale, relu_degree, cryptoContext)
    return res


def layer1(input, cryptoContext):
    scale = normalized_deltas[1][0]

    res1 = conv(input, 32, 1, 16, -1024, 1, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{1}-conv{1}bn{1}-bias", cryptoContext.L - res1.cur_limbs, 1,
                                 res1.slots, scale)
    res1 = fhe.homo_add_pt(res1, bias, cryptoContext)
    res1 = fhe.homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_relu(res1, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[1][1]
    res1 = conv(res1, 32, 1, 16, -1024, 1, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{1}-conv{2}bn{2}-bias", cryptoContext.L - res1.cur_limbs, 1,
                                 res1.slots, scale)
    res1 = fhe.homo_add_pt(res1, bias, cryptoContext)
    res1 = fhe.homo_add(
        res1, fhe.homo_mul_scalar_double(input, scale, cryptoContext), cryptoContext
    )
    res1 = fhe.homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_relu(res1, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[1][2]
    res2 = conv(res1, 32, 1, 16, -1024, 2, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{2}-conv{1}bn{1}-bias", cryptoContext.L - res2.cur_limbs, 1,
                                 res2.slots, scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_bootstrap(res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res2 = homo_relu(res2, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[1][3]
    res2 = conv(res2, 32, 1, 16, -1024, 2, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{2}-conv{2}bn{2}-bias", cryptoContext.L - res2.cur_limbs, 1,
                                 res2.slots, scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res2 = homo_relu(res2, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[1][4]
    res3 = conv(res2, 32, 1, 16, -1024, 3, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{3}-conv{1}bn{1}-bias", cryptoContext.L - res3.cur_limbs, 1,
                                 res3.slots, scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_bootstrap(res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res3 = homo_relu(res3, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[1][5]
    res3 = conv(res3, 32, 1, 16, -1024, 3, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{3}-conv{2}bn{2}-bias", cryptoContext.L - res3.cur_limbs, 1,
                                 res3.slots, scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_add(
        res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
    )
    res3 = fhe.homo_bootstrap(res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res3 = homo_relu(res3, scale, relu_degree, cryptoContext)

    return res3


def layer2(input, cryptoContext):
    scaleSx = normalized_deltas[2][0]
    scaleDx = normalized_deltas[2][1]
    boot_in = fhe.homo_bootstrap(
        input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext
    )

    res1sx0 = conv(boot_in, 32, 1, 16, -1024, 4, 1, 0, scaleSx, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{4}-conv{1}bn{1}-bias{1}", cryptoContext.L - res1sx0.cur_limbs,
                                 1, res1sx0.slots, scaleSx)
    res1sx0 = fhe.homo_add_pt(res1sx0, bias, cryptoContext)
    res1sx1 = conv(boot_in, 32, 1, 16, -1024, 4, 1, 16, scaleSx, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{4}-conv{1}bn{1}-bias{2}", cryptoContext.L - res1sx1.cur_limbs,
                                 1, res1sx1.slots, scaleSx)
    res1sx1 = fhe.homo_add_pt(res1sx1, bias, cryptoContext)

    res1dx0 = convbn_dx(boot_in, 16, -1024, 4, 1, 0, "1", scaleDx, cryptoContext)

    res1dx1 = convbn_dx(boot_in, 16, -1024, 4, 1, 16, "2", scaleDx, cryptoContext)

    fullpackSx = downsample1024to256(res1sx0, res1sx1, 16, 1, cryptoContext)
    fullpackDx = downsample1024to256(res1dx0, res1dx1, 16, 1, cryptoContext)

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    fullpackSx = homo_relu(fullpackSx, scaleSx, relu_degree, cryptoContext)
    fullpackSx = conv(fullpackSx, 16, 1, 32, -256, 4, 2, 0, scaleDx, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{4}-conv{2}bn{2}-bias", cryptoContext.L - fullpackSx.cur_limbs,
                                 1, fullpackSx.slots, scaleDx)
    fullpackSx = fhe.homo_add_pt(fullpackSx, bias, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res1 = homo_relu(res1, scaleDx, relu_degree, cryptoContext)

    scale = normalized_deltas[2][2]
    res2 = conv(res1, 16, 1, 32, -256, 5, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{5}-conv{1}bn{1}-bias", cryptoContext.L - res2.cur_limbs, 1,
                                 res1.slots, scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res2 = homo_relu(res2, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    res2 = conv(res2, 16, 1, 32, -256, 5, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{5}-conv{2}bn{2}-bias", cryptoContext.L - res2.cur_limbs, 1,
                                 res2.slots, scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res2 = homo_relu(res2, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[2][4]
    res3 = conv(res2, 16, 1, 32, -256, 6, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{6}-conv{1}bn{1}-bias", cryptoContext.L - res3.cur_limbs, 1,
                                 res2.slots, scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res3 = homo_relu(res3, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    res3 = conv(res3, 16, 1, 32, -256, 6, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{6}-conv{2}bn{2}-bias", cryptoContext.L - res3.cur_limbs, 1,
                                 res3.slots, scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_add(
        res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
    )
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res3 = homo_relu(res3, scale, relu_degree, cryptoContext)

    return res3


def layer3(input, cryptoContext):
    scaleSx = normalized_deltas[3][0]
    scaleDx = normalized_deltas[3][1]

    boot_in = fhe.homo_bootstrap(
        input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13

    res1sx0 = conv(boot_in, 16, 1, 32, -256, 7, 1, 0, scaleSx, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{7}-conv{1}bn{1}-bias{1}", cryptoContext.L - res1sx0.cur_limbs,
                                 1, res1sx0.slots, scaleSx)
    res1sx0 = fhe.homo_add_pt(res1sx0, bias, cryptoContext)
    res1sx1 = conv(boot_in, 16, 1, 32, -256, 7, 1, 32, scaleSx, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{7}-conv{1}bn{1}-bias{2}", cryptoContext.L - res1sx1.cur_limbs,
                                 1, res1sx1.slots, scaleSx)
    res1sx1 = fhe.homo_add_pt(res1sx1, bias, cryptoContext)

    res1dx0 = convbn_dx(boot_in, 32, -256, 7, 1, 0, "1", scaleDx, cryptoContext)

    res1dx1 = convbn_dx(boot_in, 32, -256, 7, 1, 32, "2", scaleDx, cryptoContext)

    fullpackSx = downsample256to64(res1sx0, res1sx1, 32, cryptoContext)
    fullpackDx = downsample256to64(res1dx0, res1dx1, 32, cryptoContext)

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    fullpackSx = homo_relu(fullpackSx, scaleSx, relu_degree, cryptoContext)
    fullpackSx = conv(fullpackSx, 8, 1, 64, -64, 7, 2, 0, scaleDx, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{7}-conv{2}bn{2}-bias", cryptoContext.L - fullpackSx.cur_limbs,
                                 1, fullpackSx.slots, scaleDx)
    fullpackSx = fhe.homo_add_pt(fullpackSx, bias, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    res1 = homo_relu(res1, scaleDx, relu_degree, cryptoContext)

    scale = normalized_deltas[3][2]
    res2 = conv(res1, 8, 1, 64, -64, 8, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{8}-conv{1}bn{1}-bias", cryptoContext.L - res2.cur_limbs, 1,
                                 res2.slots, scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    res2 = homo_relu(res2, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    res2 = conv(res2, 8, 1, 64, -64, 8, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-bias", cryptoContext.L - res2.cur_limbs, 1,
                                 res2.slots, scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    res2 = homo_relu(res2, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    res3 = conv(res2, 8, 1, 64, -64, 9, 1, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{9}-conv{1}bn{1}-bias", cryptoContext.L - res3.cur_limbs, 1,
                                 res3.slots, scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    res3 = homo_relu(res3, scale, relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    res3 = conv(res3, 8, 1, 64, -64, 9, 2, 0, scale, cryptoContext)
    bias = read_values_from_file(cryptoContext, f"layer{9}-conv{2}bn{2}-bias", cryptoContext.L - res3.cur_limbs, 1,
                                 res3.slots, scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_add(
        res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
    )
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    res3 = homo_relu(res3, scale, relu_degree, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 12
    return res3


def final_layer(input, cryptoContext):
    weight = read_fc_weight(
        64, 64, cryptoContext, cryptoContext.L - input.cur_limbs, 1, input.slots
    )
    res = rotsum(input, 64, cryptoContext)
    res = fhe.homo_mul_pt(
        res,
        mask_mod(64, res.cur_limbs, 1.0 / 64.0, res.slots, cryptoContext),
        cryptoContext,
    )
    res = repeat(res, 16, cryptoContext)

    res = fhe.homo_mul_pt(res, weight, cryptoContext)
    res = rotsum_padded(res, 64, 64, cryptoContext)

    bias = read_fc_bias(64, 16, cryptoContext, cryptoContext.L - res.cur_limbs, 1, res.slots)
    res = fhe.homo_add_pt(res, bias, cryptoContext)

    return res


def executeResNet20(cryptoContext):
    openfhe_context = cryptoContext.openfhe_context
    cryptoContext.zero_32K = openfhe_context.encrypt(np.zeros(2 ** 15), cryptoContext.device, 1, 0, 2 ** 15)
    cryptoContext.zero_16K = openfhe_context.encrypt(np.zeros(2 ** 14), cryptoContext.device, 1, 0, 2 ** 14)

    print("=====================================================")
    correct = 0
    for i in range(total):
        image_vector, label, _ = read_image(i)
        in_ct = openfhe_context.encrypt(
            image_vector,
            cryptoContext.device,
            1,
            cryptoContext.L - 11,
            16 * 32 * 32,
        )

        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()
        firstLayer = initial_layer(in_ct, cryptoContext)
        resLayer1 = layer1(firstLayer, cryptoContext)
        resLayer2 = layer2(resLayer1, cryptoContext)
        resLayer3 = layer3(resLayer2, cryptoContext)
        finalRes = final_layer(resLayer3, cryptoContext)
        print("after processing image ", i, "time: ", datetime.datetime.now())
        print("time: ", time.time() - start_time)
        try:
            clear_result = openfhe_context.decrypt(finalRes)
            clear_result = clear_result.cpu().numpy().reshape(-1)
            clear_result = clear_result[:10]
            print('clear_result', clear_result)
            max_element_idx = np.argmax(clear_result)
        except RuntimeError as e:
            print(f"Decryption failed: {e}")
            clear_result = None
            max_element_idx = 11

        print("For image ", i)
        # if clear_result is not None:
        #     print(clear_result)
        # else:
        #     print("Decryption failed, clear_result is None.")
        print("ground truth: ", label, "prediction: ", max_element_idx)
        if label == max_element_idx:
            correct += 1
        message = f"correct/total: {correct}/{(i + 1)}"
        print(colored(message, "red"))
        if (i + 1) % 100 == 0:
            print("\n\n")


def resnet20():
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    # max_relu_degree = relu_degree
    # maxLevelsRemaining = approx.get_relu_depth(max_relu_degree) + 3
    # if max_relu_degree < 59:
    #     diff = approx.get_relu_depth(59)-approx.get_relu_depth(max_relu_degree)
    #     maxLevelsRemaining +=diff

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True,
                                     SAVE_MIDDLE=SAVE_MIDDLE,
                                     SAVE_END=SAVE_END,
                                     )
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, device, save_dir=DATA_DIR,
                             config=config))

    cryptoContext.weight_path = weight_dir  # fixme: workaround only

    pkl_path = None
    if config.SAVE_MIDDLE == False:
        # cryptoContext.pre_encode_type = "middle"
        # file_name = ""
        # load_encode_pkl(file_name, weight_dir)
        # pkl_path = os.path.join(weight_dir, file_name + ".pkl")

        cryptoContext.pre_encode_type = "middle"
        pkl_path = "/data/yhh/data/encode_20250612_201417.pkl"

    load_weight(pkl_path, cryptoContext)

    print("start executeResNet20")
    cryptoContext.openfhe_context = openfhe_context
    executeResNet20(cryptoContext)


if __name__ == "__main__":
    resnet20()
