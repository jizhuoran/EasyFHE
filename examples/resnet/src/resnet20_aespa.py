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

# for debug
from examples.resnet.gen_aespa_weights.HerPN import get_resnet20_HerPN, change_all_HerPN_by_PAF_MutalChannel

DATA_DIR = os.environ["DATA_DIR"]

# # config1
# total=1000
# SAVE_END=False
# SAVE_MIDDLE=False
# pre_encode_type = "middle"
# pkl_path = "/data/yhh/data/encode_20250612_171807.pkl"

# # config2
total = 10
SAVE_END = False
SAVE_MIDDLE = False
pre_encode_type = "middle"
pkl_path = "/data/yhh/data/encode_20250613_155340.pkl"

# # config3
# total=1
# SAVE_END=False
# SAVE_MIDDLE=True
# pre_encode_type = None
# pkl_path = None

#######################
#######################
weight_dir = "../weights_aespa_20/"

rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                     1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]
maxLevelsRemaining = 11
logBsSlots_list = [14]
logN = 16
dnum = 3
dcrtBits = 56
firstMod = 60
levelBudget_list = [[4, 4]]
secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
device = "cuda"
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


def initial_layer(input, cryptoContext):
    scale = 1  # normalized_deltas[0][0]
    res = conv_initial(input, 32, 1, 16, scale, cryptoContext)
    res = fhe.homo_rescale(res, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res = homo_Aespa_perfect_square(res, "conv1bn1", cryptoContext)
    return res


def layer1(input, cryptoContext):
    scale = 1  # normalized_deltas[1][0]
    # layer[0],block[0],conv1
    res1 = conv(input, 32, 1, 16, -1024, 1, 1, 0, scale, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{1}-conv{1}bn{1}", cryptoContext)

    # layer[0],block[0],conv2 and shorcut
    scale = 1  # normalized_deltas[1][1]
    # res1 = a1*x,shortcut = input = y
    res1 = conv(res1, 32, 1, 16, -1024, 1, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        input = fhe.drop_last_elements(input, input.cur_limbs - res1.cur_limbs,
                                       cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(f"layer{1}-conv{2}bn{2}-A2", cryptoContext.L - input.cur_limbs, 1, input.slots,
                               cryptoContext, scale)
    A2y = fhe.homo_mul_pt(input, A2, cryptoContext)
    res1 = fhe.homo_add(res1, A2y, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{1}-conv{2}bn{2}", cryptoContext)

    scale = 1  # normalized_deltas[1][2]
    # layer[0],block[1],conv1
    res2 = conv(res1, 32, 1, 16, -1024, 2, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{2}-conv{1}bn{1}", cryptoContext)

    # layer[0],block[1],conv2 and shorcut
    scale = 1  # normalized_deltas[1][3]
    res2 = conv(res2, 32, 1, 16, -1024, 2, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(f"layer{2}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1, res1.slots,
                               cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{2}-conv{2}bn{2}", cryptoContext)

    # layer[0],block[2],conv1
    scale = 1  # normalized_deltas[1][4]
    res3 = conv(res2, 32, 1, 16, -1024, 3, 1, 0, scale, cryptoContext)
    res3 = fhe.homo_rescale(res3, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{3}-conv{1}bn{1}", cryptoContext)

    scale = 1  # normalized_deltas[1][5]
    res3 = conv(res3, 32, 1, 16, -1024, 3, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res2 = fhe.drop_last_elements(res2, res2.cur_limbs - res3.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(f"layer{3}-conv{2}bn{2}-A2", cryptoContext.L - res2.cur_limbs, 1, res2.slots,
                               cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res2, A2, cryptoContext)
    res3 = fhe.homo_add(res3, A2y, cryptoContext)
    res3 = fhe.homo_rescale(res3, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{3}-conv{2}bn{2}", cryptoContext)

    return res3


def layer2(input, cryptoContext):
    scaleSx = 1  # normalized_deltas[2][0]
    scaleDx = 1  # normalized_deltas[2][1]
    boot_in = fhe.homo_bootstrap(input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1sx0 = conv(boot_in, 32, 1, 16, -1024, 4, 1, 0, scaleSx, cryptoContext)
    res1sx1 = conv(boot_in, 32, 1, 16, -1024, 4, 1, 16, scaleSx, cryptoContext)
    res1sx0 = fhe.homo_rescale(res1sx0, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.homo_rescale(res1sx1, 1, cryptoContext)  # RESCALE ADD BY ZRJI

    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.drop_last_elements(boot_in, 2, cryptoContext)  # RESCALE ADD BY ZRJI
    res1dx0 = convbn_dx(boot_in, 16, -1024, 4, 1, 0, "1", scaleDx, cryptoContext)

    res1dx1 = convbn_dx(boot_in, 16, -1024, 4, 1, 16, "2", scaleDx, cryptoContext)

    fullpackSx = downsample1024to256(res1sx0, res1sx1, 16, 1, cryptoContext)
    fullpackDx = downsample1024to256(res1dx0, res1dx1, 16, 1, cryptoContext)
    fullpackSx = fhe.homo_rescale(fullpackSx, 1, cryptoContext)  # RESCALE ADD BY ZRJI

    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{4}-conv{1}bn{1}", cryptoContext)

    fullpackSx = conv(fullpackSx, 16, 1, 32, -256, 4, 2, 0, scaleDx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_Aespa_perfect_square(res1, f"layer{4}-conv{2}bn{2}", cryptoContext)

    # layer[2]block[1]
    scale = 1  # normalized_deltas[2][2]
    res2 = conv(res1, 16, 1, 32, -256, 5, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{5}-conv{1}bn{1}", cryptoContext)

    scale = 1  # normalized_deltas[2][3]
    res2 = conv(res2, 16, 1, 32, -256, 5, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(f"layer{5}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1, res1.slots,
                               cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{5}-conv{2}bn{2}", cryptoContext)

    # layer[2]block[2]
    scale = 1  # normalized_deltas[2][4]
    res3 = conv(res2, 16, 1, 32, -256, 6, 1, 0, scale, cryptoContext)
    res3 = fhe.homo_rescale(res3, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{6}-conv{1}bn{1}", cryptoContext)

    scale = 1  # normalized_deltas[2][5]
    res3 = conv(res3, 16, 1, 32, -256, 6, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res2 = fhe.drop_last_elements(res2, res2.cur_limbs - res3.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(f"layer{6}-conv{2}bn{2}-A2", cryptoContext.L - res2.cur_limbs, 1, res2.slots,
                               cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res2, A2, cryptoContext)
    res3 = fhe.homo_add(res3, A2y, cryptoContext)
    res3 = fhe.homo_rescale(res3, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{6}-conv{2}bn{2}", cryptoContext)

    return res3


def layer3(input, cryptoContext):
    scaleSx = 1  # normalized_deltas[3][0]
    scaleDx = 1  # normalized_deltas[3][1]

    boot_in = fhe.homo_bootstrap(input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res1sx0 = conv(boot_in, 16, 1, 32, -256, 7, 1, 0, scaleSx, cryptoContext)
    res1sx1 = conv(boot_in, 16, 1, 32, -256, 7, 1, 32, scaleSx, cryptoContext)
    res1sx0 = fhe.homo_rescale(res1sx0, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.homo_rescale(res1sx1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.drop_last_elements(boot_in, 2, cryptoContext)  # drop_last_elements ADD BY ZRJI
    res1dx0 = convbn_dx(boot_in, 32, -256, 7, 1, 0, "1", scaleDx, cryptoContext)

    res1dx1 = convbn_dx(boot_in, 32, -256, 7, 1, 32, "2", scaleDx, cryptoContext)

    fullpackSx = downsample256to64(res1sx0, res1sx1, 32, cryptoContext)
    fullpackDx = downsample256to64(res1dx0, res1dx1, 32, cryptoContext)
    fullpackSx = fhe.homo_rescale(fullpackSx, 1, cryptoContext)  # RESCALE ADD BY ZRJI

    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{7}-conv{1}bn{1}", cryptoContext)

    fullpackSx = conv(fullpackSx, 8, 1, 64, -64, 7, 2, 0, scaleDx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{7}-conv{2}bn{2}", cryptoContext)
    res1 = fhe.homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    scale = 1  # normalized_deltas[3][2]
    res2 = conv(res1, 8, 1, 64, -64, 8, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{8}-conv{1}bn{1}", cryptoContext)

    scale = 1  # normalized_deltas[3][3]
    res2 = conv(res2, 8, 1, 64, -64, 8, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(f"layer{8}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1, res1.slots,
                               cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{8}-conv{2}bn{2}", cryptoContext)

    scale = 1  # normalized_deltas[3][4]
    res3 = conv(res2, 8, 1, 64, -64, 9, 1, 0, scale, cryptoContext)
    res3 = fhe.homo_rescale(res3, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{9}-conv{1}bn{1}", cryptoContext)

    scale = 1  # normalized_deltas[3][5]
    res3 = conv(res3, 8, 1, 64, -64, 9, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res2 = fhe.drop_last_elements(res2, res2.cur_limbs - res3.cur_limbs, cryptoContext)
    A2 = read_values_from_file(f"layer{9}-conv{2}bn{2}-A2", cryptoContext.L - res2.cur_limbs, 1, res2.slots,
                               cryptoContext, scale)  # drop_last_elements ADD BY ZRJI
    A2y = fhe.homo_mul_pt(res2, A2, cryptoContext)
    res3 = fhe.homo_add(res3, A2y, cryptoContext)
    res3 = fhe.homo_rescale(res3, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{9}-conv{2}bn{2}", cryptoContext)

    return res3


def final_layer(input, cryptoContext):
    # 64*8*8
    res = rotsum(input, 64, cryptoContext)
    res = fhe.homo_mul_pt(
        res,
        mask_mod(64, 1.0 / 64.0, res.cur_limbs, res.slots, cryptoContext),
        cryptoContext,
    )
    res = repeat(res, 16, cryptoContext)
    res = fhe.homo_rescale(res, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    weight = read_fc_weight(64, 64, cryptoContext.L - res.cur_limbs, 1, res.slots, cryptoContext)
    res = fhe.homo_mul_pt(res, weight, cryptoContext)
    res = rotsum_padded(res, 64, 64, cryptoContext)

    bias = read_fc_bias(64, 16, cryptoContext.L - res.cur_limbs, 1, res.slots, cryptoContext)
    res = fhe.homo_add_pt(res, bias, cryptoContext)
    return res


def executeResNet20(cryptoContext):
    openfhe_context = cryptoContext.openfhe_context
    cryptoContext.zero_32K = openfhe_context.encrypt(np.zeros(2 ** 15), cryptoContext.device, 2, 0, 2 ** 15)
    cryptoContext.zero_16K = openfhe_context.encrypt(np.zeros(2 ** 14), cryptoContext.device, 2, 0, 2 ** 14)

    # # 准备明文模型，测速时可以删除
    # model = get_resnet20_HerPN(num_classes=10)
    # device = torch.device("cuda:0")
    # model.to(device)
    # model_path = '/home/yhh/PNP/GPU-FHE/examples/resnet20/gen_aespa_weights/ResNet20_Aespa.pth'
    # stict = torch.load(model_path, map_location='cuda:0')
    # model.load_state_dict(stict, strict=False)
    # model.eval()
    # model = change_all_HerPN_by_PAF_MutalChannel(model)

    print("=====================================================")
    correct = 0
    for i in range(total):
        image_vector, label, index = read_image(i)
        # # 明文模型输出
        # input = torch.tensor(image_vector, device="cuda",dtype=torch.float32)
        # input = torch.stack([input[i * 1024: (i + 1) * 1024].view(32, 32) for i in range(3)], dim=0)
        # x , fea = model(input,fea_out=True)

        in_ct = openfhe_context.encrypt(
            image_vector,
            cryptoContext.device,
            1,
            12,
            16 * 32 * 32,
        )
        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()

        cryptoContext.openfhe_context = openfhe_context
        # 密文推理
        firstLayer = initial_layer(in_ct, cryptoContext)
        resLayer1 = layer1(firstLayer, cryptoContext)
        resLayer2 = layer2(resLayer1, cryptoContext)
        resLayer3 = layer3(resLayer2, cryptoContext)
        finalRes = final_layer(resLayer3, cryptoContext)
        print("after processing image ", i, "time: ", datetime.datetime.now())
        print("time: ", time.time() - start_time)

        # 对比明密文loss
        # conv_init = fea[0].flatten().reshape(-1)
        # init_out = openfhe_context.decrypt(firstLayer).cpu().numpy().reshape(-1)
        # init_out = torch.from_numpy(init_out).to(device)
        # loss = torch.sum((conv_init - init_out) ** 2)
        # print("loss: ", loss)

        # temp = openfhe_context.decrypt(resLayer1).cpu().numpy().reshape(-1)
        # print('name:resLayer1', temp)
        # fea_out = torch.tensor(fea[1].flatten().reshape(-1), device="cuda:0")
        # print('fea1', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer1', loss)

        # temp = openfhe_context.decrypt(resLayer2).cpu().numpy().reshape(-1)
        # print('name:resLayer2', temp)
        # fea_out = torch.tensor(fea[2].flatten().reshape(-1), device="cuda:0")
        # print('fea2', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer2',loss)
        #
        # temp = openfhe_context.decrypt(resLayer3).cpu().numpy().reshape(-1)
        # print('name:resLayer3', temp)
        # fea_out = torch.tensor(fea[3].flatten().reshape(-1), device="cuda:0")
        # print('fea3', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer3',loss)
        try:
            clear_result = openfhe_context.decrypt(finalRes)
            clear_result = clear_result.cpu().numpy().reshape(-1)
            clear_result = clear_result[:10]
            print('clear_result', clear_result)
            # print('x:',x)
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
        print("ground truth: ", label, "\tprediction: ", max_element_idx, "\tindex: ", index, )
        if label == max_element_idx:
            correct += 1
        message = f"correct/total: {correct}/{(i + 1)}"
        print(colored(message, "red"))
        if (i + 1) % 100 == 0:
            print("\n\n")

    print(f"\n\ncorrect/total: {correct}/{total}")


def resnet20():
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, CHECK_CIPHER=False,
                                     SAVE_MIDDLE=SAVE_MIDDLE,
                                     SAVE_END=SAVE_END,
                                     PTX_TWIN=False)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, device, save_dir=DATA_DIR,
                             config=config))

    cryptoContext.weight_path = weight_dir  # fixme: work around only
    cryptoContext.pre_encode_type = pre_encode_type
    load_weight(pkl_path, cryptoContext)

    print("start executeResNet20")
    cryptoContext.openfhe_context = openfhe_context
    executeResNet20(cryptoContext)


def homo_Aespa_perfect_square(x, filename, cryptoContext):
    # x = fhe.homo_rescale(x, 1, cryptoContext) #RESCALE ADD BY ZRJI
    n1_filename = filename + '-n1'
    n2_filename = filename + '-n2'
    slots = x.slots
    scale = 1  # 1
    n1 = read_values_from_file(n1_filename, cryptoContext.L - x.cur_limbs, 1, slots, cryptoContext, scale)
    temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
    perfect_squre = fhe.homo_square(temp1, cryptoContext)
    perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    n2 = read_values_from_file(n2_filename, cryptoContext.L - perfect_squre.cur_limbs, 1, slots, cryptoContext, scale)
    res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
    return res


if __name__ == "__main__":
    resnet20()
