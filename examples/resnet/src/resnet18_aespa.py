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
from examples.resnet.gen_aespa_weights.HerPN import get_resnet18_HerPN, change_all_HerPN_by_PAF_MutalChannel

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
weight_dir = "../weights_aespa_18/"

rotate_index_list = [-32768, -16384, -8192, -4096, -1024, -768, -256, -192, -64, -48, -32, -16, -15, -8, -1,
                     1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576, 49152, 98304]
maxLevelsRemaining = 15
logBsSlots_list = [15]
logN = 17
dnum = 1
dcrtBits = 55
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
    # conv3*3 [64,32,32]
    scale = 1
    res = conv_initial(input, 32, 1, 64, scale, cryptoContext)
    res = fhe.homo_rescale(res, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res = homo_Aespa_perfect_square(res, "conv1bn1", cryptoContext)
    return res


def layer1(input, cryptoContext):
    scale = 1
    # layer[0],block[0],conv1
    # input = [64,32,32]
    res1 = conv(input, 32, 1, 64, -1024, 1, 1, 0, scale, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{1}-conv{1}bn{1}", cryptoContext)
    # layer[0],block[0],conv2 and shorcut
    res1 = conv(res1, 32, 1, 64, -1024, 1, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        input = fhe.drop_last_elements(input, input.cur_limbs - res1.cur_limbs,
                                       cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(cryptoContext, f"layer{1}-conv{2}bn{2}-A2", cryptoContext.L - input.cur_limbs, 1,
                               input.slots, scale)
    A2y = fhe.homo_mul_pt(input, A2, cryptoContext)
    res1 = fhe.homo_add(res1, A2y, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{1}-conv{2}bn{2}", cryptoContext)

    # layer[0],block[1],conv1
    res2 = conv(res1, 32, 1, 64, -1024, 2, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{2}-conv{1}bn{1}", cryptoContext)

    # layer[0],block[1],conv2
    res2 = conv(res2, 32, 1, 64, -1024, 2, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(cryptoContext, f"layer{2}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1,
                               res1.slots, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{2}-conv{2}bn{2}", cryptoContext)
    return res2


def layer2(input, cryptoContext):
    # after down sample [128,16,16]
    scaleSx = 1
    scaleDx = 1

    boot_in = input

    res1sx0 = conv(boot_in, 32, 1, 64, -1024, 3, 1, 0, scaleSx, cryptoContext)
    res1sx1 = conv(boot_in, 32, 1, 64, -1024, 3, 1, 64, scaleSx, cryptoContext)
    res1sx0 = fhe.homo_rescale(res1sx0, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.homo_rescale(res1sx1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.drop_last_elements(boot_in, 2, cryptoContext)  # RESCALE ADD BY ZRJI
    res1dx0 = convbn_dx(boot_in, 64, -1024, 3, 1, 0, "1", scaleDx, cryptoContext)
    res1dx1 = convbn_dx(boot_in, 64, -1024, 3, 1, 64, "2", scaleDx, cryptoContext)
    fullpackSx = downsample1024to256(res1sx0, res1sx1, 64, 2, cryptoContext)
    fullpackDx = downsample1024to256(res1dx0, res1dx1, 64, 2, cryptoContext)
    fullpackSx = fhe.homo_bootstrap(fullpackSx, cryptoContext.L, 15, [4, 4], cryptoContext)
    fullpackDx = fhe.homo_bootstrap(fullpackDx, cryptoContext.L, 15, [4, 4], cryptoContext)

    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{3}-conv{1}bn{1}", cryptoContext)
    fullpackSx = conv(fullpackSx, 16, 1, 128, -256, 3, 2, 0, scaleDx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = homo_Aespa_perfect_square(res1, f"layer{3}-conv{2}bn{2}", cryptoContext)
    # layer[2]block[1]
    scale = 1
    res2 = conv(res1, 16, 1, 128, -256, 4, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{4}-conv{1}bn{1}", cryptoContext)
    res2 = conv(res2, 16, 1, 128, -256, 4, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(cryptoContext, f"layer{4}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1,
                               res1.slots, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{4}-conv{2}bn{2}", cryptoContext)
    return res2


def layer3(input, cryptoContext):
    scaleSx = 1
    scaleDx = 1

    boot_in = input
    res1sx0 = conv(boot_in, 16, 1, 128, -256, 5, 1, 0, scaleSx, cryptoContext)
    res1sx1 = conv(boot_in, 16, 1, 128, -256, 5, 1, 128, scaleSx, cryptoContext)
    res1sx0 = fhe.homo_rescale(res1sx0, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.homo_rescale(res1sx1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.drop_last_elements(boot_in, 2, cryptoContext)  # drop_last_elements ADD BY ZRJI
    res1dx0 = convbn_dx(boot_in, 128, -256, 5, 1, 0, "1", scaleDx, cryptoContext)

    res1dx1 = convbn_dx(boot_in, 128, -256, 5, 1, 128, "2", scaleDx, cryptoContext)

    fullpackSx = downsample256to64(res1sx0, res1sx1, 128, cryptoContext)
    fullpackDx = downsample256to64(res1dx0, res1dx1, 128, cryptoContext)
    fullpackSx = fhe.homo_bootstrap(fullpackSx, cryptoContext.L, 15, [4, 4], cryptoContext)
    fullpackDx = fhe.homo_bootstrap(fullpackDx, cryptoContext.L, 15, [4, 4], cryptoContext)

    fullpackSx = fhe.homo_rescale(fullpackSx, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{5}-conv{1}bn{1}", cryptoContext)

    fullpackSx = conv(fullpackSx, 8, 1, 256, -64, 5, 2, 0, scaleDx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{5}-conv{2}bn{2}", cryptoContext)

    scale = 1
    res2 = conv(res1, 8, 1, 256, -64, 6, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{6}-conv{1}bn{1}", cryptoContext)

    res2 = conv(res2, 8, 1, 256, -64, 6, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(cryptoContext, f"layer{6}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1,
                               res1.slots, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{6}-conv{2}bn{2}", cryptoContext)

    return res2


def layer4(input, cryptoContext):
    scaleSx = 1
    scaleDx = 1

    boot_in = fhe.homo_bootstrap(input, cryptoContext.L, 15, [4, 4], cryptoContext)
    res1sx0 = conv(boot_in, 8, 1, 256, -64, 7, 1, 0, scaleSx, cryptoContext)
    res1sx1 = conv(boot_in, 8, 1, 256, -64, 7, 1, 256, scaleSx, cryptoContext)
    res1sx0 = fhe.homo_rescale(res1sx0, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.homo_rescale(res1sx1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.drop_last_elements(boot_in, 2, cryptoContext)  # drop_last_elements ADD BY ZRJI
    res1dx0 = convbn_dx(boot_in, 256, -64, 7, 1, 0, "1", scaleDx, cryptoContext)

    res1dx1 = convbn_dx(boot_in, 256, -64, 7, 1, 256, "2", scaleDx, cryptoContext)

    fullpackSx = downsample64to16(res1sx0, res1sx1, 256, cryptoContext)
    fullpackDx = downsample64to16(res1dx0, res1dx1, 256, cryptoContext)

    fullpackSx = fhe.homo_rescale(fullpackSx, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{7}-conv{1}bn{1}", cryptoContext)
    fullpackSx = conv(fullpackSx, 4, 1, 512, -16, 7, 2, 0, scaleDx, cryptoContext)

    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_rescale(res1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{7}-conv{2}bn{2}", cryptoContext)

    scale = 1
    res2 = conv(res1, 4, 1, 512, -16, 8, 1, 0, scale, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{8}-conv{1}bn{1}", cryptoContext)

    scale = 1
    res2 = conv(res2, 4, 1, 512, -16, 8, 2, 0, scale, cryptoContext)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.drop_last_elements(res1, res1.cur_limbs - res2.cur_limbs,
                                      cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-A2", cryptoContext.L - res1.cur_limbs, 1,
                               res1.slots, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.homo_rescale(res2, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{8}-conv{2}bn{2}", cryptoContext)
    return res2


def final_layer(input, cryptoContext):
    # 512*4*4
    # he_res20_ctx.cur_num_slots = 8192
    # 16合1旋转&相加
    res = rotsum(input, 16, cryptoContext)
    # 仅对16倍数对应的位置有mask对应的值1/16
    res = fhe.homo_mul_pt(
        res,
        mask_mod(16, res.cur_limbs, 1.0 / 16.0, res.slots, cryptoContext),
        cryptoContext,
    )
    res = repeat(res, 16, cryptoContext)  # repeat num = 16，because 16>10 and min? For cifar100, we need repeat 128
    res = fhe.homo_rescale(res, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    weight = read_fc_weight(512, 16, cryptoContext, cryptoContext.L - res.cur_limbs, 1, res.slots)
    res = fhe.homo_mul_pt(res, weight, cryptoContext)
    res = rotsum_padded(res, 16, 512, cryptoContext)
    bias = read_fc_bias(512, 16, cryptoContext, cryptoContext.L - res.cur_limbs, 1, res.slots)
    res = fhe.homo_add_pt(res, bias, cryptoContext)
    return res


def executeResNet18(cryptoContext):
    openfhe_context = cryptoContext.openfhe_context

    cryptoContext.zero_64K = openfhe_context.encrypt(np.zeros(2 ** 16), device, 2, 0, 2 ** 16)
    cryptoContext.zero_32K = openfhe_context.encrypt(np.zeros(2 ** 15), device, 2, 0, 2 ** 15)
    cryptoContext.zero_16K = openfhe_context.encrypt(np.zeros(2 ** 14), device, 2, 0, 2 ** 14)

    # 准备明文模型，测速时可以删除
    # model = get_resnet18_HerPN(num_classes=10)
    # device = torch.device("cuda:0")
    # model.to(device)
    # model_path = '../Aespa/ResNet18_Aespa.pth'
    # stict = torch.load(model_path, map_location='cuda:0')
    # model.load_state_dict(stict, strict=False)
    # model.eval()
    # model = change_all_HerPN_by_PAF_MutalChannel(model)

    print("=====================================================")
    correct = 0
    for i in range(total):
        # input image size 64*32*32 = 2^16
        image_vector, label, index = read_image(i)
        # image_vector = torch.tensor(np.array(image_vector), device="cuda")
        # 明文模型输出
        # input = torch.tensor(image_vector, device="cuda",dtype=torch.float32)
        # input = torch.stack([input[i * 1024: (i + 1) * 1024].view(32, 32) for i in range(3)], dim=0)
        # x , fea = model(input,fea_out=True)
        in_ct = openfhe_context.encrypt(
            image_vector,
            device,
            1,
            12,
            64 * 32 * 32,
        )
        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()
        cryptoContext.openfhe_context = openfhe_context
        # 密文推理
        firstLayer = initial_layer(in_ct, cryptoContext)
        resLayer1 = layer1(firstLayer, cryptoContext)
        resLayer2 = layer2(resLayer1, cryptoContext)
        resLayer3 = layer3(resLayer2, cryptoContext)
        resLayer4 = layer4(resLayer3, cryptoContext)
        finalRes = final_layer(resLayer4, cryptoContext)
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
        #
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

        # temp = openfhe_context.decrypt(resLayer4).cpu().numpy().reshape(-1)
        # print('name:resLayer4', temp)
        # fea_out = torch.tensor(fea[4].flatten().reshape(-1), device="cuda:0")
        # print('fea4', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer4', loss)

        try:
            clear_result = openfhe_context.decrypt(finalRes)
            clear_result = clear_result.cpu().numpy().reshape(-1)
            clear_result = clear_result[:10]
            print(clear_result)
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


def resnet18():
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

    cryptoContext.weight_path = weight_dir  # fixme: workaround only

    pkl_path = None
    if config.SAVE_MIDDLE == False:
        cryptoContext.pre_encode_type = "middle"
        pkl_path = "/data/yhh/data/encode_20250611_234607.pkl"

        # cryptoContext.pre_encode_type = "end"
        # pkl_path = ""
    load_weight(pkl_path, cryptoContext)
    print("start executeResNet18")
    cryptoContext.openfhe_context = openfhe_context
    executeResNet18(cryptoContext)


def homo_Aespa_perfect_square(x, filename, cryptoContext):
    x = fhe.homo_rescale(x, x.noise_deg - 1, cryptoContext)  # RESCALE ADD BY ZRJI
    n1_filename = filename + '-n1'
    n2_filename = filename + '-n2'
    slots = x.slots
    scale = 1
    n1 = read_values_from_file(cryptoContext, n1_filename, cryptoContext.L - x.cur_limbs, 1, slots, scale)
    # input x = origin sqrt(a2) * x
    temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
    perfect_squre = fhe.homo_square(temp1, cryptoContext)
    perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    n2 = read_values_from_file(cryptoContext, n2_filename, cryptoContext.L - perfect_squre.cur_limbs, 1, slots, scale)
    res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
    return res


if __name__ == "__main__":
    resnet18()
