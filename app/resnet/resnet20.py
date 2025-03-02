import contextlib
import datetime
import torch
# from torch.fhe import utils
import approx
import torch.fhe as fhe
from app.resnet.convs import *
from app.resnet.utils import *

class HE_res20_context:
    def __init__(self, cur_num_slots, relu_degree):
        self.cur_num_slots = cur_num_slots
        self.relu_degree = relu_degree

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


def homo_relu(ciphertext, scale, degree, cryptoContext):
    def relu_function(x):
        return 0 if x < 0 else (1 / scale) * x

    coefficients = approx.eval_chebyshev_coefficients(relu_function, -1, 1, degree)
    result = approx.eval_chebyshev_series_ps(ciphertext, coefficients, -1, 1, cryptoContext)
    return result


def initial_layer(input, he_res20_ctx, cryptoContext):
    scale=normalized_deltas[0][0]
    res= convbn_initial(input, scale, he_res20_ctx, cryptoContext)
    res= homo_relu(res, scale, he_res20_ctx.relu_degree, cryptoContext)
    return res


def layer1(input, he_res20_ctx, cryptoContext):
    scale = normalized_deltas[1][0]

    res1= convbn(input, 1, 1, scale, he_res20_ctx, cryptoContext)
    res1=fhe.homo_bootstrap(res1,L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res1= homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)


    scale = normalized_deltas[1][1]
    res1= convbn(res1, 1, 2, scale, he_res20_ctx, cryptoContext)
    res1=fhe.homo_add(res1, fhe.homo_mul_scalar_double(input,scale,cryptoContext),cryptoContext)
    res1=fhe.homo_bootstrap(res1,L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res1= homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)


    scale = normalized_deltas[1][2]
    res2 = convbn(res1, 2, 1, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_bootstrap(res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)


    scale = normalized_deltas[1][3]
    res2 = convbn(res2, 2, 2, scale, he_res20_ctx, cryptoContext)
    res2=fhe.homo_add(res2,fhe.homo_mul_scalar_double(res1,scale,cryptoContext), cryptoContext)
    res2 = fhe.homo_bootstrap(res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)


    scale = normalized_deltas[1][4]
    res3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext)
    res3 = fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)



    scale = normalized_deltas[1][5]
    res3 = convbn(res3, 3, 2, scale, he_res20_ctx, cryptoContext)
    res3 = fhe.homo_add(res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext)
    res3 = fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3

def layer2(input, he_res20_ctx, cryptoContext):
    scaleSx =normalized_deltas[2][0]
    scaleDx =normalized_deltas[2][1]
    boot_in =fhe.homo_bootstrap(input, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext)
    res1sx=[None, None]
    res1dx=[None, None]
    res1sx[0], res1sx[1]  = convbn1632sx(boot_in, 4, 1, scaleSx, he_res20_ctx, cryptoContext)
    res1dx[0], res1dx[1]  = convbn1632dx(boot_in, 4, 1, scaleDx, he_res20_ctx, cryptoContext)

    fullpackSx = downsample1024to256(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext)
    fullpackDx = downsample1024to256(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)


    he_res20_ctx.cur_num_slots = 8192
    # set_bootstrapping_keys(he_res20_ctx.cur_num_slots, cryptoContext)

    fullpackSx = fhe.homo_bootstrap(fullpackSx,L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    fullpackSx = homo_relu(fullpackSx,scaleSx,he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn2(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1 = fhe.homo_bootstrap(res1, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][2]
    res2  = convbn2(res1, 5, 1, scale, he_res20_ctx, cryptoContext)
    res2  = fhe.homo_bootstrap(res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    res2  = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    res2  = convbn2(res2, 5, 2, scale, he_res20_ctx, cryptoContext)
    res2  = fhe.homo_add(res2,fhe.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2  = fhe.homo_bootstrap(res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    res2  = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][4]
    res3  = convbn2(res2, 6, 1, scale, he_res20_ctx, cryptoContext)
    res3  = fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    res3  = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    res3  = convbn2(res3, 6, 2, scale, he_res20_ctx, cryptoContext)
    res3  = fhe.homo_add(res3,fhe.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3  = fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    res3  = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer3(input,he_res20_ctx, cryptoContext):
    scaleSx=normalized_deltas[3][0]
    scaleDx=normalized_deltas[3][1]

    boot_in=fhe.homo_bootstrap(input, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext)
    res1sx=[None, None]
    res1dx=[None, None]
    res1sx[0], res1sx[1]= convbn3264sx(boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext)
    res1dx[0], res1dx[1]= convbn3264dx(boot_in, 7, 1, scaleDx, he_res20_ctx, cryptoContext)

    fullpackSx = downsample256to64(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext)
    fullpackDx = downsample256to64(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)

    he_res20_ctx.cur_num_slots = 4096
    # set_bootstrapping_keys(he_res20_ctx.cur_num_slots, cryptoContext)


    fullpackSx = fhe.homo_bootstrap(fullpackSx,L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn3(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1 = fhe.homo_bootstrap(res1, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)



    scale = normalized_deltas[3][2]
    res2= convbn3(res1, 8, 1, scale, he_res20_ctx, cryptoContext)
    res2=fhe.homo_bootstrap(res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    res2= homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    res2= convbn3(res2, 8, 2, scale, he_res20_ctx, cryptoContext)
    res2=fhe.homo_add(res2,fhe.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = fhe.homo_bootstrap(res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    res3= convbn3(res2, 9, 1, scale, he_res20_ctx, cryptoContext)
    res3=fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    res3= homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    res3= convbn3(res3, 9, 2, scale, he_res20_ctx, cryptoContext)
    res3=fhe.homo_add(res3,fhe.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3 = fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)
    res3 = fhe.homo_bootstrap(res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext)
    return res3


def final_layer(input, he_res20_ctx, cryptoContext):

    he_res20_ctx.cur_num_slots=4096
    weight=read_fc_weight(cryptoContext, cryptoContext.L - input.cur_limbs, 1, he_res20_ctx.cur_num_slots)
    # print(cryptoContext.L - input.cur_limbs, 1, he_res20_ctx.cur_num_slots)
    res = rotsum(input, 64,cryptoContext)
    res = fhe.homo_mul_pt(res,
                          mask_mod(64, res.cur_limbs, 1.0 / 64.0, he_res20_ctx, cryptoContext), cryptoContext)
    res=repeat(res,16,cryptoContext)

    res=fhe.homo_mul_pt(res,weight,cryptoContext)
    res=rotsum_padded(res,64,cryptoContext)

    return res


def executeResNet20(he_res20_ctx, cryptoContext, openfhe_context):

    he_res20_ctx.cur_num_slots = (1<<14)
    cryptoContext.openfhe_context = openfhe_context

    cryptoContext.zero_32K = openfhe_context.encrypt(np.zeros(2 ** 15), 1, 0, 2 ** 15)
    cryptoContext.zero_16K = openfhe_context.encrypt(np.zeros(2 ** 14), 1, 0, 2 ** 14)

    # for _logslot in [12, 13, 14]:
    #     load_bootstrapping_and_rotation_keys(_logslot, cryptoContext)
    # utils.load_rotation_keys(cryptoContext, "app")

    # print("resnet computation start")
    # firstLayer= initial_layer(in_ct, he_res20_ctx, cryptoContext)
    # resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext)
    # resLayer2 = layer2(resLayer1,  he_res20_ctx, cryptoContext)
    # resLayer3 = layer3(resLayer2,  he_res20_ctx, cryptoContext)
    # finalRes = final_layer(resLayer3,he_res20_ctx, cryptoContext)

    for j in range(29,60,2):
        # 创建日志文件
        log_filename = f'./logs/relu_deg_{j}.txt'
        he_res20_ctx.relu_degree = j
        print("=====================================================")
        print("=================relu_degree: {}=====================".format(j))
        print("=====================================================")

        with open(log_filename, 'w') as log_file:
            # with contextlib.redirect_stdout(log_file):
            for i in range(200):
                he_res20_ctx.cur_num_slots = (1<<14)

                image_vector, label, index = read_image(i)
                image_vector = np.array(image_vector)
                image_vector = torch.tensor(image_vector, device="cuda")
                in_ct = openfhe_context.encrypt(image_vector, 1,
                                                                cryptoContext.L - 5 - get_relu_depth(he_res20_ctx.relu_degree),
                                                                he_res20_ctx.cur_num_slots)  # note: initial level is aligned with the original open source codes

                print("start processing image ", i, "time: ", datetime.datetime.now())
                # print("current time: ", datetime.datetime.now())
                firstLayer = initial_layer(in_ct, he_res20_ctx, cryptoContext)
                # print("after initial layer")
                # print("current time: ", datetime.datetime.now())
                resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext)
                # print("after layer1")
                # print("current time: ", datetime.datetime.now())
                resLayer2 = layer2(resLayer1, he_res20_ctx, cryptoContext)
                # print("after layer2")
                # print("current time: ", datetime.datetime.now())
                resLayer3 = layer3(resLayer2, he_res20_ctx, cryptoContext)
                # print("after layer3")
                # print("current time: ", datetime.datetime.now())
                finalRes = final_layer(resLayer3, he_res20_ctx, cryptoContext)
                print("after processing image ", i, "time: ", datetime.datetime.now())

                # finalRes.slots = 10
                try:
                    clear_result = openfhe_context.decrypt(finalRes)  # 尝试解密
                    clear_result = clear_result.cpu().numpy().reshape(-1)
                    max_element_idx = np.argmax(clear_result[:10])
                except RuntimeError as e:
                    print(f"Decryption failed: {e}")  # 打印错误信息
                    clear_result = None  # 设置一个默认值或采取其他处理措施
                    max_element_idx = 11

                # clear_result = openfhe_context.decrypt(finalRes) #decrypt by cc with different slots value should be fine
                # clear_result = clear_result.cpu().numpy().reshape(-1)
                # max_element_idx = np.argmax(clear_result[:10])
                # print to console
                print("For image ", i)
                if clear_result is not None:
                    print(clear_result[:10])  # 只有在 clear_result 不是 None 时才执行切片操作
                else:
                    print("Decryption failed, clear_result is None.")  # 处理解密失败的情况
                print("ground truth: ", label, "prediction: ", max_element_idx)

                # print to log file
                print("For image ", i, "index: ", index, file=log_file)
                if clear_result is not None:
                    print(clear_result[:10], file=log_file)  # 只有在 clear_result 不是 None 时才执行切片操作
                else:
                    print("Decryption failed, clear_result is None.", file=log_file)  # 处理解密失败的情况
                print("ground truth: ", label, "prediction: ", max_element_idx, file=log_file)
                log_file.flush()



def resnet20( ):

    # generate context
    max_relu_degree = 59
    maxLevelsRemaining = get_relu_depth(max_relu_degree) + 3
    if max_relu_degree < 59:
        diff = get_relu_depth(59)-get_relu_depth(max_relu_degree)
        maxLevelsRemaining +=diff
    rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                            1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]
    logBsSlots_list = [12, 13, 14]
    logN = 16
    dnum = 3
    dcrtBits = 52
    firstMod = 56
    levelBudget_list = [[4, 4], [4, 4], [4, 4]]
    secretKeyDist = "UNIFORM_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    save_dir = "/home/ysjs1/data/yhh/data/"

    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    he_res20_context_ = HE_res20_context(None, max_relu_degree) # ini app ctx--he resnet ctx
    he_res20_context_.weight_dir = "/home/ysjs1/data/yhh/data"

    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                       levelBudget_list, secretKeyDist, rescaleTech, save_dir=save_dir,
                       autoLoadAndSetConfig=True, mode="release"))

    cryptoContext.GEN_PRECOMPUTATION = False # poor workaround, should be fixed in the future, need to be set to False now
    cryptoContext.PRELOAD_ALL = False # poor workaround, should be fixed in the future, need to be set to False/True now
    print("start executeResNet20")

    encode_weight_path = (
        he_res20_context_.weight_dir
        + "/ENCODE-VAL_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            logN,
            '-'.join(map(str, logBsSlots_list)),
            maxLevelsRemaining,
            '-'.join('-'.join(map(str, levelBudget)) for levelBudget in levelBudget_list),
            dcrtBits,
            firstMod,
            secretKeyDist,
            rescaleTech,
        )
    )

    with open(encode_weight_path, 'rb') as f:
        pre_encoded = pickle.load(f)
    if cryptoContext.PRELOAD_ALL:
        for key, _ in pre_encoded.items():
            pre_encoded[key].mv = [torch.tensor(pre_encoded[key].mv[0], dtype=torch.uint64, device="cuda")]
    cryptoContext.pre_encoded = pre_encoded

    executeResNet20(he_res20_context_, cryptoContext, openfhe_context)

if __name__ == "__main__":
    resnet20()