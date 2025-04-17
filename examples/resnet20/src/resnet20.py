import os, sys, datetime, time

from examples.resnet20.Aespa.HerPN import get_resnet20_HerPN, change_all_HerPN_by_PAF_MutalChannel

sys.path.append("/".join(os.getcwd().split("/")[:-4]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch
import numpy as np
import torch.fhe as fhe
from examples.utils import approx
from examples.resnet20.src.convs import *
from huggingface_hub import hf_hub_download
import zipfile

DATA_DIR = os.environ["DATA_DIR"]


class HE_res20_context:
    def __init__(self, data_dir,aespa_flag=False):
        self.cur_num_slots = None
        self.relu_degree = None
        self.weight_dir = data_dir
        self.aespa = aespa_flag

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
    def scaled_relu_function(x):
        return 0 if x < 0 else (1 / scale) * x

    result = approx.eval_chebyshev_function(scaled_relu_function, ciphertext, -1, 1, degree, cryptoContext)
    return result


def initial_layer(input, he_res20_ctx, cryptoContext):
    scale = normalized_deltas[0][0]
    if he_res20_ctx.aespa:
        res,bias = convbn_initial(input, scale, he_res20_ctx, cryptoContext)
        print('===================Aespa1====================')
        # res = homo_aespa(res,a2,bias,cryptoContext)
        res = Aespa(res,"conv1bn1",cryptoContext)
        return res

    else:
        res = convbn_initial(input, scale, he_res20_ctx, cryptoContext)
        res = homo_relu(res, scale, he_res20_ctx.relu_degree, cryptoContext)
        return res


def layer1(input, he_res20_ctx, cryptoContext):
    scale = normalized_deltas[1][0]
    # layer[0],block[0],conv1
    if he_res20_ctx.aespa:
        res1,bias1 = convbn(input, 1, 1, scale, he_res20_ctx, cryptoContext)
        # a2 = read_values_from_file(cryptoContext,  f"layer{1}-conv{1}bn{1}-a2",cryptoContext.L-input.cur_limbs,1,16384,scale)
        print('===================Aespa2====================')
        # res1 = homo_aespa(res1,a2,bias1,cryptoContext)
        # res1 = homo_aespa_a0(res1, a2, f"layer{1}-conv{1}bn{1}-bias",he_res20_ctx, cryptoContext)
        res1 = Aespa(res1, f"layer{1}-conv{1}bn{1}", cryptoContext)
    else:
        res1 = convbn(input, 1, 1, scale, he_res20_ctx, cryptoContext)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res1 = homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    # layer[0],block[0],conv2 and shorcut
    scale = normalized_deltas[1][1]
    if he_res20_ctx.aespa:
        # res1 = a1*x,shortcut = input = y
        res1,bias1 = convbn(res1, 1, 2, scale, he_res20_ctx, cryptoContext)
        # a1 = read_values_from_file(cryptoContext,  f"layer{1}-conv{2}bn{2}-a1",cryptoContext.L-res1.cur_limbs,1,16384,scale)
        # a1y = fhe.homo_mul_pt(input,a1,cryptoContext)
        # res1 = fhe.homo_add(res1,a1y,cryptoContext)
        res1 = fhe.homo_add(res1, input, cryptoContext)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        print('===================Aespa3====================')
        # res1 = homo_aespa(res1,a2,bias1,cryptoContext)
        res1 = Aespa(res1, f"layer{1}-conv{2}bn{2}",  cryptoContext)
    else:
        res1 = convbn(res1, 1, 2, scale, he_res20_ctx, cryptoContext)
        res1 = fhe.homo_add(
            res1, fhe.homo_mul_scalar_double(input, scale, cryptoContext), cryptoContext
        )
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res1 = homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][2]
    # layer[0],block[1],conv1
    if he_res20_ctx.aespa:
        res2,bias2 = convbn(res1, 2, 1, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        print('===================Aespa4====================')
        # res1 = homo_aespa(res1,a2,bias2,cryptoContext)
        res2 = Aespa(res2, f"layer{2}-conv{1}bn{1}", cryptoContext)
    else:
        res2 = convbn(res1, 2, 1, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)
    # layer[0],block[1],conv2 and shorcut
    scale = normalized_deltas[1][3]
    if he_res20_ctx.aespa:
        res2,bias2 = convbn(res2, 2, 2, scale, he_res20_ctx, cryptoContext)

        # a1 = read_values_from_file(cryptoContext,  f"layer{2}-conv{2}bn{2}-a1",cryptoContext.L-res2.cur_limbs,1,16384,scale)
        # a1y = fhe.homo_mul_pt(res1, a1, cryptoContext)
        print('max:res2', np.max(cryptoContext.openfhe_context.decrypt(res2).cpu().numpy().reshape(-1)))
        print('max:res1', np.max(cryptoContext.openfhe_context.decrypt(res1).cpu().numpy().reshape(-1)))
        res2 = fhe.homo_add(res2, res1,cryptoContext)
        print('max:res2', np.max(cryptoContext.openfhe_context.decrypt(res2).cpu().numpy().reshape(-1)))
        # a2 = read_values_from_file(cryptoContext,  f"layer{2}-conv{2}bn{2}-a2",cryptoContext.L-res2.cur_limbs,1,16384,scale)

        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        print('===================Aespa5====================')
        # res2 = homo_aespa(res2, a2, bias2,cryptoContext)
        res2 = Aespa(res2, f"layer{2}-conv{2}bn{2}", cryptoContext )
        print('max:res2.1', np.max(cryptoContext.openfhe_context.decrypt(res2).cpu().numpy().reshape(-1)))

    else:
        res2 = convbn(res2, 2, 2, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_add(
            res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
        )
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    # layer[0],block[2],conv1
    scale = normalized_deltas[1][4]
    if he_res20_ctx.aespa:
        res3,bias3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext)
        print('max:res3', np.max(cryptoContext.openfhe_context.decrypt(res3).cpu().numpy().reshape(-1)))
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )

        # a2 = read_values_from_file(cryptoContext,  f"layer{3}-conv{1}bn{1}-a2",cryptoContext.L-res2.cur_limbs,1,16384,scale)
        print('===================Aespa6====================')
        # res3 = homo_aespa(res3, a2, bias3,cryptoContext)
        # res3 = homo_aespa_a0(res3, a2, f"layer{3}-conv{1}bn{1}-bias",he_res20_ctx, cryptoContext)
        res3 = Aespa(res3, f"layer{3}-conv{1}bn{1}",  cryptoContext)
        print('res3,max', np.max(cryptoContext.openfhe_context.decrypt(res3).cpu().numpy().reshape(-1)))

    else:
        res3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][5]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn(res3, 3, 2, scale, he_res20_ctx, cryptoContext)
        print('res3,max', np.max(cryptoContext.openfhe_context.decrypt(res3).cpu().numpy().reshape(-1)))
        # a1 = read_values_from_file(cryptoContext,  f"layer{3}-conv{2}bn{2}-a1",cryptoContext.L-res3.cur_limbs,1,16384,scale)
        # a1y = fhe.homo_mul_pt(res2, a1, cryptoContext)
        res3 = fhe.homo_add(res3, res2,cryptoContext)
        # a2 = read_values_from_file(cryptoContext,  f"layer{3}-conv{2}bn{2}-a2",cryptoContext.L-res3.cur_limbs,1,16384,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        print('===================Aespa7====================')
        # res3 = homo_aespa(res3, a2, bias3,cryptoContext)
        res3 = Aespa(res3, f"layer{3}-conv{2}bn{2}",  cryptoContext)
    else:
        res3 = convbn(res3, 3, 2, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_add(
            res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
        )
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)
    return res3


def layer2(input, he_res20_ctx, cryptoContext):
    scaleSx = normalized_deltas[2][0]
    scaleDx = normalized_deltas[2][1]
    #scaleDx = 0.20468562391210693
    boot_in = fhe.homo_bootstrap(
        input, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
    )
    res1sx = [None, None]
    res1dx = [None, None]
    # 因为输入16通道，输出32通道，所以需要额外处理，包括短路结构
    # layer[2]block[0]
    if he_res20_ctx.aespa:
        res1sx[0], res1sx[1],bias = convbn1632sx(
            boot_in, 4, 1, scaleSx, he_res20_ctx, cryptoContext
        )
    else:
        res1sx[0], res1sx[1] = convbn1632sx(
            boot_in, 4, 1, scaleSx, he_res20_ctx, cryptoContext
        )
    res1dx[0], res1dx[1] = convbn1632dx(
        boot_in, 4, 1, scaleDx, he_res20_ctx, cryptoContext
    )
    fullpackSx = downsample1024to256(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext)

    x = np.array(cryptoContext.openfhe_context.decrypt(fullpackSx).cpu().numpy().reshape(-1))
    print('Max-conv1', x)

    fullpackDx = downsample1024to256(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)


    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    if he_res20_ctx.aespa:
        print('===================Aespa8====================')
        # fullpackSx = homo_aespa(fullpackSx, a2, bias,cryptoContext)
        fullpackSx = Aespa(fullpackSx, f"layer{4}-conv{1}bn{1}",  cryptoContext)
        x = np.array(cryptoContext.openfhe_context.decrypt(fullpackSx).cpu().numpy().reshape(-1))
        print('Max-HerPN1', x)
    else:
        fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)

    he_res20_ctx.cur_num_slots = 8192
    if he_res20_ctx.aespa:
        fullpackSx, bias = convbn2(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext)
        x = np.array(cryptoContext.openfhe_context.decrypt(fullpackSx).cpu().numpy().reshape(-1))
        print('Max-conv2', x)
        # Todo:check a1 is mix with downsample's BN
        # a1 = read_values_from_file(cryptoContext,  f"layer{4}-conv{2}bn{2}-a1",cryptoContext.L-fullpackSx.cur_limbs,1,8192,scaleSx)
        # fullpackDx = fhe.homo_mul_pt(fullpackDx, a1,cryptoContext)
        res1 = fhe.homo_add(fullpackSx, fullpackDx,cryptoContext)
        x = np.array(cryptoContext.openfhe_context.decrypt(res1).cpu().numpy().reshape(-1))
        print('Max-sum', x)
        # a2 = read_values_from_file(cryptoContext,  f"layer{4}-conv{2}bn{2}-a2",cryptoContext.L-fullpackSx.cur_limbs,1,8192,scaleSx)
        # res1 = fhe.homo_bootstrap(
        #     res1, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        # )
        print('===================Aespa9====================')
        # res1 = homo_aespa(res1,a2,bias,cryptoContext)
        res1 = Aespa(res1, f"layer{4}-conv{2}bn{2}",  cryptoContext)
    else:
        fullpackSx = convbn2(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext)
        res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)
    # layer[2]block[1]
    scale = normalized_deltas[2][2]
    if he_res20_ctx.aespa:
        res2,bias2 = convbn2(res1, 5, 1, scale, he_res20_ctx, cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{5}-conv{1}bn{1}-a2",cryptoContext.L-res1.cur_limbs,1,8192,scale)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        print('===================Aespa10====================')
        # res2 = homo_aespa(res2, a2, bias2,cryptoContext)
        # res2 = homo_aespa_a0(res2, a2, f"layer{5}-conv{1}bn{1}-bias", he_res20_ctx, cryptoContext)
        res2 = Aespa(res2, f"layer{5}-conv{1}bn{1}",  cryptoContext)
    else:
        res2 = convbn2(res1, 5, 1, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    if he_res20_ctx.aespa:
        res2,bias2 = convbn2(res2, 5, 2, scale, he_res20_ctx, cryptoContext)
        # a1 = read_values_from_file(cryptoContext,  f"layer{5}-conv{2}bn{2}-a1",cryptoContext.L-res2.cur_limbs,1,8192,scale)
        # a1y = fhe.homo_mul_pt(res1,a1,cryptoContext)
        res2 = fhe.homo_add(res2,res1,cryptoContext)
        # a2 = read_values_from_file(cryptoContext,  f"layer{5}-conv{2}bn{2}-a2",cryptoContext.L-res2.cur_limbs,1,8192,scale)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        print('===================Aespa11====================')
        # res2 = homo_aespa(res2, a2, bias2,cryptoContext)
        res2 = Aespa(res2, f"layer{5}-conv{2}bn{2}", cryptoContext)
    else:
        res2 = convbn2(res2, 5, 2, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_add(
            res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
        )
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)
    # layer[2]block[2]
    scale = normalized_deltas[2][4]
    if he_res20_ctx.aespa:
        res3,bias3 = convbn2(res2, 6, 1, scale, he_res20_ctx, cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{6}-conv{1}bn{1}-a2",cryptoContext.L-res2.cur_limbs,1,8192,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        print('===================Aespa12====================')
        # res3 = homo_aespa(res3, a2, bias3,cryptoContext)
        # res3 = homo_aespa_a0(res3, a2, f"layer{6}-conv{1}bn{1}-bias", he_res20_ctx, cryptoContext)
        res3 = Aespa(res3, f"layer{6}-conv{1}bn{1}",  cryptoContext)
    else:
        res3 = convbn2(res2, 6, 1, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn2(res3, 6, 2, scale, he_res20_ctx, cryptoContext)
        # a1 = read_values_from_file(cryptoContext,  f"layer{6}-conv{2}bn{2}-a1",cryptoContext.L-res3.cur_limbs,1,8192,scale)
        # a1y = fhe.homo_mul_pt(res2, a1, cryptoContext)
        res3 = fhe.homo_add(res3, res2,cryptoContext)
        # a2 = read_values_from_file(cryptoContext,  f"layer{6}-conv{2}bn{2}-a2",cryptoContext.L-res3.cur_limbs,1,8192,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        print('===================Aespa13====================')
        # res3 = homo_aespa(res3, a2, bias3,cryptoContext)
        res3 = Aespa(res3, f"layer{6}-conv{2}bn{2}",  cryptoContext)
    else:
        res3 = convbn2(res3, 6, 2, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_add(
            res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
        )
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer3(input, he_res20_ctx, cryptoContext):
    scaleSx = normalized_deltas[3][0]
    scaleDx = normalized_deltas[3][1]

    boot_in = fhe.homo_bootstrap(
        input, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    res1sx = [None, None]
    res1dx = [None, None]
    if he_res20_ctx.aespa:
        res1sx[0], res1sx[1], bias = convbn3264sx(
            boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext
        )
    else:
        res1sx[0], res1sx[1] = convbn3264sx(
            boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext
        )

    res1dx[0], res1dx[1] = convbn3264dx(
        boot_in, 7, 1, scaleDx, he_res20_ctx, cryptoContext
    )

    fullpackSx = downsample256to64(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext)
    fullpackDx = downsample256to64(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)


    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
    )
    if he_res20_ctx.aespa:
        a2 = read_values_from_file(cryptoContext, f"layer{7}-conv{1}bn{1}-a2",
                                      cryptoContext.L - boot_in.cur_limbs, 1, 8192, scaleSx)
        print('===================Aespa14====================')
        # fullpackSx = homo_aespa(fullpackSx, a2, bias,cryptoContext)
        fullpackSx = Aespa(fullpackSx, f"layer{7}-conv{1}bn{1}",  cryptoContext)
    else:
        fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)

    he_res20_ctx.cur_num_slots = 4096
    if he_res20_ctx.aespa:
        fullpackSx,bias = convbn3(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext)
        # a1 = read_values_from_file(cryptoContext, f"layer{7}-conv{2}bn{2}-a1",cryptoContext.L-fullpackSx.cur_limbs,1,4096,scaleDx)
        # fullpackDx = fhe.homo_mul_pt(fullpackDx, a1,cryptoContext)
        res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
        # a2 = read_values_from_file(cryptoContext, f"layer{7}-conv{2}bn{2}-a2",cryptoContext.L-fullpackSx.cur_limbs,1,4096,scaleDx)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        print('===================Aespa15====================')
        # res1 = homo_aespa(res1, a2, bias,cryptoContext)
        res1 = Aespa(res1, f"layer{7}-conv{2}bn{2}", cryptoContext)
    else:
        fullpackSx = convbn3(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext)
        res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][2]
    if he_res20_ctx.aespa:
        res2,bias2 = convbn3(res1, 8, 1, scale, he_res20_ctx, cryptoContext)
        a2 = read_values_from_file(cryptoContext, f"layer{8}-conv{1}bn{1}-a2",cryptoContext.L-res1.cur_limbs,1,4096,scale)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        print('===================Aespa16====================')
        # res2 = homo_aespa(res2, a2, bias2,cryptoContext)
        # res2 = homo_aespa_a0(res2, a2, f"layer{8}-conv{1}bn{1}-bias", he_res20_ctx, cryptoContext)
        res2 = Aespa(res2, f"layer{8}-conv{1}bn{1}",  cryptoContext)
    else:
        res2 = convbn3(res1, 8, 1, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    if he_res20_ctx.aespa:
        res2, bias2 = convbn3(res2, 8, 2, scale, he_res20_ctx, cryptoContext)
        # a1 = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-a1",cryptoContext.L-res2.cur_limbs,1,4096,scale)
        # a1y = fhe.homo_mul_pt(res1, a1, cryptoContext)
        res2 = fhe.homo_add(res2, res1,cryptoContext)
        # a2 = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-a2",cryptoContext.L-res2.cur_limbs,1,4096,scale)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        print('===================Aespa17====================')
        # res2 = homo_aespa(res2, a2, bias2,cryptoContext)
        res2 = Aespa(res2, f"layer{8}-conv{2}bn{2}", cryptoContext)
    else:
        res2 = convbn3(res2, 8, 2, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_add(
            res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
        )
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn3(res2, 9, 1, scale, he_res20_ctx, cryptoContext)
        a2 = read_values_from_file(cryptoContext, f"layer{9}-conv{1}bn{1}-a2",cryptoContext.L-res2.cur_limbs,1,4096,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        print('===================Aespa18====================')
        # res3 = homo_aespa(res3, a2, bias3,cryptoContext)
        # res3 = homo_aespa_a0(res3, a2, f"layer{9}-conv{1}bn{1}-bias", he_res20_ctx, cryptoContext)
        res3 = Aespa(res3, f"layer{9}-conv{1}bn{1}", cryptoContext)
    else:
        res3 = convbn3(res2, 9, 1, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn3(res3, 9, 2, scale, he_res20_ctx, cryptoContext)
        # a1 =read_values_from_file(cryptoContext, f"layer{9}-conv{2}bn{2}-a1",cryptoContext.L-res3.cur_limbs,1,4096,scale)
        # a1y = fhe.homo_mul_pt(res2, a1, cryptoContext)
        res3 = fhe.homo_add(res3, res2,cryptoContext)
        a2 = read_values_from_file(cryptoContext, f"layer{9}-conv{2}bn{2}-a2",cryptoContext.L-res3.cur_limbs,1,4096,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        print('===================Aespa19====================')
        # res3 = homo_aespa(res3, a2, bias3,cryptoContext)
        res3 = Aespa(res3, f"layer{9}-conv{2}bn{2}", cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
    else:
        res3 = convbn3(res3, 9, 2, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_add(
            res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
        )
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
    return res3


def final_layer(input, he_res20_ctx, cryptoContext):

    he_res20_ctx.cur_num_slots = 4096
    weight = read_fc_weight(
        cryptoContext, cryptoContext.L - input.cur_limbs, 1, he_res20_ctx.cur_num_slots
    )
    res = rotsum(input, 64, cryptoContext)
    res = fhe.homo_mul_pt(
        res,
        mask_mod(64, res.cur_limbs, 1.0 / 64.0, he_res20_ctx, cryptoContext),
        cryptoContext,
    )
    res = repeat(res, 16, cryptoContext)

    res = fhe.homo_mul_pt(res, weight, cryptoContext)
    res = rotsum_padded(res, 64, cryptoContext)

    return res


def read_image(index):
    filePath = "../cifar10/test_batch.bin"
    IMAGE_SIZE = 3072
    LABEL_SIZE = 1
    RECORD_SIZE = LABEL_SIZE + IMAGE_SIZE
    try:
        with open(filePath, "rb") as file:
            file.seek(index * RECORD_SIZE)
            label = file.read(LABEL_SIZE)
            if not label:
                raise ValueError("Failed to read label.")
            label = int.from_bytes(label, byteorder="big")
            print(f"Label: {label}")
            image_data = file.read(IMAGE_SIZE)
            if not image_data or len(image_data) != 3072:
                raise ValueError("Failed to read image data.")
        imageVector = []
        for channel in range(3):
            for i in range(1024):
                pixel = float(image_data[channel * 1024 + i]) / 255.0
                if channel == 0:
                    pixel = (pixel - 0.4914) / 0.2023
                elif channel == 1:
                    pixel = (pixel - 0.4822) / 0.1994
                elif channel == 2:
                    pixel = (pixel - 0.4465) / 0.2010
                imageVector.append(pixel)
        return imageVector, label, index
    except FileNotFoundError:
        print(f"Failed to open the file: {filePath}")


def executeResNet20(he_res20_ctx, cryptoContext, openfhe_context):

    he_res20_ctx.cur_num_slots = 1 << 14
    he_res20_ctx.relu_degree = 59
    cryptoContext.openfhe_context = openfhe_context

    cryptoContext.zero_32K = openfhe_context.encrypt(np.zeros(2**15), 1, 0, 2**15)
    cryptoContext.zero_16K = openfhe_context.encrypt(np.zeros(2**14), 1, 0, 2**14)

    print("=====================================================")
    for i in range(100):
        he_res20_ctx.cur_num_slots = 1 << 14

        image_vector, label, _ = read_image(i)
        image_vector = torch.tensor(np.array(image_vector), device="cuda")
        #
        input = torch.tensor(image_vector, device="cuda",dtype=torch.float32)
        model = get_resnet20_HerPN(num_classes=10)
        device = torch.device("cuda:0")
        model.to(device)
        model_path = '/home/fyh/PNP/GPU-FHE/examples/resnet20/Aespa/ResNet20_Aespa.pth'
        stict = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(stict, strict=False)
        model.eval()
        model = change_all_HerPN_by_PAF_MutalChannel(model)
        input = torch.stack([input[i * 1024: (i + 1) * 1024].view(32, 32) for i in range(3)], dim=0)
        x , fea = model(input,fea_out=True)

        in_ct = openfhe_context.encrypt(
            image_vector,
            1,
            cryptoContext.L - 11,
            he_res20_ctx.cur_num_slots,
        )

        conv_init = fea[0]
        conv_init= conv_init.flatten().reshape(-1)
        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()

        cryptoContext.openfhe_context = openfhe_context

        firstLayer = initial_layer(in_ct, he_res20_ctx, cryptoContext)
        conv1_out = openfhe_context.decrypt(firstLayer).cpu().numpy().reshape(-1)
        conv1_out = torch.from_numpy(conv1_out).to(device)
        loss = torch.sum((conv_init-conv1_out)**2)
        print("loss: ", loss)

        resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext)
        temp = openfhe_context.decrypt(resLayer1).cpu().numpy().reshape(-1)
        print('name:resLayer1',temp)
        print('fea1', fea[1].flatten().reshape(-1))
        fea_out = torch.tensor(fea[1].flatten().reshape(-1), device="cuda:0")
        temp = torch.tensor(temp, device="cuda:0")
        loss = torch.sum((fea_out - temp) ** 2)
        print('resLayer1',loss)

        resLayer2 = layer2(resLayer1, he_res20_ctx, cryptoContext)
        temp = openfhe_context.decrypt(resLayer2).cpu().numpy().reshape(-1)
        print('name:resLayer2', temp)
        print('fea2', fea[2].flatten().reshape(-1))
        fea_out = torch.tensor(fea[2].flatten().reshape(-1), device="cuda:0")
        temp = torch.tensor(temp, device="cuda:0")
        loss = torch.sum((fea_out - temp) ** 2)
        print(loss)

        resLayer3 = layer3(resLayer2, he_res20_ctx, cryptoContext)
        temp = openfhe_context.decrypt(resLayer3).cpu().numpy().reshape(-1)
        print('name:resLayer3', temp)
        print('fea3', fea[3].flatten().reshape(-1))
        fea_out = torch.tensor(fea[3].flatten().reshape(-1), device="cuda:0")
        temp = torch.tensor(temp, device="cuda:0")
        loss = torch.sum((fea_out - temp) ** 2)
        print(loss)

        finalRes = final_layer(resLayer3, he_res20_ctx, cryptoContext)

        print("time: ", time.time() - start_time)
        print("after processing image ", i, "time: ", datetime.datetime.now())
        try:
            clear_result = openfhe_context.decrypt(finalRes)
            clear_result = clear_result.cpu().numpy().reshape(-1)
            clear_result = clear_result[:10]
            print(clear_result)
            print('x:',x)
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


def load_encode_pkl(file_name, he_res20_context_):
    repo_id = "catslab/res20-ver_LowMem_encode_middle"
    hf_token = "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"
    pkl_path = os.path.join(he_res20_context_.weight_dir, file_name+".pkl")
    zip_path = os.path.join(he_res20_context_.weight_dir, file_name+".zip")

    if os.path.exists(pkl_path):
        print(">> Found cached pkl, skipping download.")
        return

    if os.path.exists(zip_path):
        print(">> Found cached encode zip.")
    else:
        print(f">> {file_name}.pkl not found, downloading zip from Hugging Face private repo...")

        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name + ".zip",
            repo_type="model",
            token=hf_token,
            local_dir=he_res20_context_.weight_dir,
            # local_dir_use_symlinks=False
        )
        print(">> Download complete.")

    print(">> Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(he_res20_context_.weight_dir)
    os.remove(zip_path)
    print(">> Extraction complete.")


def resnet20( ):
    # generate context
    max_relu_degree = 59
    maxLevelsRemaining = get_relu_depth(max_relu_degree) + 3
    if max_relu_degree < 59:
        diff = get_relu_depth(59)-get_relu_depth(max_relu_degree)
        maxLevelsRemaining +=diff
    rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                         1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]

    maxLevelsRemaining = 14
    logBsSlots_list = [12, 13, 14]
    logN = 16
    dnum = 1
    dcrtBits = 56
    firstMod = 60
    levelBudget_list = [[4, 4], [4, 4], [4, 4]]
    secretKeyDist = "SPARSE_TERNARY" # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
    DATA_DIR = '/home/fyh/PNP/GPU-FHE/examples/resnet20/src'
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    he_res20_context_ = HE_res20_context("./weights_Aespa",True)


    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=False,SAVE_MIDDLE=False, PTX_TWIN=False)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, save_dir=DATA_DIR,
                             config=config))



    cryptoContext.pre_encode_type = "middle"
    pkl_path = None
    if config.SAVE_MIDDLE==False:
        # file_name = "encode_20250412_221730" # baseline
        # file_name= "encode_20250415_215214"  # todo: add aespa pkl
        file_name = "encode_20250416_224643"
        pkl_path = os.path.join("/home/fyh/PNP/GPU-FHE/examples/resnet20/src", file_name + ".pkl")
        # load_encode_pkl(file_name, he_res20_context_)
    load_weight(pkl_path, cryptoContext)

    print("start executeResNet20")
    executeResNet20(he_res20_context_, cryptoContext, openfhe_context)

def homo_aespa(a1_x,a2,a0,cryptoContext):
    # get (a1x)^2
    a1_x2 = fhe.homo_square(a1_x, cryptoContext)
    # get a2_x^2
    a2_x2 = fhe.homo_mul_pt(a1_x2,a2,cryptoContext)

    res = fhe.homo_add(a2_x2,a1_x,cryptoContext)
    print('name:res_1', cryptoContext.openfhe_context.decrypt(res).cpu().numpy().reshape(-1))
    print('name:res_1_size', np.size(cryptoContext.openfhe_context.decrypt(res).cpu().numpy().reshape(-1)))

    res = fhe.homo_add_pt(res, a0, cryptoContext)

    print('name:res_2', cryptoContext.openfhe_context.decrypt(res).cpu().numpy().reshape(-1))

    return res

def homo_aespa_a0(a1_x,a2,a0_filename,he_res20_ctx,cryptoContext):
    # get (a1x)^2
    a1_x2 = fhe.homo_square(a1_x, cryptoContext)
    # get a2_x^2
    a2_x2 = fhe.homo_mul_pt(a1_x2,a2,cryptoContext)

    res = fhe.homo_add(a2_x2,a1_x,cryptoContext)
    res = cryptoContext.openfhe_context.decrypt(res).cpu().numpy().reshape(-1)

    a0 = read_aespa_value(a0_filename,np.size(res.reshape(-1)))
    res = res + a0

    res = cryptoContext.openfhe_context.encrypt(
        res,
        1,
        cryptoContext.L - 11,
        he_res20_ctx.cur_num_slots,
    )
    print('name:res_2', cryptoContext.openfhe_context.decrypt(res).cpu().numpy().reshape(-1))
    return res

# def Aespa(a1_x,a2_filename,a0_filename,he_res20_ctx,cryptoContext):
#     #     明文下的aespa，首先是将密文状态下的a1_x解密为一个tensor，然后用对应的运算
#     slots = a1_x.slots
#     a1_x = np.array(cryptoContext.openfhe_context.decrypt(a1_x).cpu().numpy().reshape(-1))
#     a1_x = a1_x[:slots]
#     a1_x = torch.tensor(a1_x)
#     a2 = torch.tensor(read_aespa_value(a2_filename,a1_x.size()[0]))
#     a0 = torch.tensor(read_aespa_value(a0_filename, a1_x.size()[0]))
#     a2_x2 = a2 * (a1_x**2)
#     res = a2_x2 + a1_x + a0
#     res = cryptoContext.openfhe_context.encrypt(res,1,0,slots)
#     return res

def Aespa(x,filename,cryptoContext):
    a1_filename = filename + '-a1'
    a2_filename = filename + '-a2'
    a0_filename = filename + '-bias'
    # 明文x
    slots = x.slots
    x = np.array(cryptoContext.openfhe_context.decrypt(x).cpu().numpy().reshape(-1))
    print('Max:{}'.format(filename), np.max(x))
    x = x[:slots]
    x = torch.tensor(x)

    a1 = torch.tensor(read_aespa_value(a1_filename, slots))
    a2 = torch.tensor(read_aespa_value(a2_filename,slots))
    a0 = torch.tensor(read_aespa_value(a0_filename, slots))
    part1 = a2 * (x**2)
    part2 = a1 * x
    res = part1 + part2 +a0
    res = cryptoContext.openfhe_context.encrypt(res,1,0,slots)
    # print('{}:a0'.format(a0_filename),a0)
    return res

def read_aespa_value(filename,target_len, scale=1.0):
    values = []
    val_name = filename
    filename = DATA_DIR + '/weights_Aespa/' + filename + '.bin'
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

    values = np.array(values, dtype=np.double)
    n = len(values)
    if target_len % n != 0:
        raise ValueError(f"target_len ({target_len}) must be a multiple of the original array's length ({n})")

    k = target_len // n
    return np.repeat(values, k)

if __name__ == "__main__":
    resnet20()

"""
当前这个程序有什么问题？
（1）无法正确解密：
部分aespa在密文add a0时出现误差过大（bug1），导致无法解密。
解法:将这部分aespa（2，6，10，12，14，18）中的加a0的部分转换为明文加法
（2）与明文模型输出不匹配：
主要现象为输出同质化过于严重，检查后表现为在aespa中的x过小，导致输出基本等于a0，所以输出/预测严重同质化；
原因：输入


"""
