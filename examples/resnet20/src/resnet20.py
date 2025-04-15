import os, sys, datetime, time
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
        a2 = read_values_from_file(cryptoContext, "conv1bn1-a2", cryptoContext.L - input.cur_limbs, 1, 16384, scale)
        res = homo_aespa(res,a2,bias,cryptoContext)
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
        a2 = read_values_from_file(cryptoContext,  f"layer{1}-conv{1}bn{1}-a2",cryptoContext.L-input.cur_limbs,1,16384,scale)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res1 = homo_aespa(res1,a2,bias1,cryptoContext)
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
        a1 = read_values_from_file(cryptoContext,  f"layer{1}-conv{2}bn{2}-a1",cryptoContext.L-res1.cur_limbs,1,16384,scale)
        a1y = fhe.homo_mul_pt(input,a1,cryptoContext)
        res1 = fhe.homo_add(res1,a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{1}-conv{2}bn{2}-a2",cryptoContext.L-res1.cur_limbs,1,16384,scale)

        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res1 = homo_aespa(res1,a2,bias1,cryptoContext)
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
        a2 = read_values_from_file(cryptoContext,  f"layer{2}-conv{1}bn{1}-a2",cryptoContext.L-res1.cur_limbs,1,16384,scale)
        res1 = homo_aespa(res1,a2,bias2,cryptoContext)
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

        a1 = read_values_from_file(cryptoContext,  f"layer{2}-conv{2}bn{2}-a1",cryptoContext.L-res2.cur_limbs,1,16384,scale)
        a1y = fhe.homo_mul_pt(res1, a1, cryptoContext)
        res2 = fhe.homo_add(res2, a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{2}-conv{2}bn{2}-a2",cryptoContext.L-res2.cur_limbs,1,16384,scale)

        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res2 = homo_aespa(res2, a2, bias2,cryptoContext)
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
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )

        a2 = read_values_from_file(cryptoContext,  f"layer{3}-conv{1}bn{1}-a2",cryptoContext.L-res2.cur_limbs,1,16384,scale)
        res3 = homo_aespa(res3, a2, bias3,cryptoContext)
    else:
        res3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][5]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn(res3, 3, 2, scale, he_res20_ctx, cryptoContext)
        a1 = read_values_from_file(cryptoContext,  f"layer{3}-conv{2}bn{2}-a1",cryptoContext.L-res3.cur_limbs,1,16384,scale)
        a1y = fhe.homo_mul_pt(res2, a1, cryptoContext)
        res3 = fhe.homo_add(res3, a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{3}-conv{2}bn{2}-a2",cryptoContext.L-res3.cur_limbs,1,16384,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
        )
        res3 = homo_aespa(res3, a2, bias3,cryptoContext)
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
    fullpackDx = downsample1024to256(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    if he_res20_ctx.aespa:
        a2 = read_values_from_file(cryptoContext, f"layer{4}-conv{1}bn{1}-a2",
                                      cryptoContext.L - boot_in.cur_limbs, 1, 16384, scaleSx)
        fullpackSx = homo_aespa(fullpackSx, a2, bias,cryptoContext)
    else:
        fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)

    he_res20_ctx.cur_num_slots = 8192
    if he_res20_ctx.aespa:
        fullpackSx, bias = convbn2(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext)
        # Todo:check a1 is mix with downsample's BN
        a1 = read_values_from_file(cryptoContext,  f"layer{4}-conv{2}bn{2}-a1",cryptoContext.L-fullpackSx.cur_limbs,1,8192,scaleSx)
        fullpackDx = fhe.homo_mul_pt(fullpackDx, a1,cryptoContext)
        res1 = fhe.homo_add(fullpackSx, fullpackDx,cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{4}-conv{2}bn{2}-a2",cryptoContext.L-fullpackSx.cur_limbs,1,8192,scaleSx)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res1 = homo_aespa(res1,a2,bias,cryptoContext)
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
        res2 = homo_aespa(res2, a2, bias2,cryptoContext)
    else:
        res2 = convbn2(res1, 5, 1, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    if he_res20_ctx.aespa:
        res2,bias2 = convbn2(res2, 5, 2, scale, he_res20_ctx, cryptoContext)
        a1 = read_values_from_file(cryptoContext,  f"layer{5}-conv{2}bn{2}-a1",cryptoContext.L-res2.cur_limbs,1,8192,scale)
        a1y = fhe.homo_mul_pt(res1,a1,cryptoContext)
        res2 = fhe.homo_add(res2,a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{5}-conv{2}bn{2}-a2",cryptoContext.L-res2.cur_limbs,1,8192,scale)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res2 = homo_aespa(res2, a2, bias2,cryptoContext)
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
        res3 = homo_aespa(res3, a2, bias3,cryptoContext)
    else:
        res3 = convbn2(res2, 6, 1, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn2(res3, 6, 2, scale, he_res20_ctx, cryptoContext)
        a1 = read_values_from_file(cryptoContext,  f"layer{6}-conv{2}bn{2}-a1",cryptoContext.L-res3.cur_limbs,1,8192,scale)
        a1y = fhe.homo_mul_pt(res2, a1, cryptoContext)
        res3 = fhe.homo_add(res3, a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext,  f"layer{6}-conv{2}bn{2}-a2",cryptoContext.L-res3.cur_limbs,1,8192,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
        )
        res3 = homo_aespa(res3, a2, bias3,cryptoContext)
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
        fullpackSx = homo_aespa(fullpackSx, a2, bias,cryptoContext)
    else:
        fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)

    he_res20_ctx.cur_num_slots = 4096
    if he_res20_ctx.aespa:
        fullpackSx,bias = convbn3(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext)
        a1 = read_values_from_file(cryptoContext, f"layer{7}-conv{2}bn{2}-a1",cryptoContext.L-fullpackSx.cur_limbs,1,4096,scaleDx)
        fullpackDx = fhe.homo_mul_pt(fullpackDx, a1,cryptoContext)
        res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
        a2 = read_values_from_file(cryptoContext, f"layer{7}-conv{2}bn{2}-a2",cryptoContext.L-fullpackSx.cur_limbs,1,4096,scaleDx)
        res1 = fhe.homo_bootstrap(
            res1, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res1 = homo_aespa(res1, a2, bias,cryptoContext)
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
        res2 = homo_aespa(res2, a2, bias2,cryptoContext)
    else:
        res2 = convbn3(res1, 8, 1, scale, he_res20_ctx, cryptoContext)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    if he_res20_ctx.aespa:
        res2, bias2 = convbn3(res2, 8, 2, scale, he_res20_ctx, cryptoContext)
        a1 = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-a1",cryptoContext.L-res2.cur_limbs,1,4096,scale)
        a1y = fhe.homo_mul_pt(res1, a1, cryptoContext)
        res2 = fhe.homo_add(res2, a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-a2",cryptoContext.L-res2.cur_limbs,1,4096,scale)
        res2 = fhe.homo_bootstrap(
            res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res2 = homo_aespa(res2, a2, bias2,cryptoContext)
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
        res3 = homo_aespa(res3, a2, bias3,cryptoContext)
    else:
        res3 = convbn3(res2, 9, 1, scale, he_res20_ctx, cryptoContext)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    if he_res20_ctx.aespa:
        res3, bias3 = convbn3(res3, 9, 2, scale, he_res20_ctx, cryptoContext)
        a1 =read_values_from_file(cryptoContext, f"layer{9}-conv{2}bn{2}-a1",cryptoContext.L-res3.cur_limbs,1,4096,scale)
        a1y = fhe.homo_mul_pt(res2, a1, cryptoContext)
        res3 = fhe.homo_add(res3, a1y,cryptoContext)
        a2 = read_values_from_file(cryptoContext, f"layer{9}-conv{2}bn{2}-a2",cryptoContext.L-res3.cur_limbs,1,4096,scale)
        res3 = fhe.homo_bootstrap(
            res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
        )
        res3 = homo_aespa(res3, a2, bias3,cryptoContext)
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
    for i in range(1):
        he_res20_ctx.cur_num_slots = 1 << 14

        image_vector, label, _ = read_image(i)
        image_vector = torch.tensor(np.array(image_vector), device="cuda")
        in_ct = openfhe_context.encrypt(
            image_vector,
            1,
            cryptoContext.L - 11,
            he_res20_ctx.cur_num_slots,
        )

        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()
        firstLayer = initial_layer(in_ct, he_res20_ctx, cryptoContext)


        resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext)
        # clear_result = openfhe_context.decrypt(resLayer1)
        # clear_result = clear_result.cpu().numpy().reshape(-1)
        # print(clear_result)

        resLayer2 = layer2(resLayer1, he_res20_ctx, cryptoContext)
        # clear_result = openfhe_context.decrypt(resLayer2)
        # clear_result = clear_result.cpu().numpy().reshape(-1)
        # print(clear_result)

        resLayer3 = layer3(resLayer2, he_res20_ctx, cryptoContext)
        # clear_result = openfhe_context.decrypt(resLayer3)
        # clear_result = clear_result.cpu().numpy().reshape(-1)
        # print(clear_result)

        finalRes = final_layer(resLayer3, he_res20_ctx, cryptoContext)
        print("time: ", time.time() - start_time)
        print("after processing image ", i, "time: ", datetime.datetime.now())
        try:
            clear_result = openfhe_context.decrypt(finalRes)
            clear_result = clear_result.cpu().numpy().reshape(-1)
            max_element_idx = np.argmax(clear_result[:10])
        except RuntimeError as e:
            print(f"Decryption failed: {e}")
            clear_result = None
            max_element_idx = 11

        print("For image ", i)
        if clear_result is not None:
            print(clear_result[:10])
        else:
            print("Decryption failed, clear_result is None.")
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
    dcrtBits = 59
    firstMod = 60
    levelBudget_list = [[4, 4], [4, 4], [4, 4]]
    secretKeyDist = "SPARSE_TERNARY" # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
    DATA_DIR = '/home/fyh/PNP/GPU-FHE/examples/resnet20/src'
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    he_res20_context_ = HE_res20_context("./weights_Aespa",True)


    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=False,SAVE_MIDDLE=False)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, save_dir=DATA_DIR,
                             config=config))



    cryptoContext.pre_encode_type = "middle"
    pkl_path = None
    if config.SAVE_MIDDLE==False:
        # file_name = "encode_20250412_221730" # baseline
        # file_name = "encode_20250415_174210" # todo: aespa pkl
        file_name= "encode_20250415_190124"
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
    res = fhe.homo_add_pt(res, a0, cryptoContext)
    return res

if __name__ == "__main__":
    resnet20()
