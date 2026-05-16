import argparse
import contextlib
import datetime
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append("/".join(os.getcwd().split("/")[:-5]))
sys.path.append("/".join(os.getcwd().split("/")[:-4]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
from termcolor import colored
import easyfhe as torch
import easyfhe.fhe as fhe
from examples.resnet.src.convs import *
from examples.resnet.src.weight_pack import WeightPack
from examples.utils.utils import *

# for debug
from examples.resnet.gen_aespa_weights.HerPN import get_resnet20_HerPN, change_all_HerPN_by_PAF_MutalChannel

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))


def _parse_args():
    parser = argparse.ArgumentParser()
    fhe.add_runtime_args(parser, default_device=os.environ.get("EASYFHE_DEVICE", "cuda"))
    fhe.add_output_args(parser)
    parser.add_argument("--total", type=int, default=int(os.environ.get("EASYFHE_TOTAL", "1")))
    return parser.parse_known_args()[0]


ARGS = _parse_args()

# # config2
total = ARGS.total
SAVE_END = ARGS.save_end
SAVE_MIDDLE = ARGS.save_middle
weights_path = os.environ.get(
    "EASYFHE_RESNET20_AESPA_WEIGHTS",
    str(SCRIPT_DIR.parent / "resnet20_aespa_weights.npz"),
)

rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                     1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]
maxLevelsRemaining = 12
logBsSlots_list = [14]
logN = 16
dnum = int(os.environ.get("EASYFHE_DNUM", "3"))
dcrtBits = 52
firstMod = 55
levelBudget_list = [[4, 4]]
secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
rescaleTech = "FIXEDMANUAL"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
device = ARGS.device
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
print("device: ", device)
print("weights_path=", weights_path)

BOOTSTRAP_CONSTANTS = {}
BOOTSTRAP_TRACE_COUNT = 0
REFERENCE_INTERNALS = {}


def _trace_bootstrap_io(cipher, result, cryptoContext):
    before = cryptoContext.decrypt(cipher).cpu().numpy().reshape(-1)
    after = cryptoContext.decrypt(result).cpu().numpy().reshape(-1)
    size = min(before.size, after.size)
    diff = after[:size] - before[:size]
    print(
        "[bootstrap trace]",
        "idx=", BOOTSTRAP_TRACE_COUNT,
        "before_state=", (cipher.slots, cipher.cur_limbs, cipher.noise_deg, cipher.scaling_factor),
        "after_state=", (result.slots, result.cur_limbs, result.noise_deg, result.scaling_factor),
        "max_abs=", float(np.max(np.abs(diff))),
        "mean_abs=", float(np.mean(np.abs(diff))),
        "rmse=", float(np.sqrt(np.mean(diff * diff))),
        "before[:10]=", before[:10],
        "after[:10]=", after[:10],
    )


def _load_reference_model():
    model = get_resnet20_HerPN(num_classes=10)
    model_path = SCRIPT_DIR.parent / "gen_aespa_weights" / "ResNet20_Aespa.pth"
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    model = change_all_HerPN_by_PAF_MutalChannel(model)
    model.eval()
    return model


def _reference_features(image_vector):
    ref_input = torch.tensor(image_vector, dtype=torch.float32)
    ref_input = torch.stack(
        [ref_input[i * 1024: (i + 1) * 1024].view(32, 32) for i in range(3)],
        dim=0,
    ).unsqueeze(0)
    model = _load_reference_model()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        features = []
        internals = {}

        x = model.conv1(ref_input)
        x = model.HerPN1(x)
        features.append(x)

        for block_idx, block in enumerate(model.layer1, start=1):
            identity = x
            out = block.conv1(x)
            out = block.HerPN1(out)
            internals[f"layer1.block{block_idx}.herpn1"] = out.reshape(-1).numpy()
            out = block.conv2(out)
            out = out + identity
            internals[f"layer1.block{block_idx}.pre_herpn2"] = out.reshape(-1).numpy()
            out = block.HerPN2(out)
            internals[f"layer1.block{block_idx}"] = out.reshape(-1).numpy()
            x = out
        features.append(x)

        x = model.layer2(x)
        features.append(x)

        x = model.layer3(x)
        features.append(x)

        x = model.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = model.fc(x)

    return (
        logits.reshape(-1).numpy(),
        [feature.reshape(-1).numpy() for feature in features],
        internals,
    )


def _trace_feature(name, cipher, expected, cryptoContext):
    actual = cryptoContext.decrypt(cipher).cpu().numpy().reshape(-1)
    size = min(actual.size, expected.size)
    diff = actual[:size] - expected[:size]
    print(
        "[feature trace]",
        name,
        "slots=", cipher.slots,
        "limbs=", cipher.cur_limbs,
        "noise=", cipher.noise_deg,
        "max_abs=", float(np.max(np.abs(diff))),
        "mean_abs=", float(np.mean(np.abs(diff))),
        "rmse=", float(np.sqrt(np.mean(diff * diff))),
        "actual[:10]=", actual[:10],
        "expected[:10]=", expected[:10],
    )


def _trace_internal_feature(name, cipher, cryptoContext):
    if os.environ.get("EASYFHE_TRACE_FEATURES", "0") != "1":
        return
    if name not in REFERENCE_INTERNALS:
        return
    _trace_feature(name, cipher, REFERENCE_INTERNALS[name], cryptoContext)
    _maybe_stop_after_trace(name)


def _maybe_stop_after_trace(name):
    if os.environ.get("EASYFHE_TRACE_STOP_AFTER", "") == name:
        raise SystemExit(0)


def homo_bootstrap(cipher, L0, log_bs_slots, level_budget, cryptoContext):
    global BOOTSTRAP_TRACE_COUNT
    if os.environ.get("EASYFHE_SKIP_BOOTSTRAP", "0") == "1":
        return cipher
    result = fhe.homo_bootstrap(
        cipher,
        cryptoContext,
        BOOTSTRAP_CONSTANTS[int(log_bs_slots)],
        L0=L0,
    )
    if os.environ.get("EASYFHE_TRACE_BOOTSTRAP", "0") == "1":
        BOOTSTRAP_TRACE_COUNT += 1
        _trace_bootstrap_io(cipher, result, cryptoContext)
        limit = int(os.environ.get("EASYFHE_TRACE_BOOTSTRAP_LIMIT", "0"))
        if limit and BOOTSTRAP_TRACE_COUNT >= limit:
            raise SystemExit(0)
    return result


def initial_layer(input, cryptoContext, weights):
    scale = 1  # normalized_deltas[0][0]
    res = conv_initial(input, 32, 1, 16, scale, cryptoContext, weights)
    res = fhe.align_to(res, fhe.CipherState(res.cur_limbs - (1), res.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res = homo_Aespa_perfect_square(res, "conv1bn1", cryptoContext, weights)
    return res

def layer1(input, cryptoContext, weights):
    scale = 1  # normalized_deltas[1][0]
    # layer[0],block[0],conv1
    res1 = conv(input, 32, 1, 16, -1024, 1, 1, 0, scale, cryptoContext, weights)
    res1 = fhe.align_to(res1, fhe.CipherState(res1.cur_limbs - (1), res1.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_Aespa_perfect_square(res1, f"layer{1}-conv{1}bn{1}", cryptoContext, weights)

    # layer[0],block[0],conv2 and shorcut
    scale = 1  # normalized_deltas[1][1]
    # res1 = a1*x,shortcut = input = y
    res1 = conv(res1, 32, 1, 16, -1024, 1, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        input = fhe.align_to(input, fhe.CipherState(input.cur_limbs - (input.cur_limbs - res1.cur_limbs), input.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = weights.encode_for_cipher(f"layer{1}-conv{2}bn{2}-A2", input, cryptoContext, scale)
    A2y = fhe.homo_mul_pt(input, A2, cryptoContext)
    res1 = fhe.homo_add(res1, A2y, cryptoContext)
    res1 = fhe.align_to(res1, fhe.CipherState(res1.cur_limbs - (1), res1.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    _trace_internal_feature("layer1.block1.pre_herpn2", res1, cryptoContext)
    res1 = homo_Aespa_perfect_square(res1, f"layer{1}-conv{2}bn{2}", cryptoContext, weights)
    _trace_internal_feature("layer1.block1", res1, cryptoContext)

    scale = 1  # normalized_deltas[1][2]
    # layer[0],block[1],conv1
    res2 = conv(res1, 32, 1, 16, -1024, 2, 1, 0, scale, cryptoContext, weights)
    res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (1), res2.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{2}-conv{1}bn{1}", cryptoContext, weights)
    _trace_internal_feature("layer1.block2.herpn1", res2, cryptoContext)

    # layer[0],block[1],conv2 and shorcut
    scale = 1  # normalized_deltas[1][3]
    res2 = conv(res2, 32, 1, 16, -1024, 2, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.align_to(res1, fhe.CipherState(res1.cur_limbs - (res1.cur_limbs - res2.cur_limbs), res1.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = weights.encode_for_cipher(f"layer{2}-conv{2}bn{2}-A2", res1, cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (1), res2.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    _trace_internal_feature("layer1.block2.pre_herpn2", res2, cryptoContext)

    drop = maxLevelsRemaining - 5
    res2 = homo_bootstrap(res2, cryptoContext.L - drop, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    res2 = homo_Aespa_perfect_square(res2, f"layer{2}-conv{2}bn{2}", cryptoContext, weights)
    _trace_internal_feature("layer1.block2", res2, cryptoContext)

    # layer[0],block[2],conv1
    scale = 1  # normalized_deltas[1][4]
    res3 = conv(res2, 32, 1, 16, -1024, 3, 1, 0, scale, cryptoContext, weights)
    res3 = fhe.align_to(res3, fhe.CipherState(res3.cur_limbs - (1), res3.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{3}-conv{1}bn{1}", cryptoContext, weights)

    scale = 1  # normalized_deltas[1][5]
    res3 = conv(res3, 32, 1, 16, -1024, 3, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (res2.cur_limbs - res3.cur_limbs), res2.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = weights.encode_for_cipher(f"layer{3}-conv{2}bn{2}-A2", res2, cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res2, A2, cryptoContext)
    res3 = fhe.homo_add(res3, A2y, cryptoContext)
    res3 = fhe.align_to(res3, fhe.CipherState(res3.cur_limbs - (1), res3.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_bootstrap(res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res3 = homo_Aespa_perfect_square(res3, f"layer{3}-conv{2}bn{2}", cryptoContext, weights)
    _trace_internal_feature("layer1.block3", res3, cryptoContext)

    return res3

def layer2(input, cryptoContext, weights):
    scaleSx = 1  # normalized_deltas[2][0]
    scaleDx = 1  # normalized_deltas[2][1]
    # boot_in = homo_bootstrap(input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    boot_in = input
    res1sx0 = conv(boot_in, 32, 1, 16, -1024, 4, 1, 0, scaleSx, cryptoContext, weights)
    res1sx1 = conv(boot_in, 32, 1, 16, -1024, 4, 1, 16, scaleSx, cryptoContext, weights)
    res1sx0 = fhe.align_to(res1sx0, fhe.CipherState(res1sx0.cur_limbs - (1), res1sx0.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.align_to(res1sx1, fhe.CipherState(res1sx1.cur_limbs - (1), res1sx1.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI

    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.align_to(boot_in, fhe.CipherState(boot_in.cur_limbs - (2), boot_in.noise_deg), cryptoContext)  # RESCALE ADD BY ZRJI

    res1dx0 = convbn_dx(boot_in, 16, -1024, 4, 1, 0, "1", scaleDx, cryptoContext, weights)
    res1dx1 = convbn_dx(boot_in, 16, -1024, 4, 1, 16, "2", scaleDx, cryptoContext, weights)

    fullpackSx = downsample1024to256(res1sx0, res1sx1, 16, 1, cryptoContext, weights)
    fullpackDx = downsample1024to256(res1dx0, res1dx1, 16, 1, cryptoContext, weights)
    fullpackSx = fhe.align_to(fullpackSx, fhe.CipherState(fullpackSx.cur_limbs - (1), fullpackSx.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI

    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{4}-conv{1}bn{1}", cryptoContext, weights)

    fullpackSx = conv(fullpackSx, 16, 1, 32, -256, 4, 2, 0, scaleDx, cryptoContext, weights)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)

    drop = maxLevelsRemaining - 9
    res1 = homo_bootstrap(res1, cryptoContext.L-drop, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_Aespa_perfect_square(res1, f"layer{4}-conv{2}bn{2}", cryptoContext, weights)

    # layer[2]block[1]
    scale = 1  # normalized_deltas[2][2]
    res2 = conv(res1, 16, 1, 32, -256, 5, 1, 0, scale, cryptoContext, weights)
    res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (1), res2.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{5}-conv{1}bn{1}", cryptoContext, weights)

    scale = 1  # normalized_deltas[2][3]
    res2 = conv(res2, 16, 1, 32, -256, 5, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.align_to(res1, fhe.CipherState(res1.cur_limbs - (res1.cur_limbs - res2.cur_limbs), res1.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = weights.encode_for_cipher(f"layer{5}-conv{2}bn{2}-A2", res1, cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (1), res2.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{5}-conv{2}bn{2}", cryptoContext, weights)

    # layer[2]block[2]
    scale = 1  # normalized_deltas[2][4]
    res3 = conv(res2, 16, 1, 32, -256, 6, 1, 0, scale, cryptoContext, weights)
    res3 = fhe.align_to(res3, fhe.CipherState(res3.cur_limbs - (1), res3.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{6}-conv{1}bn{1}", cryptoContext, weights)

    scale = 1  # normalized_deltas[2][5]
    res3 = conv(res3, 16, 1, 32, -256, 6, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (res2.cur_limbs - res3.cur_limbs), res2.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = weights.encode_for_cipher(f"layer{6}-conv{2}bn{2}-A2", res2, cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res2, A2, cryptoContext)
    res3 = fhe.homo_add(res3, A2y, cryptoContext)
    res3 = fhe.align_to(res3, fhe.CipherState(res3.cur_limbs - (1), res3.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_bootstrap(res3, cryptoContext.L-1, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res3 = homo_Aespa_perfect_square(res3, f"layer{6}-conv{2}bn{2}", cryptoContext, weights)

    return res3

def layer3(input, cryptoContext, weights):
    scaleSx = 1  # normalized_deltas[3][0]
    scaleDx = 1  # normalized_deltas[3][1]

    boot_in = input
    # boot_in = homo_bootstrap(input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)  # 13
    res1sx0 = conv(boot_in, 16, 1, 32, -256, 7, 1, 0, scaleSx, cryptoContext, weights)
    res1sx1 = conv(boot_in, 16, 1, 32, -256, 7, 1, 32, scaleSx, cryptoContext, weights)
    res1sx0 = fhe.align_to(res1sx0, fhe.CipherState(res1sx0.cur_limbs - (1), res1sx0.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res1sx1 = fhe.align_to(res1sx1, fhe.CipherState(res1sx1.cur_limbs - (1), res1sx1.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        boot_in = fhe.align_to(boot_in, fhe.CipherState(boot_in.cur_limbs - (2), boot_in.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    res1dx0 = convbn_dx(boot_in, 32, -256, 7, 1, 0, "1", scaleDx, cryptoContext, weights)

    res1dx1 = convbn_dx(boot_in, 32, -256, 7, 1, 32, "2", scaleDx, cryptoContext, weights)

    fullpackSx = downsample256to64(res1sx0, res1sx1, 32, cryptoContext, weights)
    fullpackDx = downsample256to64(res1dx0, res1dx1, 32, cryptoContext, weights)
    fullpackSx = fhe.align_to(fullpackSx, fhe.CipherState(fullpackSx.cur_limbs - (1), fullpackSx.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI

    fullpackSx = homo_Aespa_perfect_square(fullpackSx, f"layer{7}-conv{1}bn{1}", cryptoContext, weights)

    fullpackSx = conv(fullpackSx, 8, 1, 64, -64, 7, 2, 0, scaleDx, cryptoContext, weights)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.align_to(res1, fhe.CipherState(res1.cur_limbs - (1), res1.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res1 = homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_Aespa_perfect_square(res1, f"layer{7}-conv{2}bn{2}", cryptoContext, weights)

    scale = 1  # normalized_deltas[3][2]
    res2 = conv(res1, 8, 1, 64, -64, 8, 1, 0, scale, cryptoContext, weights)
    res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (1), res2.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{8}-conv{1}bn{1}", cryptoContext, weights)

    scale = 1  # normalized_deltas[3][3]
    res2 = conv(res2, 8, 1, 64, -64, 8, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res1 = fhe.align_to(res1, fhe.CipherState(res1.cur_limbs - (res1.cur_limbs - res2.cur_limbs), res1.noise_deg), cryptoContext)  # drop_last_elements ADD BY ZRJI
    A2 = weights.encode_for_cipher(f"layer{8}-conv{2}bn{2}-A2", res1, cryptoContext, scale)
    A2y = fhe.homo_mul_pt(res1, A2, cryptoContext)
    res2 = fhe.homo_add(res2, A2y, cryptoContext)
    res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (1), res2.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res2 = homo_Aespa_perfect_square(res2, f"layer{8}-conv{2}bn{2}", cryptoContext, weights)

    scale = 1  # normalized_deltas[3][4]
    res3 = conv(res2, 8, 1, 64, -64, 9, 1, 0, scale, cryptoContext, weights)
    res3 = fhe.align_to(res3, fhe.CipherState(res3.cur_limbs - (1), res3.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{9}-conv{1}bn{1}", cryptoContext, weights)

    scale = 1  # normalized_deltas[3][5]
    res3 = conv(res3, 8, 1, 64, -64, 9, 2, 0, scale, cryptoContext, weights)
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        res2 = fhe.align_to(res2, fhe.CipherState(res2.cur_limbs - (res2.cur_limbs - res3.cur_limbs), res2.noise_deg), cryptoContext)
    A2 = weights.encode_for_cipher(f"layer{9}-conv{2}bn{2}-A2", res2, cryptoContext, scale)  # drop_last_elements ADD BY ZRJI
    A2y = fhe.homo_mul_pt(res2, A2, cryptoContext)
    res3 = fhe.homo_add(res3, A2y, cryptoContext)
    res3 = fhe.align_to(res3, fhe.CipherState(res3.cur_limbs - (1), res3.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    res3 = homo_Aespa_perfect_square(res3, f"layer{9}-conv{2}bn{2}", cryptoContext, weights)

    return res3

def final_layer(input, cryptoContext, weights):
    # 64*8*8
    res = rotsum(input, 64, cryptoContext)
    res = fhe.homo_mul_pt(
        res,
        weights.encode(f"mask_mod_64_{1.0 / 64.0}_{res.slots}", cryptoContext.L - res.cur_limbs, res.slots, cryptoContext),
        cryptoContext,
    )
    res = repeat(res, 16, cryptoContext)
    res = fhe.align_to(res, fhe.CipherState(res.cur_limbs - (1), res.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    weight = weights.encode_for_cipher(f"fc_{res.slots}", res, cryptoContext)
    res = fhe.homo_mul_pt(res, weight, cryptoContext)
    res = fhe.align_to(res, fhe.CipherState(res.cur_limbs - (1), res.noise_deg - (1)), cryptoContext)
    res = rotsum_padded(res, 64, 64, cryptoContext)

    bias = weights.encode_for_cipher(f"bias_{res.slots}", res, cryptoContext)
    res = fhe.homo_add_pt(res, bias, cryptoContext)
    return res


def executeResNet20(cryptoContext, weights):
    cryptoContext.zero_32K = cryptoContext.encrypt(np.zeros(2 ** 15), cryptoContext.device, 2, 0, 2 ** 15)
    cryptoContext.zero_16K = cryptoContext.encrypt(np.zeros(2 ** 14), cryptoContext.device, 2, 0, 2 ** 14)

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
    time_list = []
    correct = 0
    for i in range(total):
        global REFERENCE_INTERNALS
        image_vector, label, index = read_image(i)
        trace_features = os.environ.get("EASYFHE_TRACE_FEATURES", "0") == "1"
        if trace_features:
            ref_logits, ref_features, REFERENCE_INTERNALS = _reference_features(image_vector)
            print(
                "[feature trace]",
                "ref_logits[:10]=", ref_logits[:10],
                "pred=", int(np.argmax(ref_logits[:10])),
            )
        else:
            REFERENCE_INTERNALS = {}
        # # 明文模型输出
        # input = torch.tensor(image_vector, device="cuda",dtype=torch.float32)
        # input = torch.stack([input[i * 1024: (i + 1) * 1024].view(32, 32) for i in range(3)], dim=0)
        # x , fea = model(input,fea_out=True)

        in_ct = cryptoContext.encrypt(
            image_vector,
            cryptoContext.device,
            1,
            19,
            16 * 32 * 32,
        )
        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()

        # 密文推理
        if cryptoContext.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        firstLayer = initial_layer(in_ct, cryptoContext, weights)
        if trace_features:
            _trace_feature("firstLayer", firstLayer, ref_features[0], cryptoContext)
            _maybe_stop_after_trace("firstLayer")
        resLayer1 = layer1(firstLayer, cryptoContext, weights)
        if trace_features:
            _trace_feature("resLayer1", resLayer1, ref_features[1], cryptoContext)
            _maybe_stop_after_trace("resLayer1")
        resLayer2 = layer2(resLayer1, cryptoContext, weights)
        if trace_features:
            _trace_feature("resLayer2", resLayer2, ref_features[2], cryptoContext)
            _maybe_stop_after_trace("resLayer2")
        resLayer3 = layer3(resLayer2, cryptoContext, weights)
        if trace_features:
            _trace_feature("resLayer3", resLayer3, ref_features[3], cryptoContext)
            _maybe_stop_after_trace("resLayer3")
            print("[feature trace]", "ref final logits[:10]=", ref_logits[:10])
        finalRes = final_layer(resLayer3, cryptoContext, weights)
        if cryptoContext.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        print("time: ", end_time - start_time)
        print("after processing image ", i, "time: ", datetime.datetime.now())
        time_list.append(end_time - start_time)
        # 对比明密文loss
        # conv_init = fea[0].flatten().reshape(-1)
        # init_out = cryptoContext.decrypt(firstLayer).cpu().numpy().reshape(-1)
        # init_out = torch.from_numpy(init_out).to(device)
        # loss = torch.sum((conv_init - init_out) ** 2)
        # print("loss: ", loss)

        # temp = cryptoContext.decrypt(resLayer1).cpu().numpy().reshape(-1)
        # print('name:resLayer1', temp)
        # fea_out = torch.tensor(fea[1].flatten().reshape(-1), device="cuda:0")
        # print('fea1', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer1', loss)

        # temp = cryptoContext.decrypt(resLayer2).cpu().numpy().reshape(-1)
        # print('name:resLayer2', temp)
        # fea_out = torch.tensor(fea[2].flatten().reshape(-1), device="cuda:0")
        # print('fea2', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer2',loss)
        #
        # temp = cryptoContext.decrypt(resLayer3).cpu().numpy().reshape(-1)
        # print('name:resLayer3', temp)
        # fea_out = torch.tensor(fea[3].flatten().reshape(-1), device="cuda:0")
        # print('fea3', fea_out)
        # temp = torch.tensor(temp, device="cuda:0")
        # loss = torch.sum((fea_out - temp) ** 2)
        # print('resLayer3',loss)
        try:
            clear_result = cryptoContext.decrypt(finalRes)
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
    avg = sum(time_list[1:]) / (len(time_list)-1) if len(time_list) > 1 else time_list[0]
    min_val = min(time_list)
    print(f"!!!ver2: {time_list}")
    print("avg:", avg)
    print("min_val:", min_val)

def resnet20():
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    options = fhe.runtime_options_from_args(ARGS)
    bootstrap_specs = tuple(
        fhe.BootstrapSpec(log_bs_slots, tuple(level_budget))
        for log_bs_slots, level_budget in zip(logBsSlots_list, levelBudget_list)
    )
    cryptoContext = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=fhe.bootstrap_depth(maxLevelsRemaining, bootstrap_specs, secretKeyDist),
            log_n=logN,
            dnum=dnum,
            dcrt_bits=dcrtBits,
            first_mod=firstMod,
            secret_key_dist=secretKeyDist,
            rescale_tech=rescaleTech,
            rotations=tuple(rotate_index_list),
        ),
        device=device,
        options=options,
    )
    for log_bs_slots, level_budget in zip(logBsSlots_list, levelBudget_list):
        BOOTSTRAP_CONSTANTS[int(log_bs_slots)] = fhe.generate_bootstrap_constants(
            cryptoContext, log_bs_slots, level_budget, maxLevelsRemaining
        )
    print("cryptoContext: ", cryptoContext)
    weights = WeightPack.from_npz(weights_path)
    print("weights loaded:", len(weights))

    print("start executeResNet20")
    executeResNet20(cryptoContext, weights)


def homo_Aespa_perfect_square(x, filename, cryptoContext, weights):
    if x.noise_deg >1:
        x = fhe.align_to(x, fhe.CipherState(x.cur_limbs - (1), x.noise_deg - (1)), cryptoContext) #RESCALE ADD BY ZRJI
    n1_filename = filename + '-n1'
    n2_filename = filename + '-n2'
    slots = x.slots
    scale = 1  # 1
    n1 = weights.encode_for_cipher(n1_filename, x, cryptoContext, scale)
    temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
    perfect_squre = fhe.homo_square(temp1, cryptoContext)
    perfect_squre = fhe.align_to(perfect_squre, fhe.CipherState(perfect_squre.cur_limbs - (1), perfect_squre.noise_deg - (1)), cryptoContext)  # RESCALE ADD BY ZRJI
    n2 = weights.encode_for_cipher(n2_filename, perfect_squre, cryptoContext, scale)
    res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
    return res


if __name__ == "__main__":
    resnet20()
