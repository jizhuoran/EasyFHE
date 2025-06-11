import os, sys, datetime, time


sys.path.append("/".join(os.getcwd().split("/")[:-5]))
sys.path.append("/".join(os.getcwd().split("/")[:-4]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch
import numpy as np
import torch.fhe as fhe
from examples.utils import approx
from examples.resnet20.src.resnet20_convs import *
from huggingface_hub import hf_hub_download
import zipfile

DATA_DIR = os.environ["DATA_DIR"]
logBsSlots_list = [14, 13, 12]
levelBudget_list = [[4, 4], [4, 4], [4, 4]]

class HE_res20_context:
    def __init__(self, data_dir):
        self.cur_num_slots = None
        self.relu_degree = None
        self.weight_dir = data_dir

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
    res = convbn_initial(input, 16, scale, he_res20_ctx, cryptoContext, 32, 1)
    bias=read_values_from_file(cryptoContext, "conv1bn1-bias", cryptoContext.L-res.cur_limbs, 1, 16384, scale)
    res=fhe.homo_add_pt(res,bias,cryptoContext)
    res = homo_relu(res, scale, he_res20_ctx.relu_degree, cryptoContext)
    return res


def layer1(input, he_res20_ctx, cryptoContext):
    scale = normalized_deltas[1][0]

    res1 = convbn(input, 1, 1, scale, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0)
    bias = read_values_from_file(cryptoContext,  f"layer{1}-conv{1}bn{1}-bias",cryptoContext.L-res1.cur_limbs,1,16384,scale)
    res1 = fhe.homo_add_pt(res1,bias,cryptoContext)
    res1 = fhe.homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][1]
    res1 = convbn(res1, 1, 2, scale, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0)
    bias = read_values_from_file(cryptoContext, f"layer{1}-conv{2}bn{2}-bias",cryptoContext.L-res1.cur_limbs,1,16384,scale)
    res1 = fhe.homo_add_pt(res1, bias, cryptoContext)
    res1 = fhe.homo_add(
        res1, fhe.homo_mul_scalar_double(input, scale, cryptoContext), cryptoContext
    )
    res1 = fhe.homo_bootstrap(res1, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res1 = homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][2]
    res2 = convbn(res1, 2, 1, scale, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0)
    bias = read_values_from_file(cryptoContext, f"layer{2}-conv{1}bn{1}-bias",cryptoContext.L-res2.cur_limbs,1,16384,scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_bootstrap(res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][3]
    res2 = convbn(res2, 2, 2, scale, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0)
    bias = read_values_from_file(cryptoContext, f"layer{2}-conv{2}bn{2}-bias",cryptoContext.L-res2.cur_limbs,1,16384,scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(res2, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][4]
    res3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0)
    bias = read_values_from_file(cryptoContext, f"layer{3}-conv{1}bn{1}-bias",cryptoContext.L-res3.cur_limbs,1,16384,scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_bootstrap(res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][5]
    res3 = convbn(res3, 3, 2, scale, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0)
    bias = read_values_from_file(cryptoContext, f"layer{3}-conv{2}bn{2}-bias",cryptoContext.L-res3.cur_limbs,1,16384,scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_add(
        res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
    )
    res3 = fhe.homo_bootstrap(res3, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer2(input, he_res20_ctx, cryptoContext):
    scaleSx = normalized_deltas[2][0]
    scaleDx = normalized_deltas[2][1]
    boot_in = fhe.homo_bootstrap(
        input, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext
    )

    res1sx0 = convbn(boot_in, 4, 1, scaleSx, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 0, "1")
    bias = read_values_from_file(cryptoContext, f"layer{4}-conv{1}bn{1}-bias{1}",cryptoContext.L-res1sx0.cur_limbs,1,16384,scaleSx)
    res1sx0 = fhe.homo_add_pt(res1sx0, bias, cryptoContext)
    res1sx1 = convbn(boot_in, 4, 1, scaleSx, he_res20_ctx, cryptoContext, 32, 1, 16384, 16, -1024, 16, "2")
    bias = read_values_from_file(cryptoContext, f"layer{4}-conv{1}bn{1}-bias{2}",cryptoContext.L-res1sx1.cur_limbs,1,16384,scaleSx)
    res1sx1 = fhe.homo_add_pt(res1sx1, bias, cryptoContext)

    res1dx0 = convbn_dx(
        boot_in, 4, 1, scaleDx, he_res20_ctx, cryptoContext, 16384, 16, -1024, 0, "1"
    )

    res1dx1 = convbn_dx(
        boot_in, 4, 1, scaleDx, he_res20_ctx, cryptoContext, 16384, 16, -1024, 16, "2"
    )

    fullpackSx = downsample1024to256(res1sx0, res1sx1, he_res20_ctx, cryptoContext)
    fullpackDx = downsample1024to256(res1dx0, res1dx1, he_res20_ctx, cryptoContext)

    he_res20_ctx.cur_num_slots = 8192

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )
    fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 0)
    bias = read_values_from_file(cryptoContext, f"layer{4}-conv{2}bn{2}-bias",cryptoContext.L-fullpackSx.cur_limbs,1,8192,scaleDx)
    fullpackSx = fhe.homo_add_pt(fullpackSx, bias, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][2]
    res2 = convbn(res1, 5, 1, scale, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 0)
    bias = read_values_from_file(cryptoContext, f"layer{5}-conv{1}bn{1}-bias",cryptoContext.L-res2.cur_limbs,1,8192,scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    res2 = convbn(res2, 5, 2, scale, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 0)
    bias = read_values_from_file(cryptoContext, f"layer{5}-conv{2}bn{2}-bias",cryptoContext.L-res2.cur_limbs,1,8192,scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][4]
    res3 = convbn(res2, 6, 1, scale, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 0)
    bias = read_values_from_file(cryptoContext, f"layer{6}-conv{1}bn{1}-bias",cryptoContext.L-res3.cur_limbs,1,8192,scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    res3 = convbn(res3, 6, 2, scale, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 0)
    bias = read_values_from_file(cryptoContext, f"layer{6}-conv{2}bn{2}-bias",cryptoContext.L-res3.cur_limbs,1,8192,scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_add(
        res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
    )
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer3(input, he_res20_ctx, cryptoContext):
    scaleSx = normalized_deltas[3][0]
    scaleDx = normalized_deltas[3][1]

    boot_in = fhe.homo_bootstrap(
        input, cryptoContext.L, logBsSlots_list[1], levelBudget_list[1], cryptoContext
    )

    res1sx0 = convbn(boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 0, "1")
    bias = read_values_from_file(cryptoContext, f"layer{7}-conv{1}bn{1}-bias{1}",cryptoContext.L-res1sx0.cur_limbs,1,8192,scaleSx)
    res1sx0 = fhe.homo_add_pt(res1sx0, bias, cryptoContext)
    res1sx1 = convbn(boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext, 16, 1, 8192, 32, -256, 32, "2")
    bias = read_values_from_file(cryptoContext, f"layer{7}-conv{1}bn{1}-bias{2}",cryptoContext.L-res1sx1.cur_limbs,1,8192,scaleSx)
    res1sx1 = fhe.homo_add_pt(res1sx1, bias, cryptoContext)

    res1dx0 = convbn_dx(
        boot_in, 7, 1, scaleDx, he_res20_ctx, cryptoContext, 8192, 32, -256, 0, "1"
    )

    res1dx1 = convbn_dx(
        boot_in, 7, 1, scaleDx, he_res20_ctx, cryptoContext, 8192, 32, -256, 32, "2"
    )

    fullpackSx = downsample256to64(res1sx0, res1sx1, he_res20_ctx, cryptoContext)
    fullpackDx = downsample256to64(res1dx0, res1dx1, he_res20_ctx, cryptoContext)

    he_res20_ctx.cur_num_slots = 4096

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
    )
    fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext, 8, 1, 4096, 64, -64, 0)
    bias = read_values_from_file(cryptoContext, f"layer{7}-conv{2}bn{2}-bias",cryptoContext.L-fullpackSx.cur_limbs,1,4096,scaleDx)
    fullpackSx = fhe.homo_add_pt(fullpackSx, bias, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
    )
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][2]
    res2 = convbn(res1, 8, 1, scale, he_res20_ctx, cryptoContext, 8, 1, 4096, 64, -64, 0)
    bias = read_values_from_file(cryptoContext, f"layer{8}-conv{1}bn{1}-bias",cryptoContext.L-res2.cur_limbs,1,4096,scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    res2 = convbn(res2, 8, 2, scale, he_res20_ctx, cryptoContext, 8, 1, 4096, 64, -64, 0)
    bias = read_values_from_file(cryptoContext, f"layer{8}-conv{2}bn{2}-bias",cryptoContext.L-res2.cur_limbs,1,4096,scale)
    res2 = fhe.homo_add_pt(res2, bias, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    res3 = convbn(res2, 9, 1, scale, he_res20_ctx, cryptoContext, 8, 1, 4096, 64, -64, 0)
    bias = read_values_from_file(cryptoContext, f"layer{9}-conv{1}bn{1}-bias",cryptoContext.L-res3.cur_limbs,1,4096,scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    res3 = convbn(res3, 9, 2, scale, he_res20_ctx, cryptoContext, 8, 1, 4096, 64, -64, 0)
    bias = read_values_from_file(cryptoContext, f"layer{9}-conv{2}bn{2}-bias",cryptoContext.L-res3.cur_limbs,1,4096,scale)
    res3 = fhe.homo_add_pt(res3, bias, cryptoContext)
    res3 = fhe.homo_add(
        res3, fhe.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext
    )
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, cryptoContext.L, logBsSlots_list[2], levelBudget_list[2], cryptoContext
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
    res = rotsum_padded(res, 64, 64, cryptoContext)

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

    cryptoContext.zero_32K = openfhe_context.encrypt(np.zeros(2**15), cryptoContext.device, 1, 0, 2**15)
    cryptoContext.zero_16K = openfhe_context.encrypt(np.zeros(2**14), cryptoContext.device,  1, 0, 2**14)

    print("=====================================================")
    for i in range(1):
        he_res20_ctx.cur_num_slots = 1 << 14

        image_vector, label, _ = read_image(i)
        in_ct = openfhe_context.encrypt(
            image_vector,
            cryptoContext.device,
            1,
            cryptoContext.L - 11,
            he_res20_ctx.cur_num_slots,
        )

        print("start processing image ", i, "time: ", datetime.datetime.now())
        start_time = time.time()
        firstLayer = initial_layer(in_ct, he_res20_ctx, cryptoContext)
        resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext)
        resLayer2 = layer2(resLayer1, he_res20_ctx, cryptoContext)
        resLayer3 = layer3(resLayer2, he_res20_ctx, cryptoContext)
        finalRes = final_layer(resLayer3, he_res20_ctx, cryptoContext)
        print("after processing image ", i, "time: ", datetime.datetime.now())
        print("time: ", time.time() - start_time)
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
    # logBsSlots_list = [14, 13, 12]
    logN = 16
    dnum = 3
    dcrtBits = 59
    firstMod = 60
    # levelBudget_list = [[4, 4], [4, 4], [4, 4]]
    secretKeyDist = "SPARSE_TERNARY" # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO" #todo: FIXEDMANUAL is not supported yet!
    device = "cuda"
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    he_res20_context_ = HE_res20_context("../weights")

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, SAVE_MIDDLE=True)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, device, save_dir=DATA_DIR,
                             config=config))

    cryptoContext.weight_path = '../weights/' # fixme: workaround only

    pkl_path = None
    if config.SAVE_MIDDLE==False:
        file_name = "encode_20250412_221730" #italian-res20-cifar10 encode middle pkl
        cryptoContext.pre_encode_type = "middle"
        load_encode_pkl(file_name, he_res20_context_)
        pkl_path = os.path.join(he_res20_context_.weight_dir, file_name + ".pkl")

        # pkl_path = "" #italian-res20-cifar10 encode end pkl, should be generated from encode middle pkl
        # cryptoContext.pre_encode_type = "end"
    load_weight(pkl_path, cryptoContext)

    print("start executeResNet20")
    executeResNet20(he_res20_context_, cryptoContext, openfhe_context)

if __name__ == "__main__":
    resnet20()
