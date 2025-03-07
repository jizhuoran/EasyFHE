import os, sys, datetime
import numpy as np

sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from utils import *
from convs import *
import torch.fhe as fhe
import approx
import torch

DATA_DIR = os.environ["DATA_DIR"]


class HE_res20_context:
    def __init__(self, data_dir):
        self.cur_num_slots = None
        self.relu_degree = None
        self.weight_dir = data_dir


def homo_relu(ciphertext, scale, degree, cryptoContext):
    def relu_function(x):
        return 0 if x < 0 else (1 / scale) * x

    coefficients = approx.eval_chebyshev_coefficients(relu_function, -1, 1, degree)
    result = approx.eval_chebyshev_series_ps(
        ciphertext, coefficients, -1, 1, cryptoContext
    )
    return result


def initial_layer(input, he_res20_ctx, cryptoContext):
    scale = normalized_deltas[0][0]
    res = convbn_initial(input, scale, he_res20_ctx, cryptoContext)
    res = homo_relu(res, scale, he_res20_ctx.relu_degree, cryptoContext)
    return res


def layer1(input, he_res20_ctx, cryptoContext):
    scale = normalized_deltas[1][0]

    res1 = convbn(input, 1, 1, scale, he_res20_ctx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
    )
    res1 = homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][1]
    res1 = convbn(res1, 1, 2, scale, he_res20_ctx, cryptoContext)
    res1 = fhe.homo_add(
        res1, fhe.homo_mul_scalar_double(input, scale, cryptoContext), cryptoContext
    )
    res1 = fhe.homo_bootstrap(
        res1, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
    )
    res1 = homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][2]
    res2 = convbn(res1, 2, 1, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][3]
    res2 = convbn(res2, 2, 2, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][4]
    res3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, L0=cryptoContext.L, logBsSlots=14, cryptoContext=cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][5]
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
    res1sx[0], res1sx[1] = convbn1632sx(
        boot_in, 4, 1, scaleSx, he_res20_ctx, cryptoContext
    )
    res1dx[0], res1dx[1] = convbn1632dx(
        boot_in, 4, 1, scaleDx, he_res20_ctx, cryptoContext
    )

    fullpackSx = downsample1024to256(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext)
    fullpackDx = downsample1024to256(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)

    he_res20_ctx.cur_num_slots = 8192

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn2(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][2]
    res2 = convbn2(res1, 5, 1, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    res2 = convbn2(res2, 5, 2, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][4]
    res3 = convbn2(res2, 6, 1, scale, he_res20_ctx, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, L0=cryptoContext.L, logBsSlots=13, cryptoContext=cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
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
    res1sx[0], res1sx[1] = convbn3264sx(
        boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext
    )
    res1dx[0], res1dx[1] = convbn3264dx(
        boot_in, 7, 1, scaleDx, he_res20_ctx, cryptoContext
    )

    fullpackSx = downsample256to64(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext)
    fullpackDx = downsample256to64(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext)

    he_res20_ctx.cur_num_slots = 4096

    fullpackSx = fhe.homo_bootstrap(
        fullpackSx, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
    )
    fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn3(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext)
    res1 = fhe.homo_add(fullpackSx, fullpackDx, cryptoContext)
    res1 = fhe.homo_bootstrap(
        res1, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
    )
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][2]
    res2 = convbn3(res1, 8, 1, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_bootstrap(
        res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    res2 = convbn3(res2, 8, 2, scale, he_res20_ctx, cryptoContext)
    res2 = fhe.homo_add(
        res2, fhe.homo_mul_scalar_double(res1, scale, cryptoContext), cryptoContext
    )
    res2 = fhe.homo_bootstrap(
        res2, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
    )
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    res3 = convbn3(res2, 9, 1, scale, he_res20_ctx, cryptoContext)
    res3 = fhe.homo_bootstrap(
        res3, L0=cryptoContext.L, logBsSlots=12, cryptoContext=cryptoContext
    )
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
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
    filePath = DATA_DIR + "/cifar10/test_batch.bin"
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
        firstLayer = initial_layer(in_ct, he_res20_ctx, cryptoContext)
        resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext)
        resLayer2 = layer2(resLayer1, he_res20_ctx, cryptoContext)
        resLayer3 = layer3(resLayer2, he_res20_ctx, cryptoContext)
        finalRes = final_layer(resLayer3, he_res20_ctx, cryptoContext)
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


def resnet20():

    rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1, 1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]
    logBsSlots_list = [12, 13, 14]
    logN = 16
    levelBudget_list = [[4, 4], [4, 4], [4, 4]]

    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    he_res20_context_ = HE_res20_context(DATA_DIR)

    maxLevelsRemaining = 14
    cryptoContext, openfhe_context = fhe.try_load_context(
        maxLevelsRemaining,
        rotate_index_list,
        logBsSlots_list,
        logN,
        levelBudget_list,
        save_dir=DATA_DIR,
        autoLoadAndSetConfig=True,
    )
    cryptoContext.openfhe_context = openfhe_context
    cryptoContext.PRELOAD_ALL = True
    print("start executeResNet20")

    encode_weight_path = (
        he_res20_context_.weight_dir
        + "/ENCODE-VAL_{}_{}_{}_{}.pkl".format(
            logN,
            "-".join(map(str, logBsSlots_list)),
            maxLevelsRemaining,
            "-".join(
                "-".join(map(str, levelBudget)) for levelBudget in levelBudget_list
            ),
        )
    )

    with open(encode_weight_path, "rb") as f:
        pre_encoded = pickle.load(f)
    if cryptoContext.PRELOAD_ALL:
        for key, _ in pre_encoded.items():
            pre_encoded[key].cv = [
                torch.tensor(pre_encoded[key].cv[0], dtype=torch.uint64, device="cuda")
            ]
    cryptoContext.pre_encoded = pre_encoded

    executeResNet20(he_res20_context_, cryptoContext, openfhe_context)


if __name__ == "__main__":
    resnet20()
