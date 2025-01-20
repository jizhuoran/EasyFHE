import datetime
import torch
from torch.fhe import utils
from torch.fhe import approx
from torch.fhe.bootstrapping import homo_bootstrap
from torch.fhe.resnet.convs import *
from torch.fhe.resnet.utils import *

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


def initial_layer(input, he_res20_ctx, cryptoContext, openfhe_context_dict):
    scale=normalized_deltas[0][0]
    res=convbn_initial(input, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res= homo_relu(res, scale, he_res20_ctx.relu_degree, cryptoContext)
    return res


def layer1(input, he_res20_ctx, cryptoContext, openfhe_context_dict):
    scale = normalized_deltas[1][0]

    res1= convbn(input, 1, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res1=homo_bootstrap(res1,L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res1= homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][1]
    res1= convbn(res1, 1, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res1=homo_ops.homo_add(res1, homo_ops.homo_mul_scalar_double(input,scale,cryptoContext),cryptoContext)
    res1=homo_bootstrap(res1,L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res1= homo_relu(res1, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][2]
    res2 = convbn(res1, 2, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][3]
    res2 = convbn(res2, 2, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext), cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][4]
    res3 = convbn(res2, 3, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[1][5]
    res3 = convbn(res3, 3, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res3 = homo_ops.homo_add(res3, homo_ops.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer2(input, he_res20_ctx, cryptoContext, openfhe_context_dict):
    scaleSx =normalized_deltas[2][0]
    scaleDx =normalized_deltas[2][1]
    boot_in =homo_bootstrap(input, L0=cryptoContext.L, logSlots=14, cryptoContext=cryptoContext)
    res1sx=[None, None]
    res1dx=[None, None]
    res1sx[0], res1sx[1]  = convbn1632sx(boot_in,4,1, scaleSx, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res1dx[0], res1dx[1]  = convbn1632dx(boot_in,4,1, scaleDx, he_res20_ctx, cryptoContext, openfhe_context_dict)

    fullpackSx = downsample1024to256(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext, openfhe_context_dict)
    fullpackDx = downsample1024to256(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext, openfhe_context_dict)


    he_res20_ctx.cur_num_slots = 8192
    load_bootstrapping_and_rotation_keys(he_res20_ctx.cur_num_slots, cryptoContext)

    fullpackSx = homo_bootstrap(fullpackSx,L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    fullpackSx = homo_relu(fullpackSx,scaleSx,he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn2(fullpackSx, 4, 2, scaleDx, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res1 = homo_ops.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1 = homo_bootstrap(res1, L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][2]
    res2  = convbn2(res1, 5, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res2  = homo_bootstrap(res2, L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    res2  = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    res2  = convbn2(res2, 5, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res2  = homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2  = homo_bootstrap(res2, L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    res2  = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][4]
    res3  = convbn2(res2, 6, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res3  = homo_bootstrap(res3, L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    res3  = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    res3  = convbn2(res3, 6, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res3  = homo_ops.homo_add(res3,homo_ops.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3  = homo_bootstrap(res3, L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    res3  = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer3(input,he_res20_ctx, cryptoContext, openfhe_context_dict):
    scaleSx=normalized_deltas[3][0]
    scaleDx=normalized_deltas[3][1]

    boot_in=homo_bootstrap(input, L0=cryptoContext.L, logSlots=13, cryptoContext=cryptoContext)
    res1sx=[None, None]
    res1dx=[None, None]
    res1sx[0], res1sx[1]= convbn3264sx(boot_in, 7, 1, scaleSx, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res1dx[0], res1dx[1]= convbn3264dx(boot_in, 7, 1, scaleDx, he_res20_ctx, cryptoContext, openfhe_context_dict)

    fullpackSx = downsample256to64(res1sx[0], res1sx[1], he_res20_ctx, cryptoContext, openfhe_context_dict)
    fullpackDx = downsample256to64(res1dx[0], res1dx[1], he_res20_ctx, cryptoContext, openfhe_context_dict)

    he_res20_ctx.cur_num_slots = 4096
    load_bootstrapping_and_rotation_keys(he_res20_ctx.cur_num_slots, cryptoContext)


    fullpackSx = homo_bootstrap(fullpackSx,L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    fullpackSx = homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx = convbn3(fullpackSx, 7, 2, scaleDx, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res1 = homo_ops.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1 = homo_bootstrap(res1, L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    res1 = homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)



    scale = normalized_deltas[3][2]
    res2= convbn3(res1, 8, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res2=homo_bootstrap(res2, L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    res2= homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    res2= convbn3(res2, 8, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    res3= convbn3(res2, 9, 1, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res3=homo_bootstrap(res3, L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    res3= homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    res3= convbn3(res3, 9, 2, scale, he_res20_ctx, cryptoContext, openfhe_context_dict)
    res3=homo_ops.homo_add(res3,homo_ops.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, logSlots=12, cryptoContext=cryptoContext)
    return res3


def final_layer(input, he_res20_ctx, cryptoContext, openfhe_context_dict):

    he_res20_ctx.cur_num_slots=4096
    openfhe_context = openfhe_context_dict[str(log2_int(he_res20_ctx.cur_num_slots))]
    weight=read_fc_weight(cryptoContext, cryptoContext.L - input.cur_limbs, 1, he_res20_ctx.cur_num_slots)
    res = rotsum(input, 64,cryptoContext)
    res = homo_ops.homo_mul_pt(res,
                               mask_mod(64, res.cur_limbs, 1.0/64.0, he_res20_ctx, cryptoContext, openfhe_context),cryptoContext)
    res=repeat(res,16,cryptoContext)

    res=homo_ops.homo_mul_pt(res,weight,cryptoContext)
    res=rotsum_padded(res,64,cryptoContext)

    return res


def executeResNet20(he_res20_ctx, cryptoContext, openfhe_context_dict):

    he_res20_ctx.cur_num_slots = (1<<14)
    openfhe_context = openfhe_context_dict[str(log2_int(he_res20_ctx.cur_num_slots))]

    image_vector = read_image(0)
    image_vector = torch.tensor(image_vector, device="cuda")
    in_ct, in_ct_openfhe = openfhe_context.encrypt(image_vector, 1, cryptoContext.L - 5 - get_relu_depth(he_res20_ctx.relu_degree), he_res20_ctx.cur_num_slots) # note: initial level is aligned with the original open source codes

    load_bootstrapping_and_rotation_keys(he_res20_ctx.cur_num_slots, cryptoContext)

    utils.load_rotation_keys(cryptoContext, "app")


    with open('torch/fhe/resnet/weights.pkl', 'rb') as f:
        weight_map = pickle.load(f)
    cryptoContext.weight_map = weight_map

    with open('torch/fhe/resnet/encode_val.pkl', 'rb') as f:
        weight_map = pickle.load(f)
    for key, _ in weight_map.items():
        weight_map[key].mx = [torch.tensor(weight_map[key].mx, device="cuda")]
    cryptoContext.pre_encoded = weight_map    

    # print("resnet computation start")
    # firstLayer= initial_layer(in_ct, he_res20_ctx, cryptoContext, openfhe_context_dict)
    # resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext, openfhe_context_dict)
    # resLayer2 = layer2(resLayer1,  he_res20_ctx, cryptoContext, openfhe_context_dict)
    # resLayer3 = layer3(resLayer2,  he_res20_ctx, cryptoContext, openfhe_context_dict)
    # finalRes = final_layer(resLayer3,he_res20_ctx, cryptoContext, openfhe_context_dict)


    print("resnet computation start")
    print("current time: ", datetime.datetime.now())
    firstLayer = initial_layer(in_ct, he_res20_ctx, cryptoContext, openfhe_context_dict)
    print("after initial layer")
    print("current time: ", datetime.datetime.now())
    resLayer1 = layer1(firstLayer, he_res20_ctx, cryptoContext, openfhe_context_dict)
    print("after layer1")
    print("current time: ", datetime.datetime.now())
    resLayer2 = layer2(resLayer1, he_res20_ctx, cryptoContext, openfhe_context_dict)
    print("after layer2")
    print("current time: ", datetime.datetime.now())
    resLayer3 = layer3(resLayer2, he_res20_ctx, cryptoContext, openfhe_context_dict)
    print("after layer3")
    print("current time: ", datetime.datetime.now())
    finalRes = final_layer(resLayer3, he_res20_ctx, cryptoContext, openfhe_context_dict)
    print("after final layer")
    print("current time: ", datetime.datetime.now())

    finalRes.slots = 10
    clear_result = openfhe_context.decrypt(finalRes) #decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print(clear_result[:10]) # should be of len 10

    max_element_idx = np.argmax(clear_result)
    print(max_element_idx)


def resnet20( ):
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

    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    he_res20_context_ = HE_res20_context(None, max_relu_degree) # ini app ctx--he resnet ctx

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

    print("start executeResNet20")
    #print all the key of openfhe_context_dict
    # print("key of openfhe_context_dict ", openfhe_context_dict.keys())
    executeResNet20(he_res20_context_, cryptoContext, openfhe_context_dict)
