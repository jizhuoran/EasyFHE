import torch

from torch.fhe import utils
from torch.fhe import approx
from torch.fhe.bootstrapping import homo_bootstrap
from torch.fhe.resnet.convs import *

# global input_cnt
# global_num_slots=4096 #todo: should not put in global
# he_res20_ctx.relu_degree=59 #todo: should not put in global
# input_filename = ""

class he_res20_ctx:
    def __init__(self, cur_num_slots, relu_degree):
        self.cur_num_slots = cur_num_slots
        self.relu_degree = relu_degree

#todo: to be delete
def decrypt_tovector(input,slots,cryptoContext):
    if slots==0:
        slots=global_num_slots
    #Todo:   context->Decrypt(key_pair.secretKey, c, &p);
    # p->SetSlots(slots);
    # p.slots=slots
    # p->SetLength(slots);
    # vector<double> vec = p->GetRealPackedValue();
    vec=[]
    return vec

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


def layer2(input,cryptoContext):
    scaleSx=normalized_deltas[2][0]
    scaleDx=normalized_deltas[2][1]
    boot_in=homo_bootstrap(input, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1sx=convbn1632sx(boot_in,4,1,scaleSx,cryptoContext)
    res1dx=convbn1632dx(boot_in,4,1,scaleDx,cryptoContext)

    #Todo:        controller.clear_bootstrapping_and_rotation_keys(16384);
    #Todo：       controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);
    utils.load_rotation_keys(cryptoContext, "app")

    fullpackSx = downsample1024to256(res1sx[0], res1sx[1], cryptoContext)
    fullpackDx = downsample1024to256(res1dx[0], res1dx[1], cryptoContext)

    # Todo:res1sx.clear();
    # Todo:res1dx.clear();
    #Todo:    controller.clear_rotation_keys();
    #Todo:controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", 8192, verbose > 1);
    load_bootstrapping_and_rotation_keys("app",8192,cryptoContext)

    global_num_slots=8192
    fullpackSx=homo_bootstrap(fullpackSx,L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    fullpackSx=homo_relu(fullpackSx,scaleSx,he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx=convbn2(fullpackSx,4,2,scaleDx,cryptoContext)
    res1=homo_ops.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1=homo_bootstrap(res1, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1= homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)



    scale = normalized_deltas[2][2]
    res2=convbn2(res1,5,1,scale,cryptoContext)
    res2=homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2= homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][3]
    res2=convbn2(res2,5,2,scale,cryptoContext)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][4]
    res3=convbn2(res2,6,1,scale,cryptoContext)
    res3=homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3= homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[2][5]
    res3=convbn2(res3,6,2,scale,cryptoContext)
    res3=homo_ops.homo_add(res3,homo_ops.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    return res3


def layer3(input,cryptoContext):
    scaleSx=normalized_deltas[3][0]
    scaleDx=normalized_deltas[3][1]
    boot_in=homo_bootstrap(input, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1sx=convbn3264sx(boot_in,7,1,scaleSx,cryptoContext)
    res1dx=convbn3264dx(boot_in,7,1,scaleDx,cryptoContext)


    #Todo:        controller.clear_bootstrapping_and_rotation_keys(16384);
    #Todo：       controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);
    utils.load_rotation_keys(cryptoContext, "app")
    fullpackSx = downsample256to64(res1sx[0], res1sx[1], cryptoContext)
    fullpackDx = downsample256to64(res1dx[0], res1dx[1], cryptoContext)
    # Todo:res1sx.clear();
    # Todo:res1dx.clear();
    #Todo:    controller.clear_rotation_keys();
    #Todo:controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", 8192, verbose > 1);
    load_bootstrapping_and_rotation_keys("app",4096,cryptoContext)

    global_num_slots=4096
    fullpackSx=homo_bootstrap(fullpackSx,L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    fullpackSx= homo_relu(fullpackSx, scaleSx, he_res20_ctx.relu_degree, cryptoContext)
    fullpackSx=convbn3(fullpackSx,7,2,scaleDx,cryptoContext)
    res1=homo_ops.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1=homo_bootstrap(res1, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1= homo_relu(res1, scaleDx, he_res20_ctx.relu_degree, cryptoContext)



    scale = normalized_deltas[3][2]
    res2=convbn3(res1,8,1,scale,cryptoContext)
    res2=homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2= homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][3]
    res2=convbn3(res2,8,2,scale,cryptoContext)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2 = homo_relu(res2, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][4]
    res3=convbn3(res2,9,1,scale,cryptoContext)
    res3=homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3= homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)

    scale = normalized_deltas[3][5]
    res3=convbn3(res3,9,2,scale,cryptoContext)
    res3=homo_ops.homo_add(res3,homo_ops.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3 = homo_relu(res3, scale, he_res20_ctx.relu_degree, cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    return res3


def final_layer(input,cryptoContext):
#Todo:encode 未定义函数：clear_bootstrapping_and_rotation_keys   load_rotation_keys
    # controller.clear_bootstrapping_and_rotation_keys(4096);
    # controller.load_rotation_keys("rotations-finallayer.bin", false);
    utils.load_rotation_keys(cryptoContext, "app")

    global_num_slots=4096

    weight=openfhe_context.encode(read_fc_weight("../weights/fc.bin"),cryptoContext.L-input.cur_limbs,global_num_slots)
    res = rotsum( input, 64,cryptoContext)
    res = homo_ops.homo_mul_pt(res,mask_mod(64,res.cur_limbs,1.0/64.0,cryptoContext),cryptoContext)
    res=repeat(res,16,cryptoContext)

    res=homo_ops.homo_mul_pt(res,weight,cryptoContext)
    res=rotsum_padded(res,64,cryptoContext)
    clear_result=[]
    clear_result=decrypt_tovector(res,10,cryptoContext)
    max_element_iterator=clear_result.index(max(clear_result))
#Todo 不确定index_max的取值：    int index_max = distance(clear_result.begin(), max_element_iterator);
    index_max=max_element_iterator

#Todo：输出相关问题
# if (verbose >= 0) {
#         cout << "The input image is classified as " << YELLOW_TEXT << utils::get_class(index_max) << RESET_COLOR << "" << endl;
#         cout << "The index of max element is " << YELLOW_TEXT << index_max << RESET_COLOR << "" << endl;
#         if (plain) {
#             string command = "python3 ../src/plain/script.py \"" + input_filename + "\"";
#             int return_sys = system(command.c_str());
#             if (return_sys == 1) {
#                 cout << "There was an error launching src/plain/script.py. Run it from Python in order to debug it." << endl;
#             }
#         }
#     }
    return res


def executeResNet20(he_res20_ctx, cryptoContext, openfhe_context_dict):

    image_vector = read_image(0)
    image_vector = torch.tensor(image_vector, device="cuda")
    cur_num_slots = he_res20_ctx.cur_num_slots
    openfhe_context = openfhe_context_dict[str(cur_num_slots)]
    in_ct, in_ct_openfhe = openfhe_context.encrypt(image_vector, 1, cryptoContext.L - 5 - get_relu_depth(he_res20_ctx.relu_degree), cur_num_slots) # note: initial level is aligned with the original open source codes
    utils.load_rotation_keys(cryptoContext, "app")

    firstLayer= initial_layer(in_ct, he_res20_ctx, cryptoContext, openfhe_context_dict)
    resLayer1=layer1(firstLayer,he_res20_ctx, cryptoContext, openfhe_context_dict)
    resLayer2=layer2(resLayer1,cryptoContext)
    resLayer3=layer3(resLayer2,cryptoContext)
    finalRes=final_layer(resLayer3,cryptoContext) #todo: deal with result


def resnet20( ):
    logN = 16,
    logSlots_list = [14, 13, 12],
    levelBudget_list = [[4, 4], [4, 4], [4, 4]],
    dnum = 3,
    dcrtBits = 59,
    firstMod = 60,
    max_relu_degree = 59,
    secretKeyDist = "UNIFORM_TERNARY",
    rescaleTech = "FLEXIBLEAUTO",  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    save_dir = "torch/fhe/data/"

    # generate context
    approxModDepth = 9,
    maxLevelsRemaining = get_relu_depth(max_relu_degree) + 3,
    if max_relu_degree < 59:
        diff = get_relu_depth(59)-get_relu_depth(max_relu_degree)
        maxLevelsRemaining +=diff

    rotate_index_list = [-8192, -4096, -1024, -768, -256, -192, -64, -32, -16, -15, -8, -1,
                         1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 12288, 24576]

    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    he_res20_ctx((1<<14),max_relu_degree) # ini app ctx--he resnet ctx

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
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
                                                                 save_dir=save_dir)

    executeResNet20(he_res20_ctx, cryptoContext, openfhe_context_dict)

    # specify_slots = logSlots_list[0] # logslots = 11
    # openfhe_context = openfhe_context_dict[str(specify_slots)]
    # values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    # x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    # x = torch.tensor(x, device="cuda")
    # cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)
    #
    # # do the application computation
    # utils.load_rotation_keys(cryptoContext, "app")
    # cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    # cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    # # compute golden answer
    # cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, -1)
    # cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,2)
    # is_euqal = utils.compare_bs_ct_with_openfhe(cipher, cipher_openfhe)
    # if is_euqal:
    #     print("homo_rotate: Test passed!")
    # else:
    #     print("homo_rotate: Test failed!")
    #
    # # bootstrapping, logSlots = 11
    # cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    # cryptoContext.BsContext.to_cuda()
    # utils.load_rotation_keys(cryptoContext, specify_slots)
    # result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    # # compute golden answer
    # cipher_openfhe.SetSlots((1<<specify_slots))
    # openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
    # is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    # if is_euqal:
    #     print("BootstrapTest_logslots11: Test passed!")
    # else:
    #     print("BootstrapTest_logslots11: Test failed!")
    #
    # # #####################################
    # # # ..., omit some homomorphic computation
    # # #####################################
    #
    # # bootstrapping, logSlots = 12
    # specify_slots = logSlots_list[1]
    # openfhe_context1 = openfhe_context_dict[str(specify_slots)]
    #
    # cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    # cryptoContext.BsContext.to_cuda()
    # utils.load_rotation_keys(cryptoContext, specify_slots)
    # result1 = eval_bootstrap(result, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    #
    # # compute golden answer
    # openfhe_boot.SetSlots((1 << specify_slots)) # to cheat openfhe boot with (1<<specify_slots)
    # openfhe_boot1 = openfhe_context1.cc.EvalBootstrap(openfhe_boot)
    # is_euqal = utils.compare_bs_ct_with_openfhe(result1, openfhe_boot1)
    # if is_euqal:
    #     print("BootstrapTest_logslots12: Test passed!")
    # else:
    #     print("BootstrapTest_logslots12: Test failed!")
