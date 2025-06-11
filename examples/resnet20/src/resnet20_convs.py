import torch.fhe as fhe
from resnet20_utils import *

def rot_input(input, img_width, padding, cryptoContext):
    digits=fhe.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)
    c_rotations=[]
    c_rotations.append(fhe.homo_rotate(fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),-img_width,cryptoContext))
    c_rotations.append(fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext))
    c_rotations.append(
        fhe.homo_rotate(fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext))
    c_rotations.append(input)#这里旋转什么的都只需要对cv1吗？
    c_rotations.append(fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext))
    c_rotations.append(fhe.homo_rotate(fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),img_width,cryptoContext))
    c_rotations.append(fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext))
    c_rotations.append(
        fhe.homo_rotate(fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext), img_width, cryptoContext))
    return c_rotations

@fhe.utils.profile_python_function
def convbn_initial(input,num_channel,scale, he_res20_ctx, cryptoContext, img_width, padding):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)
    
    c_rotations = rot_input(input, img_width, padding, cryptoContext)

    for j in range(num_channel):
        k_rows=[]
        for k in range(9):
            encoded=read_values_from_file(cryptoContext, f"conv1bn1-ch{j}-k{k+1}",cryptoContext.L-input.cur_limbs,1,16384,scale)
            k_rows.append(encoded)
        partial_sum = fhe.fused_pairwise_mac(c_rotations, k_rows, cryptoContext)
        partial_sum = fhe.homo_rescale(partial_sum, 1, cryptoContext) #RESCALE ADD BY ZRJI
        sum_rot = fhe.homo_rotate(partial_sum,1024,cryptoContext)
        partial_sum = fhe.homo_add(partial_sum,sum_rot,cryptoContext)
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext)
        partial_sum = fhe.homo_mul_pt(partial_sum, mask_from_to(0, 1024, partial_sum.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, 1024, cryptoContext)
       
    # bias=read_values_from_file(cryptoContext, "conv1bn1-bias", cryptoContext.L-input.cur_limbs, 1, 16384, scale)
    # finalsum=fhe.homo_add_pt(finalsum,bias,cryptoContext)

    return finalsum

@fhe.utils.profile_python_function
def convbn(input, layer, n, scale, he_res20_ctx, cryptoContext, img_width, padding, slots, num_channel, rot_offset, channel_offset, biasoff=""):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    c_rotations = rot_input(input, img_width, padding, cryptoContext)
    for j in range(num_channel):
        k_rows=[]
        for k in range(9):
            encoded=read_values_from_file(cryptoContext, f"layer{layer}-conv{n}bn{n}-ch{j+channel_offset}-k{k+1}",cryptoContext.L-input.cur_limbs,1,slots,scale)
            k_rows.append(encoded)
        partial_sum = fhe.fused_pairwise_mac(c_rotations, k_rows, cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)
    return finalsum

@fhe.utils.profile_python_function
def convbn_dx(input, layer, n, scale, he_res20_ctx, cryptoContext, slots, num_channel, rot_offset, channel_offset, biasoff=""):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    for j in range(num_channel):
        encoded = read_values_from_file(cryptoContext, f"layer{layer}dx-conv{n}bn{n}-ch{j+channel_offset}-k1", cryptoContext.L - input.cur_limbs,1,
                                                                    he_res20_ctx.cur_num_slots, scale)
        partial_sum = fhe.homo_mul_pt(input, encoded, cryptoContext)
        
        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = fhe.homo_rescale(finalsum, 1, cryptoContext) #RESCALE ADD BY ZRJI
    bias1 = read_values_from_file(cryptoContext, f"layer{layer}dx-conv{n}bn{n}-bias"+biasoff, cryptoContext.L - finalsum.cur_limbs,1, slots, scale)
    finalsum =fhe.homo_add_pt(finalsum,bias1,cryptoContext)

    return finalsum


@fhe.utils.profile_python_function
def downsample1024to256(c1, c2, he_res20_ctx, cryptoContext):

    c1.slots=32768
    c2.slots=32768
    he_res20_ctx.cur_num_slots = 16384 * 2
    fullpack=fhe.homo_add(fhe.homo_mul_pt(c1, mask_first_n(16384, c1.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext),
                          fhe.homo_mul_pt(c2, mask_scecond_n(16384, c2.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext), cryptoContext)

    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_mul_pt(fhe.homo_add(fullpack,
                                                    fhe.homo_rotate(fullpack,1,cryptoContext),cryptoContext),
                             gen_mask(2, fullpack.cur_limbs, he_res20_ctx, cryptoContext),
                             cryptoContext)
    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack,
                                                      fhe.homo_rotate(
                                                          fhe.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext),
                               gen_mask(4, fullpack.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext)
    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_mul_pt(fhe.homo_add(fullpack,fhe.homo_rotate(fullpack,4,cryptoContext),cryptoContext),
                             gen_mask(8, fullpack.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext)
    fullpack = fhe.force_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_add(fullpack,fhe.homo_rotate(fullpack,8,cryptoContext),cryptoContext)

    assert fullpack.noise_deg == 1
    downsampledrows = cryptoContext.zero_32K
    downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI
    
    for i in range(16):
        masked=fhe.homo_mul_pt(fullpack,
                               mask_first_n_mod(16, 1024, i, fullpack.cur_limbs, cryptoContext), cryptoContext)
        downsampledrows=fhe.homo_add(downsampledrows,masked,cryptoContext)
        if i<15:
            fullpack=fhe.homo_rotate(fullpack,64-16,cryptoContext)

    downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext)

    assert downsampledrows.noise_deg == 1

    downsampledchannels = cryptoContext.zero_32K
    downsampledchannels = fhe.drop_last_elements(downsampledchannels, downsampledchannels.cur_limbs - downsampledrows.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI
    for i in range(32):
        masked=fhe.homo_mul_pt(downsampledrows, mask_channel(i, downsampledrows.cur_limbs, cryptoContext), cryptoContext)
        downsampledchannels=fhe.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels=fhe.homo_rotate(downsampledchannels,-(1024-256),cryptoContext)

    downsampledchannels=fhe.homo_rotate(downsampledchannels,(1024-256)*32,cryptoContext)
    downsampledchannels=fhe.homo_add(downsampledchannels,fhe.homo_rotate(downsampledchannels,-8192,cryptoContext),cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                            fhe.homo_rotate(
                                                fhe.homo_rotate(downsampledchannels,-8192,cryptoContext), -8192, cryptoContext),
                                            cryptoContext)
    downsampledchannels.slots=8192

    return downsampledchannels

@fhe.utils.profile_python_function
# @fhe.utils.profile_pytorch_function
def downsample256to64(c1, c2, he_res20_ctx, cryptoContext):

    c1.slots=16384
    c2.slots=16384
    he_res20_ctx.cur_num_slots = 8192 * 2
    fullpack=fhe.homo_add(fhe.homo_mul_pt(c1,
                                          mask_first_n(8192, c1.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext),
                          fhe.homo_mul_pt(c2, mask_scecond_n(8192, c2.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext), cryptoContext)

    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_mul_pt(
        fhe.homo_add(fullpack,fhe.homo_rotate(fullpack,1,cryptoContext),cryptoContext),
        gen_mask(2, fullpack.cur_limbs, he_res20_ctx, cryptoContext),cryptoContext)
    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(fullpack,
                          fhe.homo_rotate(
                              fhe.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext),
        gen_mask(4, fullpack.cur_limbs, he_res20_ctx, cryptoContext), cryptoContext)
    fullpack = fhe.force_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_add(fullpack,
                               fhe.homo_rotate(fullpack,4,cryptoContext),cryptoContext)

    downsampledrows = cryptoContext.zero_16K.deep_copy()
    downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI

    assert fullpack.noise_deg == 1
    for i in range(32):
        masked=fhe.homo_mul_pt(fullpack, mask_first_n_mod2(8, 256, i, fullpack.cur_limbs, cryptoContext), cryptoContext)
        downsampledrows=fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i<31:
            fullpack = fhe.homo_rotate(fullpack, 24, cryptoContext)

    downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext) #RESCALE ADD BY ZRJI
    downsampledchannels = cryptoContext.zero_16K.deep_copy()
    downsampledchannels = fhe.drop_last_elements(downsampledchannels, downsampledchannels.cur_limbs - downsampledrows.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI
    assert downsampledrows.noise_deg == 1

    for i in range(64):
        masked=fhe.homo_mul_pt(downsampledrows,
                               mask_channel2(i, downsampledrows.cur_limbs, cryptoContext), cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels,-(256-64),cryptoContext)

    downsampledchannels = fhe.homo_rotate(downsampledchannels,(256-64)*64,cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,fhe.homo_rotate(downsampledchannels,-4096,cryptoContext),cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                            fhe.homo_rotate(fhe.homo_rotate(downsampledchannels,-4096,cryptoContext), -4096, cryptoContext),
                                            cryptoContext)
    downsampledchannels.slots=4096

    return downsampledchannels
