import torch.fhe as fhe
from resnet18_utils import *

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
            encoded=read_values_from_file(cryptoContext, f"conv1bn1-ch{j}-k{k+1}",cryptoContext.L-input.cur_limbs,1,input.slots,scale)
            k_rows.append(encoded)
        partial_sum = fhe.fused_pairwise_mac(c_rotations, k_rows, cryptoContext)
        partial_sum = fhe.homo_rescale(partial_sum, 1, cryptoContext) #RESCALE ADD BY ZRJI
        sum_rot = fhe.homo_rotate(partial_sum,1024,cryptoContext)
        partial_sum = fhe.homo_add(partial_sum,sum_rot,cryptoContext)
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext)
        partial_sum = fhe.homo_mul_pt(partial_sum, mask_from_to(0, 1024, partial_sum.cur_limbs, partial_sum.slots, cryptoContext), cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, 1024, cryptoContext)
    return finalsum

@fhe.utils.profile_python_function
def convbn(input, layer, n, scale, he_res20_ctx, cryptoContext, img_width, padding, slots, num_channel, rot_offset, channel_offset, biasoff=""):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    c_rotations = rot_input(input, img_width, padding, cryptoContext)
    for j in range(num_channel):
        k_rows=[]
        for k in range(9):
            encoded=read_values_from_file(cryptoContext, f"layer{layer}-conv{n}bn{n}-ch{j+channel_offset}-k{k+1}",cryptoContext.L-input.cur_limbs,1,input.slots,scale)
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
                                                                    input.slots, scale)
        partial_sum = fhe.homo_mul_pt(input, encoded, cryptoContext)
        
        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = fhe.homo_rescale(finalsum, 1, cryptoContext) #RESCALE ADD BY ZRJI
    bias1 = read_values_from_file(cryptoContext, f"layer{layer}dx-conv{n}bn{n}-bias"+biasoff, cryptoContext.L - finalsum.cur_limbs,1, finalsum.slots, scale)
    finalsum =fhe.homo_add_pt(finalsum,bias1,cryptoContext)

    return finalsum


def choose_zero(slots, cryptoContext):
    if slots == 65536:
        return cryptoContext.zero_64K.deep_copy()
    elif slots == 32768:
        return cryptoContext.zero_32K.deep_copy()
    elif slots == 16384:
        return cryptoContext.zero_16K.deep_copy()
    else:
        raise ValueError(f"Unsupported slots value: {slots}")


# old downsample
# def downsample1024to256(c1, c2, num_channel, num_cipher, he_res20_ctx, cryptoContext):
#     c1.slots = 32768
#     c2.slots = 32768
#     he_res20_ctx.cur_num_slots = 16384 * 2
#     fullpack = fhe.homo_add(
#         fhe.homo_mul_pt(c1, mask_first_n(16384, c1.cur_limbs, 16384*2, cryptoContext), cryptoContext),
#         fhe.homo_mul_pt(c2, mask_scecond_n(16384, c2.cur_limbs, 32768, cryptoContext), cryptoContext),
#         cryptoContext)
#
#     fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
#     fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack,
#                                             fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext),
#                                gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext),
#                                cryptoContext)
#     fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
#     fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack,
#                                             fhe.homo_rotate(
#                                                 fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext),
#                                             cryptoContext),
#                                gen_mask(4, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
#     fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
#     fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext),
#                                gen_mask(8, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
#     fullpack = fhe.force_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
#     fullpack = fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 8, cryptoContext), cryptoContext)
#
#     assert fullpack.noise_deg == 1
#     downsampledrows = cryptoContext.zero_32K
#     downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs,
#                                              cryptoContext)  # drop_last_elements ADD BY ZRJI
#
#     for i in range(16):
#         masked = fhe.homo_mul_pt(fullpack,
#                                  mask_first_n_mod(16, 1024, i, 32, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
#         downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
#         if i < 15:
#             fullpack = fhe.homo_rotate(fullpack, 64 - 16, cryptoContext)
#
#     downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext)
#
#     assert downsampledrows.noise_deg == 1
#
#     downsampledchannels = cryptoContext.zero_32K
#     downsampledchannels = fhe.drop_last_elements(downsampledchannels,
#                                                  downsampledchannels.cur_limbs - downsampledrows.cur_limbs,
#                                                  cryptoContext)  # drop_last_elements ADD BY ZRJI
#     for i in range(32):
#         masked = fhe.homo_mul_pt(downsampledrows, mask_channel(i, 16,1024,1, downsampledrows.cur_limbs, cryptoContext),
#                                  cryptoContext)
#         downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
#         downsampledchannels = fhe.homo_rotate(downsampledchannels, -(1024 - 256), cryptoContext)
#
#     downsampledchannels = fhe.homo_rotate(downsampledchannels, (1024 - 256) * 32, cryptoContext)
#     downsampledchannels = fhe.homo_add(downsampledchannels, fhe.homo_rotate(downsampledchannels, -8192, cryptoContext),
#                                        cryptoContext)
#     downsampledchannels = fhe.homo_add(downsampledchannels,
#                                        fhe.homo_rotate(
#                                            fhe.homo_rotate(downsampledchannels, -8192, cryptoContext), -8192,
#                                            cryptoContext),
#                                        cryptoContext)
#     downsampledchannels.slots = 8192
#
#     return downsampledchannels

@fhe.utils.profile_python_function
def downsample1024to256(c1, c2, num_channel, num_cipher, he_res20_ctx, cryptoContext):

    assert num_cipher ==2 or num_cipher==1

    cipher_list =[]
    downsampledchannels_list =[]
    if num_cipher==1:
        old_slots = c1.slots
        c1 = torch.fhe.homo_ops.slot_resize(c1, c1.slots*2, cryptoContext)
        c2 = torch.fhe.homo_ops.slot_resize(c2, c2.slots*2, cryptoContext)
        fullpack = fhe.homo_add(
            fhe.homo_mul_pt(c1, mask_first_n(old_slots, c1.cur_limbs, c1.slots, cryptoContext), cryptoContext),
            fhe.homo_mul_pt(c2, mask_scecond_n(old_slots, c2.cur_limbs, c2.slots, cryptoContext), cryptoContext),
            cryptoContext)
        cipher_list.append(fullpack)
    else:
        cipher_list.append(c1)
        cipher_list.append(c2)

    for cipher in cipher_list:
        fullpack = fhe.homo_rescale(cipher, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack,
                                                fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext),
                                   gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext),
                                   cryptoContext)
        # 相邻两个相加
        fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack,
                                                fhe.homo_rotate(
                                                    fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext),
                                                cryptoContext),
                                   gen_mask(4, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
        fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        fullpack = fhe.homo_mul_pt(fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext),
                                   gen_mask(8, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
        fullpack = fhe.force_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        fullpack = fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 8, cryptoContext), cryptoContext)

        assert fullpack.noise_deg == 1
        downsampledrows = choose_zero(cipher.slots, cryptoContext)
        downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs,
                                                 cryptoContext)  # drop_last_elements ADD BY ZRJI

        for i in range(16):
            #  每个i取得1024中第i个16的数，每64取16，最终得到256的通道
            masked = fhe.homo_mul_pt(fullpack,
                                     mask_first_n_mod(16, 1024, i, 2*num_channel//num_cipher,fullpack.cur_limbs, fullpack.slots, cryptoContext),
                                     cryptoContext)
            downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
            if i < 15:
                fullpack = fhe.homo_rotate(fullpack, 64 - 16, cryptoContext)

        downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext)

        assert downsampledrows.noise_deg == 1

        downsampledchannels = choose_zero(cipher.slots, cryptoContext)
        downsampledchannels = fhe.drop_last_elements(downsampledchannels,
                                                     downsampledchannels.cur_limbs - downsampledrows.cur_limbs,
                                                     cryptoContext)  # drop_last_elements ADD BY ZRJI
        for i in range(2*num_channel//num_cipher):
            # 将128通道的更紧凑
            masked = fhe.homo_mul_pt(downsampledrows,
                                     mask_channel(i, num_channel, 1024, num_cipher, downsampledrows.cur_limbs, cryptoContext),
                                     cryptoContext)
            downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
            downsampledchannels = fhe.homo_rotate(downsampledchannels, -(1024 - 256), cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels, 2*num_channel//num_cipher * (1024-256), cryptoContext)
        downsampledchannels_list.append(downsampledchannels)
    ######################################


    if num_cipher==1: #resnet20
        downsampledchannels = downsampledchannels_list[0]
        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_rotate(downsampledchannels, - downsampledchannels_list[0].slots//4, cryptoContext), cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_rotate(
                                               fhe.homo_rotate(downsampledchannels, - downsampledchannels_list[0].slots//4, cryptoContext), - downsampledchannels_list[0].slots//4,
                                               cryptoContext),
                                           cryptoContext)
        downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, downsampledchannels.slots//4, cryptoContext)
        return downsampledchannels
    else: #resnet18
        downsampledchannels = choose_zero(downsampledchannels_list[0].slots, cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,
            fhe.homo_mul_pt(downsampledchannels_list[0], mask_first_n(16384, downsampledchannels_list[0].cur_limbs, downsampledchannels_list[0].slots, cryptoContext), cryptoContext),cryptoContext)

        downsampledchannels = fhe.homo_add(downsampledchannels,
            fhe.homo_mul_pt(fhe.homo_rotate(downsampledchannels_list[1], 16384, cryptoContext),mask_scecond_n(16384, downsampledchannels_list[1].cur_limbs, downsampledchannels_list[1].slots,cryptoContext), cryptoContext),cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,fhe.homo_rotate(downsampledchannels, 16384*2, cryptoContext),cryptoContext)
        downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, 32768, cryptoContext)


        return downsampledchannels



@fhe.utils.profile_python_function
def downsample256to64(c1, c2, num_channel, he_res20_ctx, cryptoContext):
    old_slots = c1.slots
    c1 = torch.fhe.homo_ops.slot_resize(c1, c1.slots*2, cryptoContext)
    c2 = torch.fhe.homo_ops.slot_resize(c2, c2.slots*2, cryptoContext)
    fullpack=fhe.homo_add(fhe.homo_mul_pt(c1,mask_first_n(old_slots, c1.cur_limbs, c1.slots, cryptoContext), cryptoContext),
                          fhe.homo_mul_pt(c2, mask_scecond_n(old_slots, c2.cur_limbs, c2.slots, cryptoContext), cryptoContext), cryptoContext)

    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_mul_pt(
        fhe.homo_add(fullpack,fhe.homo_rotate(fullpack,1,cryptoContext),cryptoContext),
        gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext),cryptoContext)
    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(fullpack,
                          fhe.homo_rotate(
                              fhe.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext),
        gen_mask(4, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
    fullpack = fhe.force_rescale(fullpack, 1, cryptoContext) #RESCALE ADD BY ZRJI
    fullpack=fhe.homo_add(fullpack,
                               fhe.homo_rotate(fullpack,4,cryptoContext),cryptoContext)

    downsampledrows = choose_zero(fullpack.slots, cryptoContext)
    downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI

    assert fullpack.noise_deg == 1
    for i in range(32):
        masked=fhe.homo_mul_pt(fullpack, mask_first_n_mod2(8, 256, i, 2*num_channel, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
        downsampledrows=fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i<31:
            fullpack = fhe.homo_rotate(fullpack, 24, cryptoContext)

    downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext) #RESCALE ADD BY ZRJI
    downsampledchannels = choose_zero(fullpack.slots, cryptoContext)
    downsampledchannels = fhe.drop_last_elements(downsampledchannels, downsampledchannels.cur_limbs - downsampledrows.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI
    assert downsampledrows.noise_deg == 1

    for i in range(num_channel*2):
        masked=fhe.homo_mul_pt(downsampledrows,
                               mask_channel(i, num_channel, 256, 1, downsampledrows.cur_limbs, cryptoContext), cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels,-(256-64),cryptoContext)

    downsampledchannels = fhe.homo_rotate(downsampledchannels,(256-64)*(num_channel*2),cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,fhe.homo_rotate(downsampledchannels,-(downsampledchannels.slots//4),cryptoContext),cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                            fhe.homo_rotate(fhe.homo_rotate(downsampledchannels,-(downsampledchannels.slots//4),cryptoContext), -(downsampledchannels.slots//4), cryptoContext),
                                            cryptoContext)
    downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, (downsampledchannels.slots//4), cryptoContext)

    return downsampledchannels

@fhe.utils.profile_python_function
def downsample64to16(c1, c2, num_channel, he_res20_ctx, cryptoContext):
    old_slots = c1.slots
    c1 = torch.fhe.homo_ops.slot_resize(c1, c1.slots*2, cryptoContext)
    c2 = torch.fhe.homo_ops.slot_resize(c2, c2.slots*2, cryptoContext)
    fullpack=fhe.homo_add(fhe.homo_mul_pt(c1,mask_first_n(old_slots, c1.cur_limbs, c1.slots, cryptoContext), cryptoContext),
                          fhe.homo_mul_pt(c2, mask_scecond_n(old_slots, c2.cur_limbs, c2.slots, cryptoContext), cryptoContext), cryptoContext)

    fullpack=fhe.homo_mul_pt(
        fhe.homo_add(fullpack,fhe.homo_rotate(fullpack,1,cryptoContext),cryptoContext),
        gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext),cryptoContext)
    fullpack = fhe.homo_add(fullpack,fhe.homo_rotate(fhe.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext)
    downsampledrows = choose_zero(fullpack.slots, cryptoContext)

    for i in range(4):
        masked=fhe.homo_mul_pt(fullpack, mask_first_n_mod3(4, 64, i, num_channel*2, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
        downsampledrows=fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i<3:
            fullpack = fhe.homo_rotate(fullpack, 12, cryptoContext)

    downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext) #RESCALE ADD BY ZRJI
    downsampledchannels = choose_zero(fullpack.slots, cryptoContext)
    # downsampledchannels = fhe.drop_last_elements(downsampledchannels, downsampledchannels.cur_limbs - downsampledrows.cur_limbs, cryptoContext) #drop_last_elements ADD BY ZRJI
    assert downsampledrows.noise_deg == 1

    for i in range(num_channel*2):
        masked=fhe.homo_mul_pt(downsampledrows,mask_channel(i, num_channel, 64, 1, downsampledrows.cur_limbs, cryptoContext), cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels,-(64-16),cryptoContext)

    downsampledchannels = fhe.homo_rotate(downsampledchannels,(64-16)*num_channel*2,cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,fhe.homo_rotate(downsampledchannels,-(downsampledchannels.slots//4),cryptoContext),cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                            fhe.homo_rotate(fhe.homo_rotate(downsampledchannels,-(downsampledchannels.slots//4),cryptoContext), -(downsampledchannels.slots//4), cryptoContext),
                                            cryptoContext)
    downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, (downsampledchannels.slots//4), cryptoContext)
    return downsampledchannels
