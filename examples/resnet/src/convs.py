import torch.fhe as fhe
from utils import *


def _pairwise_mac(ctxs, ptxs, cryptoContext):
    if len(ctxs) != len(ptxs) or len(ctxs) == 0:
        raise ValueError(f"ctxs and ptxs must have the same non-zero length, but got {len(ctxs)} and {len(ptxs)}")

    partial_sum = fhe.homo_mul_pt(ctxs[0], ptxs[0], cryptoContext)
    for ctx, ptx in zip(ctxs[1:], ptxs[1:]):
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_mul_pt(ctx, ptx, cryptoContext), cryptoContext)
    return partial_sum


def rot_input(input, img_width, padding, cryptoContext):
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)
    c_rotations = []
    digits_neg_padding = fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    digits_padding = fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    digits_neg_img_width = fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    digits_img_width = fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)

    c_rotations.append(fhe.homo_rotate(digits_neg_padding, -img_width, cryptoContext))
    c_rotations.append(digits_neg_img_width)
    c_rotations.append(fhe.homo_rotate(digits_padding, -img_width, cryptoContext))
    c_rotations.append(digits_neg_padding)
    c_rotations.append(input)
    c_rotations.append(digits_padding)
    c_rotations.append(fhe.homo_rotate(digits_neg_padding, img_width, cryptoContext))
    c_rotations.append(digits_img_width)
    c_rotations.append(fhe.homo_rotate(digits_padding, img_width, cryptoContext))
    return c_rotations


@fhe.utils.profile_python_function
def conv_initial(input, img_width, padding, num_channel, scale, cryptoContext):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    c_rotations = rot_input(input, img_width, padding, cryptoContext)

    for j in range(num_channel):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(f"conv1bn1-ch{j}-k{k + 1}", cryptoContext.L - input.cur_limbs, input.slots,
                                            cryptoContext, scale)
            k_rows.append(encoded)
        partial_sum = _pairwise_mac(c_rotations, k_rows, cryptoContext)
        partial_sum = fhe.homo_rescale(partial_sum, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        sum_rot = fhe.homo_rotate(partial_sum, 1024, cryptoContext)
        partial_sum = fhe.homo_add(partial_sum, sum_rot, cryptoContext)
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext)
        partial_sum = fhe.homo_mul_pt(partial_sum,
                                      mask_from_to(0, 1024, partial_sum.cur_limbs, partial_sum.slots, cryptoContext),
                                      cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, 1024, cryptoContext) # note: 退出前最后一次轮转就把out0转到左数第一个channel上来了
    return finalsum


@fhe.utils.profile_python_function
def conv_initial_32K(input, img_width, padding, num_channel, num_cipher, scale, cryptoContext):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    c_rotations = rot_input(input, img_width, padding, cryptoContext)

    channel_per_cipher = num_channel//num_cipher

    for j in range(num_channel):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(f"conv1bn1-ch{j}-k{k + 1}", cryptoContext.L - input.cur_limbs, input.slots,
                                            cryptoContext, scale)
            k_rows.append(encoded)
        partial_sum = _pairwise_mac(c_rotations, k_rows, cryptoContext)
        partial_sum = fhe.homo_rescale(partial_sum, 1, cryptoContext)  # RESCALE ADD BY ZRJI

        sum_rot = fhe.homo_rotate(partial_sum, 1024, cryptoContext)
        partial_sum = fhe.homo_add(partial_sum, sum_rot, cryptoContext)
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext)


        partial_sum = fhe.homo_mul_pt(partial_sum,
                                      mask_from_to(0, 1024, partial_sum.cur_limbs, partial_sum.slots, cryptoContext),
                                      cryptoContext)
        if j < channel_per_cipher:
            finalsum0 = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum0, partial_sum, cryptoContext)
            finalsum0 = fhe.homo_rotate(finalsum0, 1024, cryptoContext)
        else:
            finalsum1 = partial_sum.deep_copy() if j == num_channel//num_cipher else fhe.homo_add(finalsum1, partial_sum, cryptoContext)
            finalsum1 = fhe.homo_rotate(finalsum1, 1024, cryptoContext)
    return [finalsum0, finalsum1]


@fhe.utils.profile_python_function
def conv(input, img_width, padding, num_channel, rot_offset, layer, n, channel_offset, scale, cryptoContext):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    c_rotations = rot_input(input, img_width, padding, cryptoContext)
    for j in range(num_channel):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset}-k{k + 1}",
                                            cryptoContext.L - input.cur_limbs, input.slots, cryptoContext, scale)
            k_rows.append(encoded)
        partial_sum = _pairwise_mac(c_rotations, k_rows, cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)
    return finalsum


#separate example
# @fhe.utils.profile_python_function
# def conv_merge2(input, img_width, padding, num_channel, rot_offset, layer, n, channel_offset, scale, cryptoContext):
#     if input.noise_deg > 1:
#         input = fhe.force_rescale(input, 1, cryptoContext)
#
#     c_rotations = rot_input(input, img_width, padding, cryptoContext)
#     intra = 2
#     intra_offset = num_channel//intra
#     for j in range(num_channel//intra):
#         k_rows = []
#         for k in range(9):
#             encoded = read_values_from_file(f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset}-k{k + 1}",
#                                             cryptoContext.L - input.cur_limbs, input.slots, cryptoContext, scale)
#             k_rows.append(encoded)
#         partial_sum = _pairwise_mac(c_rotations, k_rows, cryptoContext)
#
#         k_rows = []
#         for k in range(9):
#             encoded = read_values_from_file(f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset + intra_offset}-k{k + 1}",
#                                             cryptoContext.L - input.cur_limbs, input.slots, cryptoContext, scale)
#             k_rows.append(encoded)
#         partial_sum1 = _pairwise_mac(c_rotations, k_rows, cryptoContext)
#
#         finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
#         finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)
#
#         finalsum1 = partial_sum1.deep_copy() if j == 0 else fhe.homo_add(finalsum1, partial_sum1,
#                                                                                             cryptoContext)
#         finalsum1 = fhe.homo_rotate(finalsum1, rot_offset, cryptoContext)
#         if j==num_channel//intra-1:
#             finalsum1 = fhe.homo_rotate(finalsum1,8192, cryptoContext)
#
#
#     finalsum = fhe.homo_add(finalsum1, finalsum, cryptoContext)
#     finalsum = fhe.homo_rotate(finalsum, -8192, cryptoContext) #fixme: 可以合并到前面最后一次转里头去
#
#     return finalsum

@fhe.utils.profile_python_function
def conv_bsgs(input, img_width, padding, num_channel, rot_offset, layer, n, channel_offset, scale, b_step, cryptoContext):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    g_step = num_channel // b_step
    c_rotations_list = []
    for i in range(b_step): #fixme: use more keys in `rot_input`, and add hoisting for `tmp`
        tmp = input
        if i != 0:
            tmp = fhe.homo_rotate(input, (input.slots//b_step)*i, cryptoContext)
        c_rotations = rot_input(tmp, img_width, padding, cryptoContext)
        c_rotations_list.append(c_rotations)

    for j in range(g_step):
        for i in range(b_step):
            k_rows = []
            for k in range(9):
                encoded = read_values_from_file_bsgs(f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset + i * g_step}-k{k + 1}",
                                                cryptoContext.L - input.cur_limbs, input.slots, b_step, i, cryptoContext, scale)
                k_rows.append(encoded)
            tmp_partial_sum = _pairwise_mac(c_rotations_list[i], k_rows, cryptoContext)

            partial_sum = tmp_partial_sum.deep_copy() if i==0 else fhe.homo_add(partial_sum, tmp_partial_sum, cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = fhe.homo_rotate(finalsum, (input.slots//b_step), cryptoContext) #fixme: 可以合并到前面最后一次转里头去

    return finalsum


@fhe.utils.profile_python_function
def conv_32K(input, img_width, padding, num_channel, rot_offset, layer, n, channel_offset, num_cipher, scale, cryptoContext):
    if input[0].noise_deg > 1:
        input[0] = fhe.force_rescale(input[0], 1, cryptoContext)
        input[1] = fhe.force_rescale(input[1], 1, cryptoContext)

    c_rotations0 = rot_input(input[0], img_width, padding, cryptoContext)
    c_rotations1 = rot_input(input[1], img_width, padding, cryptoContext)
    channel_offset_by_yhh = num_channel // num_cipher
    left_len = 0
    for j in range(num_channel):
        k_rows = [[],[]]
        if j < num_channel//2:
            val_name_list = [f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset}",
                             f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset + channel_offset_by_yhh}"]
        else:
            val_name_list = [f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset}",
                             f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset - channel_offset_by_yhh}"]
        for k in range(9):
            encoded = read_values_from_file_32K_conv([val_name_list[0] + f"-k{k + 1}", val_name_list[1] + f"-k{k + 1}"],
                                                     cryptoContext.L - input[0].cur_limbs, input[0].slots, num_channel,
                                                     left_len, cryptoContext, scale=1.0)
            k_rows[0].append(encoded[0])
            k_rows[1].append(encoded[1])

        partial_sum0 = _pairwise_mac(c_rotations0, k_rows[0], cryptoContext)
        partial_sum1 = _pairwise_mac(c_rotations1, k_rows[1], cryptoContext)


        finalsum0 = partial_sum0.deep_copy() if j == 0 else fhe.homo_add(finalsum0, partial_sum0, cryptoContext)
        finalsum1 = partial_sum1.deep_copy() if j == 0 else fhe.homo_add(finalsum1, partial_sum1, cryptoContext)
        finalsum0 = fhe.homo_rotate(finalsum0, rot_offset, cryptoContext)
        finalsum1 = fhe.homo_rotate(finalsum1, rot_offset, cryptoContext)

        left_len += 1
        if left_len == num_channel // num_cipher:
            left_len = 0
            finalsum1, finalsum0 = finalsum0, finalsum1

    return [finalsum0, finalsum1]


@fhe.utils.profile_python_function
def convbn_dx(input, num_channel, rot_offset, layer, n, channel_offset, biasoff, scale, cryptoContext):
    if input.noise_deg > 1:
        input = fhe.force_rescale(input, 1, cryptoContext)

    for j in range(num_channel):
        encoded = read_values_from_file(f"layer{layer}dx-conv{n}bn{n}-ch{j + channel_offset}-k1",
                                        cryptoContext.L - input.cur_limbs, input.slots, cryptoContext, scale)
        partial_sum = fhe.homo_mul_pt(input, encoded, cryptoContext)

        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = fhe.homo_rescale(finalsum, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    bias1 = read_values_from_file(f"layer{layer}dx-conv{n}bn{n}-bias" + biasoff, cryptoContext.L - finalsum.cur_limbs,
                                  finalsum.slots, cryptoContext, scale)
    finalsum = fhe.homo_add_pt(finalsum, bias1, cryptoContext)

    return finalsum

@fhe.utils.profile_python_function
def convbn_dx_32K(input, num_channel, rot_offset, layer, n, channel_offset, biasoff, num_cipher, scale, cryptoContext):
    if input[0].noise_deg > 1:
        input[0] = fhe.force_rescale(input[0], 1, cryptoContext)
        input[1] = fhe.force_rescale(input[1], 1, cryptoContext)

    channel_offset_by_yhh = num_channel // num_cipher
    left_len = 0

    for j in range(num_channel):
        if j < num_channel//2:
            val_name_list = [f"layer{layer}dx-conv{n}bn{n}-ch{j + channel_offset}",
                            f"layer{layer}dx-conv{n}bn{n}-ch{j + channel_offset + channel_offset_by_yhh}"]
        else:
            val_name_list = [f"layer{layer}dx-conv{n}bn{n}-ch{j + channel_offset}",
                            f"layer{layer}dx-conv{n}bn{n}-ch{j + channel_offset - channel_offset_by_yhh}"]

        encoded = read_values_from_file_32K_conv([val_name_list[0] + f"-k1", val_name_list[1] + f"-k1"],
                                                 cryptoContext.L - input[0].cur_limbs, input[0].slots, num_channel,
                                                 left_len, cryptoContext, scale=1.0)

        partial_sum0 = fhe.homo_mul_pt(input[0], encoded[0], cryptoContext)
        partial_sum1 = fhe.homo_mul_pt(input[1], encoded[1], cryptoContext)

        finalsum0 = partial_sum0.deep_copy() if j == 0 else fhe.homo_add(finalsum0, partial_sum0, cryptoContext)
        finalsum1 = partial_sum1.deep_copy() if j == 0 else fhe.homo_add(finalsum1, partial_sum1, cryptoContext)
        finalsum0 = fhe.homo_rotate(finalsum0, rot_offset, cryptoContext)
        finalsum1 = fhe.homo_rotate(finalsum1, rot_offset, cryptoContext)

        left_len += 1
        if left_len == num_channel // num_cipher:
            left_len = 0
            finalsum1, finalsum0 = finalsum0, finalsum1

    finalsum0 = fhe.homo_rescale(finalsum0, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    finalsum1 = fhe.homo_rescale(finalsum1, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    bias1 = read_values_from_file_32K_Aespa(f"layer{layer}dx-conv{n}bn{n}-bias" + biasoff, cryptoContext.L - finalsum0.cur_limbs,
                                  finalsum0.slots, 2, cryptoContext, scale)
    finalsum0 = fhe.homo_add_pt(finalsum0, bias1[0], cryptoContext)
    finalsum1 = fhe.homo_add_pt(finalsum1, bias1[1], cryptoContext)

    return [finalsum0, finalsum1]


def choose_zero(slots, cryptoContext):
    if slots == 65536:
        return cryptoContext.zero_64K.deep_copy()
    elif slots == 32768:
        return cryptoContext.zero_32K.deep_copy()
    elif slots == 16384:
        return cryptoContext.zero_16K.deep_copy()
    else:
        raise ValueError(f"Unsupported slots value: {slots}")


@fhe.utils.profile_python_function
def downsample1024to256(c1, c2, num_channel, num_cipher, cryptoContext):
    assert num_cipher == 2 or num_cipher == 1

    cipher_list = []
    downsampledchannels_list = []
    if num_cipher == 1:
        old_slots = c1.slots
        c1 = torch.fhe.homo_ops.slot_resize(c1, c1.slots * 2, cryptoContext)
        c2 = torch.fhe.homo_ops.slot_resize(c2, c2.slots * 2, cryptoContext)
        fullpack = fhe.homo_add(
            fhe.homo_mul_pt(c1, mask_first_n(old_slots, c1.cur_limbs, c1.slots, cryptoContext), cryptoContext),
            fhe.homo_mul_pt(c2, mask_scecond_n(old_slots, c2.cur_limbs, c2.slots, cryptoContext), cryptoContext),
            cryptoContext)
        cipher_list.append(fullpack)
    else:
        cipher_list.append(c1)
        cipher_list.append(c2)

    for cipher in cipher_list:
        if cipher.noise_deg > 1:
            fullpack = fhe.homo_rescale(cipher, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        else:
            fullpack = cipher
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
                                     mask_first_n_mod(16, 1024, i, 2 * num_channel // num_cipher, fullpack.cur_limbs,
                                                      fullpack.slots, cryptoContext),
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
        for i in range(2 * num_channel // num_cipher):
            # 将128通道的更紧凑
            masked = fhe.homo_mul_pt(downsampledrows,
                                     mask_channel(i, num_channel, 1024, num_cipher, downsampledrows.cur_limbs,
                                                  cryptoContext),
                                     cryptoContext)
            downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
            downsampledchannels = fhe.homo_rotate(downsampledchannels, -(1024 - 256), cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels, 2 * num_channel // num_cipher * (1024 - 256),
                                              cryptoContext)
        downsampledchannels = fhe.force_rescale(downsampledchannels, 1, cryptoContext)
        downsampledchannels_list.append(downsampledchannels)

    if num_cipher == 1:  # resnet20
        downsampledchannels = downsampledchannels_list[0]
        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_rotate(downsampledchannels,
                                                           - downsampledchannels_list[0].slots // 4, cryptoContext),
                                           cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_rotate(
                                               fhe.homo_rotate(downsampledchannels,
                                                               - downsampledchannels_list[0].slots // 4, cryptoContext),
                                               - downsampledchannels_list[0].slots // 4,
                                               cryptoContext),
                                           cryptoContext)
        downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, downsampledchannels.slots // 4,
                                                             cryptoContext)
        return downsampledchannels
    else:  # resnet18
        downsampledchannels = choose_zero(downsampledchannels_list[0].slots, cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_mul_pt(downsampledchannels_list[0],
                                                           mask_first_n(16384, downsampledchannels_list[0].cur_limbs,
                                                                        downsampledchannels_list[0].slots,
                                                                        cryptoContext), cryptoContext), cryptoContext)

        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_mul_pt(
                                               fhe.homo_rotate(downsampledchannels_list[1], 16384, cryptoContext),
                                               mask_scecond_n(16384, downsampledchannels_list[1].cur_limbs,
                                                              downsampledchannels_list[1].slots, cryptoContext),
                                               cryptoContext), cryptoContext)
        # fixme: 因为前面的choose_zero按照65536slots选的，但这里实际是32768，
        #  因此为了正确解密要按照 repeated packing的方式转32768构造上slots中的数据。
        #  其实就是一个slots_resize的过程。
        #  感觉应该通过管理避免这里/这个函数里出现resize？
        #  注：如果没有最后的slots_resize变小就不需要在downsample里处理前面计算的rescale。
        downsampledchannels = fhe.homo_add(downsampledchannels,
                                           fhe.homo_rotate(downsampledchannels, 16384 * 2, cryptoContext),
                                           cryptoContext)
        downsampledchannels.slots = 32768 # to replace the following two lines
        # downsampledchannels = fhe.force_rescale(downsampledchannels, 1, cryptoContext)
        # downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, 32768, cryptoContext)

        return downsampledchannels


@fhe.utils.profile_python_function
def downsample1024to256_32K(c1, c2, num_channel, num_cipher, cryptoContext):
    assert num_cipher == 2 or num_cipher == 1

    cipher_list = []
    downsampledchannels_list = []

    cipher_list.append(c1)
    cipher_list.append(c2)

    for cipher in cipher_list:
        if cipher.noise_deg > 1:
            fullpack = fhe.homo_rescale(cipher, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        else:
            fullpack = cipher
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
                                     mask_first_n_mod(16, 1024, i, 2 * num_channel // num_cipher, fullpack.cur_limbs,
                                                      fullpack.slots, cryptoContext),
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
        for i in range(2 * num_channel // num_cipher):
            # 将128通道的更紧凑
            masked = fhe.homo_mul_pt(downsampledrows,
                                     mask_channel(i, num_channel, 1024, num_cipher, downsampledrows.cur_limbs,
                                                  cryptoContext),
                                     cryptoContext)
            downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
            downsampledchannels = fhe.homo_rotate(downsampledchannels, -(1024 - 256), cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels, 2 * num_channel // num_cipher * (1024 - 256),
                                              cryptoContext)
        downsampledchannels = fhe.force_rescale(downsampledchannels, 1, cryptoContext)
        downsampledchannels_list.append(downsampledchannels)


    # resnet18
    data_per_cipher = 8192 # fixme: poor work around
    # todo: in this function, all the xx.slots is coincidentally euqal to 32768, which is the final required value,
    #  therefore no need to further resize, and outside we could merge the cts without further masks
    #  [!!!]change with caution
    downsampledchannels = choose_zero(downsampledchannels_list[0].slots, cryptoContext) # fixme: hardcode to 32768 regardless the number of ciphers?
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                       fhe.homo_mul_pt(downsampledchannels_list[0],
                                                       mask_first_n(data_per_cipher, downsampledchannels_list[0].cur_limbs,
                                                                    downsampledchannels_list[0].slots,
                                                                    cryptoContext), cryptoContext), cryptoContext)

    downsampledchannels = fhe.homo_add(downsampledchannels,
                                       fhe.homo_mul_pt(
                                           fhe.homo_rotate(downsampledchannels_list[1], -data_per_cipher, cryptoContext),
                                           mask_from_to(data_per_cipher, data_per_cipher*2, downsampledchannels_list[1].cur_limbs,
                                                          downsampledchannels_list[1].slots, cryptoContext),
                                           cryptoContext), cryptoContext)

    return downsampledchannels


@fhe.utils.profile_python_function
def downsample256to64(c1, c2, num_channel, cryptoContext):
    old_slots = c1.slots
    c1 = torch.fhe.homo_ops.slot_resize(c1, c1.slots * 2, cryptoContext)
    c2 = torch.fhe.homo_ops.slot_resize(c2, c2.slots * 2, cryptoContext)
    fullpack = fhe.homo_add(
        fhe.homo_mul_pt(c1, mask_first_n(old_slots, c1.cur_limbs, c1.slots, cryptoContext), cryptoContext),
        fhe.homo_mul_pt(c2, mask_scecond_n(old_slots, c2.cur_limbs, c2.slots, cryptoContext), cryptoContext),
        cryptoContext)

    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext),
        gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
    fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(fullpack,
                     fhe.homo_rotate(
                         fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext), cryptoContext),
        gen_mask(4, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
    fullpack = fhe.force_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    fullpack = fhe.homo_add(fullpack,
                            fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext)

    downsampledrows = choose_zero(fullpack.slots, cryptoContext)
    downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs,
                                             cryptoContext)  # drop_last_elements ADD BY ZRJI

    assert fullpack.noise_deg == 1
    for i in range(32):
        masked = fhe.homo_mul_pt(fullpack,
                                 mask_first_n_mod2(8, 256, i, 2 * num_channel, fullpack.cur_limbs, fullpack.slots,
                                                   cryptoContext), cryptoContext)
        downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i < 31:
            fullpack = fhe.homo_rotate(fullpack, 24, cryptoContext)

    downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    downsampledchannels = choose_zero(fullpack.slots, cryptoContext)
    downsampledchannels = fhe.drop_last_elements(downsampledchannels,
                                                 downsampledchannels.cur_limbs - downsampledrows.cur_limbs,
                                                 cryptoContext)  # drop_last_elements ADD BY ZRJI
    assert downsampledrows.noise_deg == 1

    for i in range(num_channel * 2):
        masked = fhe.homo_mul_pt(downsampledrows,
                                 mask_channel(i, num_channel, 256, 1, downsampledrows.cur_limbs, cryptoContext),
                                 cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels, -(256 - 64), cryptoContext)

    downsampledchannels = fhe.homo_rotate(downsampledchannels, (256 - 64) * (num_channel * 2), cryptoContext)

    # fixme: 原版这部分在显式的做slot resize的工作，和最后的slot resize操作是重复的，这个问题和1024-256 version是一致的。
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                       fhe.homo_rotate(downsampledchannels, -(downsampledchannels.slots // 4),
                                                       cryptoContext), cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                       fhe.homo_rotate(
                                           fhe.homo_rotate(downsampledchannels, -(downsampledchannels.slots // 4),
                                                           cryptoContext), -(downsampledchannels.slots // 4),
                                           cryptoContext),
                                       cryptoContext)
    downsampledchannels = fhe.force_rescale(downsampledchannels, 1, cryptoContext)
    downsampledchannels = fhe.homo_ops.slot_resize(downsampledchannels, (downsampledchannels.slots // 4),
                                                         cryptoContext)

    return downsampledchannels


@fhe.utils.profile_python_function
def downsample256to64_32K(c1, c2, num_channel, cryptoContext):
    cipher_list = [c1, c2]
    num_cipher = 2
    result_list = []
    for cipher in cipher_list:
        fullpack = cipher # now slots is 32768, for 32K version
        fullpack = fhe.homo_mul_pt(
            fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext),
            gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
        fullpack = fhe.homo_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        fullpack = fhe.homo_mul_pt(
            fhe.homo_add(fullpack,
                         fhe.homo_rotate(
                             fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext), cryptoContext),
            gen_mask(4, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
        fullpack = fhe.force_rescale(fullpack, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        fullpack = fhe.homo_add(fullpack,
                                fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext)

        downsampledrows = choose_zero(fullpack.slots, cryptoContext)
        downsampledrows = fhe.drop_last_elements(downsampledrows, downsampledrows.cur_limbs - fullpack.cur_limbs,
                                                 cryptoContext)  # drop_last_elements ADD BY ZRJI

        assert fullpack.noise_deg == 1
        for i in range(32):
            masked = fhe.homo_mul_pt(fullpack,
                                     mask_first_n_mod2(8, 256, i, num_channel, fullpack.cur_limbs, fullpack.slots,
                                                       cryptoContext), cryptoContext) # fixme: note: remove `2*` in original version cause we do not merge cipher when logN=16
            downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
            if i < 31:
                fullpack = fhe.homo_rotate(fullpack, 24, cryptoContext)

        downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext)  # RESCALE ADD BY ZRJI
        downsampledchannels = choose_zero(fullpack.slots, cryptoContext)
        downsampledchannels = fhe.drop_last_elements(downsampledchannels,
                                                     downsampledchannels.cur_limbs - downsampledrows.cur_limbs,
                                                     cryptoContext)  # drop_last_elements ADD BY ZRJI
        assert downsampledrows.noise_deg == 1

        for i in range(num_channel): # fixme: note: remove `2*` in original version cause we do not merge cipher when logN=16
            masked = fhe.homo_mul_pt(downsampledrows,
                                     mask_channel(i, num_channel, 256, num_cipher, downsampledrows.cur_limbs, cryptoContext),
                                     cryptoContext) # fixme: input 瞎凑的，不知道这个函数在干什么
            downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
            downsampledchannels = fhe.homo_rotate(downsampledchannels, -(256 - 64), cryptoContext)

        downsampledchannels = fhe.homo_rotate(downsampledchannels, (256 - 64) * num_channel, cryptoContext) # fixme: note: remove `2*` in original version cause we do not merge cipher when logN=16

        result_list.append(downsampledchannels)

    cipher_merged = fhe.homo_add(result_list[0],
                                 fhe.homo_rotate(result_list[1], -8192, cryptoContext),
                                 cryptoContext)
    cipher_merged = fhe.homo_rescale(cipher_merged, 1, cryptoContext)
    cipher_merged = fhe.homo_ops.slot_resize(cipher_merged, 16384, cryptoContext) # fixme: could be removed, should not hardcode
    return cipher_merged



@fhe.utils.profile_python_function
def downsample64to16(c1, c2, num_channel, cryptoContext):
    old_slots = c1.slots
    c1 = torch.fhe.homo_ops.slot_resize(c1, c1.slots * 2, cryptoContext)
    c2 = torch.fhe.homo_ops.slot_resize(c2, c2.slots * 2, cryptoContext)
    fullpack = fhe.homo_add(
        fhe.homo_mul_pt(c1, mask_first_n(old_slots, c1.cur_limbs, c1.slots, cryptoContext), cryptoContext),
        fhe.homo_mul_pt(c2, mask_scecond_n(old_slots, c2.cur_limbs, c2.slots, cryptoContext), cryptoContext),
        cryptoContext)
    fullpack = fhe.force_rescale(fullpack, 1, cryptoContext)

    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext),
        gen_mask(2, fullpack.cur_limbs, fullpack.slots, cryptoContext), cryptoContext)
    fullpack = fhe.force_rescale(fullpack, 1, cryptoContext)
    fullpack = fhe.homo_add(fullpack, fhe.homo_rotate(fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext),
                            cryptoContext)
    downsampledrows = choose_zero(fullpack.slots, cryptoContext)

    for i in range(4):
        masked = fhe.homo_mul_pt(fullpack,
                                 mask_first_n_mod3(4, 64, i, num_channel * 2, fullpack.cur_limbs, fullpack.slots,
                                                   cryptoContext), cryptoContext)
        downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i < 3:
            fullpack = fhe.homo_rotate(fullpack, 12, cryptoContext)

    downsampledrows = fhe.force_rescale(downsampledrows, 1, cryptoContext)  # RESCALE ADD BY ZRJI
    downsampledchannels = choose_zero(fullpack.slots, cryptoContext)
    downsampledchannels = fhe.drop_last_elements(downsampledchannels,
                                                 downsampledchannels.cur_limbs - downsampledrows.cur_limbs,
                                                 cryptoContext)  # drop_last_elements ADD BY ZRJI
    assert downsampledrows.noise_deg == 1

    for i in range(num_channel * 2):
        masked = fhe.homo_mul_pt(downsampledrows,
                                 mask_channel(i, num_channel, 64, 1, downsampledrows.cur_limbs, cryptoContext),
                                 cryptoContext)
        downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
        downsampledchannels = fhe.homo_rotate(downsampledchannels, -(64 - 16), cryptoContext)

    downsampledchannels = fhe.homo_rotate(downsampledchannels, (64 - 16) * num_channel * 2, cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                       fhe.homo_rotate(downsampledchannels, -(downsampledchannels.slots // 4),
                                                       cryptoContext), cryptoContext)
    downsampledchannels = fhe.homo_add(downsampledchannels,
                                       fhe.homo_rotate(
                                           fhe.homo_rotate(downsampledchannels, -(downsampledchannels.slots // 4),
                                                           cryptoContext), -(downsampledchannels.slots // 4),
                                           cryptoContext),
                                       cryptoContext)
    downsampledchannels = fhe.force_rescale(downsampledchannels, 1, cryptoContext)
    downsampledchannels = torch.fhe.homo_ops.slot_resize(downsampledchannels, (downsampledchannels.slots // 4),
                                                         cryptoContext)
    return downsampledchannels
