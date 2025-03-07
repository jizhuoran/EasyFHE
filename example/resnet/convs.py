import torch.fhe as fhe
from utils import *


def eval_add_many(ciphertexts, cryptoContext):
    inSize = len(ciphertexts)
    if inSize < 1:
        raise ValueError("Input ciphertext vector size should be 1 or more")
    sum = ciphertexts[0].deep_copy()
    for i in range(1, inSize):
        sum = fhe.homo_add(sum, ciphertexts[i], cryptoContext)
    return sum


def convbn_initial(input, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    img_width = 32
    padding = 1
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)
    c_rotations = []
    digit_pos_pad = fhe.eval_fast_rotate(
        digits, input, padding, True, True, cryptoContext
    )
    digit_neg_pad = fhe.eval_fast_rotate(
        digits, input, -padding, True, True, cryptoContext
    )
    c_rotations.append(fhe.homo_rotate(digit_neg_pad, -img_width, cryptoContext))
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    )
    c_rotations.append(fhe.homo_rotate(digit_pos_pad, -img_width, cryptoContext))
    c_rotations.append(digit_neg_pad)
    c_rotations.append(input)
    c_rotations.append(digit_pos_pad)
    c_rotations.append(fhe.homo_rotate(digit_neg_pad, img_width, cryptoContext))
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)
    )
    c_rotations.append(fhe.homo_rotate(digit_pos_pad, img_width, cryptoContext))

    bias = read_values_from_file(
        cryptoContext,
        "conv1bn1-bias",
        cryptoContext.L - input.cur_limbs,
        1,
        16384,
        scale,
    )

    for j in range(16):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(
                cryptoContext,
                f"conv1bn1-ch{j}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                16384,
                scale,
            )
            k_rows.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

        sum = eval_add_many(k_rows, cryptoContext)
        res = sum.deep_copy()
        sum_rot = fhe.homo_rotate(sum, 1024, cryptoContext)
        res = fhe.homo_add(res, sum_rot, cryptoContext)
        res = fhe.homo_add(
            res, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext
        )

        res = fhe.homo_mul_pt(
            res,
            mask_from_to(0, 1024, res.cur_limbs, he_res20_ctx, cryptoContext),
            cryptoContext,
        )

        if j == 0:
            finalsum = res.deep_copy()
            finalsum = fhe.homo_rotate(finalsum, 1024, cryptoContext)

        else:
            finalsum = fhe.homo_add(finalsum, res, cryptoContext)
            finalsum = fhe.homo_rotate(finalsum, 1024, cryptoContext)

    finalsum = fhe.homo_add_pt(finalsum, bias, cryptoContext)

    return finalsum


def convbn(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    img_width = 32
    padding = 1

    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)

    c_rotations = []
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),
            -img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext),
            -img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    )
    c_rotations.append(input)
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),
            img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext),
            img_width,
            cryptoContext,
        )
    )

    bias = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias",
        cryptoContext.L - input.cur_limbs,
        1,
        16384,
        scale,
    )

    for j in range(16):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                16384,
                scale,
            )
            k_rows.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

        sum = eval_add_many(k_rows, cryptoContext)
        if j == 0:
            finalsum = sum.deep_copy()
            finalsum = fhe.homo_rotate(finalsum, -1024, cryptoContext)
        else:
            finalsum = fhe.homo_add(finalsum, sum, cryptoContext)
            finalsum = fhe.homo_rotate(finalsum, -1024, cryptoContext)
    finalsum = fhe.homo_add_pt(finalsum, bias, cryptoContext)

    return finalsum


def convbn2(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    img_width = 16
    padding = 1
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)

    c_rotations = []
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),
            -img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext),
            -img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    )
    c_rotations.append(input)
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),
            img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext),
            img_width,
            cryptoContext,
        )
    )

    bias = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias",
        cryptoContext.L - input.cur_limbs,
        1,
        8192,
        scale,
    )

    for j in range(32):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                8192,
                scale,
            )
            k_rows.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

        sum = eval_add_many(k_rows, cryptoContext)
        if j == 0:
            finalsum = sum.deep_copy()
            finalsum = fhe.homo_rotate(finalsum, -256, cryptoContext)
        else:
            finalsum = fhe.homo_add(finalsum, sum, cryptoContext)
            finalsum = fhe.homo_rotate(finalsum, -256, cryptoContext)
    finalsum = fhe.homo_add_pt(finalsum, bias, cryptoContext)

    return finalsum


def convbn3(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    img_width = 8
    padding = 1
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)
    c_rotations = []
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),
            -img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext),
            -img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    )
    c_rotations.append(input)
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext),
            img_width,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext),
            img_width,
            cryptoContext,
        )
    )

    bias = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias",
        cryptoContext.L - input.cur_limbs,
        1,
        4096,
        scale,
    )

    for j in range(64):
        k_rows = []
        for k in range(9):
            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                4096,
                scale,
            )
            k_rows.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))
        sum = eval_add_many(k_rows, cryptoContext)
        if j == 0:
            finalsum = sum.deep_copy()
            finalsum = fhe.homo_rotate(finalsum, -64, cryptoContext)
        else:
            finalsum = fhe.homo_add(finalsum, sum, cryptoContext)
            finalsum = fhe.homo_rotate(finalsum, -64, cryptoContext)

    finalsum = fhe.homo_add_pt(finalsum, bias, cryptoContext)
    return finalsum


def convbn1632sx(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    img_width = 32
    padding = 1
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)

    c_rotations = []
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext),
            -padding,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext),
            padding,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    )
    c_rotations.append(input)
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext),
            -padding,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext),
            padding,
            cryptoContext,
        )
    )

    bias1 = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias1",
        cryptoContext.L - input.cur_limbs,
        1,
        16384,
        scale,
    )
    bias2 = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias2",
        cryptoContext.L - input.cur_limbs,
        1,
        16384,
        scale,
    )

    for j in range(16):
        k_rows016 = []
        k_rows1632 = []
        for k in range(9):
            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                16384,
                scale,
            )
            k_rows016.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j+16}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                16384,
                scale,
            )
            k_rows1632.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

        sum016 = eval_add_many(k_rows016, cryptoContext)
        sum1632 = eval_add_many(k_rows1632, cryptoContext)

        if j == 0:
            finalsum016 = sum016.deep_copy()
            finalsum016 = fhe.homo_rotate(finalsum016, -1024, cryptoContext)
            finalsum1632 = sum1632.deep_copy()
            finalsum1632 = fhe.homo_rotate(finalsum1632, -1024, cryptoContext)
        else:
            finalsum016 = fhe.homo_add(finalsum016, sum016, cryptoContext)
            finalsum016 = fhe.homo_rotate(finalsum016, -1024, cryptoContext)
            finalsum1632 = fhe.homo_add(finalsum1632, sum1632, cryptoContext)
            finalsum1632 = fhe.homo_rotate(finalsum1632, -1024, cryptoContext)

    finalsum016 = fhe.homo_add_pt(finalsum016, bias1, cryptoContext)
    finalsum1632 = fhe.homo_add_pt(finalsum1632, bias2, cryptoContext)

    return finalsum016, finalsum1632


def convbn1632dx(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    bias1 = read_values_from_file(
        cryptoContext,
        f"layer{layer}dx-conv{n}bn{n}-bias1",
        cryptoContext.L - input.cur_limbs,
        1,
        16384,
        scale,
    )
    bias2 = read_values_from_file(
        cryptoContext,
        f"layer{layer}dx-conv{n}bn{n}-bias2",
        cryptoContext.L - input.cur_limbs,
        1,
        16384,
        scale,
    )

    for j in range(16):
        k_rows016 = []
        k_rows1632 = []

        encoded = read_values_from_file(
            cryptoContext,
            f"layer{layer}dx-conv{n}bn{n}-ch{j}-k1",
            cryptoContext.L - input.cur_limbs,
            1,
            he_res20_ctx.cur_num_slots,
            scale,
        )
        k_rows016.append(fhe.homo_mul_pt(input, encoded, cryptoContext))

        encoded = read_values_from_file(
            cryptoContext,
            f"layer{layer}dx-conv{n}bn{n}-ch{j+16}-k1",
            cryptoContext.L - input.cur_limbs,
            1,
            he_res20_ctx.cur_num_slots,
            scale,
        )
        k_rows1632.append(fhe.homo_mul_pt(input, encoded, cryptoContext))

        sum016 = eval_add_many(k_rows016, cryptoContext)
        sum1632 = eval_add_many(k_rows1632, cryptoContext)

        if j == 0:
            finalsum016 = sum016.deep_copy()
            finalsum016 = fhe.homo_rotate(finalsum016, -1024, cryptoContext)
            finalsum1632 = sum1632.deep_copy()
            finalsum1632 = fhe.homo_rotate(finalsum1632, -1024, cryptoContext)
        else:
            finalsum016 = fhe.homo_add(finalsum016, sum016, cryptoContext)
            finalsum016 = fhe.homo_rotate(finalsum016, -1024, cryptoContext)
            finalsum1632 = fhe.homo_add(finalsum1632, sum1632, cryptoContext)
            finalsum1632 = fhe.homo_rotate(finalsum1632, -1024, cryptoContext)

    finalsum016 = fhe.homo_add_pt(finalsum016, bias1, cryptoContext)
    finalsum1632 = fhe.homo_add_pt(finalsum1632, bias2, cryptoContext)

    return finalsum016, finalsum1632


def convbn3264sx(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    img_width = 16
    padding = 1
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)

    c_rotations = []
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext),
            -padding,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext),
            padding,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    )
    c_rotations.append(input)
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext),
            -padding,
            cryptoContext,
        )
    )
    c_rotations.append(
        fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)
    )
    c_rotations.append(
        fhe.homo_rotate(
            fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext),
            padding,
            cryptoContext,
        )
    )

    bias1 = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias1",
        cryptoContext.L - input.cur_limbs,
        1,
        8192,
        scale,
    )
    bias2 = read_values_from_file(
        cryptoContext,
        f"layer{layer}-conv{n}bn{n}-bias2",
        cryptoContext.L - input.cur_limbs,
        1,
        8192,
        scale,
    )
    for j in range(32):
        k_rows032 = []
        k_rows3264 = []
        for k in range(9):
            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                8192,
                scale,
            )
            k_rows032.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

            encoded = read_values_from_file(
                cryptoContext,
                f"layer{layer}-conv{n}bn{n}-ch{j+32}-k{k+1}",
                cryptoContext.L - input.cur_limbs,
                1,
                8192,
                scale,
            )
            k_rows3264.append(fhe.homo_mul_pt(c_rotations[k], encoded, cryptoContext))

        sum032 = eval_add_many(k_rows032, cryptoContext)
        sum3264 = eval_add_many(k_rows3264, cryptoContext)

        if j == 0:
            finalsum032 = sum032.deep_copy()
            finalsum032 = fhe.homo_rotate(finalsum032, -256, cryptoContext)
            finalsum3264 = sum3264.deep_copy()
            finalsum3264 = fhe.homo_rotate(finalsum3264, -256, cryptoContext)
        else:
            finalsum032 = fhe.homo_add(finalsum032, sum032, cryptoContext)
            finalsum032 = fhe.homo_rotate(finalsum032, -256, cryptoContext)
            finalsum3264 = fhe.homo_add(finalsum3264, sum3264, cryptoContext)
            finalsum3264 = fhe.homo_rotate(finalsum3264, -256, cryptoContext)

    finalsum032 = fhe.homo_add_pt(finalsum032, bias1, cryptoContext)
    finalsum3264 = fhe.homo_add_pt(finalsum3264, bias2, cryptoContext)

    return finalsum032, finalsum3264


def convbn3264dx(input, layer, n, scale, he_res20_ctx, cryptoContext):
    if input.noise_deg > 1:
        input = homo_ops.homo_rescale_internal(input, 1, cryptoContext)
    bias1 = read_values_from_file(
        cryptoContext,
        f"layer{layer}dx-conv{n}bn{n}-bias1",
        cryptoContext.L - input.cur_limbs,
        1,
        8192,
        scale,
    )
    bias2 = read_values_from_file(
        cryptoContext,
        f"layer{layer}dx-conv{n}bn{n}-bias2",
        cryptoContext.L - input.cur_limbs,
        1,
        8192,
        scale,
    )
    for j in range(32):
        k_rows032 = []
        k_rows3264 = []

        encoded = read_values_from_file(
            cryptoContext,
            f"layer{layer}dx-conv{n}bn{n}-ch{j}-k1",
            cryptoContext.L - input.cur_limbs,
            1,
            8192,
            scale,
        )
        k_rows032.append(fhe.homo_mul_pt(input, encoded, cryptoContext))

        encoded = read_values_from_file(
            cryptoContext,
            f"layer{layer}dx-conv{n}bn{n}-ch{j+32}-k1",
            cryptoContext.L - input.cur_limbs,
            1,
            8192,
            scale,
        )
        k_rows3264.append(fhe.homo_mul_pt(input, encoded, cryptoContext))

        sum032 = eval_add_many(k_rows032, cryptoContext)
        sum3264 = eval_add_many(k_rows3264, cryptoContext)

        if j == 0:
            finalsum032 = sum032.deep_copy()
            finalsum032 = fhe.homo_rotate(finalsum032, -256, cryptoContext)
            finalsum3264 = sum3264.deep_copy()
            finalsum3264 = fhe.homo_rotate(finalsum3264, -256, cryptoContext)
        else:
            finalsum032 = fhe.homo_add(finalsum032, sum032, cryptoContext)
            finalsum032 = fhe.homo_rotate(finalsum032, -256, cryptoContext)
            finalsum3264 = fhe.homo_add(finalsum3264, sum3264, cryptoContext)
            finalsum3264 = fhe.homo_rotate(finalsum3264, -256, cryptoContext)

    finalsum032 = fhe.homo_add_pt(finalsum032, bias1, cryptoContext)
    finalsum3264 = fhe.homo_add_pt(finalsum3264, bias2, cryptoContext)

    return finalsum032, finalsum3264


def downsample1024to256(c1, c2, he_res20_ctx, cryptoContext):

    c1.slots = 32768
    c2.slots = 32768
    he_res20_ctx.cur_num_slots = 16384 * 2
    fullpack = fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            mask_first_n(16384, c1.cur_limbs, he_res20_ctx, cryptoContext),
            cryptoContext,
        ),
        fhe.homo_mul_pt(
            c2,
            mask_scecond_n(16384, c2.cur_limbs, he_res20_ctx, cryptoContext),
            cryptoContext,
        ),
        cryptoContext,
    )

    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(
            fullpack, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext
        ),
        gen_mask(2, fullpack.cur_limbs, he_res20_ctx, cryptoContext),
        cryptoContext,
    )
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(
            fullpack,
            fhe.homo_rotate(
                fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext
            ),
            cryptoContext,
        ),
        gen_mask(4, fullpack.cur_limbs, he_res20_ctx, cryptoContext),
        cryptoContext,
    )
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(
            fullpack, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext
        ),
        gen_mask(8, fullpack.cur_limbs, he_res20_ctx, cryptoContext),
        cryptoContext,
    )
    fullpack = fhe.homo_add(
        fullpack, fhe.homo_rotate(fullpack, 8, cryptoContext), cryptoContext
    )

    if fullpack.noise_deg > 1:
        fullpack = homo_ops.homo_rescale_internal(fullpack, 1, cryptoContext)
    downsampledrows = cryptoContext.zero_32K
    for i in range(16):
        masked = fhe.homo_mul_pt(
            fullpack,
            mask_first_n_mod(16, 1024, i, fullpack.cur_limbs, cryptoContext),
            cryptoContext,
        )
        downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i < 15:
            fullpack = fhe.homo_rotate(fullpack, 64 - 16, cryptoContext)

    if downsampledrows.noise_deg > 1:
        downsampledrows = homo_ops.homo_rescale_internal(
            downsampledrows, 1, cryptoContext
        )
    downsampledchannels = cryptoContext.zero_32K
    for i in range(32):
        masked = fhe.homo_mul_pt(
            downsampledrows,
            mask_channel(i, downsampledrows.cur_limbs, cryptoContext),
            cryptoContext,
        )
        downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
        downsampledchannels = fhe.homo_rotate(
            downsampledchannels, -(1024 - 256), cryptoContext
        )

    downsampledchannels = fhe.homo_rotate(
        downsampledchannels, (1024 - 256) * 32, cryptoContext
    )
    downsampledchannels = fhe.homo_add(
        downsampledchannels,
        fhe.homo_rotate(downsampledchannels, -8192, cryptoContext),
        cryptoContext,
    )
    downsampledchannels = fhe.homo_add(
        downsampledchannels,
        fhe.homo_rotate(
            fhe.homo_rotate(downsampledchannels, -8192, cryptoContext),
            -8192,
            cryptoContext,
        ),
        cryptoContext,
    )
    downsampledchannels.slots = 8192

    return downsampledchannels


def downsample256to64(c1, c2, he_res20_ctx, cryptoContext):

    c1.slots = 16384
    c2.slots = 16384
    he_res20_ctx.cur_num_slots = 8192 * 2
    fullpack = fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            mask_first_n(8192, c1.cur_limbs, he_res20_ctx, cryptoContext),
            cryptoContext,
        ),
        fhe.homo_mul_pt(
            c2,
            mask_scecond_n(8192, c2.cur_limbs, he_res20_ctx, cryptoContext),
            cryptoContext,
        ),
        cryptoContext,
    )

    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(
            fullpack, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext
        ),
        gen_mask(2, fullpack.cur_limbs, he_res20_ctx, cryptoContext),
        cryptoContext,
    )
    fullpack = fhe.homo_mul_pt(
        fhe.homo_add(
            fullpack,
            fhe.homo_rotate(
                fhe.homo_rotate(fullpack, 1, cryptoContext), 1, cryptoContext
            ),
            cryptoContext,
        ),
        gen_mask(4, fullpack.cur_limbs, he_res20_ctx, cryptoContext),
        cryptoContext,
    )
    fullpack = fhe.homo_add(
        fullpack, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext
    )

    downsampledrows = cryptoContext.zero_16K

    if fullpack.noise_deg > 1:
        fullpack = homo_ops.homo_rescale_internal(fullpack, 1, cryptoContext)
    for i in range(32):
        masked = fhe.homo_mul_pt(
            fullpack,
            mask_first_n_mod2(8, 256, i, fullpack.cur_limbs, cryptoContext),
            cryptoContext,
        )
        downsampledrows = fhe.homo_add(downsampledrows, masked, cryptoContext)
        if i < 31:
            fullpack = fhe.homo_rotate(fullpack, 24, cryptoContext)

    downsampledchannels = cryptoContext.zero_16K
    if downsampledrows.noise_deg > 1:
        downsampledrows = homo_ops.homo_rescale_internal(
            downsampledrows, 1, cryptoContext
        )

    for i in range(64):
        masked = fhe.homo_mul_pt(
            downsampledrows,
            mask_channel2(i, downsampledrows.cur_limbs, cryptoContext),
            cryptoContext,
        )
        downsampledchannels = fhe.homo_add(downsampledchannels, masked, cryptoContext)
        downsampledchannels = fhe.homo_rotate(
            downsampledchannels, -(256 - 64), cryptoContext
        )

    downsampledchannels = fhe.homo_rotate(
        downsampledchannels, (256 - 64) * 64, cryptoContext
    )
    downsampledchannels = fhe.homo_add(
        downsampledchannels,
        fhe.homo_rotate(downsampledchannels, -4096, cryptoContext),
        cryptoContext,
    )
    downsampledchannels = fhe.homo_add(
        downsampledchannels,
        fhe.homo_rotate(
            fhe.homo_rotate(downsampledchannels, -4096, cryptoContext),
            -4096,
            cryptoContext,
        ),
        cryptoContext,
    )
    downsampledchannels.slots = 4096

    return downsampledchannels
