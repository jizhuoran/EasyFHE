import torch
import numpy as np

def altalena(v):
    new_v = []
    for i in range(len(v)):
        if i % 2 != 0:
            new_v.append(0)
        elif i % 64 >= 32 and i % 64 < 64:
            new_v.append(0)
        else:
            new_v.append(v[i])
    return new_v


def altalena2(v):
    new_v = []
    for i in range(len(v)):
        if i % 2 != 0:
            new_v.append(0)
        elif i % 32 >= 16 and i % 32 < 32:
            new_v.append(0)
        else:
            new_v.append(v[i])
    return new_v


def altalena3(v):
    new_v = []
    for i in range(len(v)):
        if i % 2 != 0:
            new_v.append(0)
        elif i % 16 >= 8 and i % 16 < 16:
            new_v.append(0)
        else:
            new_v.append(v[i])
    return new_v


def _export_conv_block_3x3(
    *,
    get_weight_vec,
    out_channels,
    in_channels,
    repeat_len,
    roll_step,
    multiply_step,
    path,
    file_prefix,
    bin_masks=None,
    A2_scaling=None,
    post_k_callback=None,
):
    for out_ch in range(out_channels):
        ks = [np.array([]) for _ in range(9)]
        for in_ch in range(in_channels):
            w9 = get_weight_vec(out_ch, in_ch)
            for idx in range(9):
                ks[idx] = np.append(ks[idx], np.repeat(w9[idx], repeat_len))

        if bin_masks is not None:
            for idx in range(9):
                ks[idx] = np.multiply(ks[idx], bin_masks[idx])

        if A2_scaling is not None:
            scale_val = A2_scaling.detach().cpu().numpy()
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], np.repeat(scale_val, multiply_step))

        if post_k_callback is not None:
            ks = post_k_callback(ks, out_ch)

        for idx, k in enumerate(ks, start=1):
            np.savetxt(
                f"{path}/{file_prefix}-ch{out_ch}-k{idx}.bin", k, delimiter=",")
    print(file_prefix, "done.")


def _export_conv_block_3x3_1(
    *,
    get_weight_vec,
    out_channels,
    in_channels,
    repeat_len,
    roll_step,
    multiply_step,
    path,
    file_prefix,
    bin_masks=None,
    A2_scaling=None,
    post_k_callback=None,
):
    for out_ch in range(out_channels):
        ks = [np.array([]) for _ in range(9)]
        for in_ch in range(in_channels):
            w9 = get_weight_vec(out_ch, in_ch)
            for idx in range(9):
                ks[idx] = np.append(ks[idx], np.repeat(w9[idx], repeat_len))
        for _ in range(out_channels - in_channels):
            for idx in range(9):
                ks[idx] = np.append(ks[idx], np.repeat(0, 1024))

        if bin_masks is not None:
            for idx in range(9):
                ks[idx] = np.multiply(ks[idx], bin_masks[idx])

        if A2_scaling is not None:
            scale_val = A2_scaling[out_ch].detach().cpu().numpy()
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], np.repeat(scale_val, multiply_step))

        if post_k_callback is not None:
            ks = post_k_callback(ks, out_ch)

        for idx, k in enumerate(ks, start=1):
            np.savetxt(
                f"{path}/{file_prefix}-ch{out_ch}-k{idx}.bin", k, delimiter=",")
    print(file_prefix, "done.")


def _export_conv_block_3x3_2(
    *,
    get_weight_vec,
    out_channels,
    in_channels,
    repeat_len,
    roll_step,
    multiply_step,
    path,
    file_prefix,
    bin_masks=None,
    A2_scaling=None,
    post_k_callback=None,
):
    for out_ch in range(in_channels):
        ks = [np.array([]) for _ in range(9)]
        for in_ch in range(out_channels):
            w9 = get_weight_vec(out_ch, in_ch)
            for idx in range(9):
                ks[idx] = np.append(ks[idx], np.repeat(w9[idx], repeat_len))

        if bin_masks is not None:
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], altalena(np.tile(bin_masks[idx], 2)))

        if A2_scaling is not None:
            scale_val = A2_scaling.detach().cpu().numpy()
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], np.repeat(scale_val, multiply_step))

        if post_k_callback is not None:
            ks = post_k_callback(ks, out_ch)

        for idx, k in enumerate(ks, start=1):
            rolled = altalena(np.roll(k, roll_step * out_ch))
            rolled2 = altalena(np.roll(k, roll_step * out_ch - 1))
            np.savetxt(f"{path}/{file_prefix}-ch{out_ch}-k{idx}.bin",
                       rolled, delimiter=",")
            np.savetxt(
                f"{path}/{file_prefix}-ch{out_ch+in_channels}-k{idx}.bin", rolled2, delimiter=",")
    print(file_prefix, "done.")


def _export_conv_block_3x3_2_20(
    *,
    get_weight_vec,
    out_channels,
    in_channels,
    repeat_len,
    roll_step,
    multiply_step,
    path,
    file_prefix,
    bin_masks=None,
    A2_scaling=None,
    post_k_callback=None,
):
    for out_ch in range(in_channels):
        ks = [np.array([]) for _ in range(9)]
        for in_ch in range(out_channels):
            w9 = get_weight_vec(out_ch, in_ch)
            for idx in range(9):
                ks[idx] = np.append(ks[idx], np.repeat(w9[idx], repeat_len))

        if bin_masks is not None:
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], altalena2(np.tile(bin_masks[idx], 2)))

        if A2_scaling is not None:
            scale_val = A2_scaling.detach().cpu().numpy()
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], np.repeat(scale_val, multiply_step))

        if post_k_callback is not None:
            ks = post_k_callback(ks, out_ch)

        for idx, k in enumerate(ks, start=1):
            rolled = altalena2(np.roll(k, roll_step * out_ch))
            rolled2 = altalena2(np.roll(k, roll_step * out_ch - 1))
            np.savetxt(f"{path}/{file_prefix}-ch{out_ch}-k{idx}.bin",
                       rolled, delimiter=",")
            np.savetxt(
                f"{path}/{file_prefix}-ch{out_ch+in_channels}-k{idx}.bin", rolled2, delimiter=",")
    print(file_prefix, "done.")


def _export_conv_block_3x3_2_30(
    *,
    get_weight_vec,
    out_channels,
    in_channels,
    repeat_len,
    roll_step,
    multiply_step,
    path,
    file_prefix,
    bin_masks=None,
    A2_scaling=None,
    post_k_callback=None,
):
    for out_ch in range(in_channels):
        ks = [np.array([]) for _ in range(9)]
        for in_ch in range(out_channels):
            w9 = get_weight_vec(out_ch, in_ch)
            for idx in range(9):
                ks[idx] = np.append(ks[idx], np.repeat(w9[idx], repeat_len))

        if bin_masks is not None:
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], altalena3(np.tile(bin_masks[idx], 2)))

        if A2_scaling is not None:
            scale_val = A2_scaling.detach().cpu().numpy()
            for idx in range(9):
                ks[idx] = np.multiply(
                    ks[idx], np.repeat(scale_val, multiply_step))

        if post_k_callback is not None:
            ks = post_k_callback(ks, out_ch)

        for idx, k in enumerate(ks, start=1):
            rolled = altalena3(np.roll(k, roll_step * out_ch))
            rolled2 = altalena3(np.roll(k, roll_step * out_ch - 1))
            np.savetxt(f"{path}/{file_prefix}-ch{out_ch}-k{idx}.bin",
                       rolled, delimiter=",")
            np.savetxt(
                f"{path}/{file_prefix}-ch{out_ch+in_channels}-k{idx}.bin", rolled2, delimiter=",")
    print(file_prefix, "done.")


def _compute_paf_params(paf_module):
    A2 = paf_module.a2.detach() ** 0.5
    n1 = paf_module.a1.detach() * 0.5 * (A2 ** -1)
    n2 = paf_module.a0.detach() - (paf_module.a1.detach() ** 2) * \
        0.25 * (paf_module.a2.detach() ** -1)
    return A2, n1, n2


def bn_to_affine(bn: torch.nn.BatchNorm2d):
    A = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = -(bn.weight * bn.running_mean /
          torch.sqrt(bn.running_var + bn.eps)) + bn.bias
    return A.detach().cpu().numpy(), b.detach().cpu().numpy()


def export_conv_block(
    *,
    # 1) 权重来源：可以只给 conv，也可以自己传 get_weight_vec 覆盖
    conv=None,
    get_weight_vec=None,

    # 2) 卷积形状和导出参数
    out_channels,
    in_channels,
    repeat_len,
    roll_step,
    multiply_step,
    path,
    file_prefix,

    # 3) mask：可以传 img_width + tile_times 自动算，也可以直接传 bin_masks 复用
    img_width=None,
    tile_times=None,
    bin_masks=None,

    # 4) 缩放：BN / PAF / override
    bn=None,
    paf=None,
    A2_scaling_override=None,

    # 5) 卷积块类型：决定用哪个底层实现
    #   "plain"        -> _export_conv_block_3x3
    #   "pad_in"       -> _export_conv_block_3x3_1
    #   "double_alt0"  -> _export_conv_block_3x3_2
    #   "double_alt20" -> _export_conv_block_3x3_2_20
    #   "double_alt30" -> _export_conv_block_3x3_3_30（如果你有的话）
    block_type="plain",

    # 6) 通道索引模式（自动生成 get_weight_vec，用于 Aespa 那种 in/out shuffle）
    #   "direct"                 -> conv.weight[out_ch][in_ch]
    #   "shift_in_plus_out_mod"  -> conv.weight[in_ch][(in_ch + out_ch) % out_channels]
    weight_pattern="direct",
    # for "shift_in_plus_out_mod": True 表示 weight[in][...], False 表示 weight[out][...]
    weight_swap_in_out=True,

    # 7) 输出后处理模式（自动生成 post_k_callback，用于 roll by out_ch）
    #   "none"           -> 不处理
    #   "roll_by_outch"  -> 对每个 k 做 np.roll(k, post_roll_step * out_ch)
    post_pattern="none",
    post_roll_step=None,

    # 8) 如果你仍然想手动控制后处理，可以传自定义 post_k_callback 覆盖 pattern
    post_k_callback=None,
):

    # =========================
    # 1) 准备 get_weight_vec
    # =========================
    if get_weight_vec is None:
        if conv is None:
            raise ValueError(
                "export_conv_block 需要传 conv 或 get_weight_vec 至少一个")

        # 这里 out_channels 就是传进来的参数，通常 = conv.weight.shape[0]
        C = out_channels

        if weight_pattern == "direct":
            # 标准 conv：weight[out][in]
            def get_weight_vec(out_ch, in_ch):
                w = conv.weight[out_ch][in_ch]
                return w.reshape(9).detach().cpu().numpy()

        elif weight_pattern == "shift_in_plus_out_mod":
            # Aespa：weight[in][(in+out)%C] 或 weight[out][(in+out)%C]
            if weight_swap_in_out:
                # weight[in][(in+out)%C]
                def get_weight_vec(out_ch, in_ch):
                    w = conv.weight[in_ch][(in_ch + out_ch) % C]
                    return w.reshape(9).detach().cpu().numpy()
            else:
                # weight[out][(in+out)%C]
                def get_weight_vec(out_ch, in_ch):
                    w = conv.weight[out_ch][(in_ch + out_ch) % C]
                    return w.reshape(9).detach().cpu().numpy()
        else:
            raise ValueError(f"未知的 weight_pattern: {weight_pattern}")

    # =========================
    # 2) 准备 bin_masks
    # =========================
    if bin_masks is None:
        if img_width is not None and tile_times is not None:
            bin_masks = _build_conv_masks(img_width, tile_times)
        else:
            bin_masks = None

    # =========================
    # 3) 统一计算缩放 & BN / PAF 参数
    # =========================
    bn_affine = None
    paf_params = None
    A2_scaling = None

    # 3.1 PAF：Aespa 模型
    if paf is not None:
        A2, n1, n2 = _compute_paf_params(paf)
        paf_params = (A2, n1, n2)
        A2_scaling = A2  # torch.Tensor

    # 3.2 BN：普通 ResNet
    if bn is not None:
        # bias 导出用 numpy
        A_np, b_np = bn_to_affine(bn)
        bn_affine = (A_np, b_np)
        # 卷积权重缩放用 torch.Tensor
        if A2_scaling is None:
            A2_scaling = bn.weight / torch.sqrt(bn.running_var + bn.eps)

    # 3.3 override 优先级最高
    if A2_scaling_override is not None:
        A2_scaling = A2_scaling_override

    # =========================
    # 4) 准备 post_k_callback
    # =========================
    if post_k_callback is None:
        if post_pattern == "none":
            def post_k_callback(ks, out_ch):
                return ks

        elif post_pattern == "roll_by_outch":
            # 默认每个 out_ch roll repeat_len；也可以外面显式指定
            if post_roll_step is None:
                post_roll_step = repeat_len

            def post_k_callback(ks, out_ch):
                return [np.roll(k, post_roll_step * out_ch) for k in ks]
        else:
            raise ValueError(f"未知的 post_pattern: {post_pattern}")

    # =========================
    # 5) 组装公共参数，调用具体导出实现
    # =========================
    core_kwargs = dict(
        get_weight_vec=get_weight_vec,
        out_channels=out_channels,
        in_channels=in_channels,
        repeat_len=repeat_len,
        roll_step=roll_step,
        multiply_step=multiply_step,
        path=path,
        file_prefix=file_prefix,
        bin_masks=bin_masks,
        A2_scaling=A2_scaling,
        post_k_callback=post_k_callback,
    )

    if block_type == "plain":
        _export_conv_block_3x3(**core_kwargs)
    elif block_type == "pad_in":
        _export_conv_block_3x3_1(**core_kwargs)
    elif block_type == "double_alt0":
        _export_conv_block_3x3_2(**core_kwargs)
    elif block_type == "double_alt20":
        _export_conv_block_3x3_2_20(**core_kwargs)
    elif block_type == "double_alt30":
        _export_conv_block_3x3_2_30(**core_kwargs)
    else:
        raise ValueError(f"未知的 block_type: {block_type}")

    # =========================
    # 6) 返回有用的信息（给外面导出 bias / PAF）
    # =========================
    return {
        "bn_affine": bn_affine,      # (A_np, b_np) 或 None
        "paf_params": paf_params,    # (A2, n1, n2) 或 None
        "A2_scaling": A2_scaling,    # torch.Tensor 或 None
        "bin_masks": bin_masks,      # 实际使用的 mask（如果你想复用）
    }


def build_mask(starting_padding, ending_padding, window_length, max_length):
    mask = []
    for i in range(starting_padding):
        mask.append(0)
    while len(mask) < (max_length - ending_padding):
        for j in range(window_length):
            mask.append(1)
        mask.append(0)

    while len(mask) > max_length:
        mask.pop()
    while len(mask) < max_length:
        mask.append(0)

    for i in range(ending_padding):
        mask[max_length - i - 1] = 0

    return mask


def _build_conv_masks(img_width, tile_times):
    bin_mask1 = np.tile(np.array(build_mask(
        img_width + 1, 0, img_width - 1, img_width ** 2)), tile_times)
    bin_mask2 = np.tile(
        np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), tile_times)
    bin_mask3 = np.tile(
        np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), tile_times)
    bin_mask4 = np.tile(
        np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), tile_times)
    bin_mask5 = np.tile(
        np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), tile_times)
    bin_mask6 = np.tile(
        np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), tile_times)
    bin_mask7 = np.tile(np.array(build_mask(
        1, img_width - 1, img_width - 1, img_width ** 2)), tile_times)
    bin_mask8 = np.tile(
        np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), tile_times)
    bin_mask9 = np.tile(np.array(build_mask(
        0, img_width + 1, img_width - 1, img_width ** 2)), tile_times)
    return [bin_mask1, bin_mask2, bin_mask3, bin_mask4, bin_mask5,
            bin_mask6, bin_mask7, bin_mask8, bin_mask9]
