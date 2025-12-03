"""
本文件将notebooks中的同名文件转换为python代码并增加中文注释
"""
import torch
import numpy as np
import os
from HerPN import get_Aespa_MutalChannel_PAF_resnet18, get_original_resnet20, get_Aespa_MutalChannel_PAF_resnet20, \
    change_HerPN2d_into_PAF_MutalChannel
from export_conv import export_conv_block, _build_conv_masks, altalena, altalena2, altalena3, bn_to_affine
import os
import shutil


def save_bias_np(bias_np, repeat_len: int, path: str, prefix: str):
    np.savetxt(
        os.path.join(path, f"{prefix}.bin"),
        np.repeat(bias_np, repeat_len),
        delimiter=",",
    )


def save_paf_params(A2, n1, n2, repeat_len: int, path: str, prefix: str):
    n1_np = n1.detach().cpu().numpy()
    n2_np = n2.detach().cpu().numpy()
    A2_np = A2.detach().cpu().numpy()

    save_bias_np(n1_np, repeat_len, path, prefix + "-n1")
    save_bias_np(n2_np, repeat_len, path, prefix + "-n2")
    save_bias_np(A2_np, repeat_len, path, prefix + "-A2")


def generate_resnet20_bin_files(path):
    model = get_original_resnet20()
    model.eval()

    # ---------- conv1 + bn1 ----------
    img_width = 32
    conv_masks_64 = _build_conv_masks(img_width, 16)

    result = export_conv_block(
        conv=model.conv1,
        out_channels=16,
        in_channels=3,
        repeat_len=1024,
        roll_step=16384,
        multiply_step=16384,
        path=path,
        file_prefix="conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.bn1,
        paf=None,
        block_type="pad_in",
    )
    A_np, b_np = result["bn_affine"]
    save_bias_np(b_np, repeat_len=1024, path=path, prefix="conv1bn1-bias")

    # ---------- layer1[0].conv1bn1 ----------
    result = export_conv_block(
        conv=model.layer1[0].conv1,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer1-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer1[0].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer1-conv1bn1-bias.bin",
        np.repeat(b_np, 1024),
        delimiter=",",
    )

    # ---------- layer1[0].conv2bn2 ----------
    result = export_conv_block(
        conv=model.layer1[0].conv2,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer1-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer1[0].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer1-conv2bn2-bias.bin",
        np.repeat(b_np, 1024),
        delimiter=",",
    )

    # ---------- layer1[1].conv1bn1 ----------
    result = export_conv_block(
        conv=model.layer1[1].conv1,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer2-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer1[1].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer2-conv1bn1-bias.bin",
        np.repeat(b_np, 1024),
        delimiter=",",
    )

    # ---------- layer1[1].conv2bn2 ----------
    result = export_conv_block(
        conv=model.layer1[1].conv2,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer2-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer1[1].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer2-conv2bn2-bias.bin",
        np.repeat(b_np, 1024),
        delimiter=",",
    )

    # ---------- layer1[2].conv1bn1 ----------
    result = export_conv_block(
        conv=model.layer1[2].conv1,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer3-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer1[2].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer3-conv1bn1-bias.bin",
        np.repeat(b_np, 1024),
        delimiter=",",
    )

    # ---------- layer1[2].conv2bn2 ----------
    result = export_conv_block(
        conv=model.layer1[2].conv2,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer3-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer1[2].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer3-conv2bn2-bias.bin",
        np.repeat(b_np, 1024),
        delimiter=",",
    )

    # ---------- layer2[0].conv1bn1（downsample, double_alt0） ----------
    def post_l4_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -16384 + 1))[:16384]
            out.append(k2)
        return out

    def get_l4_0_conv1(out_ch, in_ch):
        return model.layer2[0].conv1.weight[in_ch][(in_ch + out_ch) % 16].reshape(9).detach().cpu().numpy()
    result = export_conv_block(
        conv=model.layer2[0].conv1,
        out_channels=32,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer4-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer2[0].bn1,
        paf=None,
        block_type="double_alt0",
        # weight_pattern="shift_in_plus_out_mod",
        post_k_callback=post_l4_0_conv1,
        get_weight_vec=get_l4_0_conv1,
    )
    _, b_np = result["bn_affine"]
    bias_corrected016 = altalena(np.repeat(b_np[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b_np[16:32], 1024))
    np.savetxt(f"{path}/layer4-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=",")
    np.savetxt(f"{path}/layer4-conv1bn1-bias2.bin",
               bias_corrected1632, delimiter=",")

    A_np, b_np = bn_to_affine(model.layer2[0].downsample[1])
    A = torch.from_numpy(A_np)
    b = torch.from_numpy(b_np)
    
    for i in range(16):
        k1 = np.array([])
        for j in range(32):
            k1 = np.append(
                k1,
                np.repeat(
                    model.layer2[0].downsample[0].weight[j][(j + i) % 16]
                    .reshape(1)[0]
                    .detach(),
                    1024,
                ),
            )
        k1 = np.multiply(k1, altalena(np.tile(conv_masks_64[4], 2)))
        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        np.savetxt(
            path + "/layer4dx-conv1bn1-ch{}-k1.bin".format(i),
            altalena(np.roll(k1, 1024 * i)),
            delimiter=",",
        )
        np.savetxt(
            path + "/layer4dx-conv1bn1-ch{}-k1.bin".format(i + 16),
            altalena(np.roll(k1, 1024 * i - 1)),
            delimiter=",",
        )
    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    np.savetxt(
        f"{path}/layer4dx-conv1bn1-bias1.bin", bias_corrected016, delimiter=","
    )
    np.savetxt(
        f"{path}/layer4dx-conv1bn1-bias2.bin", bias_corrected1632, delimiter=","
    )

    # ---------- 后续 stage2 ----------
    img_width = 16
    conv_masks_64 = _build_conv_masks(img_width, 32)

    # layer2[0].conv2bn2
    result = export_conv_block(
        conv=model.layer2[0].conv2,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer4-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer2[0].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer4-conv2bn2-bias.bin",
        np.repeat(b_np, 256),
        delimiter=",",
    )

    # layer2[1].conv1bn1
    result = export_conv_block(
        conv=model.layer2[1].conv1,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer5-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer2[1].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer5-conv1bn1-bias.bin",
        np.repeat(b_np, 256),
        delimiter=",",
    )

    # layer2[1].conv2bn2
    result = export_conv_block(
        conv=model.layer2[1].conv2,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer5-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer2[1].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer5-conv2bn2-bias.bin",
        np.repeat(b_np, 256),
        delimiter=",",
    )

    # layer2[2].conv1bn1
    result = export_conv_block(
        conv=model.layer2[2].conv1,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer6-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer2[2].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer6-conv1bn1-bias.bin",
        np.repeat(b_np, 256),
        delimiter=",",
    )

    # layer2[2].conv2bn2
    result = export_conv_block(
        conv=model.layer2[2].conv2,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer6-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer2[2].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer6-conv2bn2-bias.bin",
        np.repeat(b_np, 256),
        delimiter=",",
    )

    # ---------- stage3 前半 ----------
    img_width = 16
    conv_masks_64 = _build_conv_masks(img_width, 32)

    def get_l7_0_conv1(out_ch, in_ch):
        return model.layer3[0].conv1.weight[in_ch][(in_ch + out_ch) % 32].reshape(9).detach().cpu().numpy()

    def post_l7_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -8192 + 1))[:8192]
            out.append(k2)
        return out

    # layer3[0].conv1bn1（double_alt20）
    result = export_conv_block(
        get_weight_vec=get_l7_0_conv1,
        conv=model.layer3[0].conv1,
        out_channels=64,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer7-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer3[0].bn1,
        paf=None,
        block_type="double_alt20",
        post_k_callback=post_l7_0_conv1
    )
    _, b_np = result["bn_affine"]
    bias_corrected016 = altalena2(np.repeat(b_np[:32], 256))
    bias_corrected1632 = altalena2(
        np.roll(np.repeat(b_np[32:64], 256), -1)
    )
    np.savetxt(
        f"{path}/layer7-conv1bn1-bias1.bin", bias_corrected016, delimiter=","
    )
    np.savetxt(
        f"{path}/layer7-conv1bn1-bias2.bin", bias_corrected1632, delimiter=","
    )
    A_np, b_np = bn_to_affine(model.layer3[0].downsample[1])
    A = torch.from_numpy(A_np)
    b = torch.from_numpy(b_np)
    for i in range(32):
        k1 = np.array([])
        for j in range(64):
            k1 = np.append(
                k1,
                np.repeat(
                    model.layer3[0].downsample[0].weight[j][(j + i) % 32]
                    .reshape(1)[0]
                    .detach(),
                    256,
                ),
            )
        k1 = np.multiply(k1, altalena2(np.tile(conv_masks_64[4], 2)))
        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]
        np.savetxt(
            path + "/layer7dx-conv1bn1-ch{}-k1.bin".format(i),
            altalena2(np.roll(k1, 256 * i)),
            delimiter=",",
        )
        np.savetxt(
            path + "/layer7dx-conv1bn1-ch{}-k1.bin".format(i + 32),
            altalena2(np.roll(k1, 256 * i - 1)),
            delimiter=",",
        )
    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(
        np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1)
    )
    np.savetxt(
        f"{path}/layer7dx-conv1bn1-bias1.bin", bias_corrected016, delimiter=","
    )
    np.savetxt(
        f"{path}/layer7dx-conv1bn1-bias2.bin", bias_corrected1632, delimiter=","
    )

    img_width = 8
    conv_masks_64 = _build_conv_masks(img_width, 64)

    # layer3[0].conv2bn2
    result = export_conv_block(
        conv=model.layer3[0].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer7-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer3[0].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer7-conv2bn2-bias.bin",
        np.repeat(b_np, 64),
        delimiter=",",
    )

    # layer3[1].conv1bn1
    result = export_conv_block(
        conv=model.layer3[1].conv1,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer8-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer3[1].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer8-conv1bn1-bias.bin",
        np.repeat(b_np, 64),
        delimiter=",",
    )

    # layer3[1].conv2bn2
    result = export_conv_block(
        conv=model.layer3[1].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer8-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer3[1].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer8-conv2bn2-bias.bin",
        np.repeat(b_np, 64),
        delimiter=",",
    )

    # layer3[2].conv1bn1
    result = export_conv_block(
        conv=model.layer3[2].conv1,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer9-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer3[2].bn1,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer9-conv1bn1-bias.bin",
        np.repeat(b_np, 64),
        delimiter=",",
    )

    # layer3[2].conv2bn2
    result = export_conv_block(
        conv=model.layer3[2].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer9-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=model.layer3[2].bn2,
        paf=None,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    _, b_np = result["bn_affine"]
    np.savetxt(
        path + "/layer9-conv2bn2-bias.bin",
        np.repeat(b_np, 64),
        delimiter=",",
    )

    # ---------- fc ----------
    np.savetxt(
        path + "/fc.bin",
        model.fc.weight.t().reshape(-1).detach().cpu().numpy(),
    )
    np.savetxt(
        path + "/bias.bin",
        model.fc.bias.reshape(-1).detach().cpu().numpy(),
    )


def generate_resnet20_Aespa_bin_files_complete_square(path):
    model = get_Aespa_MutalChannel_PAF_resnet20()
    model.eval()

    img_width = 32
    conv_masks_64 = _build_conv_masks(img_width, 16)

    def get_conv1(out_ch, in_ch):
        return model.conv1.weight[out_ch][in_ch].reshape(9).detach().cpu().numpy()

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_conv1,
        out_channels=16,
        in_channels=3,
        repeat_len=1024,
        roll_step=16384,
        multiply_step=16384,
        path=path,
        file_prefix="conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.HerPN1,
        block_type="pad_in",
        post_k_callback=None,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024, path=path, prefix="conv1bn1")

    result = export_conv_block(
        conv=model.layer1[0].conv1,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer1-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[0].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer1-conv1bn1")

    result = export_conv_block(
        conv=model.layer1[0].conv2,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer1-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]

    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer1-conv2bn2")
    result = export_conv_block(
        conv=model.layer1[1].conv1,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer2-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]

    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer2-conv1bn1")

    result = export_conv_block(
        conv=model.layer1[1].conv2,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer2-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer2-conv2bn2")

    result = export_conv_block(
        conv=model.layer1[2].conv1,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer3-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[2].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer3-conv1bn1")

    b = n1

    result = export_conv_block(
        conv=model.layer1[2].conv2,
        out_channels=16,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer3-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[2].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer3-conv2bn2")

    def get_l4_0_conv1(out_ch, in_ch):
        return model.layer2[0].conv1.weight[in_ch][(in_ch + out_ch) % 16].reshape(9).detach().cpu().numpy()

    def post_l4_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -16384 + 1))[:16384]
            out.append(k2)
        return out

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_l4_0_conv1,
        out_channels=32,
        in_channels=16,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer4-conv1bn1",

        img_width=img_width,
        bin_masks=conv_masks_64,

        bn=None,
        paf=model.layer2[0].HerPN1,
        block_type="double_alt0",
        post_k_callback=post_l4_0_conv1,
    )

    A2, n1, n2 = result["paf_params"]
    b = n1

    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    np.savetxt(f"{path}/layer4-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=',')
    np.savetxt(f"{path}/layer4-conv1bn1-bias2.bin",
               bias_corrected1632, delimiter=',')

    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer4-conv1bn1")
    A_np, b_np = bn_to_affine(model.layer2[0].downsample[1])
    A1 = model.layer2[0].HerPN2.a2.detach() ** 0.5
    A = torch.from_numpy(A_np) * A1
    b = torch.from_numpy(b_np) * A1
    
    for i in range(16):
        k1 = np.array([])
        for j in range(32):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 16].reshape(1)[0].detach(), 1024))
        k1 = np.multiply(k1, altalena(np.tile(conv_masks_64[4], 2)))
        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        np.savetxt(path + '/layer4dx-conv1bn1-ch{}-k1.bin'.format(i),
                   altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt(path + '/layer4dx-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')
    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    np.savetxt(f"{path}/layer4dx-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=',')
    np.savetxt(f"{path}/layer4dx-conv1bn1-bias2.bin",
               bias_corrected1632, delimiter=',')

    img_width = 16
    conv_masks_64 = _build_conv_masks(img_width, 32)

    result = export_conv_block(
        conv=model.layer2[0].conv2,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer4-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer4-conv2bn2")

    result = export_conv_block(
        conv=model.layer2[1].conv1,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer5-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )

    A2, n1, n2 = result["paf_params"]

    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer5-conv1bn1")

    result = export_conv_block(
        conv=model.layer2[1].conv2,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer5-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer5-conv2bn2")

    result = export_conv_block(
        conv=model.layer2[2].conv1,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer6-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[2].HerPN1,
        block_type="plain",

        # 等价于 conv.weight[in][(in+out) % 32]
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,

        # 等价于 np.roll(k, 256 * out_ch)
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    A2, n1, n2 = result["paf_params"]

    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer6-conv1bn1")

    def get_l6_0_conv2(out_ch, in_ch):
        return model.layer2[2].conv2.weight[in_ch][(in_ch + out_ch) % 32].reshape(9).detach().cpu().numpy()

    def post_l6_0_conv2(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.roll(k, 256 * out_ch)
            out.append(k2)
        return out

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_l6_0_conv2,
        out_channels=32,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer6-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[2].HerPN2,
        block_type="plain",
        post_k_callback=post_l6_0_conv2,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer6-conv2bn2")

    img_width = 16
    conv_masks_64 = _build_conv_masks(img_width, 32)

    def get_l7_0_conv1(out_ch, in_ch):
        return model.layer3[0].conv1.weight[in_ch][(in_ch + out_ch) % 32].reshape(9).detach().cpu().numpy()

    def post_l7_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -8192 + 1))[:8192]
            out.append(k2)
        return out

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_l7_0_conv1,
        out_channels=64,
        in_channels=32,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer7-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[0].HerPN1,
        block_type="double_alt20",
        post_k_callback=post_l7_0_conv1,
    )

    A2, n1, n2 = result["paf_params"]

    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(
        np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))
    np.savetxt(f"{path}/layer7-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=',')
    np.savetxt(f"{path}/layer7-conv1bn1-bias2.bin",
               bias_corrected1632, delimiter=',')
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer7-conv1bn1")
    A_np, b_np = bn_to_affine(model.layer3[0].downsample[1])
    A1 = model.layer3[0].HerPN2.a2.detach() ** 0.5
    A = torch.from_numpy(A_np) * A1
    b = torch.from_numpy(b_np) * A1

    for i in range(32):
        k1 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].downsample[0].weight[j][(
                j + i) % 32].reshape(1)[0].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(conv_masks_64[4], 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]
        np.savetxt(path + '/layer7dx-conv1bn1-ch{}-k1.bin'.format(i),
                   altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt(path + '/layer7dx-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')
    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(
        np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))
    np.savetxt(f"{path}/layer7dx-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=',')
    np.savetxt(f"{path}/layer7dx-conv1bn1-bias2.bin",
               bias_corrected1632, delimiter=',')

    img_width = 8
    conv_masks_64 = _build_conv_masks(img_width, 64)

    result = export_conv_block(
        conv=model.layer3[0].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer7-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer7-conv2bn2")

    result = export_conv_block(
        conv=model.layer3[1].conv1,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer8-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer8-conv1bn1")

    result = export_conv_block(
        conv=model.layer3[1].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer8-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer8-conv2bn2")

    result = export_conv_block(
        conv=model.layer3[2].conv1,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer9-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[2].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer9-conv1bn1")

    result = export_conv_block(
        conv=model.layer3[2].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer9-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[2].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer9-conv2bn2")

    np.savetxt(path + "/fc.bin",
               model.fc.weight.t().reshape(-1).detach().cpu().numpy())
    np.savetxt(path + "/bias.bin",
               model.fc.bias.reshape(-1).detach().cpu().numpy())


def generate_resnet18_Aespa_bin_files_complete_square(path):
    model = get_Aespa_MutalChannel_PAF_resnet18()
    model.eval()

    img_width = 32
    conv_masks_64 = _build_conv_masks(img_width, 64)

    def get_conv1(out_ch, in_ch):
        return model.conv1.weight[out_ch][in_ch].reshape(9).detach().cpu().numpy()

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_conv1,
        out_channels=64,
        in_channels=3,
        repeat_len=1024,
        roll_step=65536,
        multiply_step=65536,
        path=path,
        file_prefix="conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.HerPN1,
        block_type="pad_in",
        post_k_callback=None,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024, path=path, prefix="conv1bn1")

    result = export_conv_block(
        conv=model.layer1[0].conv1,
        out_channels=64,
        in_channels=64,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer1-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[0].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",  # weight[in][(in+out)%64]
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer1-conv1bn1")

    result = export_conv_block(
        conv=model.layer1[0].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer1-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer1-conv2bn2")

    result = export_conv_block(
        conv=model.layer1[1].conv1,
        out_channels=64,
        in_channels=64,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer2-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer2-conv1bn1")

    result = export_conv_block(
        conv=model.layer1[1].conv2,
        out_channels=64,
        in_channels=64,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer2-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer1[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=1024,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=1024,
                    path=path, prefix="layer2-conv2bn2")

    def get_l3_0_conv1(out_ch, in_ch):
        return model.layer2[0].conv1.weight[in_ch][(in_ch + out_ch) % 64].reshape(9).detach().cpu().numpy()

    def post_l3_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -65536 + 1))[:65536]
            out.append(k2)
        return out

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_l3_0_conv1,
        out_channels=128,
        in_channels=64,
        repeat_len=1024,
        roll_step=1024,
        multiply_step=1024,
        path=path,
        file_prefix="layer3-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[0].HerPN1,
        block_type="double_alt0",
        post_k_callback=post_l3_0_conv1,
    )

    A2, n1, n2 = result["paf_params"]
    b = n1

    bias_corrected064 = altalena(
        np.repeat(b.detach().cpu().numpy()[:64], 1024))
    bias_corrected64128 = altalena(
        np.roll(np.repeat(b.detach().cpu().numpy()[64:128], 1024), -1))
    np.savetxt(f"{path}/layer3-conv1bn1-bias1.bin",
               bias_corrected064, delimiter=',')
    np.savetxt(f"{path}/layer3-conv1bn1-bias2.bin",
               bias_corrected64128, delimiter=',')

    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer3-conv1bn1")
    A_np, b_np = bn_to_affine(model.layer2[0].downsample[1])
    A1 = model.layer2[0].HerPN2.a2.detach() ** 0.5
    A = torch.from_numpy(A_np) * A1
    b = torch.from_numpy(b_np) * A1

    for i in range(64):
        k1 = np.array([])
        for j in range(128):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 64].reshape(1)[0].detach().cpu(),
                                     1024))
        k1 = np.multiply(k1, altalena(np.tile(conv_masks_64[4], 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().cpu().numpy(), 1024))
        k1 = np.add(k1, np.roll(k1, -65536 + 1))[:65536]
        np.savetxt(path + '/layer3dx-conv1bn1-ch{}-k1.bin'.format(i),
                   altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt(path + '/layer3dx-conv1bn1-ch{}-k1.bin'.format(i + 64), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')
    bias_corrected064 = altalena(
        np.repeat(b.detach().cpu().numpy()[:64], 1024))
    bias_corrected64128 = altalena(
        np.repeat(b.detach().cpu().numpy()[64:128], 1024))
    np.savetxt(f"{path}/layer3dx-conv1bn1-bias1.bin",
               bias_corrected064, delimiter=',')
    np.savetxt(f"{path}/layer3dx-conv1bn1-bias2.bin",
               bias_corrected64128, delimiter=',')

    img_width = 16
    conv_masks_64 = _build_conv_masks(img_width, 128)

    result = export_conv_block(
        conv=model.layer2[0].conv2,
        out_channels=128,
        in_channels=128,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer3-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )

    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer3-conv2bn2")

    # layer4-conv1bn1
    result = export_conv_block(
        conv=model.layer2[1].conv1,
        out_channels=128,
        in_channels=128,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer4-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer4-conv1bn1")

    # layer4-conv2bn2
    result = export_conv_block(
        conv=model.layer2[1].conv2,
        out_channels=128,
        in_channels=128,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer4-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer2[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=256,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=256,
                    path=path, prefix="layer4-conv2bn2")

    # layer5-conv1bn1（double_alt20，需要自定义 get/post）
    def get_l5_0_conv1(out_ch, in_ch):
        return (
            model.layer3[0].conv1.weight[in_ch][(in_ch + out_ch) % 128]
            .reshape(9)
            .detach()
            .cpu()
            .numpy()
        )

    def post_l5_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -32768 + 1))[:32768]
            out.append(k2)
        return out

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_l5_0_conv1,
        out_channels=256,
        in_channels=128,
        repeat_len=256,
        roll_step=256,
        multiply_step=256,
        path=path,
        file_prefix="layer5-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[0].HerPN1,
        block_type="double_alt20",
        post_k_callback=post_l5_0_conv1,
    )
    A2, n1, n2 = result["paf_params"]

    bias_corrected016 = altalena2(
        np.repeat(b.detach().cpu().numpy()[:128], 256)
    )
    bias_corrected16128 = altalena2(
        np.roll(np.repeat(b.detach().cpu().numpy()[128:256], 256), -1)
    )
    np.savetxt(f"{path}/layer5-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=",")
    np.savetxt(f"{path}/layer5-conv1bn1-bias2.bin",
               bias_corrected16128, delimiter=",")

    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer5-conv1bn1")
    A_np, b_np = bn_to_affine(model.layer3[0].downsample[1])
    A1 = model.layer3[0].HerPN2.a2.detach() ** 0.5
    A = torch.from_numpy(A_np) * A1
    b = torch.from_numpy(b_np) * A1
    for i in range(128):
        k1 = np.array([])
        for j in range(256):
            k1 = np.append(k1, np.repeat(model.layer3[0].downsample[0].weight[j]
                           [(j + i) % 128].reshape(1)[0].detach().cpu(), 256,),)
        k1 = np.multiply(k1, altalena2(np.tile(conv_masks_64[4], 2)))
        k1 = np.multiply(k1, np.repeat(A.detach().cpu().numpy(), 256))
        k1 = np.add(k1, np.roll(k1, -32768 + 1))[:32768]
        np.savetxt(
            path + "/layer5dx-conv1bn1-ch{}-k1.bin".format(i),
            altalena2(np.roll(k1, 256 * i)),
            delimiter=",",
        )
        np.savetxt(
            path + "/layer5dx-conv1bn1-ch{}-k1.bin".format(i + 128),
            altalena2(np.roll(k1, 256 * i - 1)),
            delimiter=",",
        )
    bias_corrected016 = altalena2(
        np.repeat(b.detach().cpu().numpy()[:128], 256))
    bias_corrected16128 = altalena2(
        np.roll(np.repeat(b.detach().cpu().numpy()[128:256], 256), -1)
    )
    np.savetxt(f"{path}/layer5dx-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=",")
    np.savetxt(f"{path}/layer5dx-conv1bn1-bias2.bin",
               bias_corrected16128, delimiter=",")

    img_width = 8
    conv_masks_64 = _build_conv_masks(img_width, 256)

    # layer5-conv2bn2
    result = export_conv_block(
        conv=model.layer3[0].conv2,
        out_channels=256,
        in_channels=256,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer5-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer5-conv2bn2")

    # layer6-conv1bn1
    result = export_conv_block(
        conv=model.layer3[1].conv1,
        out_channels=256,
        in_channels=256,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer6-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer6-conv1bn1")

    # layer6-conv2bn2
    result = export_conv_block(
        conv=model.layer3[1].conv2,
        out_channels=256,
        in_channels=256,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer6-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer3[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=64,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=64,
                    path=path, prefix="layer6-conv2bn2")

    # layer7-conv1bn1（double_alt30，需要自定义 get/post）
    def get_l7_0_conv1(out_ch, in_ch):
        return (
            model.layer4[0].conv1.weight[in_ch][(in_ch + out_ch) % 256]
            .reshape(9)
            .detach()
            .cpu()
            .numpy()
        )

    def post_l7_0_conv1(ks, out_ch):
        out = []
        for k in ks:
            k2 = np.add(k, np.roll(k, -16384 + 1))[:16384]
            out.append(k2)
        return out

    result = export_conv_block(
        conv=None,
        get_weight_vec=get_l7_0_conv1,
        out_channels=512,
        in_channels=256,
        repeat_len=64,
        roll_step=64,
        multiply_step=64,
        path=path,
        file_prefix="layer7-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer4[0].HerPN1,
        block_type="double_alt30",
        post_k_callback=post_l7_0_conv1,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=16,
                    path=path, prefix="layer7-conv1bn1")
    A_np, b_np = bn_to_affine(model.layer4[0].downsample[1])
    A1 = model.layer4[0].HerPN2.a2.detach() ** 0.5
    A = torch.from_numpy(A_np) * A1
    b = torch.from_numpy(b_np) * A1
    for i in range(256):
        k1 = np.array([])
        for j in range(512):
            k1 = np.append(
                k1, np.repeat(model.layer4[0].downsample[0].weight[j][(j + i) % 256].reshape(1)[0].detach().cpu(), 64,),)
        k1 = np.multiply(k1, altalena3(np.tile(conv_masks_64[4], 2)))
        k1 = np.multiply(k1, np.repeat(A.detach().cpu().numpy(), 64))
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        np.savetxt(
            path + "/layer7dx-conv1bn1-ch{}-k1.bin".format(i),
            altalena3(np.roll(k1, 64 * i)),
            delimiter=",",
        )
        np.savetxt(
            path + "/layer7dx-conv1bn1-ch{}-k1.bin".format(i + 256),
            altalena3(np.roll(k1, 64 * i - 1)),
            delimiter=",",
        )
    bias_corrected016 = altalena3(
        np.repeat(b.detach().cpu().numpy()[:256], 64))
    bias_corrected1632 = altalena3(
        np.roll(np.repeat(b.detach().cpu().numpy()[256:512], 64), -1)
    )
    np.savetxt(f"{path}/layer7dx-conv1bn1-bias1.bin",
               bias_corrected016, delimiter=",")
    np.savetxt(f"{path}/layer7dx-conv1bn1-bias2.bin",
               bias_corrected1632, delimiter=",")

    img_width = 4
    conv_masks_64 = _build_conv_masks(img_width, 512)

    # layer7-conv2bn2
    result = export_conv_block(
        conv=model.layer4[0].conv2,
        out_channels=512,
        in_channels=512,
        repeat_len=16,
        roll_step=16,
        multiply_step=16,
        path=path,
        file_prefix="layer7-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer4[0].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=16,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=16,
                    path=path, prefix="layer7-conv2bn2")

    # layer8-conv1bn1
    result = export_conv_block(
        conv=model.layer4[1].conv1,
        out_channels=512,
        in_channels=512,
        repeat_len=16,
        roll_step=16,
        multiply_step=16,
        path=path,
        file_prefix="layer8-conv1bn1",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer4[1].HerPN1,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=16,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=16,
                    path=path, prefix="layer8-conv1bn1")

    # layer8-conv2bn2
    result = export_conv_block(
        conv=model.layer4[1].conv2,
        out_channels=512,
        in_channels=512,
        repeat_len=16,
        roll_step=16,
        multiply_step=16,
        path=path,
        file_prefix="layer8-conv2bn2",
        img_width=img_width,
        bin_masks=conv_masks_64,
        bn=None,
        paf=model.layer4[1].HerPN2,
        block_type="plain",
        weight_pattern="shift_in_plus_out_mod",
        weight_swap_in_out=True,
        post_pattern="roll_by_outch",
        post_roll_step=16,
    )
    A2, n1, n2 = result["paf_params"]
    save_paf_params(A2, n1, n2, repeat_len=16,
                    path=path, prefix="layer8-conv2bn2")

    np.savetxt(
        path + "/fc.bin",
        model.fc.weight.t().reshape(-1).detach().cpu().numpy(),
    )
    np.savetxt(
        path + "/bias.bin",
        model.fc.bias.reshape(-1).detach().cpu().numpy(),
    )


def move_files_by_keyword(source_dir, target_dir, keyword, recursive=False):
    """
    移动包含关键字的文件到目标文件夹

    参数:
        source_dir (str): 源文件夹路径
        target_dir (str): 目标文件夹路径
        keyword (str): 文件名需要包含的关键字
        recursive (bool): 是否递归搜索子目录，默认为False
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    # 文件计数器
    moved_count = 0
    if recursive:
        # 递归搜索所有子目录
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if keyword in file:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                    print(f"已移动: {src_path} → {dst_path}")
    else:
        # 仅搜索当前目录
        for file in os.listdir(source_dir):
            src_path = os.path.join(source_dir, file)
            if os.path.isfile(src_path) and keyword in file:
                dst_path = os.path.join(target_dir, file)

                # 处理文件名冲突
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    new_name = f"{base}_copy{ext}"
                    dst_path = os.path.join(target_dir, new_name)

                shutil.move(src_path, dst_path)
                moved_count += 1
                print(f"已移动: {src_path} → {dst_path}")

    print(f"\n操作完成! 共移动 {moved_count} 个文件")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")


if __name__ == "__main__":
    # 配置参数 - 根据需要修改这些值
    # SOURCE_DIR = "./weights_Aespa"  # 源文件夹路径
    # TARGET_DIR = "./weights_Aespa"  # 目标文件夹路径
    # KEYWORD = "layer7-conv1bn1"  # 文件名包含的关键字
    # RECURSIVE = True  # 是否搜索子目录
    #
    # 执行文件移动
    # move_files_by_keyword(SOURCE_DIR, TARGET_DIR, KEYWORD, RECURSIVE)

    # # ## gen res18 bin files
    # path = '../weights_aespa_18/'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # generate_resnet18_Aespa_bin_files_complete_square(path)
    # print('success')

    # gen res20 aespa bin files // pass
    path = '../weights_Aespa_20'
    if not os.path.exists(path):
        os.mkdir(path)
    generate_resnet20_Aespa_bin_files_complete_square(path)
    print('success')

    # # gen res20 bin files //pass
    # path = '../weights_20/'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # generate_resnet20_bin_files(path)
    # print('success')

    # generate specific bin for res18
    # model = get_Aespa_MutalChannel_PAF_resnet18()
    # model.eval()
    # np.savetxt('../weights_aespa_18/fc.bin', model.fc.weight.t().reshape(-1).detach().numpy())
