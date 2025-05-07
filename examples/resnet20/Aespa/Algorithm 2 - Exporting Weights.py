"""
本文件将notebooks中的同名文件转换为python代码并增加中文注释
"""
import torch
import numpy as np
import  os
from HerPN import get_Aespa_MutalChannel_PAF_resnet18,get_Aespa_MutalChannel_PAF_resnet20

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

def generate_resnet20_bin_files():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model.eval()
    img_width = 32
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 16)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 16)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 16)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 16)
    # conv1bn1
    A = model.bn1.weight / torch.sqrt(model.bn1.running_var + model.bn1.eps)
    b = -(model.bn1.weight * model.bn1.running_mean / torch.sqrt(model.bn1.running_var + model.bn1.eps)) + model.bn1.bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(3):
            k1 = np.append(k1, np.repeat(model.conv1.weight[i][j].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.conv1.weight[i][j].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.conv1.weight[i][j].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.conv1.weight[i][j].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.conv1.weight[i][j].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.conv1.weight[i][j].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.conv1.weight[i][j].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.conv1.weight[i][j].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.conv1.weight[i][j].reshape(9)[8].detach(), 1024))

        for j in range(16 - 3):
            k1 = np.append(k1, np.repeat(0, 1024))
            k2 = np.append(k2, np.repeat(0, 1024))
            k3 = np.append(k3, np.repeat(0, 1024))
            k4 = np.append(k4, np.repeat(0, 1024))
            k5 = np.append(k5, np.repeat(0, 1024))
            k6 = np.append(k6, np.repeat(0, 1024))
            k7 = np.append(k7, np.repeat(0, 1024))
            k8 = np.append(k8, np.repeat(0, 1024))
            k9 = np.append(k9, np.repeat(0, 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A[i].detach(), 16384))
        k2 = np.multiply(k2, np.repeat(A[i].detach(), 16384))
        k3 = np.multiply(k3, np.repeat(A[i].detach(), 16384))
        k4 = np.multiply(k4, np.repeat(A[i].detach(), 16384))
        k5 = np.multiply(k5, np.repeat(A[i].detach(), 16384))
        k6 = np.multiply(k6, np.repeat(A[i].detach(), 16384))
        k7 = np.multiply(k7, np.repeat(A[i].detach(), 16384))
        k8 = np.multiply(k8, np.repeat(A[i].detach(), 16384))
        k9 = np.multiply(k9, np.repeat(A[i].detach(), 16384))

        """
        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)
        """

        np.savetxt('weights/conv1bn1-ch{}-k1.bin'.format(i), k1, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k2.bin'.format(i), k2, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k3.bin'.format(i), k3, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k4.bin'.format(i), k4, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k5.bin'.format(i), k5, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k6.bin'.format(i), k6, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k7.bin'.format(i), k7, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k8.bin'.format(i), k8, delimiter=',')
        np.savetxt('weights/conv1bn1-ch{}-k9.bin'.format(i), k9, delimiter=',')

    np.savetxt('weights/conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')

    ## Layer1[0]: Conv1+Bn1
    A = model.layer1[0].bn1.weight / torch.sqrt(model.layer1[0].bn1.running_var + model.layer1[0].bn1.eps)
    b = -(model.layer1[0].bn1.weight * model.layer1[0].bn1.running_mean / torch.sqrt(
        model.layer1[0].bn1.running_var + model.layer1[0].bn1.eps)) + model.layer1[0].bn1.bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        # The binary mask is required to put zeros in values coming from other channels and so on
        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights/layer1-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer1-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer1-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')

    ## Layer1[0]: Conv2+Bn2
    A = model.layer1[0].bn2.weight / torch.sqrt(model.layer1[0].bn2.running_var + model.layer1[0].bn2.eps)
    b = -(model.layer1[0].bn2.weight * model.layer1[0].bn2.running_mean / torch.sqrt(
        model.layer1[0].bn2.running_var + model.layer1[0].bn2.eps)) + model.layer1[0].bn2.bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights/layer1-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer1-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer1-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')

    ## Layer1[1]: Conv1+Bn1
    A = model.layer1[1].bn1.weight / torch.sqrt(model.layer1[1].bn1.running_var + model.layer1[1].bn1.eps)
    b = -(model.layer1[1].bn1.weight * model.layer1[1].bn1.running_mean / torch.sqrt(
        model.layer1[1].bn1.running_var + model.layer1[1].bn1.eps)) + model.layer1[1].bn1.bias
    print("A: {}\n\nb: {}".format(A, b))
    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights/layer2-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer2-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer2-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv2+Bn2

    A = model.layer1[1].bn2.weight / torch.sqrt(model.layer1[1].bn2.running_var + model.layer1[1].bn2.eps)
    b = -(model.layer1[1].bn2.weight * model.layer1[1].bn2.running_mean / torch.sqrt(
        model.layer1[1].bn2.running_var + model.layer1[1].bn2.eps)) + model.layer1[1].bn2.bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights/layer2-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer2-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer2-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')

    ## Layer1[2]: Conv1+Bn1
    A = model.layer1[2].bn1.weight / torch.sqrt(model.layer1[2].bn1.running_var + model.layer1[2].bn1.eps)
    b = -(model.layer1[2].bn1.weight * model.layer1[2].bn1.running_mean / torch.sqrt(
        model.layer1[2].bn1.running_var + model.layer1[2].bn1.eps)) + model.layer1[2].bn1.bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights/layer3-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer3-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer3-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')

    ## Layer1[2]: Conv2+Bn2
    A = model.layer1[2].bn2.weight / torch.sqrt(model.layer1[2].bn2.running_var + model.layer1[2].bn2.eps)
    b = -(model.layer1[2].bn2.weight * model.layer1[2].bn2.running_mean / torch.sqrt(
        model.layer1[2].bn2.running_var + model.layer1[2].bn2.eps)) + model.layer1[2].bn2.bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights/layer3-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer3-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer3-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')


    ## Layer2[0]: Conv1+Bn1 SX DownsamplingFast
    # Dispari sempre 0, riempe i primi 32 mod 64
    # Con singola rotazione
    #
    # A = model.layer2[0].bn1.weight / torch.sqrt(model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)
    # b = -(model.layer2[0].bn1.weight * model.layer2[0].bn1.running_mean / torch.sqrt(
    #     model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)) + model.layer2[0].bn1.bias
    #
    # for i in range(16):
    #     k1 = np.array([])
    #     k2 = np.array([])
    #     k3 = np.array([])
    #     k4 = np.array([])
    #     k5 = np.array([])
    #     k6 = np.array([])
    #     k7 = np.array([])
    #     k8 = np.array([])
    #     k9 = np.array([])
    #
    #     for j in range(32):
    #         k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
    #         k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
    #         k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
    #         k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
    #         k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
    #         k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
    #         k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
    #         k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
    #         k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))
    #
    #     k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
    #     k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
    #     k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
    #     k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
    #     k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
    #     k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
    #     k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
    #     k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
    #     k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))
    #
    #     k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
    #     k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
    #     k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
    #     k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
    #     k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
    #     k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
    #     k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
    #     k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
    #     k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))
    #
    #     # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
    #     k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
    #     k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
    #     k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
    #     k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
    #     k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
    #     k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
    #     k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
    #     k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
    #     k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]
    #
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
    #
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')
    #
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
    #                         np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    #
    # np.savetxt('weights/layer4-conv1bn1-bias1.bin', bias_corrected016, delimiter=',')
    # np.savetxt('weights/layer4-conv1bn1-bias2.bin', bias_corrected1632, delimiter=',')

    # N.b il 1632 deve essere ruotato a DX di uno

    ## Layer2[0]: Conv1+Bn1

    A = model.layer2[0].bn1.weight / torch.sqrt(model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)
    b = -(model.layer2[0].bn1.weight * model.layer2[0].bn1.running_mean / torch.sqrt(
        model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)) + model.layer2[0].bn1.bias
    print("A: {}\n\nb: {}".format(A, b))

    channels = []

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
        k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
        k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
        k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
        k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
        k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
        k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
        k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]

        np.savetxt('weights/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
                            np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))

    np.savetxt('weights/layer4-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights/layer4-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    ## Layer2[0]: Conv1+Bn1 DX

    A = model.layer2[0].downsample[1].weight / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
    b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + model.layer2[0].downsample[1].bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])

        for j in range(32):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 16].reshape(1)[0].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]

        print(k1[0])

        np.savetxt('weights/layer4dx-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights/layer4dx-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))

    np.savetxt('weights/layer4dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights/layer4dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    ## Layer2[0]: Conv2+Bn2

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    A = model.layer2[0].bn2.weight / torch.sqrt(model.layer2[0].bn2.running_var + model.layer2[0].bn2.eps)
    b = -(model.layer2[0].bn2.weight * model.layer2[0].bn2.running_mean / torch.sqrt(
        model.layer2[0].bn2.running_var + model.layer2[0].bn2.eps)) + model.layer2[0].bn2.bias

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights/layer4-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer4-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer4-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')

    ## Layer2[1]: Conv1+Bn1

    A = model.layer2[1].bn1.weight / torch.sqrt(model.layer2[1].bn1.running_var + model.layer2[1].bn1.eps)
    b = -(model.layer2[1].bn1.weight * model.layer2[1].bn1.running_mean / torch.sqrt(
        model.layer2[1].bn1.running_var + model.layer2[1].bn1.eps)) + model.layer2[1].bn1.bias

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights/layer5-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer5-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer5-conv1bn1-bias.bin', np.repeat(b.detach(), 256), delimiter=',')

    ## Layer2[1]: Conv2+Bn2

    A = model.layer2[1].bn2.weight / torch.sqrt(model.layer2[1].bn2.running_var + model.layer2[1].bn2.eps)
    b = -(model.layer2[1].bn2.weight * model.layer2[1].bn2.running_mean / torch.sqrt(
        model.layer2[1].bn2.running_var + model.layer2[1].bn2.eps)) + model.layer2[1].bn2.bias

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights/layer5-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer5-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer5-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')

    ## Layer2[2]: Conv1+Bn1

    A = model.layer2[2].bn1.weight / torch.sqrt(model.layer2[2].bn1.running_var + model.layer2[2].bn1.eps)
    b = -(model.layer2[2].bn1.weight * model.layer2[2].bn1.running_mean / torch.sqrt(
        model.layer2[2].bn1.running_var + model.layer2[2].bn1.eps)) + model.layer2[2].bn1.bias

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights/layer6-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer6-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer6-conv1bn1-bias.bin', np.repeat(b.detach(), 256), delimiter=',')

    ## Layer2[2]: Conv2+Bn2

    A = model.layer2[2].bn2.weight / torch.sqrt(model.layer2[2].bn2.running_var + model.layer2[2].bn2.eps)
    b = -(model.layer2[2].bn2.weight * model.layer2[2].bn2.running_mean / torch.sqrt(
        model.layer2[2].bn2.running_var + model.layer2[2].bn2.eps)) + model.layer2[2].bn2.bias

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights/layer6-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer6-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer6-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')

    ## Layer3[0]: Conv1+Bn1 SX

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)







    A = model.layer3[0].bn1.weight / torch.sqrt(model.layer3[0].bn1.running_var + model.layer3[0].bn1.eps)
    b = -(model.layer3[0].bn1.weight * model.layer3[0].bn1.running_mean / torch.sqrt(
        model.layer3[0].bn1.running_var + model.layer3[0].bn1.eps)) + model.layer3[0].bn1.bias
    print("A: {}\n\nb: {}".format(A, b))

    channels = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena2(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena2(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena2(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena2(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena2(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena2(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena2(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena2(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]
        k2 = np.add(k2, np.roll(k2, -8192 + 1))[:8192]
        k3 = np.add(k3, np.roll(k3, -8192 + 1))[:8192]
        k4 = np.add(k4, np.roll(k4, -8192 + 1))[:8192]
        k5 = np.add(k5, np.roll(k5, -8192 + 1))[:8192]
        k6 = np.add(k6, np.roll(k6, -8192 + 1))[:8192]
        k7 = np.add(k7, np.roll(k7, -8192 + 1))[:8192]
        k8 = np.add(k8, np.roll(k8, -8192 + 1))[:8192]
        k9 = np.add(k9, np.roll(k9, -8192 + 1))[:8192]

        np.savetxt('weights/layer7-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k2.bin'.format(i), altalena2(np.roll(k2, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k2.bin'.format(i + 32), altalena2(np.roll(k2, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k3.bin'.format(i), altalena2(np.roll(k3, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k3.bin'.format(i + 32), altalena2(np.roll(k3, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k4.bin'.format(i), altalena2(np.roll(k4, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k4.bin'.format(i + 32), altalena2(np.roll(k4, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k5.bin'.format(i), altalena2(np.roll(k5, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k5.bin'.format(i + 32), altalena2(np.roll(k5, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k6.bin'.format(i), altalena2(np.roll(k6, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k6.bin'.format(i + 32), altalena2(np.roll(k6, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k7.bin'.format(i), altalena2(np.roll(k7, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k7.bin'.format(i + 32), altalena2(np.roll(k7, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k8.bin'.format(i), altalena2(np.roll(k8, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k8.bin'.format(i + 32), altalena2(np.roll(k8, 256 * i - 1)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k9.bin'.format(i), altalena2(np.roll(k9, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7-conv1bn1-ch{}-k9.bin'.format(i + 32), altalena2(np.roll(k9, 256 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena2(np.repeat(b.detach().numpy(), 256)),
                            np.roll(altalena2(np.repeat(b.detach().numpy(), 256)), -8192 + 1))[:8192]
    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))

    np.savetxt('weights/layer7-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights/layer7-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    ## Layer3[0]: Conv1+Bn1 DX

    A = model.layer3[0].downsample[1].weight / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)
    b = -(model.layer3[0].downsample[1].weight * model.layer3[0].downsample[1].running_mean / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)) + model.layer3[0].downsample[1].bias
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(32):
        k1 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].downsample[0].weight[j][(j + i) % 32].reshape(1)[0].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]

        print(k1[0])

        np.savetxt('weights/layer7dx-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights/layer7dx-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))

    np.savetxt('weights/layer7dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights/layer7dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    img_width = 8
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 64)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 64)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 64)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 64)

    ## Layer3[0]: Conv2+Bn2

    A = model.layer3[0].bn2.weight / torch.sqrt(model.layer3[0].bn2.running_var + model.layer3[0].bn2.eps)
    b = -(model.layer3[0].bn2.weight * model.layer3[0].bn2.running_mean / torch.sqrt(
        model.layer3[0].bn2.running_var + model.layer3[0].bn2.eps)) + model.layer3[0].bn2.bias

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights/layer7-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer7-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer7-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')

    ## Layer3[1]: Conv1+Bn1

    A = model.layer3[1].bn1.weight / torch.sqrt(model.layer3[1].bn1.running_var + model.layer3[1].bn1.eps)
    b = -(model.layer3[1].bn1.weight * model.layer3[1].bn1.running_mean / torch.sqrt(
        model.layer3[1].bn1.running_var + model.layer3[1].bn1.eps)) + model.layer3[1].bn1.bias

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights/layer8-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer8-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer8-conv1bn1-bias.bin', np.repeat(b.detach(), 64), delimiter=',')

    ## Layer3[1]: Conv2+Bn2

    A = model.layer3[1].bn2.weight / torch.sqrt(model.layer3[1].bn2.running_var + model.layer3[1].bn2.eps)
    b = -(model.layer3[1].bn2.weight * model.layer3[1].bn2.running_mean / torch.sqrt(
        model.layer3[1].bn2.running_var + model.layer3[1].bn2.eps)) + model.layer3[1].bn2.bias

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights/layer8-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer8-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer8-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')

    ## Layer3[2]: Conv1+Bn1

    A = model.layer3[2].bn1.weight / torch.sqrt(model.layer3[2].bn1.running_var + model.layer3[2].bn1.eps)
    b = -(model.layer3[2].bn1.weight * model.layer3[2].bn1.running_mean / torch.sqrt(
        model.layer3[2].bn1.running_var + model.layer3[2].bn1.eps)) + model.layer3[2].bn1.bias

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights/layer9-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer9-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer9-conv1bn1-bias.bin', np.repeat(b.detach(), 64), delimiter=',')

    ## Layer3[2]: Conv2+Bn2

    A = model.layer3[2].bn2.weight / torch.sqrt(model.layer3[2].bn2.running_var + model.layer3[2].bn2.eps)
    b = -(model.layer3[2].bn2.weight * model.layer3[2].bn2.running_mean / torch.sqrt(
        model.layer3[2].bn2.running_var + model.layer3[2].bn2.eps)) + model.layer3[2].bn2.bias

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights/layer9-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights/layer9-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights/layer9-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')

    ## Fully Connected Layer

    np.savetxt('weights/fc.bin', model.fc.weight.t().reshape(-1).detach().numpy())


def generate_resnet20_Aespa_bin_files():
    model = get_Aespa_MutalChannel_PAF_resnet20()
    model.eval()
    img_width = 32
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 16)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 16)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 16)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 16)
    # conv1bn1
    # A = model.bn1.weight / torch.sqrt(model.bn1.running_var + model.bn1.eps)
    # b = -(model.bn1.weight * model.bn1.running_mean / torch.sqrt(model.bn1.running_var + model.bn1.eps)) + model.bn1.bias
    # changes：将对应的A和b转换为对应的a1和a0，另外改变存储的路径
    A = model.HerPN1.a1.detach()
    b = model.HerPN1.a0.detach()
    # a2设置为原本的a2/a1^2方便下一步计算
    a2 = model.HerPN1.a2.detach()
    A2  = a2
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(3):
            k1 = np.append(k1, np.repeat(model.conv1.weight[i][j].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.conv1.weight[i][j].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.conv1.weight[i][j].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.conv1.weight[i][j].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.conv1.weight[i][j].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.conv1.weight[i][j].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.conv1.weight[i][j].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.conv1.weight[i][j].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.conv1.weight[i][j].reshape(9)[8].detach(), 1024))

        for j in range(16 - 3):
            k1 = np.append(k1, np.repeat(0, 1024))
            k2 = np.append(k2, np.repeat(0, 1024))
            k3 = np.append(k3, np.repeat(0, 1024))
            k4 = np.append(k4, np.repeat(0, 1024))
            k5 = np.append(k5, np.repeat(0, 1024))
            k6 = np.append(k6, np.repeat(0, 1024))
            k7 = np.append(k7, np.repeat(0, 1024))
            k8 = np.append(k8, np.repeat(0, 1024))
            k9 = np.append(k9, np.repeat(0, 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A[i].detach(), 16384))
        # k2 = np.multiply(k2, np.repeat(A[i].detach(), 16384))
        # k3 = np.multiply(k3, np.repeat(A[i].detach(), 16384))
        # k4 = np.multiply(k4, np.repeat(A[i].detach(), 16384))
        # k5 = np.multiply(k5, np.repeat(A[i].detach(), 16384))
        # k6 = np.multiply(k6, np.repeat(A[i].detach(), 16384))
        # k7 = np.multiply(k7, np.repeat(A[i].detach(), 16384))
        # k8 = np.multiply(k8, np.repeat(A[i].detach(), 16384))
        # k9 = np.multiply(k9, np.repeat(A[i].detach(), 16384))

        """
        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)
        """
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k1.bin'.format(i), k1, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k2.bin'.format(i), k2, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k3.bin'.format(i), k3, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k4.bin'.format(i), k4, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k5.bin'.format(i), k5, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k6.bin'.format(i), k6, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k7.bin'.format(i), k7, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k8.bin'.format(i), k8, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k9.bin'.format(i), k9, delimiter=',')

    np.savetxt('weights_Aespa/conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[0]: Conv1+Bn1
    temp_PAF = model.layer1[0].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        # The binary mask is required to put zeros in values coming from other channels and so on
        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[0]: Conv2+Bn2
    temp_PAF = model.layer1[0].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv1+Bn1
    temp_PAF = model.layer1[1].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))
    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv2+Bn2

    temp_PAF = model.layer1[1].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[2]: Conv1+Bn1
    temp_PAF = model.layer1[2].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer3-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')

    ## Layer1[2]: Conv2+Bn2
    temp_PAF = model.layer1[2].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer3-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')

    ## Layer2[0]: Conv1+Bn1 SX DownsamplingFast
    # Dispari sempre 0, riempe i primi 32 mod 64
    # Con singola rotazione
    #
    # A = model.layer2[0].bn1.weight / torch.sqrt(model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)
    # b = -(model.layer2[0].bn1.weight * model.layer2[0].bn1.running_mean / torch.sqrt(
    #     model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)) + model.layer2[0].bn1.bias
    #
    # for i in range(16):
    #     k1 = np.array([])
    #     k2 = np.array([])
    #     k3 = np.array([])
    #     k4 = np.array([])
    #     k5 = np.array([])
    #     k6 = np.array([])
    #     k7 = np.array([])
    #     k8 = np.array([])
    #     k9 = np.array([])
    #
    #     for j in range(32):
    #         k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
    #         k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
    #         k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
    #         k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
    #         k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
    #         k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
    #         k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
    #         k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
    #         k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))
    #
    #     k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
    #     k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
    #     k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
    #     k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
    #     k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
    #     k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
    #     k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
    #     k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
    #     k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))
    #
    #     k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
    #     k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
    #     k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
    #     k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
    #     k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
    #     k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
    #     k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
    #     k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
    #     k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))
    #
    #     # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
    #     k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
    #     k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
    #     k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
    #     k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
    #     k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
    #     k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
    #     k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
    #     k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
    #     k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]
    #
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
    #
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')
    #
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
    #                         np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    #
    # np.savetxt('weights_Aespa/layer4-conv1bn1-bias1.bin', bias_corrected016, delimiter=',')
    # np.savetxt('weights_Aespa/layer4-conv1bn1-bias2.bin', bias_corrected1632, delimiter=',')

    # N.b il 1632 deve essere ruotato a DX di uno

    ## Layer2[0]: Conv1+Bn1

    temp_PAF = model.layer2[0].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    channels = []

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))

        # k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
        # k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
        # k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
        # k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
        # k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
        # k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
        # k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
        # k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
        # k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
        k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
        k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
        k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
        k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
        k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
        k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
        k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]

        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
                            np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))


    np.savetxt('weights_Aespa/layer4-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    np.savetxt('weights_Aespa/layer4-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 256), delimiter=',')

    ## Layer2[0]: Conv1+Bn1 DX

    A = model.layer2[0].downsample[1].weight / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
    b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + model.layer2[0].downsample[1].bias
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])

        for j in range(32):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 16].reshape(1)[0].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]

        # print(k1[0])

        np.savetxt('weights_Aespa/layer4dx-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4dx-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # Todo:2个版本尝试
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]

    np.savetxt('weights_Aespa/layer4dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer4dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    ## Layer2[0]: Conv2+Bn2

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    temp_PAF = model.layer2[0].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer4-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[1]: Conv1+Bn1

    temp_PAF = model.layer2[1].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv1bn1-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[1]: Conv2+Bn2

    temp_PAF = model.layer2[1].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[2]: Conv1+Bn1

    temp_PAF = model.layer2[2].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv1bn1-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[2]: Conv2+Bn2

    temp_PAF = model.layer2[2].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer3[0]: Conv1+Bn1 SX

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    temp_PAF = model.layer3[0].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    channels = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena2(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena2(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena2(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena2(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena2(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena2(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena2(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena2(np.tile(bin_mask9, 2)))

        # k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))
        # k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 256))
        # k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 256))
        # k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 256))
        # k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 256))
        # k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 256))
        # k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 256))
        # k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 256))
        # k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]
        k2 = np.add(k2, np.roll(k2, -8192 + 1))[:8192]
        k3 = np.add(k3, np.roll(k3, -8192 + 1))[:8192]
        k4 = np.add(k4, np.roll(k4, -8192 + 1))[:8192]
        k5 = np.add(k5, np.roll(k5, -8192 + 1))[:8192]
        k6 = np.add(k6, np.roll(k6, -8192 + 1))[:8192]
        k7 = np.add(k7, np.roll(k7, -8192 + 1))[:8192]
        k8 = np.add(k8, np.roll(k8, -8192 + 1))[:8192]
        k9 = np.add(k9, np.roll(k9, -8192 + 1))[:8192]

        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i), altalena2(np.roll(k2, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i + 32), altalena2(np.roll(k2, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i), altalena2(np.roll(k3, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i + 32), altalena2(np.roll(k3, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i), altalena2(np.roll(k4, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i + 32), altalena2(np.roll(k4, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i), altalena2(np.roll(k5, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i + 32), altalena2(np.roll(k5, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i), altalena2(np.roll(k6, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i + 32), altalena2(np.roll(k6, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i), altalena2(np.roll(k7, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i + 32), altalena2(np.roll(k7, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i), altalena2(np.roll(k8, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i + 32), altalena2(np.roll(k8, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i), altalena2(np.roll(k9, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i + 32), altalena2(np.roll(k9, 256 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena2(np.repeat(b.detach().numpy(), 256)),
                            np.roll(altalena2(np.repeat(b.detach().numpy(), 256)), -8192 + 1))[:8192]
    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))

    np.savetxt('weights_Aespa/layer7-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[0]: Conv1+Bn1 DX

    A = model.layer3[0].downsample[1].weight / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)
    b = -(model.layer3[0].downsample[1].weight * model.layer3[0].downsample[1].running_mean / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)) + model.layer3[0].downsample[1].bias
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(32):
        k1 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].downsample[0].weight[j][(j + i) % 32].reshape(1)[0].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]

        # print(k1[0])

        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))

    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    img_width = 8
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 64)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 64)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 64)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 64)

    ## Layer3[0]: Conv2+Bn2

    temp_PAF = model.layer3[0].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer7-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[1]: Conv1+Bn1

    temp_PAF = model.layer3[1].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv1bn1-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[1]: Conv2+Bn2

    temp_PAF = model.layer3[1].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[2]: Conv1+Bn1

    temp_PAF = model.layer3[2].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer9-conv1bn1-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv1bn1-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv1bn1-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[2]: Conv2+Bn2

    temp_PAF = model.layer3[2].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        # k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        # k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        # k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        # k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        # k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        # k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        # k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        # k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        # k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer9-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv2bn2-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv2bn2-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Fully Connected Layer

    np.savetxt('weights_Aespa/fc.bin', model.fc.weight.t().reshape(-1).detach().numpy())

def generate_resnet20_Aespa_bin_files_reduce_mult():
    model = get_Aespa_MutalChannel_PAF_resnet20()
    model.eval()
    img_width = 32
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 16)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 16)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 16)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 16)
    # conv1bn1
    # A = model.bn1.weight / torch.sqrt(model.bn1.running_var + model.bn1.eps)
    # b = -(model.bn1.weight * model.bn1.running_mean / torch.sqrt(model.bn1.running_var + model.bn1.eps)) + model.bn1.bias
    # changes：将对应的A和b转换为对应的a1和a0，另外改变存储的路径
    A = model.HerPN1.a1.detach()
    b = model.HerPN1.a0.detach()
    # a2设置为原本的a2/a1^2方便下一步计算
    A2 = model.HerPN1.a2.detach() * (A ** -2)

    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(3):
            k1 = np.append(k1, np.repeat(model.conv1.weight[i][j].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.conv1.weight[i][j].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.conv1.weight[i][j].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.conv1.weight[i][j].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.conv1.weight[i][j].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.conv1.weight[i][j].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.conv1.weight[i][j].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.conv1.weight[i][j].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.conv1.weight[i][j].reshape(9)[8].detach(), 1024))

        for j in range(16 - 3):
            k1 = np.append(k1, np.repeat(0, 1024))
            k2 = np.append(k2, np.repeat(0, 1024))
            k3 = np.append(k3, np.repeat(0, 1024))
            k4 = np.append(k4, np.repeat(0, 1024))
            k5 = np.append(k5, np.repeat(0, 1024))
            k6 = np.append(k6, np.repeat(0, 1024))
            k7 = np.append(k7, np.repeat(0, 1024))
            k8 = np.append(k8, np.repeat(0, 1024))
            k9 = np.append(k9, np.repeat(0, 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A[i].detach(), 16384))
        k2 = np.multiply(k2, np.repeat(A[i].detach(), 16384))
        k3 = np.multiply(k3, np.repeat(A[i].detach(), 16384))
        k4 = np.multiply(k4, np.repeat(A[i].detach(), 16384))
        k5 = np.multiply(k5, np.repeat(A[i].detach(), 16384))
        k6 = np.multiply(k6, np.repeat(A[i].detach(), 16384))
        k7 = np.multiply(k7, np.repeat(A[i].detach(), 16384))
        k8 = np.multiply(k8, np.repeat(A[i].detach(), 16384))
        k9 = np.multiply(k9, np.repeat(A[i].detach(), 16384))

        """
        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)
        """
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k1.bin'.format(i), k1, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k2.bin'.format(i), k2, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k3.bin'.format(i), k3, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k4.bin'.format(i), k4, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k5.bin'.format(i), k5, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k6.bin'.format(i), k6, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k7.bin'.format(i), k7, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k8.bin'.format(i), k8, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k9.bin'.format(i), k9, delimiter=',')

    np.savetxt('weights_Aespa/conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[0]: Conv1+Bn1
    temp_PAF = model.layer1[0].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        # The binary mask is required to put zeros in values coming from other channels and so on
        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[0]: Conv2+Bn2
    temp_PAF = model.layer1[0].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv1+Bn1
    temp_PAF = model.layer1[1].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))
    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv2+Bn2

    temp_PAF = model.layer1[1].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')
    ## Layer1[2]: Conv1+Bn1
    temp_PAF = model.layer1[2].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer3-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')

    ## Layer1[2]: Conv2+Bn2
    temp_PAF = model.layer1[2].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer3-conv2bn2-bias.bin'.format(i), np.repeat(b.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-a2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-a1.bin'.format(i), np.repeat(A.detach(), 1024), delimiter=',')

    ## Layer2[0]: Conv1+Bn1 SX DownsamplingFast
    # Dispari sempre 0, riempe i primi 32 mod 64
    # Con singola rotazione
    #
    # A = model.layer2[0].bn1.weight / torch.sqrt(model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)
    # b = -(model.layer2[0].bn1.weight * model.layer2[0].bn1.running_mean / torch.sqrt(
    #     model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)) + model.layer2[0].bn1.bias
    #
    # for i in range(16):
    #     k1 = np.array([])
    #     k2 = np.array([])
    #     k3 = np.array([])
    #     k4 = np.array([])
    #     k5 = np.array([])
    #     k6 = np.array([])
    #     k7 = np.array([])
    #     k8 = np.array([])
    #     k9 = np.array([])
    #
    #     for j in range(32):
    #         k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
    #         k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
    #         k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
    #         k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
    #         k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
    #         k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
    #         k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
    #         k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
    #         k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))
    #
    #     k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
    #     k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
    #     k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
    #     k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
    #     k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
    #     k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
    #     k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
    #     k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
    #     k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))
    #
    #     k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
    #     k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
    #     k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
    #     k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
    #     k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
    #     k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
    #     k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
    #     k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
    #     k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))
    #
    #     # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
    #     k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
    #     k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
    #     k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
    #     k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
    #     k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
    #     k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
    #     k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
    #     k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
    #     k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]
    #
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
    #
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')
    #
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
    #                         np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    #
    # np.savetxt('weights_Aespa/layer4-conv1bn1-bias1.bin', bias_corrected016, delimiter=',')
    # np.savetxt('weights_Aespa/layer4-conv1bn1-bias2.bin', bias_corrected1632, delimiter=',')

    # N.b il 1632 deve essere ruotato a DX di uno

    ## Layer2[0]: Conv1+Bn1

    temp_PAF = model.layer2[0].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    channels = []

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
        k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
        k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
        k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
        k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
        k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
        k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
        k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
        k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
        k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
        k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
        k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
        k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
        k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
        k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
        k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]

        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
                            np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))


    np.savetxt('weights_Aespa/layer4-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    np.savetxt('weights_Aespa/layer4-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 256), delimiter=',')

    ## Layer2[0]: Conv1+Bn1 DX

    A = model.layer2[0].downsample[1].weight / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
    b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + model.layer2[0].downsample[1].bias
    A1 = model.layer2[0].HerPN2.a1.detach()
    A = A * A1
    b = b * A1
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])

        for j in range(32):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 16].reshape(1)[0].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]

        # print(k1[0])

        np.savetxt('weights_Aespa/layer4dx-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4dx-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # Todo:2个版本尝试
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]

    np.savetxt('weights_Aespa/layer4dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer4dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    ## Layer2[0]: Conv2+Bn2

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    temp_PAF = model.layer2[0].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer4-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[1]: Conv1+Bn1

    temp_PAF = model.layer2[1].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv1bn1-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[1]: Conv2+Bn2

    temp_PAF = model.layer2[1].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[2]: Conv1+Bn1

    temp_PAF = model.layer2[2].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv1bn1-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer2[2]: Conv2+Bn2

    temp_PAF = model.layer2[2].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv2bn2-bias.bin', np.repeat(b.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-a2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-a1.bin', np.repeat(A.detach(), 256), delimiter=',')
    ## Layer3[0]: Conv1+Bn1 SX

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    temp_PAF = model.layer3[0].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)

    channels = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena2(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena2(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena2(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena2(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena2(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena2(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena2(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena2(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))
        k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 256))
        k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 256))
        k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 256))
        k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 256))
        k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 256))
        k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 256))
        k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 256))
        k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]
        k2 = np.add(k2, np.roll(k2, -8192 + 1))[:8192]
        k3 = np.add(k3, np.roll(k3, -8192 + 1))[:8192]
        k4 = np.add(k4, np.roll(k4, -8192 + 1))[:8192]
        k5 = np.add(k5, np.roll(k5, -8192 + 1))[:8192]
        k6 = np.add(k6, np.roll(k6, -8192 + 1))[:8192]
        k7 = np.add(k7, np.roll(k7, -8192 + 1))[:8192]
        k8 = np.add(k8, np.roll(k8, -8192 + 1))[:8192]
        k9 = np.add(k9, np.roll(k9, -8192 + 1))[:8192]

        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i), altalena2(np.roll(k2, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i + 32), altalena2(np.roll(k2, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i), altalena2(np.roll(k3, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i + 32), altalena2(np.roll(k3, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i), altalena2(np.roll(k4, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i + 32), altalena2(np.roll(k4, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i), altalena2(np.roll(k5, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i + 32), altalena2(np.roll(k5, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i), altalena2(np.roll(k6, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i + 32), altalena2(np.roll(k6, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i), altalena2(np.roll(k7, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i + 32), altalena2(np.roll(k7, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i), altalena2(np.roll(k8, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i + 32), altalena2(np.roll(k8, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i), altalena2(np.roll(k9, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i + 32), altalena2(np.roll(k9, 256 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena2(np.repeat(b.detach().numpy(), 256)),
                            np.roll(altalena2(np.repeat(b.detach().numpy(), 256)), -8192 + 1))[:8192]
    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))

    np.savetxt('weights_Aespa/layer7-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-bias.bin'.format(i), np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-a2.bin'.format(i), np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-a1.bin'.format(i), np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[0]: Conv1+Bn1 DX

    A = model.layer3[0].downsample[1].weight / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)
    b = -(model.layer3[0].downsample[1].weight * model.layer3[0].downsample[1].running_mean / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)) + model.layer3[0].downsample[1].bias
    A1 = model.layer3[0].HerPN2.a1.detach()
    A = A * A1
    b = b * A1
    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(32):
        k1 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].downsample[0].weight[j][(j + i) % 32].reshape(1)[0].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]

        # print(k1[0])

        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))

    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    img_width = 8
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 64)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 64)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 64)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 64)

    ## Layer3[0]: Conv2+Bn2

    temp_PAF = model.layer3[0].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach() * (A ** -2)

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer7-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[1]: Conv1+Bn1

    temp_PAF = model.layer3[1].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv1bn1-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[1]: Conv2+Bn2

    temp_PAF = model.layer3[1].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[2]: Conv1+Bn1

    temp_PAF = model.layer3[2].HerPN1
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer9-conv1bn1-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv1bn1-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv1bn1-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Layer3[2]: Conv2+Bn2

    temp_PAF = model.layer3[2].HerPN2
    A = temp_PAF.a1.detach()
    b = temp_PAF.a0.detach()
    A2 = temp_PAF.a2.detach()* (A ** -2)

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer9-conv2bn2-bias.bin', np.repeat(b.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv2bn2-a2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv2bn2-a1.bin', np.repeat(A.detach(), 64), delimiter=',')
    ## Fully Connected Layer

    np.savetxt('weights_Aespa/fc.bin', model.fc.weight.t().reshape(-1).detach().numpy())

def generate_resnet20_Aespa_bin_files_complete_square():
    model = get_Aespa_MutalChannel_PAF_resnet20()
    model.eval()
    img_width = 32
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 16)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 16)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 16)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 16)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 16)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 16)
    # conv1bn1
    # 更新权重生成，A2 = sqrt(a2),n1 = a1/(2*sqrt(a2)),n2 = a0 - a1^2/4a2
    temp_PAF = model.HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1,n2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(3):
            k1 = np.append(k1, np.repeat(model.conv1.weight[i][j].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.conv1.weight[i][j].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.conv1.weight[i][j].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.conv1.weight[i][j].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.conv1.weight[i][j].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.conv1.weight[i][j].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.conv1.weight[i][j].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.conv1.weight[i][j].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.conv1.weight[i][j].reshape(9)[8].detach(), 1024))

        for j in range(16 - 3):
            k1 = np.append(k1, np.repeat(0, 1024))
            k2 = np.append(k2, np.repeat(0, 1024))
            k3 = np.append(k3, np.repeat(0, 1024))
            k4 = np.append(k4, np.repeat(0, 1024))
            k5 = np.append(k5, np.repeat(0, 1024))
            k6 = np.append(k6, np.repeat(0, 1024))
            k7 = np.append(k7, np.repeat(0, 1024))
            k8 = np.append(k8, np.repeat(0, 1024))
            k9 = np.append(k9, np.repeat(0, 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2[i].detach(), 16384))
        k2 = np.multiply(k2, np.repeat(A2[i].detach(), 16384))
        k3 = np.multiply(k3, np.repeat(A2[i].detach(), 16384))
        k4 = np.multiply(k4, np.repeat(A2[i].detach(), 16384))
        k5 = np.multiply(k5, np.repeat(A2[i].detach(), 16384))
        k6 = np.multiply(k6, np.repeat(A2[i].detach(), 16384))
        k7 = np.multiply(k7, np.repeat(A2[i].detach(), 16384))
        k8 = np.multiply(k8, np.repeat(A2[i].detach(), 16384))
        k9 = np.multiply(k9, np.repeat(A2[i].detach(), 16384))

        """
        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)
        """
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k1.bin'.format(i), k1, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k2.bin'.format(i), k2, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k3.bin'.format(i), k3, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k4.bin'.format(i), k4, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k5.bin'.format(i), k5, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k6.bin'.format(i), k6, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k7.bin'.format(i), k7, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k8.bin'.format(i), k8, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k9.bin'.format(i), k9, delimiter=',')

    np.savetxt('weights_Aespa/conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    ## Layer1[0]: Conv1+Bn1
    temp_PAF = model.layer1[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        # The binary mask is required to put zeros in values coming from other channels and so on
        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    ## Layer1[0]: Conv2+Bn2
    temp_PAF = model.layer1[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv2bn2-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv1+Bn1
    temp_PAF = model.layer1[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    ## Layer1[1]: Conv2+Bn2

    temp_PAF = model.layer1[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv2bn2-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')
    ## Layer1[2]: Conv1+Bn1
    temp_PAF = model.layer1[2].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer3-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer1[2]: Conv2+Bn2
    temp_PAF = model.layer1[2].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(16):
            k1 = np.append(k1, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer1[2].conv2.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer3-conv2bn2-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer2[0]: Conv1+Bn1 SX DownsamplingFast
    # Dispari sempre 0, riempe i primi 32 mod 64
    # Con singola rotazione
    #
    # A = model.layer2[0].bn1.weight / torch.sqrt(model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)
    # b = -(model.layer2[0].bn1.weight * model.layer2[0].bn1.running_mean / torch.sqrt(
    #     model.layer2[0].bn1.running_var + model.layer2[0].bn1.eps)) + model.layer2[0].bn1.bias
    #
    # for i in range(16):
    #     k1 = np.array([])
    #     k2 = np.array([])
    #     k3 = np.array([])
    #     k4 = np.array([])
    #     k5 = np.array([])
    #     k6 = np.array([])
    #     k7 = np.array([])
    #     k8 = np.array([])
    #     k9 = np.array([])
    #
    #     for j in range(32):
    #         k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
    #         k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
    #         k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
    #         k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
    #         k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
    #         k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
    #         k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
    #         k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
    #         k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))
    #
    #     k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
    #     k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
    #     k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
    #     k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
    #     k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
    #     k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
    #     k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
    #     k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
    #     k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))
    #
    #     k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))
    #     k2 = np.multiply(k2, np.repeat(A.detach().numpy(), 1024))
    #     k3 = np.multiply(k3, np.repeat(A.detach().numpy(), 1024))
    #     k4 = np.multiply(k4, np.repeat(A.detach().numpy(), 1024))
    #     k5 = np.multiply(k5, np.repeat(A.detach().numpy(), 1024))
    #     k6 = np.multiply(k6, np.repeat(A.detach().numpy(), 1024))
    #     k7 = np.multiply(k7, np.repeat(A.detach().numpy(), 1024))
    #     k8 = np.multiply(k8, np.repeat(A.detach().numpy(), 1024))
    #     k9 = np.multiply(k9, np.repeat(A.detach().numpy(), 1024))
    #
    #     # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
    #     k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
    #     k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
    #     k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
    #     k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
    #     k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
    #     k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
    #     k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
    #     k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
    #     k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]
    #
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
    #
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
    #     np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')
    #
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
    #                         np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    #
    # np.savetxt('weights_Aespa/layer4-conv1bn1-bias1.bin', bias_corrected016, delimiter=',')
    # np.savetxt('weights_Aespa/layer4-conv1bn1-bias2.bin', bias_corrected1632, delimiter=',')

    # N.b il 1632 deve essere ruotato a DX di uno

    ## Layer2[0]: Conv1+Bn1

    temp_PAF = model.layer2[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    b = n1

    channels = []

    for i in range(16):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            k1 = np.append(k1, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[0].detach(), 1024))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[1].detach(), 1024))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[2].detach(), 1024))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[3].detach(), 1024))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[4].detach(), 1024))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[5].detach(), 1024))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[6].detach(), 1024))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[7].detach(), 1024))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 16].reshape(9)[8].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A2.detach().numpy(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach().numpy(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach().numpy(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach().numpy(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach().numpy(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach().numpy(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach().numpy(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach().numpy(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
        k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
        k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
        k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
        k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
        k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
        k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
        k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]

        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i + 16), altalena(np.roll(k2, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i + 16), altalena(np.roll(k3, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i + 16), altalena(np.roll(k4, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i + 16), altalena(np.roll(k5, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i + 16), altalena(np.roll(k6, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i + 16), altalena(np.roll(k7, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i + 16), altalena(np.roll(k8, 1024 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i + 16), altalena(np.roll(k9, 1024 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(), 1024)),
                            np.roll(altalena(np.repeat(b.detach().numpy(), 1024)), -16384 + 1))[:16384]
    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))


    np.savetxt('weights_Aespa/layer4-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    np.savetxt('weights_Aespa/layer4-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 256), delimiter=',')

    ## Layer2[0]: Conv1+Bn1 DX

    A = model.layer2[0].downsample[1].weight / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
    b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + model.layer2[0].downsample[1].bias
    A1 = model.layer2[0].HerPN2.a2.detach() ** 0.5
    A = A * A1
    b = b * A1
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(16):
        k1 = np.array([])

        for j in range(32):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 16].reshape(1)[0].detach(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 1024))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]

        # print(k1[0])

        np.savetxt('weights_Aespa/layer4dx-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer4dx-conv1bn1-ch{}-k1.bin'.format(i + 16), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))
    # bias_corrected1632 = altalena(np.roll(np.repeat(b.detach().numpy()[16:32], 1024), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]

    np.savetxt('weights_Aespa/layer4dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer4dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    ## Layer2[0]: Conv2+Bn2

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    temp_PAF = model.layer2[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer4-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    ## Layer2[1]: Conv1+Bn1

    temp_PAF = model.layer2[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv1bn1-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    ## Layer2[1]: Conv2+Bn2

    temp_PAF = model.layer2[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    ## Layer2[2]: Conv1+Bn1

    temp_PAF = model.layer2[2].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv1bn1-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    ## Layer2[2]: Conv2+Bn2

    temp_PAF = model.layer2[2].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(32):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer2[2].conv2.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')
    ## Layer3[0]: Conv1+Bn1 SX

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 32)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 32)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 32)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 32)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 32)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 32)

    temp_PAF = model.layer3[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    channels = []

    for i in range(32):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[0].detach(), 256))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[1].detach(), 256))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[2].detach(), 256))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[3].detach(), 256))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[4].detach(), 256))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[5].detach(), 256))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[6].detach(), 256))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[7].detach(), 256))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 32].reshape(9)[8].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena2(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena2(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena2(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena2(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena2(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena2(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena2(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena2(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A2.detach().numpy(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach().numpy(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach().numpy(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach().numpy(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach().numpy(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach().numpy(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach().numpy(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach().numpy(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]
        k2 = np.add(k2, np.roll(k2, -8192 + 1))[:8192]
        k3 = np.add(k3, np.roll(k3, -8192 + 1))[:8192]
        k4 = np.add(k4, np.roll(k4, -8192 + 1))[:8192]
        k5 = np.add(k5, np.roll(k5, -8192 + 1))[:8192]
        k6 = np.add(k6, np.roll(k6, -8192 + 1))[:8192]
        k7 = np.add(k7, np.roll(k7, -8192 + 1))[:8192]
        k8 = np.add(k8, np.roll(k8, -8192 + 1))[:8192]
        k9 = np.add(k9, np.roll(k9, -8192 + 1))[:8192]

        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i), altalena2(np.roll(k2, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i + 32), altalena2(np.roll(k2, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i), altalena2(np.roll(k3, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i + 32), altalena2(np.roll(k3, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i), altalena2(np.roll(k4, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i + 32), altalena2(np.roll(k4, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i), altalena2(np.roll(k5, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i + 32), altalena2(np.roll(k5, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i), altalena2(np.roll(k6, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i + 32), altalena2(np.roll(k6, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i), altalena2(np.roll(k7, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i + 32), altalena2(np.roll(k7, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i), altalena2(np.roll(k8, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i + 32), altalena2(np.roll(k8, 256 * i - 1)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i), altalena2(np.roll(k9, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i + 32), altalena2(np.roll(k9, 256 * i - 1)), delimiter=',')

    bias_corrected = np.add(altalena2(np.repeat(b.detach().numpy(), 256)),
                            np.roll(altalena2(np.repeat(b.detach().numpy(), 256)), -8192 + 1))[:8192]
    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))

    np.savetxt('weights_Aespa/layer7-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 64), delimiter=',')
    ## Layer3[0]: Conv1+Bn1 DX

    A = model.layer3[0].downsample[1].weight / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)
    b = -(model.layer3[0].downsample[1].weight * model.layer3[0].downsample[1].running_mean / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)) + model.layer3[0].downsample[1].bias
    A1 = model.layer3[0].HerPN2.a2.detach() ** 0.5
    A = A * A1
    b = b * A1

    print("A: {}\n\nb: {}".format(A, b))
    print("A2: {}".format(A2))

    for i in range(32):
        k1 = np.array([])

        for j in range(64):
            k1 = np.append(k1, np.repeat(model.layer3[0].downsample[0].weight[j][(j + i) % 32].reshape(1)[0].detach(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -8192 + 1))[:8192]

        # print(k1[0])

        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i + 32), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena2(np.repeat(b.detach().numpy()[:32], 256))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().numpy()[32:64], 256), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.repeat(b.detach().numpy()[16:32], 1024))

    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    img_width = 8
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 64)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 64)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 64)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 64)

    ## Layer3[0]: Conv2+Bn2

    temp_PAF = model.layer3[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer7-conv2bn2-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    ## Layer3[1]: Conv1+Bn1

    temp_PAF = model.layer3[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv1bn1-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    ## Layer3[1]: Conv2+Bn2

    temp_PAF = model.layer3[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv2bn2-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    ## Layer3[2]: Conv1+Bn1

    temp_PAF = model.layer3[2].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer9-conv1bn1-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv1bn1-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv1bn1-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    ## Layer3[2]: Conv2+Bn2

    temp_PAF = model.layer3[2].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach(), 64))
            k2 = np.append(k2, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach(), 64))
            k3 = np.append(k3, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach(), 64))
            k4 = np.append(k4, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach(), 64))
            k5 = np.append(k5, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach(), 64))
            k6 = np.append(k6, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach(), 64))
            k7 = np.append(k7, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach(), 64))
            k8 = np.append(k8, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach(), 64))
            k9 = np.append(k9, np.repeat(model.layer3[2].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer9-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer9-conv2bn2-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv2bn2-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer9-conv2bn2-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')
    ## Fully Connected Layer

    np.savetxt('weights_Aespa/fc.bin', model.fc.weight.t().reshape(-1).detach().numpy())

def generate_resnet18_Aespa_bin_files_complete_square():
    model = get_Aespa_MutalChannel_PAF_resnet18()
    model.eval()
    print(model)

    img_width = 32
    padding = 1
    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 64)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 64)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 64)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 64)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 64)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 64)

    temp_PAF = model.HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(3):
            k1 = np.append(k1, np.repeat(model.conv1.weight[i][j].reshape(9)[0].detach().cpu().cpu(), 1024))
            k2 = np.append(k2, np.repeat(model.conv1.weight[i][j].reshape(9)[1].detach().cpu().cpu(), 1024))
            k3 = np.append(k3, np.repeat(model.conv1.weight[i][j].reshape(9)[2].detach().cpu().cpu(), 1024))
            k4 = np.append(k4, np.repeat(model.conv1.weight[i][j].reshape(9)[3].detach().cpu().cpu(), 1024))
            k5 = np.append(k5, np.repeat(model.conv1.weight[i][j].reshape(9)[4].detach().cpu().cpu(), 1024))
            k6 = np.append(k6, np.repeat(model.conv1.weight[i][j].reshape(9)[5].detach().cpu().cpu(), 1024))
            k7 = np.append(k7, np.repeat(model.conv1.weight[i][j].reshape(9)[6].detach().cpu().cpu(), 1024))
            k8 = np.append(k8, np.repeat(model.conv1.weight[i][j].reshape(9)[7].detach().cpu().cpu(), 1024))
            k9 = np.append(k9, np.repeat(model.conv1.weight[i][j].reshape(9)[8].detach().cpu().cpu(), 1024))

        for j in range(64 - 3):
            k1 = np.append(k1, np.repeat(0, 1024))
            k2 = np.append(k2, np.repeat(0, 1024))
            k3 = np.append(k3, np.repeat(0, 1024))
            k4 = np.append(k4, np.repeat(0, 1024))
            k5 = np.append(k5, np.repeat(0, 1024))
            k6 = np.append(k6, np.repeat(0, 1024))
            k7 = np.append(k7, np.repeat(0, 1024))
            k8 = np.append(k8, np.repeat(0, 1024))
            k9 = np.append(k9, np.repeat(0, 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2[i].detach(), 65536))
        k2 = np.multiply(k2, np.repeat(A2[i].detach(), 65536))
        k3 = np.multiply(k3, np.repeat(A2[i].detach(), 65536))
        k4 = np.multiply(k4, np.repeat(A2[i].detach(), 65536))
        k5 = np.multiply(k5, np.repeat(A2[i].detach(), 65536))
        k6 = np.multiply(k6, np.repeat(A2[i].detach(), 65536))
        k7 = np.multiply(k7, np.repeat(A2[i].detach(), 65536))
        k8 = np.multiply(k8, np.repeat(A2[i].detach(), 65536))
        k9 = np.multiply(k9, np.repeat(A2[i].detach(), 65536))
        """
        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)
        """

        np.savetxt('weights_Aespa/conv1bn1-ch{}-k1.bin'.format(i), k1, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k2.bin'.format(i), k2, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k3.bin'.format(i), k3, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k4.bin'.format(i), k4, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k5.bin'.format(i), k5, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k6.bin'.format(i), k6, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k7.bin'.format(i), k7, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k8.bin'.format(i), k8, delimiter=',')
        np.savetxt('weights_Aespa/conv1bn1-ch{}-k9.bin'.format(i), k9, delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer1[0]: Conv1+Bn1
    temp_PAF = model.layer1[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach().cpu(), 1024))
            k2 = np.append(k2,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach().cpu(), 1024))
            k3 = np.append(k3,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach().cpu(), 1024))
            k4 = np.append(k4,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach().cpu(), 1024))
            k5 = np.append(k5,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach().cpu(), 1024))
            k6 = np.append(k6,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach().cpu(), 1024))
            k7 = np.append(k7,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach().cpu(), 1024))
            k8 = np.append(k8,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach().cpu(), 1024))
            k9 = np.append(k9,
                           np.repeat(model.layer1[0].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach().cpu(), 1024))

        # The binary mask is required to put zeros in values coming from other channels and so on
        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer1[0]: Conv2+Bn2
    temp_PAF = model.layer1[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach().cpu(), 1024))
            k2 = np.append(k2,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach().cpu(), 1024))
            k3 = np.append(k3,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach().cpu(), 1024))
            k4 = np.append(k4,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach().cpu(), 1024))
            k5 = np.append(k5,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach().cpu(), 1024))
            k6 = np.append(k6,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach().cpu(), 1024))
            k7 = np.append(k7,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach().cpu(), 1024))
            k8 = np.append(k8,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach().cpu(), 1024))
            k9 = np.append(k9,
                           np.repeat(model.layer1[0].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach().cpu(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer1-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer1-conv2bn2-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer1-conv2bn2-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer1[1]: Conv1+Bn1
    temp_PAF = model.layer1[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach().cpu(), 1024))
            k2 = np.append(k2,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach().cpu(), 1024))
            k3 = np.append(k3,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach().cpu(), 1024))
            k4 = np.append(k4,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach().cpu(), 1024))
            k5 = np.append(k5,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach().cpu(), 1024))
            k6 = np.append(k6,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach().cpu(), 1024))
            k7 = np.append(k7,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach().cpu(), 1024))
            k8 = np.append(k8,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach().cpu(), 1024))
            k9 = np.append(k9,
                           np.repeat(model.layer1[1].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach().cpu(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer1[1]: Conv2+Bn2

    temp_PAF = model.layer1[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(64):
            k1 = np.append(k1,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[0].detach().cpu(), 1024))
            k2 = np.append(k2,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[1].detach().cpu(), 1024))
            k3 = np.append(k3,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[2].detach().cpu(), 1024))
            k4 = np.append(k4,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[3].detach().cpu(), 1024))
            k5 = np.append(k5,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[4].detach().cpu(), 1024))
            k6 = np.append(k6,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[5].detach().cpu(), 1024))
            k7 = np.append(k7,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[6].detach().cpu(), 1024))
            k8 = np.append(k8,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[7].detach().cpu(), 1024))
            k9 = np.append(k9,
                           np.repeat(model.layer1[1].conv2.weight[j][(j + i) % 64].reshape(9)[8].detach().cpu(), 1024))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 1024))

        mul1 = np.roll(k1, 1024 * i)
        mul2 = np.roll(k2, 1024 * i)
        mul3 = np.roll(k3, 1024 * i)
        mul4 = np.roll(k4, 1024 * i)
        mul5 = np.roll(k5, 1024 * i)
        mul6 = np.roll(k6, 1024 * i)
        mul7 = np.roll(k7, 1024 * i)
        mul8 = np.roll(k8, 1024 * i)
        mul9 = np.roll(k9, 1024 * i)

        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer2-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer2-conv2bn2-n1.bin'.format(i), np.repeat(n1.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-n2.bin'.format(i), np.repeat(n2.detach(), 1024), delimiter=',')
    np.savetxt('weights_Aespa/layer2-conv2bn2-A2.bin'.format(i), np.repeat(A2.detach(), 1024), delimiter=',')

    ## Layer2[1]: Conv1+Bn1
    temp_PAF = model.layer2[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    b = n1

    for i in range(64):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(128):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[0].detach().cpu(), 1024))
            k2 = np.append(k2,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[1].detach().cpu(), 1024))
            k3 = np.append(k3,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[2].detach().cpu(), 1024))
            k4 = np.append(k4,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[3].detach().cpu(), 1024))
            k5 = np.append(k5,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[4].detach().cpu(), 1024))
            k6 = np.append(k6,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[5].detach().cpu(), 1024))
            k7 = np.append(k7,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[6].detach().cpu(), 1024))
            k8 = np.append(k8,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[7].detach().cpu(), 1024))
            k9 = np.append(k9,
                           np.repeat(model.layer2[0].conv1.weight[j][(j + i) % 64].reshape(9)[8].detach().cpu(), 1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu().numpy(), 1024))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu().numpy(), 1024))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu().numpy(), 1024))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu().numpy(), 1024))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu().numpy(), 1024))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu().numpy(), 1024))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu().numpy(), 1024))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu().numpy(), 1024))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu().numpy(), 1024))

        # Affianco CH1[0]-CH64[0]-CH1[1]-CH64[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -65536 + 1))[:65536]
        k2 = np.add(k2, np.roll(k2, -65536 + 1))[:65536]
        k3 = np.add(k3, np.roll(k3, -65536 + 1))[:65536]
        k4 = np.add(k4, np.roll(k4, -65536 + 1))[:65536]
        k5 = np.add(k5, np.roll(k5, -65536 + 1))[:65536]
        k6 = np.add(k6, np.roll(k6, -65536 + 1))[:65536]
        k7 = np.add(k7, np.roll(k7, -65536 + 1))[:65536]
        k8 = np.add(k8, np.roll(k8, -65536 + 1))[:65536]
        k9 = np.add(k9, np.roll(k9, -65536 + 1))[:65536]

        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k2.bin'.format(i), altalena(np.roll(k2, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k3.bin'.format(i), altalena(np.roll(k3, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k4.bin'.format(i), altalena(np.roll(k4, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k5.bin'.format(i), altalena(np.roll(k5, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k6.bin'.format(i), altalena(np.roll(k6, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k7.bin'.format(i), altalena(np.roll(k7, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k8.bin'.format(i), altalena(np.roll(k8, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k9.bin'.format(i), altalena(np.roll(k9, 1024 * i)), delimiter=',')

        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k1.bin'.format(i + 64), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k2.bin'.format(i + 64), altalena(np.roll(k2, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k3.bin'.format(i + 64), altalena(np.roll(k3, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k4.bin'.format(i + 64), altalena(np.roll(k4, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k5.bin'.format(i + 64), altalena(np.roll(k5, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k6.bin'.format(i + 64), altalena(np.roll(k6, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k7.bin'.format(i + 64), altalena(np.roll(k7, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k8.bin'.format(i + 64), altalena(np.roll(k8, 1024 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv1bn1-ch{}-k9.bin'.format(i + 64), altalena(np.roll(k9, 1024 * i - 1)),
                   delimiter=',')

    bias_corrected = np.add(altalena(np.repeat(b.detach().cpu().numpy(), 1024)),
                            np.roll(altalena(np.repeat(b.detach().cpu().numpy(), 1024)), -65536 + 1))[:65536]
    bias_corrected064 = altalena(np.repeat(b.detach().cpu().numpy()[:64], 1024))
    bias_corrected64128 = altalena(np.roll(np.repeat(b.detach().cpu().numpy()[64:128], 1024), -1))

    np.savetxt('weights_Aespa/layer3-conv1bn1-bias1.bin', bias_corrected064, delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-bias2.bin', bias_corrected64128, delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 256), delimiter=',')

    A = model.layer2[0].downsample[1].weight / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
    b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
        model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + model.layer2[0].downsample[
            1].bias
    A1 = model.layer2[0].HerPN2.a2.detach() ** 0.5
    A = A * A1
    b = b * A1
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(64):
        k1 = np.array([])

        for j in range(128):
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].downsample[0].weight[j][(j + i) % 64].reshape(1)[0].detach().cpu(),
                                     1024))

        k1 = np.multiply(k1, altalena(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().cpu().numpy(), 1024))

        # Affianco CH1[0]-CH64[0]-CH1[1]-CH64[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -65536 + 1))[:65536]

        print(k1[0])

        np.savetxt('weights_Aespa/layer3dx-conv1bn1-ch{}-k1.bin'.format(i), altalena(np.roll(k1, 1024 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer3dx-conv1bn1-ch{}-k1.bin'.format(i + 64), altalena(np.roll(k1, 1024 * i - 1)),
                   delimiter=',')

    bias_corrected064 = altalena(np.repeat(b.detach().cpu().numpy()[:64], 1024))
    bias_corrected64128 = altalena(np.repeat(b.detach().cpu().numpy()[64:128], 1024))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().cpu().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().cpu().numpy(),1024)), -65536 + 1))[:65536]
    # bias_corrected064 = altalena(np.repeat(b.detach().cpu().numpy()[:64], 1024))
    # bias_corrected64128 = altalena(np.repeat(b.detach().cpu().numpy()[64:128], 1024))

    np.savetxt('weights_Aespa/layer3dx-conv1bn1-bias1.bin'.format(i), bias_corrected064, delimiter=',')
    np.savetxt('weights_Aespa/layer3dx-conv1bn1-bias2.bin'.format(i), bias_corrected64128, delimiter=',')

    img_width = 16
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 128)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 128)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 128)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 128)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 128)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 128)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 128)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 128)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 128)

    temp_PAF = model.layer2[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(128):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(128):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[0].detach().cpu(), 256))
            k2 = np.append(k2,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[1].detach().cpu(), 256))
            k3 = np.append(k3,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[2].detach().cpu(), 256))
            k4 = np.append(k4,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[3].detach().cpu(), 256))
            k5 = np.append(k5,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[4].detach().cpu(), 256))
            k6 = np.append(k6,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[5].detach().cpu(), 256))
            k7 = np.append(k7,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[6].detach().cpu(), 256))
            k8 = np.append(k8,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[7].detach().cpu(), 256))
            k9 = np.append(k9,
                           np.repeat(model.layer2[0].conv2.weight[j][(j + i) % 128].reshape(9)[8].detach().cpu(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer3-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer3-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')

    temp_PAF = model.layer2[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(128):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(128):
            # Qua moltiplico np.repeat blabla per A2[j]
            k1 = np.append(k1,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[0].detach().cpu(), 256))
            k2 = np.append(k2,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[1].detach().cpu(), 256))
            k3 = np.append(k3,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[2].detach().cpu(), 256))
            k4 = np.append(k4,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[3].detach().cpu(), 256))
            k5 = np.append(k5,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[4].detach().cpu(), 256))
            k6 = np.append(k6,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[5].detach().cpu(), 256))
            k7 = np.append(k7,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[6].detach().cpu(), 256))
            k8 = np.append(k8,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[7].detach().cpu(), 256))
            k9 = np.append(k9,
                           np.repeat(model.layer2[1].conv1.weight[j][(j + i) % 128].reshape(9)[8].detach().cpu(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv1bn1-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')

    ## Layer2[1]: Conv2+Bn2
    temp_PAF = model.layer2[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    for i in range(128):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(128):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[0].detach().cpu(), 256))
            k2 = np.append(k2,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[1].detach().cpu(), 256))
            k3 = np.append(k3,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[2].detach().cpu(), 256))
            k4 = np.append(k4,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[3].detach().cpu(), 256))
            k5 = np.append(k5,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[4].detach().cpu(), 256))
            k6 = np.append(k6,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[5].detach().cpu(), 256))
            k7 = np.append(k7,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[6].detach().cpu(), 256))
            k8 = np.append(k8,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[7].detach().cpu(), 256))
            k9 = np.append(k9,
                           np.repeat(model.layer2[1].conv2.weight[j][(j + i) % 128].reshape(9)[8].detach().cpu(), 256))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 256))

        mul1 = np.roll(k1, 256 * i)
        mul2 = np.roll(k2, 256 * i)
        mul3 = np.roll(k3, 256 * i)
        mul4 = np.roll(k4, 256 * i)
        mul5 = np.roll(k5, 256 * i)
        mul6 = np.roll(k6, 256 * i)
        mul7 = np.roll(k7, 256 * i)
        mul8 = np.roll(k8, 256 * i)
        mul9 = np.roll(k9, 256 * i)

        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer4-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer4-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')

    temp_PAF = model.layer3[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    channels = []

    for i in range(128):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(256):
            k1 = np.append(k1,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[0].detach().cpu(), 256))
            k2 = np.append(k2,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[1].detach().cpu(), 256))
            k3 = np.append(k3,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[2].detach().cpu(), 256))
            k4 = np.append(k4,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[3].detach().cpu(), 256))
            k5 = np.append(k5,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[4].detach().cpu(), 256))
            k6 = np.append(k6,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[5].detach().cpu(), 256))
            k7 = np.append(k7,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[6].detach().cpu(), 256))
            k8 = np.append(k8,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[7].detach().cpu(), 256))
            k9 = np.append(k9,
                           np.repeat(model.layer3[0].conv1.weight[j][(j + i) % 128].reshape(9)[8].detach().cpu(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena2(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena2(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena2(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena2(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena2(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena2(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena2(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena2(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu().numpy(), 256))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu().numpy(), 256))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu().numpy(), 256))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu().numpy(), 256))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu().numpy(), 256))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu().numpy(), 256))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu().numpy(), 256))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu().numpy(), 256))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -32768 + 1))[:32768]
        k2 = np.add(k2, np.roll(k2, -32768 + 1))[:32768]
        k3 = np.add(k3, np.roll(k3, -32768 + 1))[:32768]
        k4 = np.add(k4, np.roll(k4, -32768 + 1))[:32768]
        k5 = np.add(k5, np.roll(k5, -32768 + 1))[:32768]
        k6 = np.add(k6, np.roll(k6, -32768 + 1))[:32768]
        k7 = np.add(k7, np.roll(k7, -32768 + 1))[:32768]
        k8 = np.add(k8, np.roll(k8, -32768 + 1))[:32768]
        k9 = np.add(k9, np.roll(k9, -32768 + 1))[:32768]

        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k1.bin'.format(i + 128), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k2.bin'.format(i), altalena2(np.roll(k2, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k2.bin'.format(i + 128), altalena2(np.roll(k2, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k3.bin'.format(i), altalena2(np.roll(k3, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k3.bin'.format(i + 128), altalena2(np.roll(k3, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k4.bin'.format(i), altalena2(np.roll(k4, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k4.bin'.format(i + 128), altalena2(np.roll(k4, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k5.bin'.format(i), altalena2(np.roll(k5, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k5.bin'.format(i + 128), altalena2(np.roll(k5, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k6.bin'.format(i), altalena2(np.roll(k6, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k6.bin'.format(i + 128), altalena2(np.roll(k6, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k7.bin'.format(i), altalena2(np.roll(k7, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k7.bin'.format(i + 128), altalena2(np.roll(k7, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k8.bin'.format(i), altalena2(np.roll(k8, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k8.bin'.format(i + 128), altalena2(np.roll(k8, 256 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k9.bin'.format(i), altalena2(np.roll(k9, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv1bn1-ch{}-k9.bin'.format(i + 128), altalena2(np.roll(k9, 256 * i - 1)),
                   delimiter=',')

    bias_corrected = np.add(altalena2(np.repeat(b.detach().cpu().numpy(), 256)),
                            np.roll(altalena2(np.repeat(b.detach().cpu().numpy(), 256)), -32768 + 1))[:32768]
    bias_corrected016 = altalena2(np.repeat(b.detach().cpu().numpy()[:128], 256))
    bias_corrected16128 = altalena2(np.roll(np.repeat(b.detach().cpu().numpy()[128:256], 256), -1))

    np.savetxt('weights_Aespa/layer5-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-bias2.bin'.format(i), bias_corrected16128, delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 64), delimiter=',')

    A = model.layer3[0].downsample[1].weight / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)
    b = -(model.layer3[0].downsample[1].weight * model.layer3[0].downsample[1].running_mean / torch.sqrt(
        model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)) + model.layer3[0].downsample[
            1].bias
    A1 = model.layer3[0].HerPN2.a2.detach() ** 0.5
    A = A * A1
    b = b * A1
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(128):
        k1 = np.array([])

        for j in range(256):
            k1 = np.append(k1, np.repeat(
                model.layer3[0].downsample[0].weight[j][(j + i) % 128].reshape(1)[0].detach().cpu(), 256))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().cpu().numpy(), 256))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -32768 + 1))[:32768]

        print(k1[0])

        np.savetxt('weights_Aespa/layer5dx-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 256 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer5dx-conv1bn1-ch{}-k1.bin'.format(i + 128), altalena2(np.roll(k1, 256 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena2(np.repeat(b.detach().cpu().numpy()[:128], 256))
    bias_corrected16128 = altalena2(np.roll(np.repeat(b.detach().cpu().numpy()[128:256], 256), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().cpu().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().cpu().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().cpu().numpy()[:16], 1024))
    # bias_corrected16128 = altalena(np.repeat(b.detach().cpu().numpy()[16:128], 1024))

    np.savetxt('weights_Aespa/layer5dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer5dx-conv1bn1-bias2.bin'.format(i), bias_corrected16128, delimiter=',')

    img_width = 8
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 256)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 256)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 256)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 256)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 256)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 256)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 256)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 256)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 256)

    temp_PAF = model.layer3[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(256):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(256):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[0].detach().cpu(), 64))
            k2 = np.append(k2,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[1].detach().cpu(), 64))
            k3 = np.append(k3,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[2].detach().cpu(), 64))
            k4 = np.append(k4,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[3].detach().cpu(), 64))
            k5 = np.append(k5,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[4].detach().cpu(), 64))
            k6 = np.append(k6,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[5].detach().cpu(), 64))
            k7 = np.append(k7,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[6].detach().cpu(), 64))
            k8 = np.append(k8,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[7].detach().cpu(), 64))
            k9 = np.append(k9,
                           np.repeat(model.layer3[0].conv2.weight[j][(j + i) % 256].reshape(9)[8].detach().cpu(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer5-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer5-conv2bn2-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer5-conv2bn2-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')

    temp_PAF = model.layer3[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(256):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(256):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[0].detach().cpu(), 64))
            k2 = np.append(k2,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[1].detach().cpu(), 64))
            k3 = np.append(k3,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[2].detach().cpu(), 64))
            k4 = np.append(k4,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[3].detach().cpu(), 64))
            k5 = np.append(k5,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[4].detach().cpu(), 64))
            k6 = np.append(k6,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[5].detach().cpu(), 64))
            k7 = np.append(k7,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[6].detach().cpu(), 64))
            k8 = np.append(k8,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[7].detach().cpu(), 64))
            k9 = np.append(k9,
                           np.repeat(model.layer3[1].conv1.weight[j][(j + i) % 256].reshape(9)[8].detach().cpu(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv1bn1-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv1bn1-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')

    temp_PAF = model.layer3[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(256):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(256):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[0].detach().cpu(), 64))
            k2 = np.append(k2,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[1].detach().cpu(), 64))
            k3 = np.append(k3,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[2].detach().cpu(), 64))
            k4 = np.append(k4,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[3].detach().cpu(), 64))
            k5 = np.append(k5,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[4].detach().cpu(), 64))
            k6 = np.append(k6,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[5].detach().cpu(), 64))
            k7 = np.append(k7,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[6].detach().cpu(), 64))
            k8 = np.append(k8,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[7].detach().cpu(), 64))
            k9 = np.append(k9,
                           np.repeat(model.layer3[1].conv2.weight[j][(j + i) % 256].reshape(9)[8].detach().cpu(), 64))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 64))

        mul1 = np.roll(k1, 64 * i)
        mul2 = np.roll(k2, 64 * i)
        mul3 = np.roll(k3, 64 * i)
        mul4 = np.roll(k4, 64 * i)
        mul5 = np.roll(k5, 64 * i)
        mul6 = np.roll(k6, 64 * i)
        mul7 = np.roll(k7, 64 * i)
        mul8 = np.roll(k8, 64 * i)
        mul9 = np.roll(k9, 64 * i)

        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer6-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer6-conv2bn2-n1.bin', np.repeat(n1.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-n2.bin', np.repeat(n2.detach(), 64), delimiter=',')
    np.savetxt('weights_Aespa/layer6-conv2bn2-A2.bin', np.repeat(A2.detach(), 64), delimiter=',')

    temp_PAF = model.layer4[0].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))
    channels = []

    for i in range(256):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(512):
            k1 = np.append(k1,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[0].detach().cpu(), 64))
            k2 = np.append(k2,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[1].detach().cpu(), 64))
            k3 = np.append(k3,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[2].detach().cpu(), 64))
            k4 = np.append(k4,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[3].detach().cpu(), 64))
            k5 = np.append(k5,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[4].detach().cpu(), 64))
            k6 = np.append(k6,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[5].detach().cpu(), 64))
            k7 = np.append(k7,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[6].detach().cpu(), 64))
            k8 = np.append(k8,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[7].detach().cpu(), 64))
            k9 = np.append(k9,
                           np.repeat(model.layer4[0].conv1.weight[j][(j + i) % 256].reshape(9)[8].detach().cpu(), 64))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask1, 2)))
        k2 = np.multiply(k2, altalena2(np.tile(bin_mask2, 2)))
        k3 = np.multiply(k3, altalena2(np.tile(bin_mask3, 2)))
        k4 = np.multiply(k4, altalena2(np.tile(bin_mask4, 2)))
        k5 = np.multiply(k5, altalena2(np.tile(bin_mask5, 2)))
        k6 = np.multiply(k6, altalena2(np.tile(bin_mask6, 2)))
        k7 = np.multiply(k7, altalena2(np.tile(bin_mask7, 2)))
        k8 = np.multiply(k8, altalena2(np.tile(bin_mask8, 2)))
        k9 = np.multiply(k9, altalena2(np.tile(bin_mask9, 2)))

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu().numpy(), 64))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu().numpy(), 64))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu().numpy(), 64))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu().numpy(), 64))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu().numpy(), 64))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu().numpy(), 64))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu().numpy(), 64))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu().numpy(), 64))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu().numpy(), 64))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]
        k2 = np.add(k2, np.roll(k2, -16384 + 1))[:16384]
        k3 = np.add(k3, np.roll(k3, -16384 + 1))[:16384]
        k4 = np.add(k4, np.roll(k4, -16384 + 1))[:16384]
        k5 = np.add(k5, np.roll(k5, -16384 + 1))[:16384]
        k6 = np.add(k6, np.roll(k6, -16384 + 1))[:16384]
        k7 = np.add(k7, np.roll(k7, -16384 + 1))[:16384]
        k8 = np.add(k8, np.roll(k8, -16384 + 1))[:16384]
        k9 = np.add(k9, np.roll(k9, -16384 + 1))[:16384]

        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k1.bin'.format(i + 256), altalena2(np.roll(k1, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i), altalena2(np.roll(k2, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k2.bin'.format(i + 256), altalena2(np.roll(k2, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i), altalena2(np.roll(k3, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k3.bin'.format(i + 256), altalena2(np.roll(k3, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i), altalena2(np.roll(k4, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k4.bin'.format(i + 256), altalena2(np.roll(k4, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i), altalena2(np.roll(k5, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k5.bin'.format(i + 256), altalena2(np.roll(k5, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i), altalena2(np.roll(k6, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k6.bin'.format(i + 256), altalena2(np.roll(k6, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i), altalena2(np.roll(k7, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k7.bin'.format(i + 256), altalena2(np.roll(k7, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i), altalena2(np.roll(k8, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k8.bin'.format(i + 256), altalena2(np.roll(k8, 64 * i - 1)),
                   delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i), altalena2(np.roll(k9, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv1bn1-ch{}-k9.bin'.format(i + 256), altalena2(np.roll(k9, 64 * i - 1)),
                   delimiter=',')

    bias_corrected = np.add(altalena2(np.repeat(b.detach().cpu().numpy(), 64)),
                            np.roll(altalena2(np.repeat(b.detach().cpu().numpy(), 64)), -16384 + 1))[:16384]
    bias_corrected016 = altalena2(np.repeat(b.detach().cpu().numpy()[:256], 64))
    bias_corrected16256 = altalena2(np.roll(np.repeat(b.detach().cpu().numpy()[256:512], 64), -1))

    np.savetxt('weights_Aespa/layer7-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-bias2.bin'.format(i), bias_corrected16256, delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-n1.bin'.format(i), np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-n2.bin'.format(i), np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv1bn1-A2.bin'.format(i), np.repeat(A2.detach(), 256), delimiter=',')

    A = model.layer4[0].downsample[1].weight / torch.sqrt(
        model.layer4[0].downsample[1].running_var + model.layer4[0].downsample[1].eps)
    b = -(model.layer4[0].downsample[1].weight * model.layer4[0].downsample[1].running_mean / torch.sqrt(
        model.layer4[0].downsample[1].running_var + model.layer4[0].downsample[1].eps)) + model.layer4[0].downsample[
            1].bias
    A1 = model.layer4[0].HerPN2.a2.detach() ** 0.5
    A = A * A1
    b = b * A1
    print("A: {}\n\nb: {}".format(A, b))

    for i in range(256):
        k1 = np.array([])

        for j in range(512):
            k1 = np.append(k1, np.repeat(
                model.layer4[0].downsample[0].weight[j][(j + i) % 256].reshape(1)[0].detach().cpu(), 64))

        k1 = np.multiply(k1, altalena2(np.tile(bin_mask5, 2)))

        k1 = np.multiply(k1, np.repeat(A.detach().cpu().numpy(), 64))

        # Affianco CH1[0]-CH16[0]-CH1[1]-CH16[1]-CH1[2]...
        k1 = np.add(k1, np.roll(k1, -16384 + 1))[:16384]

        print(k1[0])

        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i), altalena2(np.roll(k1, 64 * i)), delimiter=',')
        np.savetxt('weights_Aespa/layer7dx-conv1bn1-ch{}-k1.bin'.format(i + 256), altalena2(np.roll(k1, 64 * i - 1)),
                   delimiter=',')

    bias_corrected016 = altalena2(np.repeat(b.detach().cpu().numpy()[:256], 64))
    bias_corrected1632 = altalena2(np.roll(np.repeat(b.detach().cpu().numpy()[256:512], 64), -1))
    # bias_corrected = np.add(altalena(np.repeat(b.detach().cpu().numpy(),1024)), np.roll(altalena(np.repeat(b.detach().cpu().numpy(),1024)), -16384 + 1))[:16384]
    # bias_corrected016 = altalena(np.repeat(b.detach().cpu().numpy()[:16], 1024))
    # bias_corrected1632 = altalena(np.repeat(b.detach().cpu().numpy()[16:32], 1024))

    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias1.bin'.format(i), bias_corrected016, delimiter=',')
    np.savetxt('weights_Aespa/layer7dx-conv1bn1-bias2.bin'.format(i), bias_corrected1632, delimiter=',')

    img_width = 4
    padding = 1

    bin_mask1 = np.tile(np.array(build_mask(img_width + 1, 0, img_width - 1, img_width ** 2)), 512)
    bin_mask2 = np.tile(np.array(build_mask(img_width, 0, img_width ** 2, img_width ** 2)), 512)
    bin_mask3 = np.tile(np.array(build_mask(img_width, 0, img_width - 1, img_width ** 2)), 512)
    bin_mask4 = np.tile(np.array(build_mask(1, 0, img_width - 1, img_width ** 2)), 512)
    bin_mask5 = np.tile(np.array(build_mask(0, 0, img_width ** 2, img_width ** 2)), 512)
    bin_mask6 = np.tile(np.array(build_mask(0, 1, img_width - 1, img_width ** 2)), 512)
    bin_mask7 = np.tile(np.array(build_mask(1, img_width - 1, img_width - 1, img_width ** 2)), 512)
    bin_mask8 = np.tile(np.array(build_mask(0, img_width, img_width ** 2, img_width ** 2)), 512)
    bin_mask9 = np.tile(np.array(build_mask(0, img_width + 1, img_width - 1, img_width ** 2)), 512)

    temp_PAF = model.layer4[0].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(512):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(512):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[0].detach().cpu(), 16))
            k2 = np.append(k2,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[1].detach().cpu(), 16))
            k3 = np.append(k3,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[2].detach().cpu(), 16))
            k4 = np.append(k4,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[3].detach().cpu(), 16))
            k5 = np.append(k5,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[4].detach().cpu(), 16))
            k6 = np.append(k6,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[5].detach().cpu(), 16))
            k7 = np.append(k7,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[6].detach().cpu(), 16))
            k8 = np.append(k8,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[7].detach().cpu(), 16))
            k9 = np.append(k9,
                           np.repeat(model.layer4[0].conv2.weight[j][(j + i) % 512].reshape(9)[8].detach().cpu(), 16))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 16))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 16))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 16))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 16))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 16))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 16))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 16))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 16))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 16))

        mul1 = np.roll(k1, 16 * i)
        mul2 = np.roll(k2, 16 * i)
        mul3 = np.roll(k3, 16 * i)
        mul4 = np.roll(k4, 16 * i)
        mul5 = np.roll(k5, 16 * i)
        mul6 = np.roll(k6, 16 * i)
        mul7 = np.roll(k7, 16 * i)
        mul8 = np.roll(k8, 16 * i)
        mul9 = np.roll(k9, 16 * i)

        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer7-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer7-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer7-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')

    temp_PAF = model.layer4[1].HerPN1
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(512):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(512):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[0].detach().cpu(), 16))
            k2 = np.append(k2,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[1].detach().cpu(), 16))
            k3 = np.append(k3,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[2].detach().cpu(), 16))
            k4 = np.append(k4,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[3].detach().cpu(), 16))
            k5 = np.append(k5,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[4].detach().cpu(), 16))
            k6 = np.append(k6,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[5].detach().cpu(), 16))
            k7 = np.append(k7,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[6].detach().cpu(), 16))
            k8 = np.append(k8,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[7].detach().cpu(), 16))
            k9 = np.append(k9,
                           np.repeat(model.layer4[1].conv1.weight[j][(j + i) % 512].reshape(9)[8].detach().cpu(), 16))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 16))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 16))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 16))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 16))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 16))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 16))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 16))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 16))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 16))

        mul1 = np.roll(k1, 16 * i)
        mul2 = np.roll(k2, 16 * i)
        mul3 = np.roll(k3, 16 * i)
        mul4 = np.roll(k4, 16 * i)
        mul5 = np.roll(k5, 16 * i)
        mul6 = np.roll(k6, 16 * i)
        mul7 = np.roll(k7, 16 * i)
        mul8 = np.roll(k8, 16 * i)
        mul9 = np.roll(k9, 16 * i)

        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv1bn1-ch{}-k9.bin'.format(i), mul9, delimiter=',')

    np.savetxt('weights_Aespa/layer8-conv1bn1-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv1bn1-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')

    temp_PAF = model.layer4[1].HerPN2
    A2 = temp_PAF.a2.detach() ** 0.5
    n1 = temp_PAF.a1.detach() * 0.5 * (A2 ** -1)
    n2 = temp_PAF.a0.detach() - (temp_PAF.a1.detach() ** 2) * 0.25 * (temp_PAF.a2.detach() ** -1)
    print("A2: {}\n\nn1: {}\n\nn2: {}".format(A2, n1, n2))

    ks = []

    for i in range(512):
        k1 = np.array([])
        k2 = np.array([])
        k3 = np.array([])
        k4 = np.array([])
        k5 = np.array([])
        k6 = np.array([])
        k7 = np.array([])
        k8 = np.array([])
        k9 = np.array([])

        for j in range(512):
            # Qua moltiplico np.repeat blabla per A[j]
            k1 = np.append(k1,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[0].detach().cpu(), 16))
            k2 = np.append(k2,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[1].detach().cpu(), 16))
            k3 = np.append(k3,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[2].detach().cpu(), 16))
            k4 = np.append(k4,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[3].detach().cpu(), 16))
            k5 = np.append(k5,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[4].detach().cpu(), 16))
            k6 = np.append(k6,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[5].detach().cpu(), 16))
            k7 = np.append(k7,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[6].detach().cpu(), 16))
            k8 = np.append(k8,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[7].detach().cpu(), 16))
            k9 = np.append(k9,
                           np.repeat(model.layer4[1].conv2.weight[j][(j + i) % 512].reshape(9)[8].detach().cpu(), 16))

        k1 = np.multiply(k1, bin_mask1)
        k2 = np.multiply(k2, bin_mask2)
        k3 = np.multiply(k3, bin_mask3)
        k4 = np.multiply(k4, bin_mask4)
        k5 = np.multiply(k5, bin_mask5)
        k6 = np.multiply(k6, bin_mask6)
        k7 = np.multiply(k7, bin_mask7)
        k8 = np.multiply(k8, bin_mask8)
        k9 = np.multiply(k9, bin_mask9)

        k1 = np.multiply(k1, np.repeat(A2.detach().cpu(), 16))
        k2 = np.multiply(k2, np.repeat(A2.detach().cpu(), 16))
        k3 = np.multiply(k3, np.repeat(A2.detach().cpu(), 16))
        k4 = np.multiply(k4, np.repeat(A2.detach().cpu(), 16))
        k5 = np.multiply(k5, np.repeat(A2.detach().cpu(), 16))
        k6 = np.multiply(k6, np.repeat(A2.detach().cpu(), 16))
        k7 = np.multiply(k7, np.repeat(A2.detach().cpu(), 16))
        k8 = np.multiply(k8, np.repeat(A2.detach().cpu(), 16))
        k9 = np.multiply(k9, np.repeat(A2.detach().cpu(), 16))

        mul1 = np.roll(k1, 16 * i)
        mul2 = np.roll(k2, 16 * i)
        mul3 = np.roll(k3, 16 * i)
        mul4 = np.roll(k4, 16 * i)
        mul5 = np.roll(k5, 16 * i)
        mul6 = np.roll(k6, 16 * i)
        mul7 = np.roll(k7, 16 * i)
        mul8 = np.roll(k8, 16 * i)
        mul9 = np.roll(k9, 16 * i)

        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k1.bin'.format(i), mul1, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k2.bin'.format(i), mul2, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k3.bin'.format(i), mul3, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k4.bin'.format(i), mul4, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k5.bin'.format(i), mul5, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k6.bin'.format(i), mul6, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k7.bin'.format(i), mul7, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k8.bin'.format(i), mul8, delimiter=',')
        np.savetxt('weights_Aespa/layer8-conv2bn2-ch{}-k9.bin'.format(i), mul9, delimiter=',')


    np.savetxt('weights_Aespa/layer8-conv2bn2-n1.bin', np.repeat(n1.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-n2.bin', np.repeat(n2.detach(), 256), delimiter=',')
    np.savetxt('weights_Aespa/layer8-conv2bn2-A2.bin', np.repeat(A2.detach(), 256), delimiter=',')

    np.savetxt('weights_Aespa/fc.bin', model.fc.weight.t().reshape(-1).detach().cpu().numpy())
    np.savetxt('weights_Aespa/fc.bin', model.fc.weight.reshape(-1).detach().cpu().numpy())
    np.savetxt('weights_Aespa/bias.bin', model.fc.bias.reshape(-1).detach().cpu().numpy())


if __name__ == '__main__':
    path = './weights_Aespa'
    if not os.path.exists(path):
        os.mkdir(path)
    generate_resnet18_Aespa_bin_files_complete_square()
    print('success')
