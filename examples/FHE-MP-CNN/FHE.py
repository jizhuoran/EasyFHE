import math
import sys,os
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from torch.fhe.ciphertext import Cipher
import torch.fhe as fhe
import numpy as np
import torch
import os
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from examples.utils import approx


def import_parameters_cifar10(layer_num,end_num,linear_weight:np.ndarray,linear_bias:np.ndarray,conv_weight:np.ndarray,bn_bias:np.ndarray,bn_running_mean:np.ndarray,bn_running_var:np.ndarray,bn_weight:np.ndarray):
    if layer_num==20:
        dir="resnet20_new"
    elif layer_num==32:
        dir="resnet32_new"
    elif layer_num==44:
        dir="resnet44_new"
    elif layer_num==56:
        dir="resnet56_new"
    elif layer_num==110:
        dir="resnet110_new"
    num_c=0
    num_b=0
    num_m=0
    num_v=0
    num_w=0
    conv_weight=np.zeros((layer_num-1,27*16),dtype=float)
    bn_bias=np.zeros(layer_num-1,dtype=float)
    bn_running_mean=np.zeros(layer_num-1,dtype=float)
    bn_running_var=np.zeros(layer_num-1,dtype=float)
    bn_weight=np.zeros(layer_num-1,dtype=float)
    fh=3
    fw=3
    ci=0
    co=0
    ci=co=16
    file_path = os.path.join("pretrained_parameters", dir, "conv1_weight.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(fh * fw * ci * co):
        val = float(tokens[i])
        conv_weight[num_c]=np.append(conv_weight[num_c],val)
    num_c += 1
    for j in range (4):
        for k in range(end_num+1):
            if j==1:
                co=16
            elif j==2:
                co=32
            elif j==3:
                co=64
            if(j--1 or (j==2 and k==0)):
                ci=16
            elif ((j==2 and k!=0)or(j==3 and k==0)):
                ci=32
            else:
                ci=64
            file_path = os.path.join("..",  "pretrained_parameters", dir, f"layer{j}_{k}_conv1_weight.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(fh * fw * ci * co):
                val = float(tokens[i])
                conv_weight[num_c] = np.append(conv_weight[num_c], val)
            num_c += 1
            if j==1:
                ci=16
            elif j==2:
                ci=32
            elif j==3:
                ci=64
            file_path = os.path.join("..", "pretrained_parameters", dir, f"layer{j}_{k}_conv2_weight.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(fh * fw * ci * co):
                val = float(tokens[i])
                conv_weight[num_c] = np.append(conv_weight[num_c], val)
            num_c += 1
    ci=16

    file_path = os.path.join("..",  "pretrained_parameters", dir, "bn1_bias.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(ci):
        bn_bias[num_c] = np.append(bn_bias[num_c], float(tokens[i]))
    num_b += 1
    file_path = os.path.join("..",  "pretrained_parameters", dir, "bn1_running_mean.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(ci):
        bn_running_mean[num_c] = np.append(bn_running_mean[num_c], float(tokens[i]))
    num_m += 1
    file_path = os.path.join("..",  "pretrained_parameters", dir, "bn1_running_var.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(ci):
        bn_running_var[num_c] = np.append(bn_running_var[num_c], float(tokens[i]))
    num_v += 1
    file_path = os.path.join("..", "pretrained_parameters", dir, "bn1_weight.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(ci):
        bn_weight[num_c] = np.append(bn_weight[num_c], float(tokens[i]))
    num_w += 1
    for j in range(4):
        if j==1:
            ci=16
        elif j==2:
            ci=32
        elif j==3:
            ci=64
        for k in range(end_num+1):
            base_parts = ["..", "pretrained_parameters", dir]
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn1_bias.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_bias[num_b] = np.append(bn_bias[num_b], float(tokens[i]))
            num_b += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn1_running_mean.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_running_mean[num_m] = np.append(bn_running_mean[num_m], float(tokens[i]))
            num_m += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn1_running_var.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_running_var[num_v] = np.append(bn_running_var[num_v], float(tokens[i]))
            num_v += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn1_weight.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_weight[num_w] = np.append(bn_weight[num_w], float(tokens[i]))
            num_w += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn2_bias.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_bias[num_b] = np.append(bn_bias[num_b], float(tokens[i]))
            num_b += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn2_running_mean.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_running_mean[num_m] = np.append(bn_running_mean[num_m], float(tokens[i]))
            num_m += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn2_running_var.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_running_var[num_v] = np.append(bn_running_var[num_v], float(tokens[i]))
            num_v += 1
            file_path = os.path.join(*base_parts, f"layer{j}_{k}_bn2_weight.txt")
            if not os.path.exists(file_path):
                raise RuntimeError("file is not open")
            with open(file_path, "r") as f:
                tokens = f.read().split()
            for i in range(ci):
                bn_weight[num_w] = np.append(bn_weight[num_w], float(tokens[i]))
            num_w += 1
    base_parts = ["..", "pretrained_parameters", dir]
    file_path = os.path.join(*base_parts, "linear_weight.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(10 * 64):
        linear_weight = np.append(linear_weight, float(tokens[i]))
    file_path = os.path.join(*base_parts, "linear_bias.txt")
    if not os.path.exists(file_path):
        raise RuntimeError("file is not open")
    with open(file_path, "r") as f:
        tokens = f.read().split()
    for i in range(10):
        linear_bias = np.append(linear_bias, float(tokens[i]))


class TensorCipher:
    def __init__(self, k, h, w, c, t,p,logn, cipher:Cipher):
        self.k = k  # gap
        self.h = h  # height
        self.w = w  # width
        self.c = c  # number of channels
        self.t = t  # floor(c / k^2)
        self.p = p  # 2^log2(nt / k^2 hwt)
        self.logn = logn
        self.cipher = cipher

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
def multiplexed_parallel_convolution_seal(openfhe_context,cryptoContext,input:TensorCipher, co, st, fh, fw, data, running_var, constant_weight, epsilon, cipher_pool, end):
    conv_data=[]
    ki=input.k
    hi=input.h
    wi=input.w
    ci=input.c
    ti=input.t
    pi=input.p
    logn=input.logn
    ko=ho=wo=to=po=0
    encode_slots=(1<<15)#Todo:uncertain
    if(st!=0 and st!=2): raise ValueError(f"supported st is only 1 or 2")
    if(len(data)!=fh*fw*ci*co):raise ValueError(f"the size of data vector is not ker x ker x h x h")
    if(is_power_of_two(ki)!=True):raise ValueError(f"ki is not power of two")
    if(len(running_var)!=co or len(constant_weight)!=co):raise ValueError(f"the size of running_var or weight is not correct")
    for num in running_var:
        if(num<pow(10,-16) and num>-pow(10,-16)):raise ValueError(f"the size of running_var is too small. nearly zero.")
    if(st==1):
        ho=hi
        wo=wi
        ko=ki
    elif(st==2):
        if(hi % 2 == 1 or wi % 2 == 1):raise ValueError(f"hi or wi is not even")
        hp=hi/2
        wo=wi/2
        ko=2*ki
    n=1<<logn
    to = (co + ko * ko - 1) / (ko * ko)
    po=pow(math.floor(math.log2(n/(ko*ko*ho*wo*to))),2)
    q = (co + pi - 1) / pi
    if (n % pi != 0):raise ValueError(f"n is not divisible by pi")
    if (n % po != 0):raise ValueError(f"n is not divisible by po")
    if (ki * ki * hi * wi * ti * pi > n):raise ValueError(f"ki^2 hi wi ti pi is larger than n")
    if (ko * ko * ho * wo * to * po > (1 << logn)):raise ValueError(f"ko^2 ho wo to po is larger than n")
    weight=np.zeros((fh, fw, ci, co), dtype=float)
    compact_weight_vec = np.zeros((fh, fw, q, n), dtype=float)
    select_one=np.zeros((co,ko*ho,ko*ho,to),dtype=float)
    select_one_vec=np.zeros((co,1<<logn),dtype=float)
    for i1 in range(fh):
        for i2 in range(fw):
            for j3 in range(ci):
                for j4 in range(co):
                    weight[i1][i2][j3][j4]=data[fh*fw*ci*j4 + fh*fw*j3 + fw*i1 + i2]
    for i1 in range(fh):
        for i2 in range(fw):
            for i9 in range(q):
                for j8 in range(n):
                    j5 = ((j8 % (n / pi)) % (ki * ki * hi * wi)) / (ki * wi)
                    j6 = (j8 % (n / pi)) % (ki * wi)
                    i7 = (j8 % (n / pi)) / (ki * ki * hi * wi)
                    i8 = j8 / (n / pi)
                    if(j8%(n/pi)>=ki*ki*hi*wi*ti or i8+pi*i9>=co or ki*ki*i7+ki*(j5%ki)+j6%ki>=ci or (j6/ki)-(fw-1)/2+i2 < 0 or (j6/ki)-(fw-1)/2+i2 > wi-1 or (j5/ki)-(fh-1)/2+i1 < 0 or (j5/ki)-(fh-1)/2+i1 > hi-1):
                        compact_weight_vec[i1][i2][i9][j8] = 0.0
                    else:
                        compact_weight_vec[i1][i2][i9][j8] = weight[i1][i2][ki * ki * i7 + ki * (j5 % ki) + j6 % ki][i8 + pi * i9]
    for i4 in range(co):
        for v1 in range(ko*ho):
            for v2 in range(ko*wo):
                for u3 in range(to):
                    if ko*ko*u3 + ko*(v1%ko) + v2%ko == j4:
                        select_one[j4][v1][v2][u3] = constant_weight[j4] / math.sqrt(running_var[j4] + epsilon)
                    else: select_one[j4][v1][v2][u3]=0.0

    for j4 in range(co):
        for v1 in range(ko * ho):
            for v2 in range(ko * wo):
                for u3 in range(to):
                    if ko * ko * u3 + ko * (v1 % ko) + v2 % ko == j4:
                        select_one_vec[j4][ko*ko*ho*wo*u3+ko*wo*v1+v2] = select_one[j4][v1][v2][u3]
    ctxt_in=cipher_pool[0]
    ct_zero=cipher_pool[1]
    temp=cipher_pool[2]
    sum=cipher_pool[3]
    total_sum=cipher_pool[4]
    var=cipher_pool[5]
    ctxt_in=input.cipher

    ctxt_rot = [[None for _ in range(fw)] for _ in range(fh)]
    if (fh%2==0 or fw%2==0):raise ValueError(f"fh and fw should be odd")
    for i1 in range (fh):
        for i2 in range (fw):
            if(i1==(fh-1)/2 and i2==(fw-1)/2): ctxt_rot[i1][i2] = ctxt_in
            elif((i1==(fh-1)/2 and i2>(fw-1)/2) or i1>(fh-1)/2):ctxt_rot[i1][i2] =cipher_pool[6+fw*i1+i2-1]
            else:  ctxt_rot[i1][i2] = cipher_pool[6+fw*i1+i2]

    for i1 in range(fh):
        for i2 in range(fw):
            ctxt_rot[i1][i2] = ctxt_in
            ctxt_rot[i1][i2]=fhe.homo_rotate(ctxt_rot[i1][i2],ki*ki*wi*(i1-(fh-1)/2)+ki*(i2-(fw-1)/2),cryptoContext)
    zero=np.zeros((1<<logn),dtype=float)
    #zero=torch.tensor(zero,device="cuda")
    #plain=fhe.encode(zero,1,0,1<<logn,use_gpu_fft=True, cryptoContext=cryptoContext)
    x=torch.tensor(zero,device="cuda")
    ct_zero=openfhe_context.encrypt(x,1,0, 1<<logn,"release")
    for i9 in range(q):
        for i1 in range(fh):
            for i2 in range(fw):
                value = torch.tensor(compact_weight_vec[i1][i2][i9], device="cuda")
                value = fhe.encode(value, 1, 0, n, use_gpu_fft=True, cryptoContext=cryptoContext)
                temp=fhe.homo_mul_pt(ctxt_rot[i1][i2],value,cryptoContext)
                if(i1==0 and i2==0):sum=temp
                else:
                    sum=fhe.homo_add(sum,temp,cryptoContext)
        #Todo:sum=fhe.homo_rescale(sum,1,cryptoContext)
        var=sum
        d=math.log2(ki)
        c=math.log2(ti)
        for x in range(d):
            temp=var
            temp=fhe.homo_rotate(temp,math.pow(2,x),cryptoContext)
            var=fhe.homo_add(var,temp,cryptoContext)
        for x in range(d):
            temp=var
            temp=fhe.homo_rotate(temp,math.pow(2,x)*ki*wi,cryptoContext)
            var=fhe.homo_add(var,temp,cryptoContext)
        if(c==-1):
            sum=ct_zero
            for x in range(ti):
                temp = var
                temp = fhe.homo_rotate(temp, ki * ki * hi * wi*x, cryptoContext)
                sum = fhe.homo_add(sum, temp, cryptoContext)
            var=sum
        else:
            for x in range(c):
                temp=var
                temp = fhe.homo_rotate(temp, math.pow(2, x) *ki* ki * hi * wi, cryptoContext)
                var = fhe.homo_add(var, temp, cryptoContext)
        i8 = 0
        while i8 < pi and pi * i9 + i8 < co:
            i8+=1
            j4=pi*i9+i8
            if(j4>=co):raise ValueError(f"the value of j4 is out of range!")
            temp=var
            temp = fhe.homo_rotate(temp,  (n/pi)*(j4%pi) - j4%ko - (j4/(ko*ko))*ko*ko*ho*wo - ((j4%(ko*ko))/ko)*ko*wo, cryptoContext)
            value = torch.tensor(select_one_vec[j4], device="cuda")
            value = fhe.encode(value, 1, 0, 1<<logn, use_gpu_fft=True, cryptoContext=cryptoContext)
            temp = fhe.homo_mul_pt(temp, value, cryptoContext)
            if(i8==0 and i9==0):
                total_sum=temp
            else:
                total_sum=fhe.homo_add(total_sum,temp,cryptoContext)
    #Todo:total_sum=fhe.homo_rescale(total_sum,1,cryptoContext)
    var=total_sum
    if(end == False):
        sum=ct_zero
        for u6 in range(po):
            temp=var
            temp = fhe.homo_rotate(temp, -u6*(n/po), cryptoContext)
            sum = fhe.homo_add(sum, temp, cryptoContext)
        var=sum
    output=TensorCipher(ko, ho, wo, co, to, po,logn,var)
    return output

def multiplexed_parallel_batch_norm_seal(openfhe_context,cryptoContext,input:TensorCipher, bias, running_mean, running_var, weight, epsilon, B, end):
    ki=input.k
    hi=input.h
    ci=input.c
    wi=input.w
    ti=input.t
    pi=input.p
    logn=input.logn
    ko=ki
    ho=hi
    wo=wi
    co=ci
    to=ti
    po=pi
    if len(bias)!=ci or len(running_mean)!=ci or len(running_var)!=ci or len(weight)!=ci:
        raise ValueError(f"the size of bias, running_mean, running_var, or weight are not correct")
    for num in running_var:
        if(num<pow(10,-16) and num>-pow(10,-16)):raise ValueError(f"the size of running_var is too small. nearly zero.")
    if(hi*wi*ci>(1<<logn)):
        raise ValueError(f"hi*wi*ci should not be larger than n")
    g=np.zeros(1<<logn,dtype=float)
    n=1<<logn
    if n%pi!=0:
        raise ValueError(f"n is not divisible by pi")
    for v4 in range (n):
        v1 = ((v4 % (n / pi)) % (ki * ki * hi * wi)) / (ki * wi)
        v2 = (v4 % (n / pi)) % (ki * wi)
        u3 = (v4 % (n / pi)) / (ki * ki * hi * wi)
        if (ki*ki*u3+ki*(v1%ki)+v2%ki>=ci or v4%(n/pi)>=ki*ki*hi*wi*ti):
            g[v4] = 0.0
        else:
            idx = ki*ki*u3 + ki*(v1%ki) + v2%ki
            g[v4] = (running_mean[idx] * weight[idx] / math.sqrt(running_var[idx] + epsilon) - bias[idx]) / B

    temp=input.cipher
    cipher_g=openfhe_context.encrypt(g,1,0,1<<logn,"release")
    temp=fhe.homo_sub(temp,cipher_g,cryptoContext)
    output=TensorCipher(ko, ho, wo, co, to, po,logn,temp)
    return output

# def ReLu_seal(openfhe_context,cryptoContext,input,comp_no, deg, alpha, tree, scaled_val, scalingfactor,public_key, secret_key, relin_keys, B):
#     ki=input.k
#     hi=input.h
#     ci=input.c
#     wi=input.w
#     ti=input.t
#     pi=input.p
#     logn=input.logn
#     ko=ki
#     ho=hi
#     wo=wi
#     co=ci
#     to=ti
#     po=pi
#     if(hi*wi*ci>(1<<logn)):
#         raise ValueError(f"hi*wi*ci should not be larger than n")
#     temp = input.cipher
#     scale=1.7
#     deg=15
#     temp=homo_relu(temp,B,deg,cryptoContext)#Todo:这里的deg为什么是一个数组
#     output=TensorCipher(ko, ho, wo, co, to, po,logn,temp)
#     return output


def averagepooling_seal_scale(openfhe_context,cryptoContext,input:TensorCipher,B):
    ki=input.k
    hi=input.h
    ci=input.c
    wi=input.w
    ti=input.t
    pi=input.p
    logn=input.logn
    ko=1
    ho=1
    wo=1
    co=ci
    to=ti
    ct=input.cipher
    for x in range (math.log2(wi)):
        temp=ct
        temp=fhe.homo_rotate(temp,math.pow(2,x)*ki,cryptoContext)
        ct=fhe.homo_add(ct,temp,cryptoContext)
    for x in range (math.log2(hi)):
        temp=ct
        temp=fhe.homo_rotate(temp,math.pow(2,x)*ki*ki*wi,cryptoContext)
        ct=fhe.homo_add(ct,temp,cryptoContext)

    select_one=np.zeros(1<<logn,dtype=float)
    zero  = np.zeros(1 << logn, dtype=float)
    for s in range(ki):
        for u in range(ti):
            p=ki*u+s
            temp=ct
            temp=fhe.homo_rotate(temp,-p*ki + ki*ki*hi*wi*u + ki*wi*s,cryptoContext)
            select_one=zero
            for i in range(ki):
                select_one[(ki*u+s)*ki+i]=B/(hi*wi)
            value = torch.tensor(select_one, device="cuda")
            value = fhe.encode(value, 1, 0, 1<<logn, use_gpu_fft=True, cryptoContext=cryptoContext)
            temp = fhe.homo_mul_pt(temp, value, cryptoContext)
            if(u==0 or s==0):
                sum=temp
            else:
                sum=fhe.homo_add(sum, temp, cryptoContext)
    #Todo:sum= fhe.homo_rescale(sum,1,cryptoContext)
    output=TensorCipher(ko,ho,wo,co,to,1,logn,sum)
    return output

def matrix_multiplication_seal(openfhe_context,cryptoContext,input,matrix,bias,q,r):

    ki=input.k
    hi=input.h
    ci=input.c
    wi=input.w
    ti=input.t
    pi=input.p
    logn=input.logn
    ko=ki
    ho=hi
    wo=wi
    co=ci
    to=ti
    po=pi
    if (len(matrix)!=q*r):raise ValueError(f"the size of matrix is not q*r")
    if (len(bias)!=q):raise ValueError(f"the size of bias is not q")
    W=np.zeros((q+r-1,1<<logn),dtype=float)
    b=np.zeros(1<<logn,dtype=float)
    for z in range(q):
        b[z]=bias[z]
    for i in range(q):
        for j in range(r):
            W[i-j+r-1][i]=matrix[i*r+j]
            if(i-j+r-1<0 or i-j+r-1>=q+r-1):
                raise ValueError(f"i-j+r-1 is out of range")
            if(i*r+j<0 or i*r+j>=len(matrix)):
                raise ValueError(f"i*r+j is out of range")
    ct=input.cipher
    for s in range(q+r-1):
        temp=ct
        value = torch.tensor(W[s], device="cuda")
        value = fhe.encode(value, 1, 0, 1 << logn, use_gpu_fft=True, cryptoContext=cryptoContext)
        temp = fhe.homo_mul_pt(temp, value, cryptoContext)
        if s==0:
            sum=temp
        else:
            sum=fhe.homo_add(sum, temp, cryptoContext)
    #Todo:sum=fhe.homo_rescale(sum,1,cryptoContext)
    output=TensorCipher(ko, ho, wo, co, to, po,logn,sum)
    return output

def multiplexed_parallel_downsampling_seal(openfhe_context,cryptoContext,input):
    ki=input.k
    hi=input.h
    ci=input.c
    wi=input.w
    ti=input.t
    pi=input.p
    logn=input.logn
    ko=0
    ho=0
    wo=0
    co=0
    to=0
    po=0
    n=1<<logn
    ko=2*ki
    ho=hi/2
    wo=wi/2
    to=ti/2
    co=2*ci
    ratio=n / (ko * ko * ho * wo * to)
    po=pow(2,math.floor(math.log(ratio,2)))
    select_one_vec=np.zeros((ki,ti,1<<logn),dtype=float)
    ct=input.cipher
    for w1 in range(ki):
        for w2 in range(ti):
            for v4 in range(1<<logn):
                j5 = (v4 % (ki * ki * hi * wi)) / (ki * wi)
                j6 = v4 % (ki * wi)
                i7 = v4 / (ki * ki * hi * wi)
                if v4<ki*ki*hi*wi*ti and (j5/ki)%2 == 0 and (j6/ki)%2 == 0 and (j5%ki) == w1 and i7 == w2:
                    select_one_vec[w1][w2][v4] = 1.0
                else :
                    select_one_vec[w1][w2][v4] = 0.0
    for w1 in range(wi):
        for w2 in range(ti):
            temp=ct
            value = torch.tensor(select_one_vec[w1][w2], device="cuda")
            value = fhe.encode(value, 1, 0, 1 << logn, use_gpu_fft=True, cryptoContext=cryptoContext)
            temp = fhe.homo_mul_pt(temp, value, cryptoContext)
            w3 = ((ki * w2 + w1) % (2 * ko)) / 2
            w4 = (ki * w2 + w1) % 2
            w5 = (ki * w2 + w1) / (2 * ko)
            temp=fhe.homo_rotate(temp,ki*ki*hi*wi*w2 + ki*wi*w1 - ko*ko*ho*wo*w5 - ko*wo*w3 - ki*w4 - ko*ko*ho*wo*(ti/8),cryptoContext)
            if w1==0 and w2==0:
                sum=temp
            else:
                sum=fhe.homo_add(sum,temp,cryptoContext)
    #	evaluator.rescale_to_next_inplace(sum);		// added
    ct=sum
    sum=ct
    u6=1
    while u6<po:
        u6+=1
        temp=ct
        temp=fhe.homo_rotate(temp,-(n/po)*u6,cryptoContext)
        sum = fhe.homo_add(sum, temp, cryptoContext)
    ct=sum
    output=TensorCipher(ko, ho, wo, co, to, po,logn,ct)
    return output


def multiplexed_parallel_downsampling_seal_print(openfhe_context,cryptoContext,input):
    output=multiplexed_parallel_downsampling_seal(openfhe_context,cryptoContext,input)
    return output

def multiplexed_parallel_convolution_print(openfhe_context,cryptoContext,input,co,st,fh,fw,data,running_var,constant_weight,epsilon,cipher_pool,end):
    output=multiplexed_parallel_convolution_seal(openfhe_context,cryptoContext,input, co, st, fh, fw, data, running_var, constant_weight, epsilon,  cipher_pool, end)
    return output

def multiplexed_parallel_batch_norm_seal_print(openfhe_context,cryptoContext,input,bias,running_mean,running_var,weight,epsilon,B,end):
    output=multiplexed_parallel_batch_norm_seal(openfhe_context,cryptoContext,input, bias, running_mean, running_var, weight, epsilon, B, end)
    return output

# def approx_ReLU_seal_print(openfhe_context,cryptoContext,input,comp_no,deg,alpha,tree,scaled_val, scalingfactor,public_key,secret_key,relin_keys,B):
#     output=ReLu_seal(openfhe_context,cryptoContext,input,comp_no, deg, alpha, tree, scaled_val, scalingfactor,public_key, secret_key, relin_keys, B)
#     return output

def averagepooling_seal_scale_print(openfhe_context,cryptoContext,input,B):
    output=averagepooling_seal_scale(openfhe_context,cryptoContext,input,B)
    return output

def fully_connected_seal_print(openfhe_context,cryptoContext,input,matrix,bias,q,r):
    output=matrix_multiplication_seal(openfhe_context,cryptoContext,input,matrix,bias,q,r)
    return output


def ResNet_cifar10_seal_sparse(layer_num,start_image_id,end_image_id):
    B=40.0
    alpha=13
    comp_no=3
    scaled_val=1.7
    boundary_K = 25
    boot_deg = 59
    scale_factor = 2
    inverse_deg = 1
    logN = 16
    loge = 10
    logn = 15
    logn_1 = 14
    logn_2 = 13
    logn_3 = 12
    logp = 46
    logq = 51
    rotation_kinds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
		,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
		,62,63,64,66,84,124,128,132,256,512,959,960,990,991,1008,1023,1024,1036,1064,1092,1952,1982,1983,2016,2044,2047,2048,2072,2078,2100,3007,3024,3040,3052,3070,3071,3072,3080,3108,4031
		,4032,4062,4063,4095,4096,5023,5024,5054,5055,5087,5118,5119,5120,6047,6078,6079,6111,6112,6142,6143,6144,7071,7102,7103,7135
		,7166,7167,7168,8095,8126,8127,8159,8190,8191,8192,9149,9183,9184,9213,9215,9216,10173,10207,10208,10237,10239,10240,11197,11231
		,11232,11261,11263,11264,12221,12255,12256,12285,12287,12288,13214,13216,13246,13278,13279,13280,13310,13311,13312,14238,14240
		,14270,14302,14303,14304,14334,14335,15262,15264,15294,15326,15327,15328,15358,15359,15360,16286,16288,16318,16350,16351,16352
		,16382,16383,16384,17311,17375,18335,18399,18432,19359,19423,20383,20447,20480,21405,21406,21437,21469,21470,21471,21501,21504
		,22429,22430,22461,22493,22494,22495,22525,22528,23453,23454,23485,23517,23518,23519,23549,24477,24478,24509,24541,24542,24543
		,24573,24576,25501,25565,25568,25600,26525,26589,26592,26624,27549,27613,27616,27648,28573,28637,28640,28672,29600,29632,29664
		,29696,30624,30656,30688,30720,31648,31680,31712,31743,31744,31774,32636,32640,32644,32672,32702,32704,32706,32735
		,32736,32737,32759,32760,32761,32762,32763,32764,32765,32766,32767]
    log_special_prime = 51
    log_integer_part = logq - logp - loge + 5
    remaining_level = 16
    boot_level = 14
    total_level = remaining_level + boot_level
    logBsSlots_list = [14,13,12]
    levelBudget_list = [[3, 3], [3, 3],[3,3]]
    rescaleTech = "FLEXIBLEAUTO"
    save_dir = "/data/yky/data"
    mode = "release"
    # coeff_bit_vec = np.array([], dtype=int)
    # coeff_bit_vec = np.append(coeff_bit_vec, logq)
    # for i in range(remaining_level):
    #     coeff_bit_vec = np.append(coeff_bit_vec, logp)
    # for i in range(boot_level):
    #     coeff_bit_vec = np.append(coeff_bit_vec, logq)
    # coeff_bit_vec = np.append(coeff_bit_vec, log_special_prime)
    dnum=3
    print("start")
    cryptoContext, openfhe_context = (
        fhe.try_load_context(remaining_level, rotation_kinds, logBsSlots_list, logN, dnum, logp, logq,
                             levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir="/data/yky/data/new_resnet",
                             autoLoadAndSetConfig=True, mode=mode))#Todo:参数对齐？
    print("hello")
    end_num=0
    if layer_num==20:end_num=2
    elif layer_num==32:end_num=4
    elif layer_num==44:end_num=6
    elif layer_num==56:end_num=8
    elif layer_num==110:end_num=17
    image_id=start_image_id
    while image_id<=end_image_id:
        image_id=image_id+1
        if layer_num==20:output = open(f"../result/resnet20_cifar10_image{image_id}.txt", "w")
        #Todo:这里只给出了resnet20，源码中没有resnet32......
        dir=f"resnet{layer_num}_new"
        cipher_pool = np.array([Cipher for _ in range(14)], dtype=object)
        co=0
        st=0
        fh=0
        fw=0
        init_p=8
        n=1<<logn
        stage=0
        epsilon=0.00001
        image = np.array([])
        linear_weight = np.array([])
        linear_bias = np.array([])
        conv_weight = np.array([[]])
        bn_bias = np.array([[]])
        bn_running_mean = np.array([[]])
        bn_running_var = np.array([[]])
        bn_weight = np.array([[]])
        import_parameters_cifar10(layer_num,end_num,linear_weight,linear_bias,conv_weight,bn_bias,bn_running_mean,bn_running_var,bn_weight)
        with open("testFile/test_values.txt", "r") as in_file:
            image=np.zeros(1<<logn,dtype=float)
            for i in range(32 * 32 * 3 * image_id):
                val = float(next(in_file))
            for i in range(32 * 32 * 3):
                val = float(next(in_file))
                image[i] = val
        i=n/init_p
        while i<n:
            i+=1
            image[i]=image[i%(n/init_p)]
        for i in range(n):
            image[i]/=B
        with open("testFile/test_label.txt", "r") as in_label:
            for _ in range(image_id):
                image_label = int(in_label.readline().strip())
            image_label = int(in_label.readline().strip())
        vec = np.zeros(1 << logn, dtype=float)
        vec[:len(image)] = image[:len(image)]
        scale_temp=pow(2.0,logq)
        x = torch.tensor(vec, device="cuda")
        cipher_temp= openfhe_context.encrypt(x, 1, 0, len(vec), "release")#Todo:scale？
        cnn=TensorCipher(1,32,32,3,3,init_p,logn,cipher_temp)
        #ctxt=cnn.cipher
        #for i in range(boot_level-3):
            #evaluator.mod_switch_to_next_inplace(ctxt);


        # layer 0:


        cnn=multiplexed_parallel_convolution_print(openfhe_context,cryptoContext,cnn,16,1,fh,fw,conv_weight[stage],bn_running_var[stage],bn_weight[stage],epsilon,cipher_pool)



        cnn=multiplexed_parallel_batch_norm_seal_print(openfhe_context,cryptoContext,cnn,bn_bias[stage],bn_running_mean[stage],bn_running_var[stage],bn_weight[stage],epsilon,B)
        scale=1.7
        #approx_ReLU_seal_print(openfhe_context,cryptoContext,cnn,comp_no,deg,alpha,tree,scaled_val,logp,public_key,secret_key,relin_keys,B)
        cnn=homo_relu(cnn,B,29,cryptoContext)

        for j in range (3):
            if j==0:
                co=16
            elif j==1:
                co=32
            elif j==2:
                co=64
            for k in range(end_num+1):
                stage=2*((end_num+1)*j+k)+1
                temp=cnn
                if j>=1 and k==0:
                    st=2
                else:
                    st=1
                cnn = multiplexed_parallel_convolution_print(openfhe_context, cryptoContext, cnn, co, st, fh, fw,
                                                             conv_weight[stage], bn_running_var[stage],
                                                             bn_weight[stage], epsilon, cipher_pool)
                cnn=multiplexed_parallel_batch_norm_seal_print(openfhe_context,cryptoContext,cnn,bn_bias[stage],bn_running_mean[stage],bn_running_var[stage],bn_weight[stage],epsilon,B)
                if j==0:
                    fhe.homo_bootstrap(cnn.cipher,L0=cryptoContext.L, logBsSlots=logBsSlots_list[0],cryptoContext=cryptoContext)
                elif j==1:
                    fhe.homo_bootstrap(cnn.cipher,L0=cryptoContext.L, logBsSlots=logBsSlots_list[1],cryptoContext=cryptoContext)
                elif j==2:
                    fhe.homo_bootstrap(cnn.cipher,L0=cryptoContext.L, logBsSlots=logBsSlots_list[2],cryptoContext=cryptoContext)
                #approx_ReLU_seal_print(openfhe_context,cryptoContext,cnn,comp_no,deg,alpha,tree,scaled_val,logp,public_key,secret_key,relin_keys,B)
                cnn = homo_relu(cnn, B, 29, cryptoContext)

                stage=2*((end_num+1)*j+k)+2
                st=1
                cnn = multiplexed_parallel_convolution_print(openfhe_context, cryptoContext, cnn, co, st, fh, fw,
                                                             conv_weight[stage], bn_running_var[stage],
                                                             bn_weight[stage], epsilon, cipher_pool)
                cnn=multiplexed_parallel_batch_norm_seal_print(openfhe_context,cryptoContext,cnn,bn_bias[stage],bn_running_mean[stage],bn_running_var[stage],bn_weight[stage],epsilon,B)
                if j>=1 and k==0:
                    temp=multiplexed_parallel_downsampling_seal_print(openfhe_context,cryptoContext,temp)
                cnn.cipher=fhe.homo_add(temp.cipher,cnn.cipher,cryptoContext)
                if j==0:
                    cnn.cipher=fhe.homo_bootstrap(cnn.cipher,L0=cryptoContext.L, logBsSlots=logBsSlots_list[0],cryptoContext=cryptoContext)
                elif j==1:
                    cnn.cipher=fhe.homo_bootstrap(cnn.cipher,L0=cryptoContext.L, logBsSlots=logBsSlots_list[1],cryptoContext=cryptoContext)
                elif j==2:
                    cnn.cipher=fhe.homo_bootstrap(cnn.cipher,L0=cryptoContext.L, logBsSlots=logBsSlots_list[2],cryptoContext=cryptoContext)
                #approx_ReLU_seal_print(openfhe_context,cryptoContext,cnn,comp_no,deg,alpha,tree,scaled_val,logp,public_key,secret_key,relin_keys,B)
                cnn = homo_relu(cnn, B, 29, cryptoContext)

        cnn=averagepooling_seal_scale_print(openfhe_context,cryptoContext,cnn,B)
        cnn=fully_connected_seal_print(openfhe_context,cryptoContext,cnn,linear_weight,linear_bias,10,64)



if __name__ == "__main__":
    ResNet_cifar10_seal_sparse(20,1,2)

#Todo:  scale问题：scale_value取值问题 51 or 46（Relu）














































