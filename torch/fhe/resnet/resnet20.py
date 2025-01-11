from ensurepip import bootstrap

from torch.fhe import homo_ops
from torch.fhe import hoisting_keyswitch
from torch.fhe.bootstrapping import homo_bootstrap
from torch.fhe.ciphertext import Cipher
from torch.fhe.example.run_test import openfhe_context
from torch.onnx.symbolic_opset9 import clone
import numpy as np
import os
global_num_slots=4096

def read_values_from_file(filename, scale=1.0):
    values = []

    if not os.path.isfile(filename):
        print(f"无法打开文件: {filename}")
        return values

    try:
        # 打开文件并逐行读取
        with open(filename, 'r') as file:
            for row in file:
                # 按行解析
                for value in row.strip().split(','):
                    try:
                        num = float(value)
                        values.append(num * scale)
                    except ValueError:
                        print(f"unconvert:: {value}")
    except IOError as e:
        print(f"error: {e}")

    return values

def rotsum(input,slots,cryptoContext):
    result=input.clone()
    for i in range(np.log2(slots)):
        result=homo_ops.homo_add(result,homo_ops.homo_rotate(result,pow(2,i),cryptoContext),cryptoContext)
    return result


def repeat(input,slots,cryptoContext):
    return homo_ops.homo_rotate(rotsum(input,slots,cryptoContext),-slots+1,cryptoContext)


def rotsum_padded(input,slots,cryptoContext):
    result=input.clone()
    for i in range(np.log2(slots)):
        result=homo_ops.homo_add(result,homo_ops.homo_rotate(result,slots*pow(2,i),cryptoContext),cryptoContext)
    return result

def mask_mod(n,cur_limbs,custom_val,cryptoContext):
    level = cryptoContext.L-cur_limbs
    vec=[]
    for i in range(global_num_slots):
        if i%n==0:
            vec.append(custom_val)
        else:
            vec.append(0)
    return
    #Todo:encode
    return encode(vec, level, global_num_slots);
    return openfhe_context.encode()


def mask_scecond_n(n,cur_limbs,cryptoContext):
    mask=[]
    level=cryptoContext.L-cur_limbs
    for i in range(global_num_slots):
        if i >=n :
            mask.append(1)
        else:
            mask.append(0)
#Todo:encode
    #return encode(vec, level, global_num_slots);
    return

def mask_first_n(n,cur_limbs,cryptoContext):
    mask=[]
    level=cryptoContext.L-cur_limbs
    for i in range(global_num_slots):
        if i < n:
            mask.append(1)
        else:
            mask.append(0)
#Todo:encode
    #return encode(vec, level, global_num_slots);
    return



def mask_from_to(from_,to,cur_limbs,cryptoContext):
    vec=[]
    level=cryptoContext.L-cur_limbs
    for i in range(global_num_slots):
        if i>=from_ and i<to:
            vec.append(1)
        else:
            vec.append(0)

    return
    #Todo:encode
    #return encode(vec, level, global_num_slots);

def gen_mask(n,cur_limbs,cryptoContext):
    level = cryptoContext.L - cur_limbs
    mask=[]
    copy_interval=n
    for i in range(global_num_slots):
        if copy_interval>0:
            mask.append(1)
        else:
            mask.append(0)
        copy_interval-=1
        if copy_interval<= -n:
            copy_interval=n
    return
    # Todo:encode
    # return encode(vec, level, global_num_slots);

def mask_first_n_mod(n,padding,pos,cur_limbs,cryptoContext):
    mask=[]
    level = cryptoContext.L - cur_limbs
    for i in range(32):
        for j in range(pos*n):
            mask.append(0)
        for j in range(n):
            mask.append(1)
        for j in range(padding-n-(pos*n)):
            mask.append(0)
    return
            # Todo:encode
            # return encode(mask, level, 16384 * 2);

def mask_first_n_mod2(n,padding,pos,cur_limbs,cryptoContext):
    mask=[]
    level = cryptoContext.L - cur_limbs
    for i in range(64):
        for j in range(pos*n):
            mask.append(0)
        for j in range(n):
            mask.append(1)
        for j in range(padding-n-(pos*n)):
            mask.append(0)
    return
            # Todo:encode
            # return encode(mask, level, 8192 * 2);


def mask_channel(n,cur_limbs,cryptoContext):
    mask=[]
    level = cryptoContext.L - cur_limbs
    for i in range(n):
        for j in range(1024):
            mask.append(0)

    for i in range(256):
        mask.append(1)

    for i in range(1024-256):
        mask.append(0)
    for i in range(31-n):
        for j in range(1024):
            mask.append(0)
    return
            # Todo:encode
            # return encode(mask, level, 16384 * 2);

def mask_channel2(n,cur_limbs,cryptoContext):
    mask=[]
    level = cryptoContext.L - cur_limbs
    for i in range(n):
        for j in range(256):
            mask.append(0)

    for i in range(64):
        mask.append(1)

    for i in range(256-64):
        mask.append(0)
    for i in range(63-n):
        for j in range(256):
            mask.append(0)
    return
            # Todo:encode
            # return encode(mask, level, 8192 * 2);


def downsample1024to256(c1,c2,cryptoContext):
    #Todo: SetSlots()
    # c1->SetSlots(32768);
    # c2->SetSlots(32768);
    num_slots = 16384 * 2
    #Todo:这里加法和乘法为明文还是密文？
    #todo: yhh：里面的是分别明密文乘法，然后是密密文加法
    fullpack=homo_ops.homo_add(homo_ops.homo_mul(c1,mask_first_n(16384,c1.cur_limbs,cryptoContext),cryptoContext),homo_ops.homo_mul(c2,mask_scecond_n(16384,c2.cur_limbs,cryptoContext),cryptoContext),cryptoContext)
    #Todo:这里有密文乘明文
    fullpack=homo_ops.homo_mul(homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,1,cryptoContext),cryptoContext),gen_mask(2,fullpack.cur_limbs,cryptoContext),cryptoContext)
    fullpack = homo_ops.homo_mul(
        homo_ops.homo_add(fullpack, homo_ops.homo_rotate(homo_ops.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext),
        gen_mask(4, fullpack.cur_limbs, cryptoContext), cryptoContext)
    fullpack=homo_ops.homo_mul(homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,4,cryptoContext),cryptoContext),gen_mask(8,fullpack.cur_limbs,cryptoContext),cryptoContext)
    fullpack=homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,8,cryptoContext),cryptoContext)


#  Todo:Ctxt downsampledrows = encrypt({0});
    downsampledrows=0
    for i in range(16):
        #Todo明密文乘法
        masked=homo_ops.homo_mul(fullpack,mask_first_n_mod(16,1024,i,fullpack.cur_limbs,cryptoContext),cryptoContext)
        downsampledrows=homo_ops.homo_add(downsampledrows,masked,cryptoContext)
        if i<15:
            fullpack=homo_ops.homo_rotate(fullpack,64-16,cryptoContext)
    #Todo:Ctxt downsampledchannels = encrypt({0});
    downsampledchannels=0
    for i in range(32):
        #Todo 明密文乘
        masked=homo_ops.homo_mul(downsampledrows,mask_channel(i,downsampledrows.cur_limbs,cryptoContext),cryptoContext)
        downsampledchannels=homo_ops.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels=homo_ops.homo_rotate(downsampledchannels,-(1024-256),cryptoContext)

    downsampledchannels=homo_ops.homo_rotate(downsampledchannels,(1024-256)*32,cryptoContext)
    downsampledchannels=homo_ops.homo_add(downsampledchannels,homo_ops.homo_rotate(downsampledchannels,-8192,cryptoContext),cryptoContext)
    downsampledchannels = homo_ops.homo_add(downsampledchannels,
                                            homo_ops.homo_rotate(homo_ops.homo_rotate(downsampledchannels,-8192,cryptoContext), -8192, cryptoContext),
                                            cryptoContext)
    #Todo:    downsampledchannels->SetSlots(8192);

    return downsampledchannels

def decrypt_tovector(input,slots,cryptoContext):
    if slots==0:
        slots=global_num_slots
    #Todo:   context->Decrypt(key_pair.secretKey, c, &p);
    # p->SetSlots(slots);
    # p->SetLength(slots);
    # vector<double> vec = p->GetRealPackedValue();
    vec=[]
    return vec

def relu(ciphertext, scale, degree=59):
    pass
#     /**
#    * EvalAddMany - Evaluate addition on a vector of ciphertexts.
#    * It computes the addition in a binary tree manner.
#    *
#    * @param ctList is the list of ciphertexts.
#    * @return new ciphertext.
#    */
# func: EvalAddMany in cryptocontext.h, implemented in base-advancedshe.cpp
def eval_add_many(ciphertexts, cryptoContext):
    inSize = len(ciphertexts)
    if inSize < 1:
        raise ValueError("Input ciphertext vector size should be 1 or more")

    lim = inSize * 2 - 2
    ciphertextSumVec = [None] * (inSize - 1)
    ctrIndex = 0

    # see if all the ciphertexts are of the same cur_limbs
    # if not, raise an error
    cur_limbs = ciphertexts[0].cur_limbs
    for i in range(1, inSize):
        if cur_limbs != ciphertexts[i].cur_limbs:
            raise ValueError("All ciphertexts should have the same cur_limbs")
    for i in range(0, lim, 2):
        ciphertextSumVec[ctrIndex] = homo_ops.homo_add(ciphertexts[i] if i < inSize else ciphertextSumVec[i - inSize],
                                              ciphertexts[i + 1] if i + 1 < inSize else ciphertextSumVec[i + 1 - inSize],
                                              cryptoContext)
        ctrIndex += 1

    return ciphertextSumVec[-1]
def convbn3(input,layer,n,scale,cryptoContext):
    img_width=8
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)#这里旋转什么的都只需要对cv1吗？
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), img_width, cryptoContext))
    #Todo encode相关问题
    #Ptxt bias = encode(read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias.bin", scale), circuit_depth-2, 8192);

    for j in range(64):
        k_rows=[]
        # Todo encode相关问题
        #for k in range(9):
            # for (int k = 0; k < 9; k++) {
            #     vector < double > values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
            # to_string(j) + "-k" + to_string(k+1) + ".bin", scale);
            # Ptxt encoded = encode(values, circuit_depth - 2, 8192);
            # k_rows.push_back(context->EvalMult(c_rotations[k], encoded));
            # }
        sum=eval_add_many(k_rows,cryptoContext)
        if(j==0):
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
    #Todo：这里add应该为明密文相加
    #finalsum=homo_ops.homo_add(finalsum,bias,cryptoContext)
    return finalsum

def convbn_initial(input,scale,cryptoContext):
    img_width=32
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)#这里旋转什么的都只需要对cv1吗？
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), img_width, cryptoContext))
    #Todo encode相关问题

    #bias = encode(read_values_from_file("../weights/conv1bn1-bias.bin", scale), in->GetLevel(), 16384);

    for j in range(16):
        k_rows=[]
        # Todo encode相关问题
        # for (int k = 0; k < 9; k++) {
        #     vector < double > values = read_values_from_file("../weights/conv1bn1-ch" +
        # to_string(j) + "-k" + to_string(k+1) + ".bin", scale);
        # Ptxt encoded = encode(values, in ->GetLevel(), 16384);
        # k_rows.push_back(context->EvalMult(c_rotations[k], encoded));
        # }
        sum=eval_add_many(k_rows,cryptoContext)
        res=sum.clone()
        res=homo_ops.homo_add(res,homo_ops.homo_rotate(sum,1024,cryptoContext),cryptoContext)
        res = homo_ops.homo_add(res, homo_ops.homo_rotate(homo_ops.homo_rotate(sum,1024,cryptoContext), 1024, cryptoContext), cryptoContext)
        #Todo:这里应为明密文相乘
        res=homo_ops.homo_mul(res,mask_from_to(0,1024,res.cur_limbs,cryptoContext),cryptoContext)

        if (j == 0):
            finalsum = res.clone()
            finalsum = homo_ops.homo_rotate(finalsum, 1024, cryptoContext)
        else:
            finalsum = homo_ops.homo_add(finalsum, res, cryptoContext)
            finalsum = homo_ops.homo_rotate(finalsum, 1024, cryptoContext)
    #finalsum=homo_ops.homo_add(finalsum,bias,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum



def convbn(input,layer,n,scale,cryptoContext):
    img_width=32
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)#这里旋转什么的都只需要对cv1吗？
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), img_width, cryptoContext))
    #Todo encode相关问题
    #Ptxt bias = encode(read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias.bin", scale), circuit_depth-2, 8192);

    for j in range(16):
        k_rows=[]
        # Todo encode相关问题
        #for k in range(9):
            # for (int k = 0; k < 9; k++) {
            #     vector < double > values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
            # to_string(j) + "-k" + to_string(k+1) + ".bin", scale);
            # Ptxt encoded = encode(values, circuit_depth - 2, 8192);
            # k_rows.push_back(context->EvalMult(c_rotations[k], encoded));
            # }
        sum=eval_add_many(k_rows,cryptoContext)
        if(j==0):
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-1024,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-1024,cryptoContext)
    #finalsum=homo_ops.homo_add(finalsum,bias,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum


def convbn2(input,layer,n,scale,cryptoContext):
    img_width=16
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)#这里旋转什么的都只需要对cv1吗？#todo: remove this comment if solved
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), img_width, cryptoContext))
    #Todo encode相关问题
    #Ptxt bias = encode(read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias.bin", scale), circuit_depth-2, 8192);

    for j in range(32):
        k_rows=[]
        # Todo encode相关问题
        #for k in range(9):
            # for (int k = 0; k < 9; k++) {
            #     vector < double > values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
            # to_string(j) + "-k" + to_string(k+1) + ".bin", scale);
            # Ptxt encoded = encode(values, circuit_depth - 2, 8192);
            # k_rows.push_back(context->EvalMult(c_rotations[k], encoded));
            # }
        sum=eval_add_many(k_rows,cryptoContext)
        if(j==0):
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
    #finalsum=homo_ops.homo_add(finalsum,bias,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum


def convbn1632sx(input,layer,n,scale,cryptoContext):
    img_width=32
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-(img_width),digits,cryptoContext),-padding,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, -(img_width), digits, cryptoContext), padding, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input,padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,img_width,digits,cryptoContext),-padding,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, img_width, digits, cryptoContext), padding, cryptoContext))
    #Todo encode相关问题
    # vector < Ctxt > applied_filters16;
    # vector < Ctxt > applied_filters32;
    #
    # Ptxt
    # bias1 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias1.bin",
    #     scale), in->GetLevel(), 16384);
    # Ptxt
    # bias2 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias2.bin",
    #     scale), in->GetLevel(), 16384);
    #
    # Ctxt
    # finalSum016;
    # Ctxt
    # finalSum1632;
    for j in range(16):
        k_rows016=[]
        k_rows1632 = []
        # Todo encode相关问题
        # for (int k = 0; k < 9; k++) {
        #     vector < double > values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        # to_string(j) + "-k" + to_string(k+1) + ".bin", scale);
        # k_rows016.push_back(context->EvalMult(c_rotations[k], encode(values, in ->GetLevel(), 16384)));
        #
        # values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        # to_string(j+16) + "-k" + to_string(k+1) + ".bin", scale);
        # k_rows1632.push_back(context->EvalMult(c_rotations[k], encode(values, in ->GetLevel(), 16384)));
        # }
        sum016=eval_add_many(k_rows016,cryptoContext)
        sum1632 = eval_add_many(k_rows1632,cryptoContext)

        if(j==0):
            finalsum016=sum016.clone()
            finalsum016=homo_ops.homo_rotate(finalsum016,-1024,cryptoContext)
            finalsum1632 = sum1632.clone()
            finalsum1632 = homo_ops.homo_rotate(finalsum1632, -1024, cryptoContext)
        else:
            finalsum016=homo_ops.homo_add(finalsum016,sum016,cryptoContext)
            finalsum016=homo_ops.homo_rotate(finalsum016,-1024,cryptoContext)
            finalsum1632 = homo_ops.homo_add(finalsum1632, sum1632, cryptoContext)
            finalsum1632 = homo_ops.homo_rotate(finalsum1632, -1024, cryptoContext)

    #finalsum016=homo_ops.homo_add(finalsum016,bias1,cryptoContext)
    # finalsum1632=homo_ops.homo_add(finalsum1632,bias2,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum016, finalsum1632



def convbn1632dx(input,layer,n,scale,cryptoContext):
    # vector < Ctxt > applied_filters16;
    # vector < Ctxt > applied_filters32;
    #
    # Ptxt
    # bias1 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-bias1.bin",
    #     scale), in->GetLevel(), 16384);
    # Ptxt
    # bias2 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-bias2.bin",
    #     scale), in->GetLevel(), 16384);
    #
    # Ctxt
    # finalSum016;
    # Ctxt
    # finalSum1632;
    for j in range(16):
        k_rows016 = []
        k_rows1632 = []
        # vector < double > values = read_values_from_file(
        #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        #     to_string(j) + "-k" + to_string(1) + ".bin", scale);
        # k_rows016.push_back(context->EvalMult( in, encode(values, in->GetLevel(), global_num_slots)));
        #
        # values = read_values_from_file(
        #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        #     to_string(j + 16) + "-k" + to_string(1) + ".bin", scale);
        #
        # k_rows1632.push_back(context->EvalMult( in, encode(values, in->GetLevel(), global_num_slots)));
        sum016=eval_add_many(k_rows016,cryptoContext)
        sum1632 = eval_add_many(k_rows1632,cryptoContext)

        if(j==0):
            finalsum016=sum016.clone()
            finalsum016=homo_ops.homo_rotate(finalsum016,-1024,cryptoContext)
            finalsum1632 = sum1632.clone()
            finalsum1632 = homo_ops.homo_rotate(finalsum1632, -1024, cryptoContext)
        else:
            finalsum016=homo_ops.homo_add(finalsum016,sum016,cryptoContext)
            finalsum016=homo_ops.homo_rotate(finalsum016,-1024,cryptoContext)
            finalsum1632 = homo_ops.homo_add(finalsum1632, sum1632, cryptoContext)
            finalsum1632 = homo_ops.homo_rotate(finalsum1632, -1024, cryptoContext)

    #finalsum016=homo_ops.homo_add(finalsum016,bias1,cryptoContext)
    # finalsum1632=homo_ops.homo_add(finalsum1632,bias2,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum016, finalsum1632



def convbn3264sx(input,layer,n,scale,cryptoContext):
    img_width=16
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-(img_width),digits,cryptoContext),-padding,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, -(img_width), digits, cryptoContext), padding, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input,padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,img_width,digits,cryptoContext),-padding,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, img_width, digits, cryptoContext), padding, cryptoContext))
    #Todo encode相关问题
    # vector < Ctxt > applied_filters16;
    # vector < Ctxt > applied_filters32;
    #
    # Ptxt
    # bias1 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias1.bin",
    #     scale), in->GetLevel(), 16384);
    # Ptxt
    # bias2 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias2.bin",
    #     scale), in->GetLevel(), 16384);
    #
    # Ctxt
    # finalSum016;
    # Ctxt
    # finalSum1632;
    for j in range(32):
        k_rows032=[]
        k_rows3264 = []
        # Todo encode相关问题
        # for (int k = 0; k < 9; k++) {
        #     vector < double > values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        # to_string(j) + "-k" + to_string(k+1) + ".bin", scale);
        # k_rows016.push_back(context->EvalMult(c_rotations[k], encode(values, in ->GetLevel(), 16384)));
        #
        # values = read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        # to_string(j+16) + "-k" + to_string(k+1) + ".bin", scale);
        # k_rows1632.push_back(context->EvalMult(c_rotations[k], encode(values, in ->GetLevel(), 16384)));
        # }
        sum032=eval_add_many(k_rows032,cryptoContext)
        sum3264 = eval_add_many(k_rows3264,cryptoContext)

        if(j==0):
            finalsum032=sum032.clone()
            finalsum032=homo_ops.homo_rotate(finalsum032,-256,cryptoContext)
            finalsum3264 = sum3264.clone()
            finalsum3264 = homo_ops.homo_rotate(finalsum3264, -256, cryptoContext)
        else:
            finalsum032=homo_ops.homo_add(finalsum032,sum032,cryptoContext)
            finalsum032=homo_ops.homo_rotate(finalsum032,-256,cryptoContext)
            finalsum3264 = homo_ops.homo_add(finalsum3264, sum3264, cryptoContext)
            finalsum3264 = homo_ops.homo_rotate(finalsum3264, -256, cryptoContext)

    #finalsum032=homo_ops.homo_add(finalsum032,bias1,cryptoContext)
    # finalsum3264=homo_ops.homo_add(finalsum3264,bias2,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum032, finalsum3264


def convbn3264dx(input,layer,n,scale,cryptoContext):
    # vector < Ctxt > applied_filters16;
    # vector < Ctxt > applied_filters32;
    #
    # Ptxt
    # bias1 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-bias1.bin",
    #     scale), in->GetLevel(), 16384);
    # Ptxt
    # bias2 = encode(read_values_from_file(
    #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-bias2.bin",
    #     scale), in->GetLevel(), 16384);
    #
    # Ctxt
    # finalSum016;
    # Ctxt
    # finalSum1632;
    for j in range(32):
        k_rows032 = []
        k_rows3264 = []
        # vector < double > values = read_values_from_file(
        #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        #     to_string(j) + "-k" + to_string(1) + ".bin", scale);
        # k_rows016.push_back(context->EvalMult( in, encode(values, in->GetLevel(), global_num_slots)));
        #
        # values = read_values_from_file(
        #     "../weights/layer" + to_string(layer) + "dx-conv" + to_string(n) + "bn" + to_string(n) + "-ch" +
        #     to_string(j + 16) + "-k" + to_string(1) + ".bin", scale);
        #
        # k_rows1632.push_back(context->EvalMult( in, encode(values, in->GetLevel(), global_num_slots)));
        sum032=eval_add_many(k_rows032,cryptoContext)
        sum3264 = eval_add_many(k_rows3264,cryptoContext)

        if(j==0):
            finalsum032=sum032.clone()
            finalsum032=homo_ops.homo_rotate(finalsum032,-256,cryptoContext)
            finalsum3264 = sum3264.clone()
            finalsum3264 = homo_ops.homo_rotate(finalsum3264, -256, cryptoContext)
        else:
            finalsum032=homo_ops.homo_add(finalsum032,sum032,cryptoContext)
            finalsum032=homo_ops.homo_rotate(finalsum032,-256,cryptoContext)
            finalsum3264 = homo_ops.homo_add(finalsum3264, sum3264, cryptoContext)
            finalsum3264 = homo_ops.homo_rotate(finalsum3264, -256, cryptoContext)

    #finalsum032=homo_ops.homo_add(finalsum032,bias1,cryptoContext)
    # finalsum3264=homo_ops.homo_add(finalsum3264,bias2,cryptoContext)
    # Todo：这里add应该为明密文相加
    return finalsum032, finalsum3264



def initial_layer(input,cryptoContext):
    scale=0.90
    res=convbn_initial(input,scale,cryptoContext)
    res=relu(res,scale)
    return res



def layer1(input,cryptoContext):
    scale=1.00
    res1=convbn(input,1,1,scale,cryptoContext)
    #fixme: [!!!]这里不太确定slots
    #todo: yhh: 不确定slots的，可以看一下原始代码里，下面一行if(verbose>2) 中下面prt中slots的输入值，来推测上面prt的值
    #todo: 这里传slots=(1<<14)就行
    #todo：确认一下，不应该传log值
    res1=homo_bootstrap(res1,L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1=relu(res1,scale)

    scale=0.52
    res1=convbn(res1,1,2,scale,cryptoContext)
    res1=homo_ops.homo_add(res1,homo_ops.homo_mul_scalar_double(input,scale,cryptoContext),cryptoContext)
    res1=homo_bootstrap(res1,L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1=relu(res1,scale)

    scale=0.55
    res2 = convbn(res1, 2, 1, scale, cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2 = relu(res2, scale)

    scale=0.36
    res2 = convbn(res2, 2, 2, scale, cryptoContext)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2 = relu(res2, scale)

    scale=0.63
    res3 = convbn(res2, 3, 1, scale, cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3 = relu(res3, scale)

    scale = 0.42
    res3 = convbn(res2, 3, 2, scale, cryptoContext)
    res3 = homo_ops.homo_add(res3, homo_ops.homo_mul_scalar_double(res2, scale, cryptoContext), cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3 = relu(res3, scale)

    return res3

def layer2(input,cryptoContext):
    scaleSx=0.57
    scaleDx=0.40
    boot_in=homo_bootstrap(input, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1sx=convbn1632sx(boot_in,4,1,scaleSx,cryptoContext)
    res1dx=convbn1632dx(boot_in,4,1,scaleDx,cryptoContext)

    # Todo: nothing   ,   just test
    fullpackSx,fullpackDx=0


    #Todo:        controller.clear_bootstrapping_and_rotation_keys(16384);
    #Todo：       controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);

    # Todo:     Ctxt fullpackSx = controller.downsample1024to256(res1sx[0], res1sx[1]);
    # Todo:    Ctxt fullpackDx = controller.downsample1024to256(res1dx[0], res1dx[1]);
    # Todo:res1sx.clear();
    # Todo:res1dx.clear();
    #Todo:    controller.clear_rotation_keys();
    #Todo:controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", 8192, verbose > 1);
    global_num_slots=8192
    #fullpackSx=homo_bootstrap(fullpackSx,L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    #fullpackSx=relu(fullpackSx,scaleSx)
    fullpackSx=convbn2(fullpackSx,4,2,scaleDx,cryptoContext)
    res1=homo_ops.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1=homo_bootstrap(res1, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1=relu(res1,scaleDx)



    scale=0.76
    res2=convbn2(res1,5,1,scale,cryptoContext)
    res2=homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2=relu(res2,scale)

    scale=0.37
    res2=convbn2(res2,5,2,scale,cryptoContext)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2 = relu(res2, scale)

    scale=0.63
    res3=convbn2(res2,6,1,scale,cryptoContext)
    res3=homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3=relu(res3,scale)

    scale=0.25
    res3=convbn2(res3,6,2,scale,cryptoContext)
    res3=homo_ops.homo_add(res3,homo_ops.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3 = relu(res3, scale)

    return res3


def layer3(input,cryptoContext):
    scaleSx=0.63
    scaleDx=0.40
    boot_in=homo_bootstrap(input, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1sx=convbn3264sx(boot_in,7,1,scaleSx,cryptoContext)
    res1dx=convbn3264dx(boot_in,7,1,scaleDx,cryptoContext)

    # Todo: nothing   ,   just test
    fullpackSx,fullpackDx=0


    #Todo:        controller.clear_bootstrapping_and_rotation_keys(16384);
    #Todo：       controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);

    # Todo:     Ctxt fullpackSx = controller.downsample1024to256(res1sx[0], res1sx[1]);
    # Todo:    Ctxt fullpackDx = controller.downsample1024to256(res1dx[0], res1dx[1]);
    # Todo:res1sx.clear();
    # Todo:res1dx.clear();
    #Todo:    controller.clear_rotation_keys();
    #Todo:controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", 8192, verbose > 1);
    global_num_slots=4096
    fullpackSx=homo_bootstrap(fullpackSx,L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    fullpackSx=relu(fullpackSx,scaleSx)
    fullpackSx=convbn3(fullpackSx,7,2,scaleDx,cryptoContext)
    res1=homo_ops.homo_add(fullpackSx,fullpackDx,cryptoContext)
    res1=homo_bootstrap(res1, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res1=relu(res1,scaleDx)



    scale=0.57
    res2=convbn3(res1,8,1,scale,cryptoContext)
    res2=homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2=relu(res2,scale)

    scale=0.33
    res2=convbn3(res2,8,2,scale,cryptoContext)
    res2=homo_ops.homo_add(res2,homo_ops.homo_mul_scalar_double(res1,scale,cryptoContext),cryptoContext)
    res2 = homo_bootstrap(res2, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res2 = relu(res2, scale)

    scale=0.69
    res3=convbn3(res2,9,1,scale,cryptoContext)
    res3=homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3=relu(res3,scale)

    scale=0.1
    res3=convbn3(res3,9,2,scale,cryptoContext)
    res3=homo_ops.homo_add(res3,homo_ops.homo_mul_scalar_double(res2,scale,cryptoContext),cryptoContext)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    res3 = relu(res3, scale)
    res3 = homo_bootstrap(res3, L0=cryptoContext.L, slots=14, cryptoContext=cryptoContext)
    return res3


def final_layer(input,cryptoContext):
#Todo:encode 未定义函数：clear_bootstrapping_and_rotation_keys   load_rotation_keys
    # controller.clear_bootstrapping_and_rotation_keys(4096);
    # controller.load_rotation_keys("rotations-finallayer.bin", false);
    global_num_slots=4096

#Todo: encode
# Ptxt weight = controller.encode(read_fc_weight("../weights/fc.bin"), in->GetLevel(), controller.global_num_slots)
    res = rotsum( input, 64,cryptoContext);
#Todo:这里需要明密文相乘：res = controller.mult(res, controller.mask_mod(64, res->GetLevel(), 1.0 / 64.0));
    #res = homo_ops.mul(res,mask_mod(64,res.cur_limbs,1.0/64.0,cryptoContext),cryptoContext)
    res=repeat(res,16,cryptoContext)
#Todo:这里需要明密文相乘：
#   res=homo_ops.homo_mul(res,weight,cryptoContext)
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