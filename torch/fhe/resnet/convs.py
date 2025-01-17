from torch.fhe import hoisting_keyswitch
from torch.fhe import homo_ops
from ..ciphertext import Cipher
# from torch.fhe.resnet.resnet20 import global_num_slots
from torch.fhe.resnet.utils import *



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
    ciphertextSumVec = [] * (inSize - 1)
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


def convbn_initial(input, scale, he_res20_ctx, cryptoContext, openfhe_context_dict):
    img_width=32
    padding=1
    digits=hoisting_keyswitch.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), img_width, cryptoContext))

    cur_num_slots = he_res20_ctx.cur_num_slots
    openfhe_context = openfhe_context_dict[str( cur_num_slots)]
    bias=openfhe_context.encode(read_values_from_file("../weights/conv1bn1-bias.bin",scale),cryptoContext.L-input.cur_limbs,16384)

    for j in range(16):
        k_rows=[]
        for k in range(9):
            values=read_values_from_file(f"../weights/conv1bn1-ch{j}-k{k+1}.bin",scale)
            encoded=openfhe_context.encode(values,cryptoContext.L-input.cur_limbs,16384)
            k_rows.append(homo_ops.homo_mul_pt(c_rotations[k],encoded,cryptoContext))

        sum=eval_add_many(k_rows,cryptoContext)
        res=sum.deep_copy()
        res=homo_ops.homo_add(res,homo_ops.homo_rotate(sum,1024,cryptoContext),cryptoContext)
        res = homo_ops.homo_add(res, homo_ops.homo_rotate(homo_ops.homo_rotate(sum,1024,cryptoContext), 1024, cryptoContext), cryptoContext)

        res=homo_ops.homo_mul_pt(res,mask_from_to(0,1024,res.cur_limbs, cryptoContext, openfhe_context),cryptoContext)

        if (j == 0):
            finalsum = res.clone()
            finalsum = homo_ops.homo_rotate(finalsum, 1024, cryptoContext)
        else:
            finalsum = homo_ops.homo_add(finalsum, res, cryptoContext)
            finalsum = homo_ops.homo_rotate(finalsum, 1024, cryptoContext)
    finalsum=homo_ops.homo_add_pt(finalsum,bias,cryptoContext)

    return finalsum


def convbn(input, layer, n, scale, he_res20_ctx, cryptoContext, openfhe_context_dict):
    img_width=32
    padding=1

    digits=hoisting_keyswitch.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)

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

    openfhe_context = openfhe_context_dict[str(he_res20_ctx.cur_num_slots)]
    bias=openfhe_context.encode(read_values_from_file( f"../weights/layer{layer}-conv{n}bn{n}-bias.bin",scale),cryptoContext.L-input.cur_limbs,16384)

    for j in range(16):
        k_rows=[]
        for k in range(9):
            values=read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}.bin",scale)
            encoded=openfhe_context.encode(values,cryptoContext.L-input.cur_limbs,16384)
            k_rows.append(homo_ops.homo_mul_pt(c_rotations[k],encoded,cryptoContext))
        sum=eval_add_many(k_rows,cryptoContext)
        if(j==0):
            finalsum=sum.deep_copy()
            finalsum=homo_ops.homo_rotate(finalsum,-1024,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-1024,cryptoContext)
    finalsum=homo_ops.homo_add_pt(finalsum,bias,cryptoContext)

    return finalsum


def convbn2(input,layer,n,scale,cryptoContext):
    img_width=16
    padding=1
    digits=hoisting_keyswitch.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)

    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, -padding, digits, cryptoContext))
    c_rotations.append(input)
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input,-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input,img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input, padding, digits, cryptoContext), img_width, cryptoContext))
    bias=openfhe_context.encode(read_values_from_file( f"../weights/layer{layer}-conv{n}bn{n}-bias.bin",scale),cryptoContext.L-input.curr_limbs,8192)

    for j in range(32):
        k_rows=[]
        for k in range(9):
            values=read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}.bin",scale)
            encoded=openfhe_context.encode(values,cryptoContext.L-input.curr_limbs,8192)
            k_rows.append(homo_ops.homo_mul_pt(c_rotations[k],encoded,cryptoContext))


        sum=eval_add_many(k_rows,cryptoContext)
        sum=eval_add_many(k_rows,cryptoContext)
        if(j==0):
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
    finalsum=homo_ops.homo_add_pt(finalsum,bias,cryptoContext)

    return finalsum


def convbn3(input,layer,n,scale,cryptoContext):
    img_width=8
    padding=1
    digits=hoisting_keyswitch.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)
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

    #Ptxt bias = encode(read_values_from_file("../weights/layer" + to_string(layer) + "-conv" + to_string(n) + "bn" + to_string(n) + "-bias.bin", scale), circuit_depth-2, 8192);
    bias=openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-bias.bin",scale),cryptoContext.L-input.cur_limbs,4096)

    for j in range(64):
        k_rows=[]
        for k in range(9):
            values=read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}.bin",scale)
            encoded=openfhe_context.encode(values,cryptoContext.L-input.cur_limbs,4096)
            k_rows.append(homo_ops.homo_mul_pt(c_rotations[k],encoded,cryptoContext))
        sum=eval_add_many(k_rows,cryptoContext)
        if(j==0):
            #Todo:这里clone用法不确定
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-64,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-64,cryptoContext)

    finalsum=homo_ops.homo_add_pt(finalsum,bias,cryptoContext)
    return finalsum


def convbn1632sx(input,layer,n,scale,cryptoContext):
    img_width=32
    padding=1
    digits=hoisting_keyswitch.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)

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
    applied_filters32=[]
    applied_filters64=[]

    bias1=openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-bias1.bin",scale),cryptoContext.L-input.cur_limbs,16384)
    bias2=openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-bias2.bin",scale),cryptoContext.L-input.cur_limbs,16384)


    for j in range(16):
        k_rows016=[]
        k_rows1632 = []
        for k in range(9):
            values=[]
            values=read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}.bin",scale)
            k_rows016.append(homo_ops.homo_mul_pt(c_rotations[k],openfhe_context.encode(values,cryptoContext.L-input.cur_limbs,16384),cryptoContext))
            values = read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j+16}-k{k+1}.bin", scale)
            k_rows1632.append(homo_ops.homo_mul_pt(c_rotations[k],
                                                  openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                         16384), cryptoContext))

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

    finalsum016=homo_ops.homo_add_pt(finalsum016,bias1,cryptoContext)
    finalsum1632=homo_ops.homo_add_pt(finalsum1632,bias2,cryptoContext)

    return finalsum016, finalsum1632


def convbn1632dx(input,layer,n,scale,cryptoContext):
    bias1 = openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-bias1.bin", scale),
                                   cryptoContext.L - input.cur_limbs, 16384)
    bias2 = openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-bias2.bin", scale),
                                   cryptoContext.L - input.cur_limbs, 16384)

    for j in range(16):
        k_rows016 = []
        k_rows1632 = []

        values = []
        values = read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-ch{j}-k1.bin", scale)
        k_rows016.append(homo_ops.homo_mul_pt(input,
                                              openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                    global_num_slots), cryptoContext))
        values = read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-ch{j+16}-k1.bin", scale)
        k_rows1632.append(homo_ops.homo_mul_pt(input,
                                                   openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                          global_num_slots), cryptoContext))
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

    finalsum016=homo_ops.homo_add_pt(finalsum016,bias1,cryptoContext)
    finalsum1632=homo_ops.homo_add_pt(finalsum1632,bias2,cryptoContext)

    return finalsum016, finalsum1632


def convbn3264sx(input,layer,n,scale,cryptoContext):
    img_width=16
    padding=1
    digits=hoisting_keyswitch.modup_to_ext(input.cipher_like([input.cv[1]]),cryptoContext)

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



    bias1 = openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-bias1.bin", scale),
                                   cryptoContext.L - input.cur_limbs, 8192)
    bias2 = openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-bias2.bin", scale),
                                   cryptoContext.L - input.cur_limbs, 8192)
    for j in range(32):
        k_rows032=[]
        k_rows3264 = []
        for k in range(9):
            values = []
            values = read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j}-k{k+1}.bin", scale)
            k_rows032.append(homo_ops.homo_mul_pt(c_rotations[k],
                                                  openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                         8192), cryptoContext))
            values = read_values_from_file(f"../weights/layer{layer}-conv{n}bn{n}-ch{j+32}-k{k+1}.bin", scale)
            k_rows3264.append(homo_ops.homo_mul_pt(c_rotations[k],
                                                   openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                        8192), cryptoContext))



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

    finalsum032=homo_ops.homo_add_pt(finalsum032,bias1,cryptoContext)
    finalsum3264=homo_ops.homo_add_pt(finalsum3264,bias2,cryptoContext)

    return finalsum032, finalsum3264


def convbn3264dx(input,layer,n,scale,cryptoContext):
    bias1 = openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-bias1.bin", scale),
                                   cryptoContext.L - input.cur_limbs, 8192)
    bias2 = openfhe_context.encode(read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-bias2.bin", scale),
                                   cryptoContext.L - input.cur_limbs, 8192)
    for j in range(32):
        k_rows032 = []
        k_rows3264 = []

        values = []
        values = read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-ch{j}-k1.bin", scale)
        k_rows032.append(homo_ops.homo_mul_pt(input,
                                                  openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                         8192), cryptoContext))
        values = read_values_from_file(f"../weights/layer{layer}dx-conv{n}bn{n}-ch{j+32}-k1.bin", scale)
        k_rows3264.append(homo_ops.homo_mul_pt(input,
                                                   openfhe_context.encode(values, cryptoContext.L - input.cur_limbs,
                                                                          8192), cryptoContext))
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

    finalsum032=homo_ops.homo_add_pt(finalsum032,bias1,cryptoContext)
    finalsum3264=homo_ops.homo_add_pt(finalsum3264,bias2,cryptoContext)

    return finalsum032, finalsum3264


def downsample1024to256(c1,c2,cryptoContext):

    c1.slots=32768
    c2.slots=32768
    num_slots = 16384 * 2
    fullpack=homo_ops.homo_add(homo_ops.homo_mul_pt(c1,mask_first_n(16384,c1.cur_limbs,cryptoContext),cryptoContext),homo_ops.homo_mul_pt(c2,mask_scecond_n(16384,c2.cur_limbs,cryptoContext),cryptoContext),cryptoContext)

    fullpack=homo_ops.homo_mul_pt(homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,1,cryptoContext),cryptoContext),gen_mask(2,fullpack.cur_limbs,cryptoContext),cryptoContext)
    fullpack = homo_ops.homo_mul_pt(
        homo_ops.homo_add(fullpack, homo_ops.homo_rotate(homo_ops.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext),
        gen_mask(4, fullpack.cur_limbs, cryptoContext), cryptoContext)
    fullpack=homo_ops.homo_mul_pt(homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,4,cryptoContext),cryptoContext),gen_mask(8,fullpack.cur_limbs,cryptoContext),cryptoContext)
    fullpack=homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,8,cryptoContext),cryptoContext)

    masked = homo_ops.homo_mul(fullpack, mask_first_n_mod(16, 1024, 0, fullpack.cur_limbs, cryptoContext),
                               cryptoContext)
    #  Todo:level值采取默认值还是与之相加的level
    # todo: check limb setting, omit + 1?   input长度小于slot的值是否会自动填充
    temp=[0]*global_num_slots
    downsampledrows=openfhe_context.encrypt(temp,1,cryptoContext.L-masked.cur_limbs,global_num_slots)
    for i in range(16):

        masked=homo_ops.homo_mul_pt(fullpack,mask_first_n_mod(16,1024,i,fullpack.cur_limbs,cryptoContext),cryptoContext)
        downsampledrows=homo_ops.homo_add(downsampledrows,masked,cryptoContext)
        if i<15:
            fullpack=homo_ops.homo_rotate(fullpack,64-16,cryptoContext)

    masked = homo_ops.homo_mul(downsampledrows, mask_channel(0, downsampledrows.cur_limbs, cryptoContext),
                               cryptoContext)
    temp=[0]*global_num_slots
    downsampledchannels=openfhe_context.encrypt(temp,1,cryptoContext.L-masked.cur_limbs,global_num_slots)
    for i in range(32):

        masked=homo_ops.homo_mul_pt(downsampledrows,mask_channel(i,downsampledrows.cur_limbs,cryptoContext),cryptoContext)
        downsampledchannels=homo_ops.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels=homo_ops.homo_rotate(downsampledchannels,-(1024-256),cryptoContext)

    downsampledchannels=homo_ops.homo_rotate(downsampledchannels,(1024-256)*32,cryptoContext)
    downsampledchannels=homo_ops.homo_add(downsampledchannels,homo_ops.homo_rotate(downsampledchannels,-8192,cryptoContext),cryptoContext)
    downsampledchannels = homo_ops.homo_add(downsampledchannels,
                                            homo_ops.homo_rotate(homo_ops.homo_rotate(downsampledchannels,-8192,cryptoContext), -8192, cryptoContext),
                                            cryptoContext)
    downsampledchannels.slots=8192

    return downsampledchannels


def downsample256to64(c1,c2,cryptoContext):

    num_slots = 8192 * 2
    c1.slots=16384
    c2.slots=16384
    fullpack=homo_ops.homo_add(homo_ops.homo_mul_pt(c1,mask_first_n(8192,c1.cur_limbs,cryptoContext),cryptoContext),homo_ops.homo_mul_pt(c2,mask_scecond_n(8192,c2.cur_limbs,cryptoContext),cryptoContext),cryptoContext)

    fullpack=homo_ops.homo_mul_pt(homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,1,cryptoContext),cryptoContext),gen_mask(2,fullpack.cur_limbs,cryptoContext),cryptoContext)
    fullpack = homo_ops.homo_mul_pt(
        homo_ops.homo_add(fullpack, homo_ops.homo_rotate(homo_ops.homo_rotate(fullpack,1,cryptoContext), 1, cryptoContext), cryptoContext),
        gen_mask(4, fullpack.cur_limbs, cryptoContext), cryptoContext)
    fullpack=homo_ops.homo_add(fullpack,homo_ops.homo_rotate(fullpack,4,cryptoContext),cryptoContext)

    masked = homo_ops.homo_mul(fullpack, mask_first_n_mod(16, 1024, 0, fullpack.cur_limbs, cryptoContext),
                               cryptoContext)
    temp=[0]*global_num_slots
    downsampledrows = openfhe_context.encrypt(temp, 1, cryptoContext.L - masked.cur_limbs,global_num_slots)
    for i in range(32):

        masked=homo_ops.homo_mul_pt(fullpack,mask_first_n_mod2(8,256,i,fullpack.cur_limbs,cryptoContext),cryptoContext)
        downsampledrows=homo_ops.homo_add(downsampledrows,masked,cryptoContext)
        if i<31:
            fullpack=homo_ops.homo_rotate(fullpack,32-8,cryptoContext)
    masked = homo_ops.homo_mul(downsampledrows, mask_channel(0, downsampledrows.cur_limbs, cryptoContext),
                               cryptoContext)
    temp=[0]*global_num_slots
    downsampledchannels = openfhe_context.encrypt(temp, 1, cryptoContext.L - masked.cur_limbs,global_num_slots)
    for i in range(64):

        masked=homo_ops.homo_mul_pt(downsampledrows,mask_channel2(i,downsampledrows.cur_limbs,cryptoContext),cryptoContext)
        downsampledchannels=homo_ops.homo_add(downsampledchannels,masked,cryptoContext)
        downsampledchannels=homo_ops.homo_rotate(downsampledchannels,-(256-64),cryptoContext)

    downsampledchannels=homo_ops.homo_rotate(downsampledchannels,(256-64)*64,cryptoContext)
    downsampledchannels=homo_ops.homo_add(downsampledchannels,homo_ops.homo_rotate(downsampledchannels,-4096,cryptoContext),cryptoContext)
    downsampledchannels = homo_ops.homo_add(downsampledchannels,
                                            homo_ops.homo_rotate(homo_ops.homo_rotate(downsampledchannels,-4096,cryptoContext), -4096, cryptoContext),
                                            cryptoContext)
    downsampledchannels.slotes=4096

    return downsampledchannels
