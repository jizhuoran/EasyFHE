import itertools
import subprocess
from pathlib import Path
import csv
import torch.fhe as fhe
import numpy as np
import torch
import os,time
import shutil

from triton.profiler.flags import command_line

# 备注：
DATA_DIR = os.environ["DATA_DIR"]

from utils import *
import examples.resnet20.convs
from examples.utils.approx import eval_chebyshev_function

global_num_slots = 1<<14
origin_input_folder = "../src/tmp_embeddings/"
input_folder="../src/tmp_embeddings/"
logBsSlots_list = [14] # todo: do not use global variable
def read_fc_weight(filename):
    weight=read_values_from_file("../weights/fc.bin")
    weight_corrected=[]
    for i in range(64):
        for j in range(10):
            weight_corrected.append(weight[(10*i)+j])
            for j in range(54):
                weight_corrected.append(0)
    return weight_corrected

def log2_int(x):
    import math

    return int(math.log2(x))

def rotsum(input,slots,padding,cryptoContext):
    result=input.deep_copy()
    for i in range(log2_int(slots)):
        result=fhe.homo_add(result,fhe.homo_rotate(result,int(padding*pow(2,i)),cryptoContext),cryptoContext)
    return result

def matmulRE1(rows,weight,bias,cryptoContext):
    columns=[]
    for i in range(len(rows)):
        m=fhe.homo_mul_pt(rows[i],weight,cryptoContext)
        m=rotsum(m,128,128,cryptoContext)
        if bias is not None:
            m=fhe.homo_add_pt(m,bias,cryptoContext)
        columns.append(m)
    return columns

def matmulRE2(rows,weight,row_size,padding,cryptoContext):
    columns=[]
    for i in range(len(rows)):
        m=fhe.homo_mul(rows[i],weight,cryptoContext)
        m=rotsum(m,row_size,padding,cryptoContext)
        columns.append(m)
    return columns

def wrapUpRepeated(vectors,cryptoContext):
    masked=[]
    for i in range(len(vectors)):
        masked.append(mask_block(vectors[i],128*i,128*(i+1),1,cryptoContext))
    return examples.resnet20.convs.eval_add_many(masked,cryptoContext)

def matmulCR1(rows,matrix,cryptoContext):
    columns=[]
    for i in range(len(rows)):
        m=fhe.homo_mul(rows[i],matrix,cryptoContext)
        m=rotsum(m,64,1,cryptoContext)
        columns.append(m)
    return columns


def matmulCR2(rows,weight,bias,cryptoContext):
    columns=[]
    for i in range(len(rows)):
        m=fhe.homo_mul_pt(rows[i],weight,cryptoContext)
        m=rotsum(m,128,1,cryptoContext)
        if bias is not None:
            m=fhe.homo_add_pt(m,bias,cryptoContext)
        columns.append(m)
    return columns



def matmulScores(queries,key,cryptoContext):
    scores=matmulCR1(queries,key,cryptoContext)
    r=1/8.0
    scores_wrapped=mask_heads(scores[len(scores)-1],1/8.0*r,cryptoContext)
    scores_wrapped=fhe.homo_rotate(scores_wrapped,-1,cryptoContext)
    for i in range(len(scores) - 2, -1, -1):
        scores_wrapped=fhe.homo_add(scores_wrapped,mask_heads(scores[i],1/8.0*r,cryptoContext),cryptoContext)
        if i >0:
            scores_wrapped=fhe.homo_rotate(scores_wrapped,-1,cryptoContext)
    return scores_wrapped

func = lambda x: 1 / x
def eval_inverse_naive(c,min,max,cryptoContext):
    return eval_chebyshev_function(func,c,min,max,119,cryptoContext)


def repeat(input,slots,cryptoContext,padding=1):
    res=input.deep_copy()
    for i in range(log2_int(slots)):
        res=fhe.homo_add(res,fhe.homo_rotate(res,-pow(2,i)*padding,cryptoContext),cryptoContext)
    return res


def unwrap_scores_expanded(c,inputs_num,cryptoContext):
    result=[]
    for i in range(inputs_num):
        i_th_1=mask_mod_n(c,128,0,inputs_num*128,cryptoContext)
        i_th_2=mask_mod_n(c,128, 64,inputs_num*128,cryptoContext)
        i_th_1=repeat(i_th_1,64,cryptoContext)
        i_th_2 = repeat(i_th_2, 64, cryptoContext)
        if i <inputs_num-1:
            c=fhe.homo_rotate(c,1,cryptoContext)
        result.append(fhe.homo_add(i_th_1,i_th_2,cryptoContext))
    return result

def mask_mod_n2(ciphertext, n, cryptoContext):
    num_slots = ciphertext.slots # todo: check if they are equal
    mask = []
    for i in range(num_slots):
        if i%n==0:
            mask.append(1.0)
        else:
            mask.append(0.0)
    mask = torch.tensor(mask, dtype=torch.float64).cuda()
    pt = fhe.encode(mask,1, cryptoContext.L - ciphertext.cur_limbs, num_slots, False, cryptoContext)
    return fhe.homo_mul_pt(ciphertext, pt, cryptoContext)


def wrapUpExpanded(vectors,cryptoContext):
    masked=mask_mod_n2(vectors[-1],128,cryptoContext)

    if len(vectors) > 1:
        masked = fhe.homo_rotate(masked, -1, cryptoContext)

    for i in range(len(vectors)-2,-1,-1):
        masked=fhe.homo_add(masked,mask_mod_n2(vectors[i],128,cryptoContext),cryptoContext)
        if i>0:
            masked=fhe.homo_rotate(masked,-1,cryptoContext)
    return masked

def unwrapExpanded(c,inputs_num,cryptoContext):
    result=[]
    for i in range(inputs_num):
        out=mask_mod_n(c,128,0,inputs_num*128,cryptoContext)
        out=repeat(out,128,cryptoContext)
        if i<inputs_num-1:
            c=fhe.homo_rotate(c,1,cryptoContext)
        result.append(out)
    return result


def matmulRElarge(inputs,weights,bias,mask_val,cryptoContext):
    densed=[]
    for i in range(len(inputs)):
        for j in range(len(weights)-1,-1,-1):
            out=fhe.homo_mul_pt(inputs[i],weights[j],cryptoContext)
            out=rotsum(out,128,128, cryptoContext)
            out=mask_first_n(out,128,mask_val,cryptoContext)
            if j ==len(weights)-1:
                i_th_result=out
            else:
                i_th_result=fhe.homo_rotate(i_th_result,-64,cryptoContext)
                i_th_result=fhe.homo_rotate(i_th_result,-64,cryptoContext)
                i_th_result=fhe.homo_add(i_th_result,out,cryptoContext)
        i_th_result=fhe.homo_add_pt(i_th_result,bias,cryptoContext)
        densed.append(i_th_result)
    return densed

def wrap_containers(c,inputs_number,cryptoContext):
    result=c[0]
    for i in range(inputs_number):
        result=fhe.homo_rotate(result,-512,cryptoContext)
        result=fhe.homo_add(result,c[i],cryptoContext)
    return result

def generate_containers(inputs,bias,cryptoContext):
    containers=[]
    quantities=[]
    for i in range(math.ceil(len(inputs)/32)):
        quantity=32
        if(i+1)*32>len(inputs):
            quantity=len(inputs)-(i*32)
        quantities.append(quantity)
        sliced_inputs=slicing(inputs,i*32,(i+1)*32)
        sliced_inputs.reverse()
        partial_container=wrap_containers(sliced_inputs,quantity,cryptoContext)
        if bias is not None:
            partial_container=fhe.homo_add_pt(partial_container)
        containers.append(partial_container)
    return containers

def slicing(arr,X,Y):
    if(Y-X>=len(arr)):
        return arr
    if Y>len(arr):
        Y=len(arr)
    result=arr[X:Y]
    return result
import math


def eval_gelu_function(c,min,max,mult,degree,cryptoContext):
    def custom_function(x):
        return 0.5 * (x * (1 / mult)) * (1 + math.erf((x * (1 / mult)) / 1.41421356237))
    return eval_chebyshev_function(custom_function,c,min,max,degree,cryptoContext)

def unwrap_512_in_4_128(c,index,cryptoContext):
    result=[]
    shift=index*512
    score1 =mask_block(c,shift+0,shift+128,1,cryptoContext)
    score1=repeat(score1,128,cryptoContext,-128)
    score2 = mask_block(c, shift + 128, shift + 256, 1, cryptoContext)
    score2 = repeat(score2, 128, cryptoContext, -128)
    score3 =mask_block(c,shift+256,shift+384,1,cryptoContext)
    score3=repeat(score3,128,cryptoContext,-128)
    score4 =mask_block(c,shift+384,shift+512,1,cryptoContext)
    score4=repeat(score4,128,cryptoContext,-128)
    result.append(score1)
    result.append(score2)
    result.append(score3)
    result.append(score4)
    return result


def unwrapRepeatedLarge(containers,input_number,cryptoContext):
    unwrapped_output = []
    quantities=[]
    for i in range(math.ceil(input_number/32)):
        quantity=32
        if(i+1)*32>input_number:
            quantity=input_number-(i*32)
        quantities.append(quantity)
    for i in range(len(containers)):
        for j in range(quantities[i]):
            unwrapped_container = unwrap_512_in_4_128(containers[i], j, cryptoContext)  # returns list
            unwrapped_output.append(unwrapped_container)  # keep list-of-list
    return unwrapped_output

def matmulCRlarge(rows,weights,bias,cryptoContext):
    output=[]
    for i in range(len(rows)):
        p1=fhe.homo_mul_pt(rows[i][0],weights[0],cryptoContext)
        p2=fhe.homo_mul_pt(rows[i][1],weights[1],cryptoContext)
        p3=fhe.homo_mul_pt(rows[i][2],weights[2],cryptoContext)
        p4=fhe.homo_mul_pt(rows[i][3],weights[3],cryptoContext)
        res=examples.resnet20.convs.eval_add_many([p1,p2,p3,p4],cryptoContext)
        res=rotsum(res,128,1,cryptoContext)
        if bias is not None:
            res=fhe.homo_add_pt(res,bias,cryptoContext)
        output.append(res)
    return output


def encoder1(cryptoContext,openfhe_context):
    inputs_count=0
    p1 = Path(input_folder)
    inputs_count = sum(1 for _ in p1.iterdir())
    inputs=[]
    for i in range(inputs_count):
        inputs.append(read_expanded_input(openfhe_context,f"{input_folder}input_{i}.txt",global_num_slots))
    query_w=read_plain_input(cryptoContext,"../weights-sst2/layer0_attself_query_weight.txt",0,1,global_num_slots)
    query_b=read_plain_repeated_input(cryptoContext,"../weights-sst2/layer0_attself_query_bias.txt",0,1,1.0,global_num_slots)

    key_w=read_plain_input(cryptoContext,"../weights-sst2/layer0_attself_key_weight.txt",0,1,global_num_slots)
    key_b=read_plain_repeated_input(cryptoContext,"../weights-sst2/layer0_attself_key_bias.txt",0, 1, 1.0,global_num_slots)
    Q=matmulRE1(inputs,query_w,query_b,cryptoContext)
    K=matmulRE1(inputs,key_w,key_b,cryptoContext)
    K_wrapped=wrapUpRepeated(K,cryptoContext)
    scores=matmulScores(Q,K_wrapped,cryptoContext)
    scores=eval_exp(scores,len(inputs),cryptoContext)
    scores_sum=rotsum(scores,128,128,cryptoContext)
    scores_denominator= eval_inverse_naive(scores_sum,2,5000,cryptoContext)
    scores=fhe.homo_mul(scores,scores_denominator,cryptoContext)
    unwrapped_scores=unwrap_scores_expanded(scores, len(inputs), cryptoContext)
    value_w=read_plain_input(cryptoContext,"../weights-sst2/layer0_attself_value_weight.txt",cryptoContext.L-inputs[0].cur_limbs,1,global_num_slots)
    value_b=read_plain_repeated_input(cryptoContext,"../weights-sst2/layer0_attself_value_bias.txt",cryptoContext.L-inputs[0].cur_limbs,1,1.0,global_num_slots)
    V=matmulRE1(inputs,value_w,value_b,cryptoContext) #fixme: 250410, yhh debug uptil here #为什么上面的value_w的level，把cryptoContext.L-inputs[0].cur_limbs换成 cryptoContext.L-score.cur_limbs-2 可以复现，进去之后第二个会g
    V_wrapped=wrapUpRepeated(V,cryptoContext)
    output=matmulRE2(unwrapped_scores,V_wrapped,128,128,cryptoContext)
    dense_w=read_plain_input(cryptoContext,"../weights-sst2/layer0_selfoutput_weight.txt",cryptoContext.L-output[0].cur_limbs-2,1,global_num_slots)
    dense_b=read_plain_expanded_input(cryptoContext,"../weights-sst2/layer0_selfoutput_bias.txt",cryptoContext.L-output[0].cur_limbs-2,1,global_num_slots)
    output=matmulCR2(output,dense_w,dense_b,cryptoContext)
    for i in range(len(output)):
        output[i]=fhe.homo_add(output[i],inputs[i],cryptoContext)
    wrappedOutput=wrapUpExpanded(output,cryptoContext)
    precomputed_mean=read_plain_repeated_input(cryptoContext,"../weights-sst2/layer0_selfoutput_mean.txt",cryptoContext.L-wrappedOutput.cur_limbs,1, -1.0, global_num_slots)
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,precomputed_mean,cryptoContext)
    vy=read_plain_input(cryptoContext,"../weights-sst2/layer0_selfoutput_vy.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots)
    wrappedOutput=fhe.homo_mul_pt(wrappedOutput,vy,cryptoContext)
    bias=read_plain_expanded_input(cryptoContext,"../weights-sst2/layer0_selfoutput_normbias.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,1.0, len(inputs))
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,bias,cryptoContext)
    wrappedOutput=fhe.homo_bootstrap(wrappedOutput,cryptoContext.L, logBsSlots_list[0], cryptoContext)
    output_copy=wrappedOutput.deep_copy()
    output=unwrapExpanded(wrappedOutput,len(inputs),cryptoContext)
    GELU_max_abs_value = 1 / 13.5
    intermediate_w_1=read_plain_input(cryptoContext,"../weights-sst2/layer0_intermediate_weight1.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    intermediate_w_2=read_plain_input(cryptoContext,"../weights-sst2/layer0_intermediate_weight2.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    intermediate_w_3=read_plain_input(cryptoContext,"../weights-sst2/layer0_intermediate_weight3.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    intermediate_w_4=read_plain_input(cryptoContext,"../weights-sst2/layer0_intermediate_weight4.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    dense_weights=[intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4]
    intermediate_bias=read_plain_input(cryptoContext,"../weights-sst2/layer0_intermediate_bias.txt",cryptoContext.L-output[0].cur_limbs-1,1,global_num_slots,GELU_max_abs_value)
    output=matmulRElarge(output,dense_weights,intermediate_bias, 1, cryptoContext)
    output=generate_containers(output,None,cryptoContext)
    for i in range(len(output)):
        output[i]=eval_gelu_function(output[i],-1,1,GELU_max_abs_value,119,cryptoContext)
        output[i]=fhe.homo_bootstrap(output[i],cryptoContext.L, logBsSlots_list[0], cryptoContext)
    unwrappedLargeOutput=unwrapRepeatedLarge(output,len(inputs),cryptoContext)
    output_w_1=read_plain_input(cryptoContext,"../weights-sst2/layer0_output_weight1.txt",cryptoContext.L-unwrappedLargeOutput[0][0].cur_limbs,1,global_num_slots)
    output_w_2=read_plain_input(cryptoContext,"../weights-sst2/layer0_output_weight2.txt",cryptoContext.L-unwrappedLargeOutput[0][0].cur_limbs,1,global_num_slots)
    output_w_3=read_plain_input(cryptoContext,"../weights-sst2/layer0_output_weight3.txt",cryptoContext.L-unwrappedLargeOutput[0][0].cur_limbs,1,global_num_slots)
    output_w_4=read_plain_input(cryptoContext,"../weights-sst2/layer0_output_weight4.txt",cryptoContext.L-unwrappedLargeOutput[0][0].cur_limbs,1,global_num_slots)
    output_bias=read_plain_expanded_input(cryptoContext,"../weights-sst2/layer0_output_bias.txt",cryptoContext.L-unwrappedLargeOutput[0][0].cur_limbs+1,1,global_num_slots)
    output=matmulCRlarge(unwrappedLargeOutput,[output_w_1, output_w_2, output_w_3, output_w_4],output_bias,cryptoContext)
    wrappedOutput=wrapUpExpanded(output,cryptoContext)
    wrappedOutput=fhe.homo_add(wrappedOutput,output_copy,cryptoContext)
    precomputed_mean=read_plain_repeated_input(cryptoContext,"../weights-sst2/layer0_output_mean.txt", cryptoContext.L-wrappedOutput.cur_limbs-1, 1, -1.0, global_num_slots)
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,precomputed_mean,cryptoContext)
    vy=read_plain_input(cryptoContext,"../weights-sst2/layer0_output_vy.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots)
    wrappedOutput=fhe.homo_mul_pt(wrappedOutput,vy,cryptoContext)
    bias=read_plain_expanded_input(cryptoContext,"../weights-sst2/layer0_output_normbias.txt",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,1.0, len(inputs))
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,bias,cryptoContext)
    output=unwrapExpanded(wrappedOutput,len(inputs),cryptoContext)
    #Todo:    controller.save(output, "../checkpoint/encoder1output.bin");保存序列化文件， 不保存也行
    return output




def eval_inverse_naive2(c,min,max,mult,cryptoContext):
    def custom_function2(x):
        return mult / x
    return eval_chebyshev_function(custom_function2,c,min,max,200,cryptoContext)


def encoder2(inputs,cryptoContext):
    query_w = read_plain_input(cryptoContext,"../weights-sst2/layer1_attself_query_weight.txt",cryptoContext.L-inputs[0].cur_limbs,1,global_num_slots)
    query_b = read_plain_repeated_input(cryptoContext,"../weights-sst2/layer1_attself_query_bias.txt",cryptoContext.L-inputs[0].cur_limbs,1, 1.0,global_num_slots)
    key_w = read_plain_input(cryptoContext,"../weights-sst2/layer1_attself_key_weight.txt",cryptoContext.L-inputs[0].cur_limbs,1,global_num_slots)
    key_b = read_plain_repeated_input(cryptoContext,"../weights-sst2/layer1_attself_key_bias.txt",cryptoContext.L-inputs[0].cur_limbs, 1, 1.0,global_num_slots)
    Q = matmulRE1(inputs, query_w, query_b, cryptoContext)
    K = matmulRE1(inputs, key_w, key_b, cryptoContext)
    K_wrapped = wrapUpRepeated(K, cryptoContext)
    scores = matmulScores(Q, K_wrapped, cryptoContext)
    scores=fhe.homo_bootstrap(scores,cryptoContext.L, logBsSlots_list[0], cryptoContext)
    scores = eval_exp(scores, len(inputs), cryptoContext)

    scores=fhe.homo_mul_scalar_double(scores,1/500.0,cryptoContext)
    scores=fhe.homo_bootstrap(scores,cryptoContext.L, logBsSlots_list[0], cryptoContext)
    scores=fhe.homo_mul_scalar_double(scores,500.0,cryptoContext)
    # scores=fhe.homo_mul_scalar_int(scores,500,cryptoContext) #todo: check in cpp if it is correct to use mul scalar int


    scores_sum = rotsum(scores, 128, 128, cryptoContext)
    scores_denominator = eval_inverse_naive2(scores_sum, 3,145000, 1,cryptoContext)
    scores_denominator=fhe.homo_bootstrap(scores_denominator,cryptoContext.L, logBsSlots_list[0], cryptoContext)

    scores = fhe.homo_mul(scores, scores_denominator, cryptoContext)
    unwrapped_scores = unwrap_scores_expanded(scores, len(inputs), cryptoContext)



    value_w = read_plain_input(cryptoContext,"../weights-sst2/layer1_attself_value_weight.txt", cryptoContext.L -inputs[0].cur_limbs,1,global_num_slots)
    value_b = read_plain_repeated_input(cryptoContext,"../weights-sst2/layer1_attself_value_bias.txt",
                                        cryptoContext.L - inputs[0].cur_limbs, 1, 1.0,global_num_slots )
    V = matmulRE1(inputs, value_w, value_b, cryptoContext)
    V_wrapped = wrapUpRepeated(V, cryptoContext)
    output = matmulRE2(unwrapped_scores, V_wrapped, 128, 128, cryptoContext)

    copyFirst=output[0].deep_copy()
    output = [copyFirst]  # Only keep CLS token, consistent with C++ encoder2

    dense_w = read_plain_input(cryptoContext,"../weights-sst2/layer1_selfoutput_weight.txt", cryptoContext.L - output[0].cur_limbs,1,global_num_slots)
    dense_b = read_plain_expanded_input(cryptoContext, "../weights-sst2/layer1_selfoutput_bias.txt", cryptoContext.L - output[0].cur_limbs + 1, 1, global_num_slots)
    output = matmulCR2(output, dense_w, dense_b, cryptoContext)
    for i in range(len(output)):
        output[i] = fhe.homo_add(output[i], inputs[i], cryptoContext)  # Residual add only uses inputs[0] (CLS)

    wrappedOutput = wrapUpExpanded(output, cryptoContext)
    precomputed_mean = read_plain_repeated_input(cryptoContext,"../weights-sst2/layer1_selfoutput_mean.txt",
                                                 cryptoContext.L - wrappedOutput.cur_limbs, 1, -1.0, global_num_slots)
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, precomputed_mean, cryptoContext)

    wrappedOutput = fhe.homo_bootstrap(wrappedOutput, cryptoContext.L, logBsSlots_list[0], cryptoContext)

    vy = read_plain_input(cryptoContext,"../weights-sst2/layer1_selfoutput_vy.txt", cryptoContext.L - wrappedOutput.cur_limbs,1,global_num_slots)
    wrappedOutput = fhe.homo_mul_pt(wrappedOutput, vy, cryptoContext)
    bias = read_plain_expanded_input(cryptoContext,"../weights-sst2/layer1_selfoutput_normbias.txt",
                                     cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots,1.0, len(inputs))
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, bias, cryptoContext)

    output_copy = wrappedOutput.deep_copy()
    output = unwrapExpanded(wrappedOutput, len(inputs), cryptoContext)
    GELU_max_abs_value = 1 / 17.0
    dense_weights = [
        read_plain_input(cryptoContext, f"../weights-sst2/layer1_intermediate_weight{i + 1}.txt", cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots, GELU_max_abs_value)
        for i in range(4)
    ]
    intermediate_bias = read_plain_input(cryptoContext, "../weights-sst2/layer1_intermediate_bias.txt", cryptoContext.L - output[0].cur_limbs + 1, 1, global_num_slots, GELU_max_abs_value)
    output = matmulRElarge(output, dense_weights, intermediate_bias, 1, cryptoContext)
    output = generate_containers(output, None, cryptoContext)
    for i in range(len(output)):
        output[i] = eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 59, cryptoContext)
        output[i] = fhe.homo_bootstrap(output[i], cryptoContext.L, logBsSlots_list[0],
                                       cryptoContext)
    unwrappedLargeOutput = unwrapRepeatedLarge(output, len(output), cryptoContext)

    output_weights = [
        read_plain_input(cryptoContext, f"../weights-sst2/layer1_output_weight{i + 1}.txt", cryptoContext.L - output[0].cur_limbs, 1, global_num_slots)
        for i in range(4)
    ]
    output_bias = read_plain_expanded_input(cryptoContext, "../weights-sst2/layer1_output_bias.txt", cryptoContext.L - output[0].cur_limbs + 1, 1, global_num_slots)
    output = matmulCRlarge(unwrappedLargeOutput, output_weights, output_bias, cryptoContext)
    wrappedOutput = wrapUpExpanded(output, cryptoContext)
    wrappedOutput = fhe.homo_add(wrappedOutput, output_copy, cryptoContext)
    precomputed_mean = read_plain_repeated_input(cryptoContext,"../weights-sst2/layer1_output_mean.txt",
                                                 cryptoContext.L - wrappedOutput.cur_limbs, 1,-1.0,global_num_slots)
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, precomputed_mean, cryptoContext)
    vy = read_plain_input(cryptoContext,"../weights-sst2/layer1_output_vy.txt", cryptoContext.L - wrappedOutput.cur_limbs, 1,global_num_slots)
    wrappedOutput = fhe.homo_mul_pt(wrappedOutput, vy, cryptoContext)
    bias = read_plain_expanded_input(cryptoContext,"../weights-sst2/layer1_output_normbias.txt",
                                     cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots,1.0, len(inputs))
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, bias, cryptoContext)
    output = unwrapExpanded(wrappedOutput, len(inputs), cryptoContext)
    # Todo:    controller.save(output, "../checkpoint/encoder1output.bin");保存序列化文件， 不保存也行
    return output[0]


def classifier(input,cryptoContext,openfhe_context):
    weight=read_plain_input(cryptoContext,"../weights-sst2/classifier_weight.txt",cryptoContext.L - input.cur_limbs,1,global_num_slots)
    bias=read_plain_expanded_input(cryptoContext,"../weights-sst2/classifier_bias.txt",cryptoContext.L - input.cur_limbs,1,global_num_slots)
    output=fhe.homo_add_pt(input, weight, cryptoContext)
    output=rotsum(output,128,1,cryptoContext)
    output=fhe.homo_add_pt(output, bias, cryptoContext)
    mask=[]
    for i in range(global_num_slots):
        mask.append(0)
    mask[0]=1
    mask[128]=1
    x = torch.tensor(mask, device="cuda")
    temp=openfhe_context.encrypt(x, 0, cryptoContext.L - output.cur_limbs, global_num_slots, False)
    output=fhe.homo_mul(output,temp,cryptoContext)

    output=fhe.homo_add(output,fhe.homo_rotate(fhe.homo_rotate(output,-1,cryptoContext),128,cryptoContext),cryptoContext)
    return output



def eval_tanh_function(c,min,max,mult,degree,cryptoContext):
    def tanh_function(x):
        return math.tanh(x * (1 / mult))
    return eval_chebyshev_function(tanh_function,c,min,max,degree,cryptoContext)

def pooler(input,cryptoContext,openfhe_context):
    tanhScale=1/30.0
    weight = read_plain_input(cryptoContext,"../weights-sst2/pooler_dense_weight.txt", cryptoContext.L - input.cur_limbs,1,global_num_slots,tanhScale)
    bias = read_plain_repeated_input(cryptoContext,"../weights-sst2/pooler_dense_bias.txt", cryptoContext.L - input.cur_limbs,1, tanhScale, global_num_slots)
    output = fhe.homo_mul_pt(input, weight, cryptoContext)
    output = rotsum(output, 128, 128, cryptoContext)
    output = fhe.homo_add_pt(output, bias, cryptoContext)
    output = fhe.homo_bootstrap(output, cryptoContext.L, logBsSlots_list[0],
                                   cryptoContext)
    output=eval_tanh_function(output,-1,1,tanhScale,300,cryptoContext)
    return output


def BERT_Tiny():
    #todo: add setup_environment function
    #  根据BERT-TINT C++版本的代码改写为Python版本
    #
    
    # text = input_text
    # setup_environment(text)
    global_num_slots = 1<<14

    # generate context
    levelsUsedBeforeBootstrap = 12+4
    rotate_index_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1, -2, -4, -8, -16, -32, -64]

    maxLevelsRemaining = 20
    # logBsSlots_list = [14]
    logN = 15
    dnum = 4
    dcrtBits = 57 #note there is a bootstrap scale up and down by 500
    firstMod = 60
    levelBudget_list = [[4,4]]
    secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, save_dir=DATA_DIR,
                             config=config))

    cryptoContext.PRELOAD_ALL = True  # poor workaround, should be fixed in the future, need to be set to False/True now

    print("Context Done")

    # todo: copy res-20 work around implementation
    # encode_weight_path = (
    #     DATA_DIR
    #     + "/weight.pkl"
    # )
    #
    # load_weight(encode_weight_path, cryptoContext)

    print("\nSERVER-SIDE\nThe evaluation of the circuit started.")

    start = time.time()

    if not os.path.exists(input_folder):
        raise ValueError(f"Directory {input_folder} does not exist!")

    encoder1output=[]
    encoder2output=None

    encoder1output = encoder1(cryptoContext, openfhe_context)
    # encoder1output = controller.load_vector("../checkpoint/encoder1output.bin"); #todo: we dont save checkpoint now, therefore omit deserialization
    encoder2output = encoder2(encoder1output,cryptoContext)

    pooled = pooler(encoder2output, cryptoContext, openfhe_context)

    #todo: use homo_classifier in the future

    try:
        plain_pooled = openfhe_context.decrypt(pooled)
        plain_pooled = plain_pooled.cpu().numpy().reshape(-1)
    except RuntimeError as e:
        print(f"Decryption failed: {e}")
        plain_pooled = None

    # implement the plain classifier and post process here
    # plain_pooled generated above is available
    # 实现 pooler->解密->classified->判断
    classifier_result = classifier_tensor(plain_pooled)
    # 判断逻辑
    print("Outcome: ", end='')
    if classifier_result[0].item() > classifier_result[1].item():
        print(f"\033[92mnegative\033[0m sentiment!")  # 使用ANSI颜色代码
    else:
        print(f"\033[92mpositive\033[0m sentiment!")



def classifier_tensor(input):
    """
    本函数实现明文状态下的分类器，输入输出均为double数组
    :return:
    C++源代码如下：
    vector<double> weight = controller.read_plain_input_vector("../weights-sst2/classifier_weight.txt");
    vector<double> bias = controller.read_plain_expanded_input_vector("../weights-sst2/classifier_bias.txt");
    vector<double> output = controller.mult(input, weight);
    output = controller.rotsum(output, 128, 1);
    output = controller.add(output, bias);
    std::cout<<"done3!"<<std::endl;
    vector<double> mask;
    for (int i = 0; i < controller.num_slots; i++) {
        mask.push_back(0);
    }
    mask[0] = 1;
    mask[128] = 1;
    output = controller.mult(output, mask);
    std::cout<<"done4!"<<std::endl;
    output = controller.add(output, controller.rotate(controller.rotate(output, -1), 128));
    return output;
    """
    #1.读取文件
    weight = read_plain_input_tensor("../weights-sst2/classifier_weight.txt")
    bias = read_plain_expanded_input_tensor("../weights-sst2/classifier_bias.txt")

    if isinstance(input, np.ndarray):
        input = torch.tensor(input, dtype=torch.float64).cuda()

    output = torch.mul(input, weight)
    output = rotsum_tensor(output, 128, 1)
    output =torch.add(output,bias)
    mask = torch.zeros(global_num_slots, dtype=torch.double)
    # 设置索引0和128的位置为1
    mask[0] = 1.0
    mask[128] = 1.0
    output = torch.mul(output, mask)
    output = torch.add(output, rotate_tensor(rotate_tensor(output, -1), 128))
    return output

def rotsum_tensor(input,slots, padding):
    result=input
    for i in range(log2_int(slots)):
        temp  = rotate_tensor(result, padding * pow(2, i))
        result = torch.add(result, temp)
    return result

def rotate_tensor(input, shift):
    assert input.dim() == 1, "输入必须是1维张量"
    length = input.size(0)
    if length == 0:
        return input.clone()
    # 计算有效移位量（处理负数和越界）
    effective_shift = shift % length
    if effective_shift == 0:
        return input.clone()
    # 实现循环移位
    return torch.cat([
        input[effective_shift:],  # 后半部分
        input[:effective_shift]  # 前半部分
    ])

def read_plain_input_tensor(filename,scale=1):
    input=read_values_from_file(filename)
    size=len(input)
    if scale!=1:
        for i in range(size):
            input[i]=input[i]*scale

    cycled_input = itertools.cycle(input)
    temp = [next(cycled_input) for _ in range(global_num_slots)]
    x = torch.tensor(temp, dtype=torch.float64, device="cuda")
    return x

def read_plain_expanded_input_tensor(filename):
    input_values = read_values_from_file(filename)
    # 扩展阶段
    repeated = [val for val in input_values for _ in range(128)]
    x = torch.tensor(repeated, dtype=torch.float64, device="cuda")
    return  x


# def setup_environment(text:str):
#     """
#     本函数为BERT-TINY复现项目下从C++版本移植为Python版本：
#     C++版本如下：
#     void setup_environment(string text) {
#         string command;
#         filesystem::remove_all("../src/tmp_embeddings");
#         system("mkdir ../src/tmp_embeddings");
#         input_folder = "../src/tmp_embeddings/";
#         text = "[CLS] " + text + " [SEP]";
#         cout << "\nCLIENT-SIDE\nTokenizing the following sentence: '" << text << "'" << endl;
#         command = "python3 ../src/python/ExtractEmbeddings.py \"" + text + "\"";
#         system(command.c_str());
#         verbose = false;
#         return;
#     """
#     # Todo：修改脚本目录
#     # 1. 清理并重建临时目录
#     tmp_dir = "../src/tmp_embeddings"
#     if os.path.exists(tmp_dir):
#         shutil.rmtree(tmp_dir)  # 递归删除目录
#     os.makedirs(tmp_dir, exist_ok=True)  # 自动创建多级目录
#     input_folder = tmp_dir + "/"
#     # 2. 文本预处理
#     processed_text = f"[CLS] {text} [SEP]"
#     print(f"\nCLIENT-SIDE\nTokenizing the following sentence: '{processed_text}'")
#     # 3. 调用Python脚本
#     ExtractEmbeddings(processed_text)
#     # script_path = "../src/python/ExtractEmbeddings.py"
#     # command = ["python3", script_path, processed_text]
#     # try:
#     #     # 使用subprocess.run调用脚本
#     #     result = subprocess.run(
#     #         command,
#     #         check=True,
#     #         capture_output=True,
#     #         text=True
#     #     )
#     #     if result.returncode != 0:
#     #         print(f"Script execution failed: {result.stderr}")
#     #
#     # except subprocess.CalledProcessError as e:
#     #
#     #     print(f"Error executing script: {str(e)}")


if __name__ == "__main__":
    data_dirs = [entry.name for entry in os.scandir('./tmp_embeddings') if entry.is_dir()]
    data_dirs.sort(key=int)
    for data_dir in data_dirs:
        input_folder = origin_input_folder+data_dir+'/'
        BERT_Tiny()
