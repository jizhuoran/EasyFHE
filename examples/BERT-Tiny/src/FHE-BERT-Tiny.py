import itertools
import subprocess
from pathlib import Path
import easyfhe.fhe as fhe
import os,time, datetime
from utils import *
from examples.utils.approx import eval_chebyshev_function
from triton.profiler.flags import command_line
import math

DATA_DIR = os.environ["DATA_DIR"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FOLDER = SCRIPT_DIR / "tmp_embeddings" / "0"
input_folder = DEFAULT_INPUT_FOLDER
global_num_slots = 1<<14
# origin_input_folder = "../src/tmp_embeddings/"
# input_folder="src/tmp_embeddings/"
logBsSlots_list = [14]
levelBudget_list = [[4,4]]

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
    return eval_add_many(masked,cryptoContext)

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


def eval_inverse_naive(c,min,max,cryptoContext):
    func = lambda x: 1 / x
    return eval_chebyshev_function(func,c,min,max,119,cryptoContext)

def repeat(input,slots,cryptoContext,padding=1):
    res=input.deep_copy()
    for i in range(log2_int(slots)):
        res=fhe.homo_add(res,fhe.homo_rotate(res,-pow(2,i)*padding,cryptoContext),cryptoContext)
    return res

def unwrap_scores_expanded(c,inputs_num,cryptoContext):
    result=[]
    for i in range(inputs_num):
        i_th_1= mask_mod_n(c, 128, 0, cryptoContext)
        i_th_2= mask_mod_n(c, 128, 64, cryptoContext)
        i_th_1=repeat(i_th_1,64,cryptoContext)
        i_th_2 = repeat(i_th_2, 64, cryptoContext)
        if i <inputs_num-1:
            c=fhe.homo_rotate(c,1,cryptoContext)
        result.append(fhe.homo_add(i_th_1,i_th_2,cryptoContext))
    return result


def wrapUpExpanded(vectors,cryptoContext):
    masked= mask_mod_n(vectors[-1], 128, 0, cryptoContext)
    if len(vectors) > 1:
        masked = fhe.homo_rotate(masked, -1, cryptoContext)
    for i in range(len(vectors)-2,-1,-1):
        masked=fhe.homo_add(masked, mask_mod_n(vectors[i], 128, 0, cryptoContext), cryptoContext)
        if i>0:
            masked=fhe.homo_rotate(masked,-1,cryptoContext)
    return masked

def unwrapExpanded(c,inputs_num,cryptoContext):
    result=[]
    for i in range(inputs_num):
        out= mask_mod_n(c, 128, 0, cryptoContext)
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
        res=eval_add_many([p1,p2,p3,p4],cryptoContext)
        res=rotsum(res,128,1,cryptoContext)
        if bias is not None:
            res=fhe.homo_add_pt(res,bias,cryptoContext)
        output.append(res)
    return output


def encoder1(cryptoContext,openfhe_context):

    p1 = Path(input_folder)

    inputs_count = sum(1 for _ in p1.iterdir())

    print(f"{inputs_count} inputs were found")

    inputs=[]
    for i in range(inputs_count):
        inputs.append(
            read_expanded_input(
                cryptoContext,
                openfhe_context,
                str(p1 / f"input_{i}.txt"),
                0,
                1,
                global_num_slots,
            )
        )

    query_w=read_plain_input(cryptoContext,"layer0_attself_query_weight",0,1,global_num_slots)
    query_b= read_plain_repeated_input(cryptoContext, "layer0_attself_query_bias", 0, 1, global_num_slots, 1.0)
    key_w=read_plain_input(cryptoContext,"layer0_attself_key_weight",0,1,global_num_slots)
    key_b= read_plain_repeated_input(cryptoContext, "layer0_attself_key_bias", 0, 1, global_num_slots, 1.0)

    Q=matmulRE1(inputs,query_w,query_b,cryptoContext)
    K=matmulRE1(inputs,key_w,key_b,cryptoContext)

    K_wrapped=wrapUpRepeated(K,cryptoContext)

    scores=matmulScores(Q,K_wrapped,cryptoContext)
    scores=eval_exp(scores,len(inputs),cryptoContext)

    scores_sum=rotsum(scores,128,128,cryptoContext)
    scores_denominator= eval_inverse_naive(scores_sum,2,5000,cryptoContext)

    scores=fhe.homo_mul(scores,scores_denominator,cryptoContext)

    unwrapped_scores=unwrap_scores_expanded(scores, len(inputs), cryptoContext)

    value_w=read_plain_input(cryptoContext,"layer0_attself_value_weight",cryptoContext.L-inputs[0].cur_limbs,1,global_num_slots)
    value_b= read_plain_repeated_input(cryptoContext, "layer0_attself_value_bias",
                                       cryptoContext.L - inputs[0].cur_limbs, 1, global_num_slots, 1.0)

    V=matmulRE1(inputs,value_w,value_b,cryptoContext)
    V_wrapped=wrapUpRepeated(V,cryptoContext)

    output=matmulRE2(unwrapped_scores,V_wrapped,128,128,cryptoContext)


    ############## The evaluation of Self-Attention Done #################

    dense_w=read_plain_input(cryptoContext,"layer0_selfoutput_weight",cryptoContext.L-output[0].cur_limbs-2,1,global_num_slots)
    dense_b=read_plain_expanded_input(cryptoContext,"layer0_selfoutput_bias",cryptoContext.L-output[0].cur_limbs-2,1,global_num_slots)

    output=matmulCR2(output,dense_w,dense_b,cryptoContext)

    for i in range(len(output)):
        output[i]=fhe.homo_add(output[i],inputs[i],cryptoContext)

    wrappedOutput=wrapUpExpanded(output,cryptoContext)

    precomputed_mean= read_plain_repeated_input(cryptoContext, "layer0_selfoutput_mean",
                                                cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots, -1.0)
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,precomputed_mean,cryptoContext)

    vy=read_plain_input(cryptoContext,"layer0_selfoutput_vy",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots)
    wrappedOutput=fhe.homo_mul_pt(wrappedOutput,vy,cryptoContext)
    bias=read_plain_expanded_input(cryptoContext,"layer0_selfoutput_normbias",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,1.0, len(inputs))
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,bias,cryptoContext)

    wrappedOutput=fhe.homo_bootstrap(wrappedOutput,cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    output_copy=wrappedOutput.deep_copy() # Required at the last layernorm

    output=unwrapExpanded(wrappedOutput,len(inputs),cryptoContext)

    ##################### The evaluation of Self-Output Done #####################


    GELU_max_abs_value = 1 / 13.5

    intermediate_w_1=read_plain_input(cryptoContext,"layer0_intermediate_weight1",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    intermediate_w_2=read_plain_input(cryptoContext,"layer0_intermediate_weight2",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    intermediate_w_3=read_plain_input(cryptoContext,"layer0_intermediate_weight3",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)
    intermediate_w_4=read_plain_input(cryptoContext,"layer0_intermediate_weight4",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,GELU_max_abs_value)

    dense_weights=[intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4]

    intermediate_bias=read_plain_input(cryptoContext,"layer0_intermediate_bias",cryptoContext.L-output[0].cur_limbs-1,1,global_num_slots,GELU_max_abs_value)

    output=matmulRElarge(output,dense_weights,intermediate_bias, 1, cryptoContext)

    output=generate_containers(output,None,cryptoContext)

    for i in range(len(output)):
        output[i]=eval_gelu_function(output[i],-1,1,GELU_max_abs_value,119,cryptoContext)
        output[i]=fhe.homo_bootstrap(output[i],cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    unwrappedLargeOutput=unwrapRepeatedLarge(output,len(inputs),cryptoContext)

    ##################### The evaluation of Intermediate Done #####################

    weight_files = [f"layer0_output_weight{i}" for i in range(1, 5)]
    cur_limbs = cryptoContext.L - unwrappedLargeOutput[0][0].cur_limbs

    output_weights = [
        read_plain_input(cryptoContext, path, cur_limbs, 1, global_num_slots)
        for path in weight_files
    ]

    output_bias = read_plain_expanded_input(
        cryptoContext,
        "layer0_output_bias",
        cur_limbs + 1,
        1,
        global_num_slots
    )

    output = matmulCRlarge(unwrappedLargeOutput, output_weights, output_bias, cryptoContext)

    wrappedOutput=wrapUpExpanded(output,cryptoContext)
    wrappedOutput=fhe.homo_add(wrappedOutput,output_copy,cryptoContext)

    precomputed_mean= read_plain_repeated_input(cryptoContext, "layer0_output_mean",
                                                cryptoContext.L - wrappedOutput.cur_limbs - 1, 1, global_num_slots,
                                                -1.0)
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,precomputed_mean,cryptoContext)

    vy=read_plain_input(cryptoContext,"layer0_output_vy",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots)
    wrappedOutput=fhe.homo_mul_pt(wrappedOutput,vy,cryptoContext)
    bias=read_plain_expanded_input(cryptoContext,"layer0_output_normbias",cryptoContext.L-wrappedOutput.cur_limbs,1,global_num_slots,1.0, len(inputs))
    wrappedOutput=fhe.homo_add_pt(wrappedOutput,bias,cryptoContext)

    output=unwrapExpanded(wrappedOutput,len(inputs),cryptoContext)

    #################### The evaluation of Output Done ####################

    return output




def eval_inverse_naive2(c,min,max,mult,cryptoContext):
    def custom_function2(x):
        return mult / x
    return eval_chebyshev_function(custom_function2,c,min,max,200,cryptoContext)


def encoder2(inputs,cryptoContext):
    query_w = read_plain_input(cryptoContext,"layer1_attself_query_weight",cryptoContext.L-inputs[0].cur_limbs,1,global_num_slots)
    query_b = read_plain_repeated_input(cryptoContext, "layer1_attself_query_bias",
                                        cryptoContext.L - inputs[0].cur_limbs, 1, global_num_slots, 1.0)
    key_w = read_plain_input(cryptoContext,"layer1_attself_key_weight",cryptoContext.L-inputs[0].cur_limbs,1,global_num_slots)
    key_b = read_plain_repeated_input(cryptoContext, "layer1_attself_key_bias", cryptoContext.L - inputs[0].cur_limbs,
                                      1, global_num_slots, 1.0)

    Q = matmulRE1(inputs, query_w, query_b, cryptoContext)
    K = matmulRE1(inputs, key_w, key_b, cryptoContext)

    K_wrapped = wrapUpRepeated(K, cryptoContext)

    scores = matmulScores(Q, K_wrapped, cryptoContext)

    scores=fhe.homo_bootstrap(scores,cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    scores = eval_exp(scores, len(inputs), cryptoContext)

    scores=fhe.homo_mul_scalar_double(scores,1/500.0,cryptoContext) # Here values are scaled down in order to achieve better accuracy with bootstrapping
    scores=fhe.homo_bootstrap(scores,cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    # use `homo_mul_scalar_int` instead of `homo_mul_scalar_double` to avoid overflow when computing `_get_element_for_eval_mult`
    # scores=fhe.homo_mul_scalar_double(scores,500.0,cryptoContext)
    scores=fhe.homo_mul_scalar_int(scores,500,cryptoContext)


    scores_sum = rotsum(scores, 128, 128, cryptoContext)

    scores_denominator = eval_inverse_naive2(scores_sum, 3,145000, 1,cryptoContext)

    scores_denominator=fhe.homo_bootstrap(scores_denominator,cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    scores = fhe.homo_mul(scores, scores_denominator, cryptoContext)

    unwrapped_scores = unwrap_scores_expanded(scores, len(inputs), cryptoContext)

    value_w = read_plain_input(cryptoContext,"layer1_attself_value_weight", cryptoContext.L -inputs[0].cur_limbs,1,global_num_slots)
    value_b = read_plain_repeated_input(cryptoContext, "layer1_attself_value_bias",
                                        cryptoContext.L - inputs[0].cur_limbs, 1, global_num_slots, 1.0)

    V = matmulRE1(inputs, value_w, value_b, cryptoContext)
    V_wrapped = wrapUpRepeated(V, cryptoContext)

    output = matmulRE2(unwrapped_scores, V_wrapped, 128, 128, cryptoContext)

    #################### The evaluation of Self-Attention Done ####################

    copyFirst=output[0].deep_copy()
    output = [copyFirst]  # Only keep CLS token, consistent with C++ encoder2

    dense_w = read_plain_input(cryptoContext,"layer1_selfoutput_weight", cryptoContext.L - output[0].cur_limbs,1,global_num_slots)
    dense_b = read_plain_expanded_input(cryptoContext, "layer1_selfoutput_bias", cryptoContext.L - output[0].cur_limbs + 1, 1, global_num_slots) # Bias do only 12 reps.

    output = matmulCR2(output, dense_w, dense_b, cryptoContext)
    for i in range(len(output)):
        output[i] = fhe.homo_add(output[i], inputs[i], cryptoContext)  # Residual add only uses inputs[0] (CLS)

    wrappedOutput = wrapUpExpanded(output, cryptoContext)
    precomputed_mean = read_plain_repeated_input(cryptoContext, "layer1_selfoutput_mean",
                                                 cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots, -1.0)
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, precomputed_mean, cryptoContext)

    wrappedOutput = fhe.homo_bootstrap(wrappedOutput, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0], cryptoContext)

    vy = read_plain_input(cryptoContext,"layer1_selfoutput_vy", cryptoContext.L - wrappedOutput.cur_limbs,1,global_num_slots)
    wrappedOutput = fhe.homo_mul_pt(wrappedOutput, vy, cryptoContext)
    bias = read_plain_expanded_input(cryptoContext,"layer1_selfoutput_normbias",
                                     cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots,1.0, len(inputs))
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, bias, cryptoContext)

    output_copy = wrappedOutput.deep_copy() # Required at the last layernorm

    output = unwrapExpanded(wrappedOutput, len(inputs), cryptoContext)

    ###################### The evaluation of Self-Output Done ######################

    GELU_max_abs_value = 1 / 17.0
    dense_weights = [
        read_plain_input(cryptoContext, f"layer1_intermediate_weight{i + 1}", cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots, GELU_max_abs_value)
        for i in range(4)
    ]
    intermediate_bias = read_plain_input(cryptoContext, "layer1_intermediate_bias", cryptoContext.L - output[0].cur_limbs + 1, 1, global_num_slots, GELU_max_abs_value)
    output = matmulRElarge(output, dense_weights, intermediate_bias, 1, cryptoContext)
    output = generate_containers(output, None, cryptoContext)
    for i in range(len(output)):
        output[i] = eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 59, cryptoContext)
        output[i] = fhe.homo_bootstrap(output[i], cryptoContext.L, logBsSlots_list[0], levelBudget_list[0],
                                       cryptoContext)
    unwrappedLargeOutput = unwrapRepeatedLarge(output, len(output), cryptoContext)

    ######################## The evaluation of Intermediate Done ########################

    output_weights = [
        read_plain_input(cryptoContext, f"layer1_output_weight{i + 1}", cryptoContext.L - output[0].cur_limbs, 1, global_num_slots)
        for i in range(4)
    ]
    output_bias = read_plain_expanded_input(cryptoContext, "layer1_output_bias", cryptoContext.L - output[0].cur_limbs + 1, 1, global_num_slots)
    output = matmulCRlarge(unwrappedLargeOutput, output_weights, output_bias, cryptoContext)
    wrappedOutput = wrapUpExpanded(output, cryptoContext)
    wrappedOutput = fhe.homo_add(wrappedOutput, output_copy, cryptoContext)
    precomputed_mean = read_plain_repeated_input(cryptoContext, "layer1_output_mean",
                                                 cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots, -1.0)
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, precomputed_mean, cryptoContext)
    vy = read_plain_input(cryptoContext,"layer1_output_vy", cryptoContext.L - wrappedOutput.cur_limbs, 1,global_num_slots)
    wrappedOutput = fhe.homo_mul_pt(wrappedOutput, vy, cryptoContext)
    bias = read_plain_expanded_input(cryptoContext,"layer1_output_normbias",
                                     cryptoContext.L - wrappedOutput.cur_limbs, 1, global_num_slots,1.0, len(inputs))
    wrappedOutput = fhe.homo_add_pt(wrappedOutput, bias, cryptoContext)
    output = unwrapExpanded(wrappedOutput, len(inputs), cryptoContext)

    ############################ The evaluation of Output Done ############################
    return output[0]


def classifier(input, openfhe_context, cryptoContext):
    weight=read_plain_input(cryptoContext,"classifier_weight",cryptoContext.L - input.cur_limbs,1,global_num_slots)
    bias=read_plain_expanded_input(cryptoContext,"classifier_bias",cryptoContext.L - input.cur_limbs,1,global_num_slots)
    output=fhe.homo_mul_pt(input, weight, cryptoContext)

    output=rotsum(output,128,1,cryptoContext)
    output=fhe.homo_add_pt(output, bias, cryptoContext)

    mask=[]
    for i in range(global_num_slots):
        mask.append(0)
    mask[0]=1
    mask[128]=1
    temp=openfhe_context.encrypt(mask, cryptoContext.device, 1, cryptoContext.L - output.cur_limbs, global_num_slots)
    output=fhe.homo_mul(output,temp,cryptoContext)
    output=fhe.homo_add(output,fhe.homo_rotate(fhe.homo_rotate(output,-1,cryptoContext),128,cryptoContext),cryptoContext)
    return output



def eval_tanh_function(c,min,max,mult,degree,cryptoContext):
    def tanh_function(x):
        return math.tanh(x * (1 / mult))
    return eval_chebyshev_function(tanh_function,c,min,max,degree,cryptoContext)

def pooler(input,cryptoContext,openfhe_context):
    tanhScale=1/30.0
    weight = read_plain_input(cryptoContext,"pooler_dense_weight", cryptoContext.L - input.cur_limbs,1,global_num_slots,tanhScale)
    bias = read_plain_repeated_input(cryptoContext, "pooler_dense_bias", cryptoContext.L - input.cur_limbs, 1,
                                     global_num_slots, tanhScale)
    output = fhe.homo_mul_pt(input, weight, cryptoContext)
    output = rotsum(output, 128, 128, cryptoContext)
    output = fhe.homo_add_pt(output, bias, cryptoContext)
    output = fhe.homo_bootstrap(output, cryptoContext.L, logBsSlots_list[0], levelBudget_list[0],
                                   cryptoContext)
    output=eval_tanh_function(output,-1,1,tanhScale,300,cryptoContext)
    return output


def BERT_Tiny():

    if not os.path.exists(input_folder):
        raise ValueError(f"Directory {input_folder} does not exist!")

    # generate context
    # levelsUsedBeforeBootstrap = 12+4
    rotate_index_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1, -2, -4, -8, -16, -32, -64]

    maxLevelsRemaining = 20
    # logBsSlots_list = [14]
    logN = 15
    dnum = 4
    dcrtBits = 57 #note there is a bootstrap scale up and down by 500
    firstMod = 60
    # levelBudget_list = [[4,4]]
    secretKeyDist = "SPARSE_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
    device = "cuda"
    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True,
                                     SAVE_MIDDLE=False)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, device, save_dir=DATA_DIR,
                             config=config))
    print("Context Done")


    cryptoContext.cnt = 0

    cryptoContext.pre_encode_type = "middle"
    encode_weight_path = "/data/yhh/data/encode_20250607_213413.pkl" # on h100-zrji
    load_weight(encode_weight_path, cryptoContext)

    print("\nSERVER-SIDE\nThe evaluation of the circuit started.")

    print("current time", datetime.datetime.now())
    start = time.time()

    encoder1output = encoder1(cryptoContext, openfhe_context)
    encoder2output = encoder2(encoder1output,cryptoContext)
    pooled = pooler(encoder2output, cryptoContext, openfhe_context)
    result = classifier(pooled, openfhe_context, cryptoContext)

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    try:
        result = openfhe_context.decrypt(result)
    except RuntimeError as e:
        print(f"Decryption failed: {e}")
        result = None

    print("Outcome: ", end='')
    if result[0].item() > result[1].item():
        print(f"\033[92mnegative\033[0m sentiment!")  # use ANSI color
    else:
        print(f"\033[92mpositive\033[0m sentiment!")



def classifier_tensor(input):
    """
    plain classifier, input and output are both double list
    :return:
    C++源代码如下：
    vector<double> weight = controller.read_plain_input_vector("classifier_weight");
    vector<double> bias = controller.read_plain_expanded_input_vector("classifier_bias");
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
    weight = read_plain_input_tensor("classifier_weight")
    bias = read_plain_expanded_input_tensor("classifier_bias")

    if isinstance(input, np.ndarray):
        input = torch.tensor(input, dtype=torch.float64).cuda()
    output = torch.mul(input, weight)

    output = rotsum_tensor(output, 128, 1)

    output =torch.add(output,bias)

    mask = torch.zeros(global_num_slots, dtype=torch.double).to(device=output.device)
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
    repeat = []
    for value in input_values:
        for i in range(128):
            repeat.append(value)
    for i in range(128-len(input_values)):
        for j in range(128):
            repeat.append(0)
    x = torch.tensor(repeat, dtype=torch.float64, device="cuda")
    return  x


if __name__ == "__main__":
    input_folder = DEFAULT_INPUT_FOLDER  # "this is a bad movie"
    BERT_Tiny()
