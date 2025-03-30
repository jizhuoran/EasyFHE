import time
import torch.fhe as fhe
from torch.fhe.bootstrapping import eval_bootstrap
import numpy as np
import torch
import torch.fhe.client.openfhe as openfhe
from torch.fhe.ciphertext import Cipher
import random
import math
import os


maxLevelsRemaining = 26
appRotIndex_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
logBsSlots_list = [8] # todo: should be 8 for mnist data
logN = 16
dnum = 3
dcrtBits = 46
firstMod = 50
levelBudget_list = [[2, 2]]
rescaleTech = "FIXEDAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
save_dir = "/home/cys/PNP/GPU-FHE/examples/RNN/data"
mode = "release"  # "debug" or "release"
autoLoadAndSetConfig = True # note: currently only support True
encode_slots = (1 << (logN -1))


if not os.path.exists(save_dir):
    raise ValueError(f"Directory {save_dir} does not exist!")

cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                        levelBudget_list, "SPARSE_TERNARY", rescaleTech, save_dir=save_dir,
                        autoLoadAndSetConfig=True, mode=mode))

rnn_ih = []
rnn_hh = []
fc_weight = []
fc_bias = None


def rnn_layer(input, hidden, cryptoContext):
    # Evaluate tanh(rnn_ih * input + rnn_hh * hidden)
    # tanh() is approximated by x - (x^3) / 3 (Taylor Series, could be optimized)
    # Weights are always within (-1, 1), so an approximation in (-2, 2) is good enough
    num_slots = (1 << (logN -1))
    num_batch = num_slots / 128
    # Rotation are done based on batch size
    hidden_accum_ct = openfhe_context.encrypt(np.zeros(num_slots), 1, 0, encode_slots)
    # Matrix-Vector Multiplication
    for i in range(128):
        # rnn_ih * input
        rnn_ih_int_0_ct = fhe.homo_mul(rnn_ih[i], input, cryptoContext)
        rnn_ih_int_0_ct = fhe.homo_rescale(rnn_ih_int_0_ct, 1, cryptoContext)
        rnn_ih_int_1_ct = fhe.homo_rotate(rnn_ih_int_0_ct, -(num_batch * i), cryptoContext) 
        hidden_accum_ct = fhe.homo_add(hidden_accum_ct, rnn_ih_int_1_ct, cryptoContext) 
        # rnn_hh * hidden
        rnn_hh_int_0_ct = fhe.homo_mul(rnn_hh[i], hidden, cryptoContext)
        rnn_hh_int_0_ct = fhe.homo_rescale(rnn_hh_int_0_ct, 1, cryptoContext)
        rnn_hh_int_1_ct = fhe.homo_rotate(rnn_hh_int_0_ct, -(num_batch * i), cryptoContext) 
        hidden_accum_ct = fhe.homo_add(hidden_accum_ct, rnn_hh_int_1_ct, cryptoContext) 

    # Apply tanh activation (approximation: （ax + bx^3）)
    x2 = fhe.homo_square(hidden_accum_ct, cryptoContext)
    x2 = fhe.homo_rescale(x2, 1, cryptoContext)
    x3 = fhe.homo_mul_scalar_double(hidden_accum_ct, -0.10484599)
    x2 = fhe.homo_rescale(x3, 1, cryptoContext)
    hidden_accum_ct = fhe.homo_mul_scalar_double(hidden_accum_ct, 0.86501289)
    hidden_accum_ct = fhe.homo_rescale(hidden_accum_ct, cryptoContext)
    t_x3 = fhe.homo_mul(x2, x3, cryptoContext)
    t_x3 = fhe.homo_rescale(t_x3, 1, cryptoContext)
    hidden_accum_ct = fhe.homo_add(hidden_accum_ct, t_x3)
    return hidden_accum_ct

def fc_layer(input, cryptoContext):
    # Evaluate fc_weight * input + bias
    num_slots = (1 << (logN -1))
    num_batch = num_slots / 128
    output_accum_ct = openfhe_context.encrypt(np.zeros(num_slots), 1, 0, encode_slots)
    # Matrix-Vector Multiplication
    for i in range(128):
        mult_0 = fhe.homo_mul_pt(input, fc_weight[i % 2],  cryptoContext)
        mult_0 = fhe.homo_rescale(mult_0, cryptoContext)
        mult_1 = fhe.homo_rotate(mult_0, -(num_batch * i), cryptoContext) 
        output_accum_ct = fhe.homo_add(output_accum_ct, mult_1, cryptoContext)
    # Add bias to final result
    output_accum_ct = fhe.homo_add_pt(output_accum_ct, fc_bias, cryptoContext)
    return output_accum_ct

def activation(t):
    return (math.exp(t) - math.exp(-t)) / (math.exp(t) + math.exp(-t))

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def fhe_rnn(b_id, cryptoContext):
    EMBEDDING_SIZE = 128
    STEP_NUM = 128
    sample_num = 256

    # todo:初始化我提到外面去了

    # File paths
    embedding_file_name = f"/home/cys/PNP/GPU-FHE/examples/RNN/train/test_input/embedding_batch_{b_id}.bin"
    ground_truth_file_name = f"/home/cys/PNP/GPU-FHE/examples/RNN/train/test_input/ground_truth_batch_{b_id}.bin"

    # Load weights and biases from binary files
    rnn_ih_t = np.fromfile("/home/cys/PNP/GPU-FHE/examples/RNN/train/trained_rnn_ih.bin", dtype=np.float32).reshape(EMBEDDING_SIZE, STEP_NUM)
    rnn_hh_t = np.fromfile("/home/cys/PNP/GPU-FHE/examples/RNN/train/trained_rnn_hh.bin", dtype=np.float32).reshape(STEP_NUM, STEP_NUM)
    fc_weight_t = np.fromfile("/home/cys/PNP/GPU-FHE/examples/RNN/train/trained_fc_weight.bin", dtype=np.float32).reshape(2, STEP_NUM)
    fc_bias_t = np.fromfile("/home/cys/PNP/GPU-FHE/examples/RNN/train/trained_fc_bias.bin", dtype=np.float32)

    # Load embedding data and ground truth from binary files
    embedding_in = np.fromfile(embedding_file_name, dtype=np.float32).reshape(sample_num, STEP_NUM, EMBEDDING_SIZE)
    ground_truth = np.fromfile(ground_truth_file_name, dtype=np.float32)

    # Calculate batch size
    batch_size = encode_slots // EMBEDDING_SIZE

    # Pack the plaintext matrix in diagonal order
    rnn_ih_pt_vec = []
    rnn_hh_pt_vec = []
    fc_weight_pt_vec = []
    fc_bias_pt_vec = []

# TODO 这里的原有的三层循环被我替换成了向量化操作，但我不确定这样是不是对的
    rnn_ih_t_2d = rnn_ih_t.reshape((EMBEDDING_SIZE, EMBEDDING_SIZE))
    rnn_hh_t_2d = rnn_hh_t.reshape((STEP_NUM, STEP_NUM))

    rnn_ih = []
    for i in range(STEP_NUM):
        rows = (np.arange(EMBEDDING_SIZE) + i) % EMBEDDING_SIZE
        columns = np.arange(EMBEDDING_SIZE)
        elements = rnn_ih_t_2d[rows, columns]
        rnn_ih_pt = np.repeat(elements, batch_size)
        rnn_ih.append(fhe.encode(rnn_ih_pt, 1, 0, encode_slots, True, cryptoContext))

        rows1 = (np.arange(STEP_NUM) + i) % STEP_NUM
        columns1 = np.arange(STEP_NUM)
        elements = rnn_hh_t_2d[rows1, columns1]
        rnn_hh_pt = np.repeat(elements, batch_size)
        rnn_hh.append(fhe.encode(rnn_hh_pt, 1, 0, encode_slots, True, cryptoContext))

    fc_weight_t_2d = fc_weight_t.reshape(2, STEP_NUM)
    for i in range(2):
        # Compute row indices: (j + i) % 2
        rows = (np.arange(STEP_NUM) + i) % 2
        columns = np.arange(STEP_NUM)
        # Extract elements and repeat batch_size times
        elements = fc_weight_t_2d[rows, columns]
        fc_weight_pt = np.repeat(elements, batch_size)
        fc_weight.append(fhe.encode(fc_weight_pt, 1, 0, encode_slots, True, cryptoContext))

    fc_bias_pt_vec = [fc_bias_t[j % 2] for j in range(STEP_NUM) for _ in range(batch_size)]
    fc_bias = fhe.encode(fc_bias_pt_vec, 1, 0, encode_slots, True, cryptoContext)
    print("Finished building RNN weight plaintexts!")

    batched_hidden_ct = openfhe_context.encrypt(np.zeros(batch_size * STEP_NUM), 1, 0, encode_slots)
    print(f"Before rnn, batched_hidden_ct's remaining levels: {maxLevelsRemaining - batched_hidden_ct.L }")

    #todo  Perform key generation (if needed)
    batch_id = 0
    batched_embedding = np.zeros(batch_size * EMBEDDING_SIZE, dtype=np.float64)
    batched_embedding_ct = openfhe_context.encrypt(batched_embedding, 1, 0, encode_slots)

    batched_hidden_ref = np.zeros(batch_size * STEP_NUM, dtype=np.float64)

    # 1. Evaluate RNN layers
    for i in range(STEP_NUM):
        print(f"step {i}/{STEP_NUM}")
        sample_indices = np.arange(batch_size) + batch_id * batch_size
        j, k = np.mgrid[0:batch_size, 0:EMBEDDING_SIZE]
        src_indices = (sample_indices[j] * EMBEDDING_SIZE + i) * EMBEDDING_SIZE + k
        batched_embedding = embedding_in[src_indices].reshape(EMBEDDING_SIZE, batch_size).T.ravel(order='F') 
        batched_embedding_ct = openfhe_context.encrypt(batched_embedding, 1, 0, encode_slots)

    if (maxLevelsRemaining - (batched_hidden_ct.L - (batched_hidden_ct.cur_limbs - 1))<= 4):
        print("Evaluating Bootstrapping!")
        batched_hidden_ct = fhe.homo_bootstrap(batched_hidden_ct, L0=cryptoContext.L, logBsSlots=8, cryptoContext=cryptoContext)
        
    batched_hidden_ct = rnn_layer(batched_embedding_ct, batched_hidden_ct, cryptoContext)

    # Run reference RNN computation
    result_ref = np.zeros((128, batch_size))
    result_ih = rnn_ih_t @ batched_embedding
    result_hh = rnn_hh_t @ batched_hidden_ref

    result_ref = np.tanh(result_ih + result_hh)
    result_ref = result_ref.reshape(-1, order='C') 

    # See how much accuracy we are losing
    total_error = 0.0
    print(f"Before decrypt: batched_hidden_ct's true remaining levels: {maxLevelsRemaining - batched_hidden_ct.L - (batched_hidden_ct.cur_limbs - 1)}")
    result = openfhe_context.decrypt(batched_hidden_ct)

    result_ref_np = np.array(result_ref)
    ckks_values_np = np.array(result.GetRealPackedValue())

    diff = result_ref_np - ckks_values_np
    total_error = np.sum(diff ** 2)

    print(f"CT depth: {batched_hidden_ct.L()}/{maxLevelsRemaining}")
    avg_sq_err = total_error / (128 * batch_size)
    print(f"Avg Sq Err: {avg_sq_err}")

    batched_hidden_ref = result_ref.copy() 

    # 2. Evaluate FC layers
    fhe_result_pt = openfhe_context.encrypt(fc_layer(batched_hidden_ct, cryptoContext), 1, 0, encode_slots)
    # Run reference FC computation
    result_ref = np.zeros((2*batch_size), dtype=np.float64)
    result_fhe = np.zeros((2*batch_size), dtype=np.float64)
    weighted_sum = fc_weight_t @ batched_hidden_ref
    weighted_sum += fc_bias_t.reshape(2, 1)  
    result_ref = 1 / (1 + np.exp(-weighted_sum))
    fhe_result = fhe_result_pt.reshape(2, batch_size)
    result_fhe = 1 / (1 + np.exp(-fhe_result))
    result_ref = result_ref.flatten(order='C')  # shape (2*batch_size,)
    result_fhe = result_fhe.flatten(order='C') 

    
    num_inf = 0
    num_ref_correct = 0
    num_fhe_correct = 0

    # 遍历批次中的每个样本
    for k in range(batch_size):
        print(f"gt: {ground_truth[k]} ref out: [{result_ref[k]} , {result_ref[k + batch_size]}] "
            f"fhe out: [{result_fhe[k]} , {result_fhe[k + batch_size]}]")

        num_inf += 1
        if ground_truth[k] == 0:
            if result_ref[k] > 0.5:
                num_ref_correct += 1
            if result_fhe[k] > 0.5:
                num_fhe_correct += 1
        else:
            if result_ref[k + batch_size] > 0.5:
                num_ref_correct += 1
            if result_fhe[k + batch_size] > 0.5:
                num_fhe_correct += 1

    # 打印准确率
    print(f"ref accuracy:\t{num_ref_correct}/\t{num_inf}\t{100.0 * num_ref_correct / num_inf:.2f}%")
    print(f"fhe accuracy:\t{num_fhe_correct}/\t{num_inf}\t{100.0 * num_fhe_correct / num_inf:.2f}%")

if __name__ == "__main__":
    fhe_rnn()