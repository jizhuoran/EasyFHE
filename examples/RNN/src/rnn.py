import os, sys
sys.path.append("/".join(os.getcwd().split("/")[:-4]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch
import torch.fhe as fhe
import numpy as np
import math

DATA_DIR = os.environ["DATA_DIR"]


encode_slots = int(1 << (16 - 1)) #todo: to be redesigned? how to assign

rnn_ih = []
rnn_hh = []
fc_weight = []


def rnn_layer(input, hidden, cryptoContext, openfhe_context):
    # Evaluate tanh(rnn_ih * input + rnn_hh * hidden)
    # tanh() is approximated by x - (x^3) / 3 (Taylor Series, could be optimized)
    # Weights are always within (-1, 1), so an approximation in (-2, 2) is good enough
    num_slots = encode_slots
    num_batch = int(num_slots / 128) # We batch several inputs together
    # Rotation are done based on batch size

    hidden_accum_ct = openfhe_context.encrypt(np.zeros(num_slots), 1, 0, encode_slots)
    # Matrix-Vector Multiplication
    for i in range(128):
        # Perform Mult and Rotate, and accumulate
        # Everything is accumulated to hidden_accum_ct, and finally we apply tanh to it
        # rnn_ih * input
        rnn_ih_int_0_ct = fhe.homo_mul_pt(input, rnn_ih[i], cryptoContext)
        rnn_ih_int_0_ct = fhe.homo_rescale(rnn_ih_int_0_ct, 1, cryptoContext)
        rnn_ih_int_1_ct = fhe.homo_rotate(rnn_ih_int_0_ct, -(num_batch * i), cryptoContext) 
        hidden_accum_ct = fhe.homo_add(hidden_accum_ct, rnn_ih_int_1_ct, cryptoContext) 
        # rnn_hh * hidden
        rnn_hh_int_0_ct = fhe.homo_mul_pt(hidden, rnn_hh[i], cryptoContext)
        rnn_hh_int_0_ct = fhe.homo_rescale(rnn_hh_int_0_ct, 1, cryptoContext)
        rnn_hh_int_1_ct = fhe.homo_rotate(rnn_hh_int_0_ct, -(num_batch * i), cryptoContext) 
        hidden_accum_ct = fhe.homo_add(hidden_accum_ct, rnn_hh_int_1_ct, cryptoContext) 

    # Approximation of tanh
    x2 = fhe.homo_square(hidden_accum_ct, cryptoContext)
    x2 = fhe.homo_rescale(x2, 1, cryptoContext)
    # x3 = fhe.homo_mul(hidden_accum_ct, x2, cryptoContext)
    x3 = fhe.homo_mul_scalar_double(hidden_accum_ct, -0.10484599, cryptoContext)
    # x5 = fhe.homo_mul(x3, x2, cryptoContext)
    x3 = fhe.homo_rescale(x3, 1, cryptoContext)

    # a  x x  x
    #  \/   \/
    #  ax   x^2
    #     \/
    #    ax^3
    # Evaluate polynomial
    hidden_accum_ct = fhe.homo_mul_scalar_double(hidden_accum_ct, 0.86501289, cryptoContext)
    hidden_accum_ct = fhe.homo_rescale(hidden_accum_ct, 1, cryptoContext)
    t_x3 = fhe.homo_mul(x2, x3, cryptoContext)
    t_x3 = fhe.homo_rescale(t_x3, 1, cryptoContext)
    hidden_accum_ct = fhe.homo_add(hidden_accum_ct, t_x3, cryptoContext)

    # Consumes total 4 levels
    return hidden_accum_ct

def fc_layer(input, fc_bias, cryptoContext, openfhe_context):
    # Evaluate fc_weight * input + bias
    num_slots = encode_slots
    num_batch = int(num_slots / 128) # We batch several inputs together
    # Rotation are done based on batch size
    output_accum_ct = openfhe_context.encrypt(np.zeros(num_slots), 1, 0, encode_slots)
    # Matrix-Vector Multiplication
    for i in range(128):
        # Perform Mult and Rotate, and accumulate
        # Everything is accumulated to output_accum_ct
        mult_0 = fhe.homo_mul_pt(input, fc_weight[i % 2],  cryptoContext)
        mult_0 = fhe.homo_rescale(mult_0, 1, cryptoContext)
        mult_1 = fhe.homo_rotate(mult_0, -(num_batch * i), cryptoContext) 
        output_accum_ct = fhe.homo_add(output_accum_ct, mult_1, cryptoContext)
    # Add bias to final result
    output_accum_ct = fhe.homo_add_pt(output_accum_ct, fc_bias, cryptoContext)
    return output_accum_ct

def activation(t):
    return (math.exp(t) - math.exp(-t)) / (math.exp(t) + math.exp(-t))

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def fhe_rnn(b_id):
    EMBEDDING_SIZE = 128
    STEP_NUM = 128
    sample_num = 256

    # Calculate batch size
    batch_size = int(encode_slots // EMBEDDING_SIZE)

    logN = 16
    # encode_slots = int(1 << (logN - 1)) # fixme: bad assignment
    maxLevelsRemaining = 10
    appRotIndex_list = [-(i * int(batch_size)) for i in range(EMBEDDING_SIZE)]
    logBsSlots_list = [int(math.log2(encode_slots))]
    dnum = 1
    dcrtBits = 46
    firstMod = 50
    levelBudget_list = [[4, 4]]
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    secretKeyDist = "SPARSE_TERNARY"

    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True)
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, secretKeyDist, rescaleTech, save_dir=DATA_DIR,
                             config=config))

    # File paths
    embedding_file_name = f"../train/test_input/embedding_batch_{b_id}.bin"
    ground_truth_file_name = f"../train/test_input/ground_truth_batch_{b_id}.bin"

    # Load weights and biases from binary files
    rnn_ih_t = np.fromfile("../train/trained_rnn_ih.bin", dtype=np.float32).reshape(EMBEDDING_SIZE, STEP_NUM)
    rnn_hh_t = np.fromfile("../train/trained_rnn_hh.bin", dtype=np.float32).reshape(STEP_NUM, STEP_NUM)
    fc_weight_t = np.fromfile("../train/trained_fc_weight.bin", dtype=np.float32).reshape(2, STEP_NUM)
    fc_bias_t = np.fromfile("../train/trained_fc_bias.bin", dtype=np.float32)
    fc_bias_t = fc_bias_t.astype(np.float64) # todo: change back to the previous 32 after merging the new encode?

    # Load embedding data and ground truth from binary files
    embedding_in = np.fromfile(embedding_file_name, dtype=np.float32)[:sample_num*STEP_NUM*EMBEDDING_SIZE].reshape(sample_num, STEP_NUM, EMBEDDING_SIZE)
    ground_truth = np.fromfile(ground_truth_file_name, dtype=np.float32)


    # Pack the plaintext matrix in diagonal order
    rnn_ih_pt_vec = []
    rnn_hh_pt_vec = []
    fc_weight_pt_vec = []
    fc_bias_dat_vec = [0.0]*batch_size*STEP_NUM

# TODO 这里的原有的三层循环被我替换成了向量化操作，但我不确定这样是不是对的
    rnn_ih_t_2d = rnn_ih_t.reshape((EMBEDDING_SIZE, EMBEDDING_SIZE))
    rnn_hh_t_2d = rnn_hh_t.reshape((STEP_NUM, STEP_NUM))

    for i in range(STEP_NUM):
        rows = (np.arange(EMBEDDING_SIZE) + i) % EMBEDDING_SIZE
        columns = np.arange(EMBEDDING_SIZE)
        elements = rnn_ih_t_2d[rows, columns]
        rnn_ih_dat = np.repeat(elements, batch_size)
        # rnn_ih_dat = torch.tensor(rnn_ih_dat, dtype=torch.float64).cuda() # todo: should be removed  after merging the new encode
        rnn_ih.append(fhe.encode(rnn_ih_dat, f"rnn_ih_dat_{rows}_{columns}_{batch_size}", 0, encode_slots, False, cryptoContext))

        rows1 = (np.arange(STEP_NUM) + i) % STEP_NUM
        columns1 = np.arange(STEP_NUM)
        elements = rnn_hh_t_2d[rows1, columns1]
        rnn_hh_dat = np.repeat(elements, batch_size)
        # rnn_hh_dat = torch.tensor(rnn_hh_dat, dtype=torch.float64).cuda() # todo: should be removed after merging the new encode
        rnn_hh.append(fhe.encode(rnn_hh_dat, f"rnn_hh_dat_{rows1}_{columns1}_{batch_size}", 0, encode_slots, False, cryptoContext))

    fc_weight_t_2d = fc_weight_t.reshape(2, STEP_NUM)
    for i in range(2):
        # Compute row indices: (j + i) % 2
        rows = (np.arange(STEP_NUM) + i) % 2
        columns = np.arange(STEP_NUM)
        # Extract elements and repeat batch_size times
        elements = fc_weight_t_2d[rows, columns]
        fc_weight_dat = np.repeat(elements, batch_size)
        # fc_weight_dat = torch.tensor(fc_weight_dat, dtype=torch.float64).cuda() # todo: should be removed after merging the new encode
        fc_weight.append(fhe.encode(fc_weight_dat, f"fc_weight_dat_{rows}_{columns}_{batch_size}", 0, encode_slots, False, cryptoContext))

    for j in range(STEP_NUM):
        for k in range(batch_size):
            fc_bias_dat_vec[j * batch_size + k] = fc_bias_t[j % 2]
    fc_bias = fhe.encode(fc_bias_dat_vec, "fc_bias_dat_vec", 0, encode_slots, False, cryptoContext)
    print("Finished building RNN weight plaintexts!")

    batched_hidden_ct = openfhe_context.encrypt(np.zeros(batch_size * STEP_NUM), 1, 0, encode_slots)
    print(f"Before rnn, batched_hidden_ct's remaining levels: {batched_hidden_ct.cur_limbs- (batched_hidden_ct.noise_deg - 1)}")


    batch_id = 0
    batched_embedding = np.zeros(batch_size * EMBEDDING_SIZE, dtype=np.float64)
    batched_embedding_ct = openfhe_context.encrypt(batched_embedding, 1, 0, encode_slots)

    batched_hidden_ref = np.zeros(batch_size * STEP_NUM, dtype=np.float64)

    # 1. Evaluate RNN layers
    for i in range(STEP_NUM):
        print(f"step {i}/{STEP_NUM}")
        batched_embedding = np.empty(EMBEDDING_SIZE * batch_size, dtype=np.float64)
        for j in range(batch_size):
            for k in range(EMBEDDING_SIZE):
                sample_index = j + batch_id * batch_size
                batched_embedding[k * batch_size + j] = embedding_in[sample_index, i, k]
        # Embeddings are packed as (em_0_0, em_1_0, ... , em_{last_on_in_batch}_0, em_0_1, em_1_1, ... , em_0_127, ... , em_{last_on_in_batch}_127)
        # Rotations are done in granularity as batch_size
        batched_embedding_ct = openfhe_context.encrypt(batched_embedding, 1, 0, encode_slots)

    if (maxLevelsRemaining - (cryptoContext.L - (batched_hidden_ct.cur_limbs - 1))<= 4):
        print("Evaluating Bootstrapping!")
        batched_hidden_ct = fhe.homo_bootstrap(batched_hidden_ct, cryptoContext.L, 8, cryptoContext)
        
    batched_hidden_ct = rnn_layer(batched_embedding_ct, batched_hidden_ct, cryptoContext, openfhe_context)

    # Run reference RNN computation
    result_ref = np.zeros(batch_size * STEP_NUM, dtype=np.float64)

    for j in range(128):
        for k in range(batch_size):
            for l in range(128):
                result_ref[j * batch_size + k] += (
                        batched_embedding[l * batch_size + k] * rnn_ih_t[j, l]
                )
                result_ref[j * batch_size + k] += (
                        batched_hidden_ref[l * batch_size + k] * rnn_hh_t[j, l]
                )
            # Tanh activation
            result_ref[j * batch_size + k] = activation(result_ref[j * batch_size + k])

    # result_ref = np.zeros((128, batch_size))
    # result_ih = rnn_ih_t @ batched_embedding
    # result_hh = rnn_hh_t @ batched_hidden_ref
    #
    # result_ref = np.tanh(result_ih + result_hh)
    # result_ref = result_ref.reshape(-1, order='C')

    # See how much accuracy we are losing
    total_error = 0.0
    print(f"Before decrypt: batched_hidden_ct's true remaining levels: {batched_hidden_ct.cur_limbs - (batched_hidden_ct.noise_deg - 1)}")
    result = openfhe_context.decrypt(batched_hidden_ct)

    result_ref_np = np.array(result_ref)
    ckks_values_np = np.array(result.cpu().numpy().reshape(-1))

    diff = result_ref_np - ckks_values_np
    total_error = np.sum(diff ** 2)

    print(f"CT depth: {cryptoContext.L}/{maxLevelsRemaining}")
    avg_sq_err = total_error / (128 * batch_size)
    print(f"Avg Sq Err: {avg_sq_err}")

    batched_hidden_ref = result_ref.copy() 

    # 2. Evaluate FC layers
    batched_hidden_ct=fc_layer(batched_hidden_ct, fc_bias, cryptoContext, openfhe_context)
    fhe_result_pt = openfhe_context.decrypt(batched_hidden_ct)
    fhe_result_pt = fhe_result_pt.cpu().numpy().reshape(-1)
    # Run reference FC computation
    result_ref = np.zeros(batch_size * 2, dtype=np.float64)
    result_fhe = np.zeros(batch_size * 2, dtype=np.float64)

    # FC output and sigmoid activation
    for j in range(2):
        for k in range(batch_size):
            for l in range(128):
                result_ref[j * batch_size + k] += (
                        batched_hidden_ref[l * batch_size + k] * fc_weight_t[j, l]
                )
            result_ref[j * batch_size + k] += fc_bias_t[j]
            result_ref[j * batch_size + k] = sigmoid(result_ref[j * batch_size + k])
            result_fhe[j * batch_size + k] = sigmoid(fhe_result_pt[j * batch_size + k])

    # Accuracy comparison
    num_inf = 0
    num_ref_correct = 0
    num_fhe_correct = 0
    for k in range(batch_size):
        gt = ground_truth[batch_id * batch_size + k]
        print(
            f"gt: {gt} ref out: [{result_ref[k]} , {result_ref[k + batch_size]}] "
            f"fhe out: [{result_fhe[k]} , {result_fhe[k + batch_size]}]"
        )

        num_inf += 1
        if gt == 0:
            if result_ref[k] > 0.5:
                num_ref_correct += 1
            if result_fhe[k] > 0.5:
                num_fhe_correct += 1
        else:
            if result_ref[k + batch_size] > 0.5:
                num_ref_correct += 1
            if result_fhe[k + batch_size] > 0.5:
                num_fhe_correct += 1

    # Print accuracy
    print(f"ref accuracy:\t{num_ref_correct}/\t{num_inf}\t{100.0 * num_ref_correct / num_inf:.2f}%")
    print(f"fhe accuracy:\t{num_fhe_correct}/\t{num_inf}\t{100.0 * num_fhe_correct / num_inf:.2f}%")

    # result_ref = np.zeros((2*batch_size), dtype=np.float64)
    # result_fhe = np.zeros((2*batch_size), dtype=np.float64)
    # weighted_sum = fc_weight_t @ batched_hidden_ref
    # weighted_sum += fc_bias_t.reshape(2, 1)
    # result_ref = 1 / (1 + np.exp(-weighted_sum))
    # fhe_result = fhe_result_pt.reshape(2, batch_size)
    # result_fhe = 1 / (1 + np.exp(-fhe_result))
    # result_ref = result_ref.flatten(order='C')  # shape (2*batch_size,)
    # result_fhe = result_fhe.flatten(order='C')
    #
    #
    # num_inf = 0
    # num_ref_correct = 0
    # num_fhe_correct = 0
    #
    # for k in range(batch_size):
    #     print(f"gt: {ground_truth[k]} ref out: [{result_ref[k]} , {result_ref[k + batch_size]}] "
    #         f"fhe out: [{result_fhe[k]} , {result_fhe[k + batch_size]}]")
    #
    #     num_inf += 1
    #     if ground_truth[k] == 0:
    #         if result_ref[k] > 0.5:
    #             num_ref_correct += 1
    #         if result_fhe[k] > 0.5:
    #             num_fhe_correct += 1
    #     else:
    #         if result_ref[k + batch_size] > 0.5:
    #             num_ref_correct += 1
    #         if result_fhe[k + batch_size] > 0.5:
    #             num_fhe_correct += 1
    #
    # # 打印准确率
    # print(f"ref accuracy:\t{num_ref_correct}/\t{num_inf}\t{100.0 * num_ref_correct / num_inf:.2f}%")
    # print(f"fhe accuracy:\t{num_fhe_correct}/\t{num_inf}\t{100.0 * num_fhe_correct / num_inf:.2f}%")

if __name__ == "__main__":
    for b_id in range(1):
        fhe_rnn(b_id)
