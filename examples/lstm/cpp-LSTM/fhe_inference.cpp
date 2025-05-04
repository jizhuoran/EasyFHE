#include <openfhe.h>
#include <errno.h>
// #include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace lbcrypto;
const static usint RING_DIM = 1 << 16;

std::vector<Plaintext> rnn_ih, rnn_hh, fc_weight;
Plaintext fc_bias;

int num_inf, num_ref_correct, num_fhe_correct;

/**
 * Run 1 layer of RNN.
 * @param input ct vector for the input embeddings
 * @param hidden ct vector for the hidden state
 * @return ct vector for the next hidden state
*/
Ciphertext<DCRTPoly> rnn_layer(const Ciphertext<DCRTPoly> &input, const Ciphertext<DCRTPoly> &hidden, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> keyPair) {
    // Evaluate tanh(rnn_ih * input + rnn_hh * hidden)
    // tanh() is approximated by x - (x^3) / 3 (Taylor Series, could be optimized)
    // Weights are always within (-1, 1), so an approximation in (-2, 2) is good enough
    usint num_slots = RING_DIM / 2;
    usint num_batch = num_slots / 128;    // We batch several inputs together
    // Rotation are done based on batch size

    Plaintext hidden_accum_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(num_slots));
    Ciphertext<DCRTPoly> hidden_accum_ct = cc->Encrypt(keyPair.publicKey, hidden_accum_pt);
    // Matrix-Vector Multiplication
    for (int i = 0;i < 128; ++i) {
        // Perform Mult and Rotate, and accumulate
        // Everything is accumulated to hidden_accum_ct, and finally we apply tanh to it
        // rnn_ih * input
        Ciphertext<DCRTPoly> rnn_ih_int_0_ct = cc->EvalMult(rnn_ih[i], input);
        Ciphertext<DCRTPoly> rnn_ih_int_1_ct = cc->EvalRotate(rnn_ih_int_0_ct,  -(int(num_batch) * i));
        cc->EvalAddInPlace(hidden_accum_ct, rnn_ih_int_1_ct);

        // rnn_hh * hidden
        Ciphertext<DCRTPoly> rnn_hh_int_0_ct = cc->EvalMult(rnn_hh[i], hidden);
        Ciphertext<DCRTPoly> rnn_hh_int_1_ct = cc->EvalRotate(rnn_hh_int_0_ct,  -(int(num_batch) * i));
        cc->EvalAddInPlace(hidden_accum_ct, rnn_hh_int_1_ct);
    }
    // Approximation of tanh
    Ciphertext<DCRTPoly> x2 = cc->EvalMult(hidden_accum_ct, hidden_accum_ct);
    // Ciphertext<DCRTPoly> x3 = cc->EvalMult(hidden_accum_ct, x2);
    Ciphertext<DCRTPoly> x3 = cc->EvalMult(hidden_accum_ct, -0.10484599);
    // Ciphertext<DCRTPoly> x5 = cc->EvalMult(x3, x2);

    // a  x x  x
    //  \/   \/
    //  ax   x^2
    //     \/
    //    ax^3

    // Evaluate polynomial
    cc->EvalMultInPlace(hidden_accum_ct, 0.86501289);
    Ciphertext<DCRTPoly> t_x3 = cc->EvalMult(x2, x3);
    cc->EvalAddInPlace(hidden_accum_ct, t_x3);

    // Consumes total 4 levels
    return hidden_accum_ct;
}

/**
 * Run FC layer.
 * TODO: Make FC layer dimension more parameterized
 * @param input ct vector for the input of the FC layer
 * @return ct vector for the FC layer output
*/
Ciphertext<DCRTPoly> fc_layer(const Ciphertext<DCRTPoly> &input, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> keyPair) {
    // Evaluate fc_weight * input + bias
    usint num_slots = RING_DIM / 2;
    usint num_batch = num_slots / 128;    // We batch several inputs together
    // Rotation are done based on batch size

    Plaintext output_accum_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(num_slots));
    Ciphertext<DCRTPoly> output_accum_ct = cc->Encrypt(keyPair.publicKey, output_accum_pt);
    // Matrix-Vector Multiplication
    for (int i = 0;i < 128; ++i) {
        // Perform Mult and Rotate, and accumulate
        // Everything is accumulated to output_accum_ct
        Ciphertext<DCRTPoly> mult_0 = cc->EvalMult(fc_weight[i % 2], input);
        Ciphertext<DCRTPoly> mult_1 = cc->EvalRotate(mult_0,  -(int(num_batch) * i));
        cc->EvalAddInPlace(output_accum_ct, mult_1);
    }
    // Add bias to final result
    cc->EvalAddInPlace(output_accum_ct, fc_bias);

    return output_accum_ct;
}

float activation(float t) {
    return (exp(t) - exp(-t)) / (exp(t) + exp(-t));
}

float sigmoid(float t) {
    return (1) / (1 + exp(-t));
}

void fhe_rnn(int b_id) {
    const int EMBEDDING_SIZE = 128;
    const int STEP_NUM = 128;
    int sample_num = 256;
    // TODO: Fix these CKKS parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    // A. Specify main parameters
    /*  A1) Secret key distribution
    * The secret key distribution for CKKS should either be SPARSE_TERNARY or UNIFORM_TERNARY.
    * The SPARSE_TERNARY distribution was used in the original CKKS paper,
    * but in this example, we use UNIFORM_TERNARY because this is included in the homomorphic
    * encryption standard.
    */
    SecretKeyDist secretKeyDist = SPARSE_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetNumLargeDigits(3);

    /*  A2) Desired security level based on FHE standards.
    * In this example, we use the "NotSet" option, so the example can run more quickly with
    * a smaller ring dimension. Note that this should be used only in
    * non-production environments, or by experts who understand the security
    * implications of their choices. In production-like environments, we recommend using
    * HEStd_128_classic, HEStd_192_classic, or HEStd_256_classic for 128-bit, 192-bit,
    * or 256-bit security, respectively. If you choose one of these as your security level,
    * you do not need to set the ring dimension.
    */
    parameters.SetSecurityLevel(HEStd_128_classic);
    // parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(RING_DIM);

    /*  A3) Scaling parameters.
    * By default, we set the modulus sizes and rescaling technique to the following values
    * to obtain a good precision and performance tradeoff. We recommend keeping the parameters
    * below unless you are an FHE expert.
    */
#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
    ScalingTechnique rescaleTech = FIXEDAUTO;
    usint dcrtBits               = 78;
    usint firstMod               = 89;
#else
    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint dcrtBits               = 46;
    usint firstMod               = 50;
#endif

    parameters.SetScalingModSize(dcrtBits);
    // parameters.SetScalingTechnique(FIXEDAUTO);
    parameters.SetFirstModSize(firstMod);
    usint numSlots = (1<<15);
    // parameters.SetBatchSize(numSlots);

    /*  A4) Multiplicative depth.
    * The goal of bootstrapping is to increase the number of available levels we have, or in other words,
    * to dynamically increase the multiplicative depth. However, the bootstrapping procedure itself
    * needs to consume a few levels to run. We compute the number of bootstrapping levels required
    * using GetBootstrapDepth, and add it to levelsAvailableAfterBootstrap to set our initial multiplicative
    * depth. We recommend using the input parameters below to get started.
    */
    std::vector<uint32_t> levelBudget = {2, 2};

    // Note that the actual number of levels avalailable after bootstrapping before next bootstrapping
    // will be levelsAvailableAfterBootstrap - 1 because an additional level
    // is used for scaling the ciphertext before next bootstrapping (in 64-bit CKKS bootstrapping)
    uint32_t levelsAvailableAfterBootstrap = 10;
    // usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    usint depth = 26;
    parameters.SetMultiplicativeDepth(depth);

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(FHE);

    usint ringDim = cryptoContext->GetRingDimension();
    // This is the maximum number of slots that can be used for full packing.
    std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl;
    std::cout << "\tavailable number of slots " << numSlots << std::endl;
    std::cout << "depth: " << depth << std::endl;
    std::cout << "bootstrap depth: " << FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist) << std::endl;

    cryptoContext->EvalBootstrapSetup(levelBudget);

    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    // Retrieve weight from .bin files
    std::string embedding_file_name = std::string("../train/test_input/embedding_batch_") + std::to_string(b_id) + std::string(".bin");
    std::string ground_truth_file_name = std::string("../train/test_input/ground_truth_batch_") + std::to_string(b_id) + std::string(".bin");
    float *rnn_ih_t = new float[EMBEDDING_SIZE * STEP_NUM];
    float *rnn_hh_t = new float[STEP_NUM * STEP_NUM];
    float *fc_weight_t = new float[2 * STEP_NUM];
    float *fc_bias_t = new float[2];
    float *embedding_in = new float[sample_num * STEP_NUM * EMBEDDING_SIZE];
    float *ground_truth = new float[sample_num];
    FILE *file;
    file = fopen("../train/trained_rnn_ih.bin", "rb");
    fread(rnn_ih_t, sizeof(float), EMBEDDING_SIZE * STEP_NUM, file);
    fclose(file);
    file = fopen("../train/trained_rnn_hh.bin", "rb");
    fread(rnn_hh_t, sizeof(float), STEP_NUM * STEP_NUM, file);
    fclose(file);
    file = fopen("../train/trained_fc_weight.bin", "rb");
    fread(fc_weight_t, sizeof(float), 2 * STEP_NUM, file);
    fclose(file);
    file = fopen("../train/trained_fc_bias.bin", "rb");
    fread(fc_bias_t, sizeof(float), 2, file);
    fclose(file);
    file = fopen(embedding_file_name.c_str(), "rb");
    fread(embedding_in, sizeof(float), sample_num * STEP_NUM * EMBEDDING_SIZE, file);
    fclose(file);
    file = fopen(ground_truth_file_name.c_str(), "rb");
    fread(ground_truth, sizeof(float), sample_num, file);
    fclose(file);
    usint batch_size = numSlots / EMBEDDING_SIZE;
    // Pack the plaintext matrix in diagnal order
    std::vector<double> rnn_ih_pt_vec, rnn_hh_pt_vec;
    std::vector<double> fc_weight_pt_vec, fc_bias_pt_vec;
    for (int i = 0;i < STEP_NUM; ++i) {
        rnn_ih_pt_vec.clear();
        rnn_ih_pt_vec.resize(batch_size * EMBEDDING_SIZE);
        for (int j = 0;j < EMBEDDING_SIZE; ++j) {
            for (int k = 0;k < batch_size; ++k) {
                rnn_ih_pt_vec[j * batch_size + k] = rnn_ih_t[((j + i) % EMBEDDING_SIZE) * EMBEDDING_SIZE + j];
            }
        }
        rnn_ih.push_back(cryptoContext->MakeCKKSPackedPlaintext(rnn_ih_pt_vec));

        rnn_hh_pt_vec.clear();
        rnn_hh_pt_vec.resize(batch_size * STEP_NUM);
        for (int j = 0;j < STEP_NUM; ++j) {
            for (int k = 0;k < batch_size; ++k) {
                rnn_hh_pt_vec[j * batch_size + k] = rnn_hh_t[((j + i) % STEP_NUM) * STEP_NUM + j];
            }
        }
        rnn_hh.push_back(cryptoContext->MakeCKKSPackedPlaintext(rnn_hh_pt_vec));
    }
    for (int i = 0;i < 2; ++i) {
        fc_weight_pt_vec.clear();
        fc_weight_pt_vec.resize(batch_size * STEP_NUM);
        for (int j = 0;j < STEP_NUM; ++j) {
            for (int k = 0;k < batch_size; ++k) {
                fc_weight_pt_vec[j * batch_size + k] = fc_weight_t[((j + i) % 2) * STEP_NUM + j];
            }
        }
        fc_weight.push_back(cryptoContext->MakeCKKSPackedPlaintext(fc_weight_pt_vec));
    }
    fc_bias_pt_vec.clear();
    fc_bias_pt_vec.resize(batch_size * STEP_NUM);
    for (int j = 0;j < STEP_NUM; ++j) {
        for (int k = 0;k < batch_size; ++k) {
            fc_bias_pt_vec[j * batch_size + k] = fc_bias_t[j % 2];
        }
    }
    fc_bias = cryptoContext->MakeCKKSPackedPlaintext(fc_bias_pt_vec);
    std::cout << "Finished building RNN weight plaintexts!"  << std::endl;

    std::vector<int> rotate_keygen_list(EMBEDDING_SIZE);
    for (int i = 0;i < EMBEDDING_SIZE; ++i) {
        rotate_keygen_list[i] = -(i * int(batch_size));
    }
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotate_keygen_list, keyPair.publicKey);
    std::cout << "Rotate KeyGen Finishes!"  << std::endl;

    std::vector<double> batched_embedding(batch_size * EMBEDDING_SIZE);
    // Enumerate over all the input length
    // for (int batch_id = 0; batch_id < 128 / batch_size; ++batch_id) {
    // Initialize hidden states as 0s
    int batch_id = 0;
    Plaintext batched_hidden_pt = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(batch_size * STEP_NUM));
    Ciphertext<DCRTPoly> batched_hidden_ct = cryptoContext->Encrypt(keyPair.publicKey, batched_hidden_pt);
    std::cout << "Before rnn, batched_hidden_ct's remaining levels: " << depth - batched_hidden_ct->GetLevel() << std::endl;

    std::vector<double> batched_hidden_ref(batch_size * STEP_NUM);

    // 1. Evaluate RNN layers
    for (int i = 0;i < STEP_NUM; ++i) {
        // std::cout << "Evaluating batch " << batch_id << "/" << 128 / batch_size << " : token " << i << "/128 -- ";
        std::cout << "step " << i << "/" << STEP_NUM << std::endl;
        fflush(NULL);
        for (int j = 0;j < batch_size; ++j) {
            for (int k = 0;k < EMBEDDING_SIZE; ++k) {
                batched_embedding[k * batch_size + j] = embedding_in[((j + batch_id * batch_size) * EMBEDDING_SIZE + i) * EMBEDDING_SIZE + k];
            }
        }
        // Embeddings are packed as (em_0_0, em_1_0, ... , em_{last_on_in_batch}_0, em_0_1, em_1_1, ... , em_0_127, ... , em_{last_on_in_batch}_127)
        // Rotations are done in granularity as batch_size
        Plaintext batched_embedding_pt = cryptoContext->MakeCKKSPackedPlaintext(batched_embedding);
        Ciphertext<DCRTPoly> batched_embedding_ct = cryptoContext->Encrypt(keyPair.publicKey, batched_embedding_pt);

        if (depth - (batched_hidden_ct->GetLevel()-(batched_hidden_ct->GetNoiseScaleDeg() - 1)) <= 4) {
            std::cout << "Evaluating Bootstrapping!" << std::endl;
            Ciphertext<DCRTPoly> new_batched_hidden_ct = cryptoContext->EvalBootstrap(batched_hidden_ct);
            batched_hidden_ct = new_batched_hidden_ct;
            // std::cout << "CT depth after bootstrap: " << batched_hidden_ct->GetLevel() + (batched_hidden_ct->GetNoiseScaleDeg() - 1) << "/" << depth << std::endl;
        }

        // Evaluate RNN layer
        batched_hidden_ct = rnn_layer(batched_embedding_ct, batched_hidden_ct, cryptoContext, keyPair);

        // Run reference RNN computation
        std::vector<double> result_ref(batch_size * STEP_NUM);
        for (int j = 0;j < 128; ++j) {
            for (int k = 0;k < batch_size; ++k) {
                for (int l = 0;l < 128; ++l) {
                    result_ref[j * batch_size + k] += batched_embedding[l * batch_size + k] * rnn_ih_t[j * 128 + l];
                    result_ref[j * batch_size + k] += batched_hidden_ref[l * batch_size + k] * rnn_hh_t[j * 128 + l];
                }
                // Tanh activation
                result_ref[j * batch_size + k] = activation(result_ref[j * batch_size + k]);
            }
        }

        // See how much accuracy we are losing
        double total_error = 0;
        Plaintext result;
        std::cout << "Before decrypt: batched_hidden_ct's true remaining levels: " << (depth - batched_hidden_ct->GetLevel() - (batched_hidden_ct->GetNoiseScaleDeg() - 1)) << std::endl;

        cryptoContext->Decrypt(keyPair.secretKey, batched_hidden_ct, &result);
        for (int j = 0;j < 128; ++j) {
            for (int k = 0;k < batch_size; ++k) {
                // std::cout << "slot " << j * batch_size + k << "\tref/ckks :" << result_ref[j * batch_size + k] << " / " << result->GetRealPackedValue()[j * batch_size + k] << std::endl;
                total_error += (result_ref[j * batch_size + k] - result->GetRealPackedValue()[j * batch_size + k]) * (result_ref[j * batch_size + k] - result->GetRealPackedValue()[j * batch_size + k]);
            }
        }
        std::cout << "CT depth: " << batched_hidden_ct->GetLevel() << "/" << depth << std::endl;

        std::cout << "Avg Sq Err: " << total_error / (128 * batch_size) << std::endl;
        batched_hidden_ref = result_ref;
        // char t = getchar();
    }

    // 2. Evaluate FC layer
    batched_hidden_ct = fc_layer(batched_hidden_ct, cryptoContext, keyPair);
    Plaintext fhe_result_pt;
    cryptoContext->Decrypt(keyPair.secretKey, batched_hidden_ct, &fhe_result_pt);
    // Run reference RNN computation
    std::vector<double> result_ref(batch_size * 2), result_fhe(batch_size * 2);
    for (int j = 0;j < 2; ++j) {
        for (int k = 0;k < batch_size; ++k) {
            for (int l = 0;l < 128; ++l) {
                result_ref[j * batch_size + k] += batched_hidden_ref[l * batch_size + k] * fc_weight_t[j * 128 + l];
            }
            result_ref[j * batch_size + k] += fc_bias_t[j];

            // Sigmoid activation
            result_ref[j * batch_size + k] = sigmoid(result_ref[j * batch_size + k]);
            result_fhe[j * batch_size + k] = sigmoid(fhe_result_pt->GetRealPackedValue()[j * batch_size + k]);
        }
    }
    for (int k = 0;k < batch_size; ++k) {
        std::cout << "gt: " << ground_truth[batch_id * batch_size + k] << " ref out: [" << result_ref[k] << " ," << result_ref[k + batch_size] << "]"
                  << " fhe out: [" << result_fhe[k] << " ," << result_fhe[k + batch_size] << "]" << std::endl;

        ++num_inf;
        if (ground_truth[batch_id * batch_size + k] == 0) {
            if (result_ref[k] > 0.5) ++ num_ref_correct;
            if (result_fhe[k] > 0.5) ++ num_fhe_correct;
        } else {
            if (result_ref[k + batch_size] > 0.5) ++ num_ref_correct;
            if (result_fhe[k + batch_size] > 0.5) ++ num_fhe_correct;
        }
    }
    std::cout << "ref accuracy:\t" << num_ref_correct << "/\t" << num_inf << "\t" << 100.0 * num_ref_correct / num_inf << "%" << std::endl
              << "fhe accuracy:\t" << num_fhe_correct << "/\t" << num_inf << "\t" << 100.0 * num_fhe_correct / num_inf << "%" << std::endl;
    // }

    delete[] rnn_ih_t;
    delete[] rnn_hh_t;
    delete[] fc_weight_t;
    delete[] fc_bias_t;
    delete[] embedding_in;
    delete[] ground_truth;
}

int main() {
    // for (int i = 0;i < 50; ++i) {
    fhe_rnn(0);
    // }
}
