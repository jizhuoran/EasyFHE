import os, sys
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from examples.utils import approx
import torch.fhe as fhe
import torch
import math
import numpy as np

DATA_DIR = os.environ["DATA_DIR"]
DEBUG = True
polyDeg_in_compare_and_swap = 119  # todo: should be set outside?

def compare_and_swap(a1, a2, a3, a4, cryptoContext):
    a1_sub_a2 = fhe.homo_sub(a1, a2,cryptoContext)
    a2_sub_a1 = fhe.homo_sub(a2, a1,cryptoContext)
    lowerBound = -5
    upperBound = 5

    a1_gt_a2 = approx.eval_chebyshev_function(lambda x: 1 if x>=0 else 0,
                                              a1_sub_a2,
                                              lowerBound, upperBound, polyDeg_in_compare_and_swap, cryptoContext )
    a2_gt_a1 = approx.eval_chebyshev_function(lambda x: 1 if x>0 else 0,
                                              a2_sub_a1,
                                              lowerBound, upperBound, polyDeg_in_compare_and_swap, cryptoContext)
    return fhe.homo_add(fhe.homo_mul(a1_gt_a2,a3, cryptoContext), fhe.homo_mul(a2_gt_a1,a4, cryptoContext), cryptoContext)

def Sort(input_length=8):
    print("--------------------------------- Sorting ---------------------------------")

    # Selecting CKKS parameters
    maxLevelsRemaining = 34
    rotate_index_list = []
    i = 1
    while i < input_length:
        rotate_index_list.append(i)
        rotate_index_list.append(-i)
        i <<= 1  # 左移等效于乘以 2
    logBsSlots_list = [int(math.log2(input_length))] # fixme: align with encode slots(encode slots should align with n)?
    logN = 14
    dnum = 3
    dcrtBits = 59
    firstMod = 60
    levelBudget_list = [[2,2]]
    secretKeyDist = "SPARSE_TERNARY"
    rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # todo: note that original is FLEXIBLEAUTOEXT

    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    # generate context
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                       levelBudget_list, secretKeyDist, rescaleTech, save_dir=DATA_DIR,
                       autoLoadAndSetConfig=True, mode="release"))

    total_steps = int((1+math.log2(input_length))*math.log2(input_length)/2)
    print("Total steps: {}".format(total_steps))

    # input preparation
    input_msg = np.random.uniform(3, 4, input_length)  # Generate a random number within the range
    input_ct = openfhe_context.encrypt(input_msg, 1, 0, input_length, "release")
    # print("Generated input vector: ", input_msg)
    print("Initial number of mult depth remaining: ", input_ct.cur_limbs-1)

    # Sorting
    n = input_length
    k = 2
    step = 1
    sorted_input_msg = sorted(input_msg)
    while k<=n:
        j = int(math.floor(k/2))
        while j>0:
            cur_lRemain = input_ct.cur_limbs - (input_ct.noise_deg - 1)
            if cur_lRemain <= 10: # equivalent to multDepth - trueLevel(input_ct)
                print("lRemain before bootstrapping: ", cur_lRemain)
                input_ct = fhe.homo_bootstrap(input_ct, cryptoContext.L, logBsSlots_list[0], cryptoContext) #todo: update framework bootstrapping with numiterations
                print("lRemain after bootstrapping: ", cur_lRemain)

            print("step: {}".format(step))
            print(f"[APP TRACE] step: {step} (k={k}, j={j})", file=sys.stderr)
            step += 1

            mask1 = [0.0] * n
            mask2 = [0.0] * n
            mask3 = [0.0] * n
            mask4 = [0.0] * n

            for i in range(n):
                l = i ^ j
                if i < l:
                    if (i & k) == 0:
                        mask1[i] = 1
                        mask2[l] = 1
                    else:
                        mask3[i] = 1
                        mask4[l] = 1
            mask1 = torch.tensor(mask1, dtype=torch.float64).cuda()
            mask2 = torch.tensor(mask2, dtype=torch.float64).cuda()
            mask3 = torch.tensor(mask3, dtype=torch.float64).cuda()
            mask4 = torch.tensor(mask4, dtype=torch.float64).cuda()
            # fixme: note that the following encode with use_gpu_fft=True may lead to larger error
            arr1 = fhe.homo_mul_pt(input_ct, fhe.encode(mask1, 1, 0, input_length, use_gpu_fft=True, cryptoContext=cryptoContext), cryptoContext)
            arr2 = fhe.homo_mul_pt(input_ct, fhe.encode(mask2, 1, 0, input_length, use_gpu_fft=True, cryptoContext=cryptoContext), cryptoContext)
            arr3 = fhe.homo_mul_pt(input_ct, fhe.encode(mask3, 1, 0, input_length, use_gpu_fft=True, cryptoContext=cryptoContext), cryptoContext)
            arr4 = fhe.homo_mul_pt(input_ct, fhe.encode(mask4, 1, 0, input_length, use_gpu_fft=True, cryptoContext=cryptoContext), cryptoContext)
            arr5_1 = fhe.homo_rotate(arr1,-j,cryptoContext)
            arr5_2 = fhe.homo_rotate(arr3,-j,cryptoContext)
            arr6_1 = fhe.homo_rotate(arr2,j,cryptoContext)
            arr6_2 = fhe.homo_rotate(arr4,j,cryptoContext)
            arr7 = fhe.homo_add(fhe.homo_add(arr5_1,arr5_2,cryptoContext),fhe.homo_add(arr6_1,arr6_2,cryptoContext), cryptoContext)
            arr8 = input_ct.deep_copy()
            arr9 = fhe.homo_add(fhe.homo_add(arr5_1,arr1,cryptoContext), fhe.homo_add(arr6_2,arr4,cryptoContext), cryptoContext)
            arr10 = fhe.homo_add(fhe.homo_add(arr5_2,arr3, cryptoContext),fhe.homo_add(arr6_1,arr2,cryptoContext), cryptoContext)

            input_ct = compare_and_swap(arr7, arr8, arr9, arr10, cryptoContext)

            j =int(j/2)

            print("remaining level: ", input_ct.cur_limbs - (input_ct.noise_deg - 1))

            if DEBUG:
                clear_result = openfhe_context.decrypt(input_ct)
                clear_result = clear_result.cpu().numpy().reshape(-1)
                # print("HE decryption result: ", clear_result[:10])
                total_error = 0.0
                for i in range(n):
                    total_error += (clear_result[i]-sorted_input_msg[i])**2
                print("Avg error: ", total_error/n)

            # if (DEBUG) {
            #     cc->Decrypt(keyPair.secretKey, input_ct, &plaintextDec);
            #     plaintextDec->SetLength(encodedLength);
            #     auto tmp_result = plaintextDec->GetRealPackedValue();
            #     std::sort(tmp_result.begin(), tmp_result.end(), std::less<double>()); #todo: ??? why sort the encrypted result
            #     double total_error = 0;
            #     for (size_t i = 0; i < encodedLength; i++) {
            #       total_error += (tmp_result[i]-input_msg[i]) * (tmp_result[i]-input_msg[i]);
            #     }
            #    std::cout<<"Avg error: "<< total_error/encodedLength << std::endl;
            # }

        k *= 2

    # // Level consumption: ~12 level
    print("[APP TRACE] finish")

    # // Output computation
    print("Expected output: ", sorted_input_msg[:10])
    clear_result = openfhe_context.decrypt(input_ct)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("Actual output: ", clear_result[:10])

    # total_error = 0.0
    # for i in range(n):
    #     total_error += (clear_result[i] - sorted_input_msg[i]) ** 2
    # print("Avg error: ", total_error / n)


    # todo: to be continued
    # // Compute the average difference and magnitude of the complex numbers
    # std::vector<double> differences;
    # std::vector<double> magnitudes;
    # std::vector<double> ratio_avg;
    #
    # for (size_t i = 0; i < input_msg.size(); ++i)
    # {
    #     double magnitude = std::abs(finalResult[i]);
    #     magnitudes.push_back(magnitude);
    #     differences.push_back(std::abs(input_msg[i] - magnitude));
    #     ratio_avg.push_back(differences[i] / magnitude);
    # }
    #
    # // Compute the average difference
    # double avgDifference = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();
    #
    # // Compute the average magnitude of the complex numbers
    # double avgMagnitude = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) / magnitudes.size();
    #
    # double ratioavg = std::accumulate(ratio_avg.begin(), ratio_avg.end(), 0.0) / ratio_avg.size();
    # // Compute the average difference divided by the average magnitude
    # double ratio = avgDifference / avgMagnitude;
    # FILE* fp = fopen("input_msg.bin", "wb");
    # fwrite(input_msg.data(), sizeof(double), input_msg.size(), fp);
    # fclose(fp);
    # fp = fopen("output.bin", "wb");
    # fwrite(tmp_result.data(), sizeof(double), tmp_result.size(), fp);
    # fclose(fp);
    #
    # std::cout << "Average difference: " << avgDifference << std::endl;
    # std::cout << "Average magnitude of complex numbers: " << avgMagnitude << std::endl;
    # std::cout << "Average difference / Average magnitude: " << ratio << std::endl;
    # std::cout << "Average (difference / magnitude): " << ratioavg << std::endl;




if __name__ == "__main__":
    Sort(1<<3)


