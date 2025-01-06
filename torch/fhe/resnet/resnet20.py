from torch.fhe import homo_ops
from torch.fhe import hoisting_keyswitch
from torch.fhe.ciphertext import Cipher
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
def convbn3(input,layer,n,scale,timing,cryptoContext):
    img_width=8
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1],-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input.cv[1],-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1], padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input.cv[1], -padding, digits, cryptoContext))
    c_rotations.append(input.cv[1])#这里旋转什么的都只需要对cv1吗？
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input.cv[1], padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1],-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input.cv[1],img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1], padding, digits, cryptoContext), img_width, cryptoContext))
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
        sum=eval_add_many(k_rows)
        if(j==0):
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-256,cryptoContext)
    #finalsum=homo_ops.homo_add(finalsum,bias,cryptoContext)
    return finalsum


def convbn3(input,layer,n,scale,timing,cryptoContext):
    img_width=32
    padding=1
    digits = hoisting_keyswitch.eval_fast_rotation_precompute(input.cv[1],input.curr_limbs,cryptoContext)
    #使用list代替vector
    c_rotations=[]
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1],-padding,digits,cryptoContext),-img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input.cv[1],-img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1], padding, digits, cryptoContext), -img_width, cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input.cv[1], -padding, digits, cryptoContext))
    c_rotations.append(input.cv[1])#这里旋转什么的都只需要对cv1吗？
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation(input.cv[1], padding, digits, cryptoContext))
    c_rotations.append(homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1],-padding,digits,cryptoContext),img_width,cryptoContext))
    c_rotations.append(hoisting_keyswitch.eval_fast_rotation( input.cv[1],img_width,digits,cryptoContext))
    c_rotations.append(
        homo_ops.homo_rotate(hoisting_keyswitch.eval_fast_rotation(input.cv[1], padding, digits, cryptoContext), img_width, cryptoContext))
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
        sum=eval_add_many(k_rows)
        if(j==0):
            finalsum=sum.clone()
            finalsum=homo_ops.homo_rotate(finalsum,-1024,cryptoContext)
        else:
            finalsum=homo_ops.homo_add(finalsum,sum,cryptoContext)
            finalsum=homo_ops.homo_rotate(finalsum,-1024,cryptoContext)
    #finalsum=homo_ops.homo_add(finalsum,bias,cryptoContext)
    return finalsum