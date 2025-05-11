import numpy as np
import math
from program import * # therefore we dont need to import it when using the miniMax_relu
from PolyUpdate import *
from MinicompFunc import *
import torch.fhe as fhe

def geneT0T1(cipher, cryptoContext):
    """
    construct T0 = encrypt(1.0), T1 = cipher
    """
    # scale = cipher.scaling_factor
    T0 = cryptoContext.ones_Nh.deep_copy()
    T1 = cipher.deep_copy()

    return T0, T1


def evalT(Tm, Tn, Tmminusn, cryptoContext):
    """
    compute T_{m+n}(x) = 2 * Tm * Tn - T_{|m-n|}, corresponding to Chebyshev recurrence formula
    """
    temp = fhe.homo_mul(Tm, Tn, cryptoContext)         # Tm * Tn
    temp = fhe.homo_add(temp, temp, cryptoContext)     # 2 * Tm * Tn
    temp = fhe.homo_rescale(temp, 1, cryptoContext)
    Tmplusn = fhe.homo_sub(temp, Tmminusn, cryptoContext)
    return Tmplusn


def eval_polynomial_integrate(cipher, deg, decomp_coeff, tree, cryptoContext):
    """
    Evaluate a polynomial of degree `deg` on encrypted input `cipher` using a decomposition tree.
    The result is a homomorphic ciphertext encoding the evaluated polynomial.
    """
    scale = cipher.scaling_factor #original seal: 2^42. exact value.
    Nh = cryptoContext.N // 2
    total_depth = int(math.ceil(math.log2(deg + 1)))

    eval_type = tree.type

    decomp_deg = [-1] * (2 ** (tree.depth + 1))
    start_index = [-1] * (2 ** (tree.depth + 1))

    T = [None for _ in range(100)]
    pt = [None for _ in range(100)]

    ctxt_zero = cryptoContext.zeros_Nh

    # Initial term index for coefficient mapping
    if eval_type == EvalType.ODDBABY:
        temp_index = 1
    elif eval_type == EvalType.BABY:
        temp_index = 0
    else:
        raise ValueError("Unknown evaluation type.")

    # Initialize degree decomposition tree
    decomp_deg[1] = deg
    for i in range(1, tree.depth + 1):
        for j in range(2 ** i, 2 ** (i + 1)):
            if j >= len(decomp_deg):
                raise ValueError("Invalid tree index")
            parent = j // 2
            if j % 2 == 0:
                decomp_deg[j] = tree.tree[parent] - 1
            else:
                decomp_deg[j] = decomp_deg[parent] - tree.tree[parent]

    # Compute start index for each leaf
    for i in range(1, 2 ** (tree.depth + 1)):
        if tree.tree[i] == 0:
            start_index[i] = temp_index
            temp_index += decomp_deg[i] + 1

    # Generate T[0], T[1]
    T[0], T[1] = geneT0T1(cipher, cryptoContext)

    # Will continue with stage loop based on eval_type...
    # Part 2 will implement ODDBABY and BABY branches

    if eval_type == EvalType.ODDBABY:
        # i = stage index, from 1 to total_depth
        for i in range(1, total_depth + 1):
            # Evaluate at leaves: depth-i stage, leaf nodes (tree.tree[j] == 0)
            for j in range(1, 2 ** (tree.depth + 1)):
                if tree.tree[j] == 0 and total_depth + 1 - num_one(j) == i:
                    temp_idx = start_index[j]
                    pt[j] = fhe.homo_mul_scalar_double(T[1], decomp_coeff[temp_idx], cryptoContext)
                    temp_idx += 2

                    for k in range(3, decomp_deg[j] + 1, 2):
                        if T[k] is None:
                            raise ValueError(f"T[{k}] is None")
                        term = fhe.homo_mul_scalar_double(T[k], decomp_coeff[temp_idx], cryptoContext)
                        pt[j] = fhe.homo_add(pt[j], term, cryptoContext)  # Lazy scaling
                        temp_idx += 2

                    pt[j] = fhe.homo_rescale(pt[j], 1, cryptoContext)

            # Evaluate at internal intersections (odd indices, tree.tree[j] > 0)
            for j in range(1, 2 ** (tree.depth + 1)):
                if tree.tree[j] > 0 and total_depth + 1 - num_one(j) == i and j % 2 == 1:
                    k = j
                    if T[tree.tree[k]] is None or pt[2 * k + 1] is None:
                        raise ValueError("Required T or pt missing at internal node")

                    pt[j] = fhe.homo_mul(T[tree.tree[k]], pt[2 * k + 1], cryptoContext)
                    k *= 2
                    while tree.tree[k] > 0:
                        term = fhe.homo_mul(T[tree.tree[k]], pt[2 * k + 1], cryptoContext)
                        pt[j] = fhe.homo_add(pt[j], term, cryptoContext)
                        k *= 2

                    pt[j] = fhe.homo_rescale(pt[j], 1, cryptoContext)
                    pt[j] = fhe.homo_add(pt[j], pt[k], cryptoContext)

            # Evaluate T powers needed for next stage
            if i <= tree.m - 1:
                T[2 ** i] = evalT(T[2 ** (i - 1)], T[2 ** (i - 1)], T[0], cryptoContext)

            if i <= tree.l:
                for j in range(2 ** (i - 1) + 1, 2 ** i, 2):  # Odd indices
                    T[j] = evalT(T[2 ** (i - 1)], T[j - 2 ** (i - 1)], T[2 ** i - j], cryptoContext)

        return pt[1]
    elif eval_type == EvalType.BABY:
        for i in range(1, total_depth + 1):
            # Evaluate leaf endpoints
            for j in range(1, 2 ** (tree.depth + 1)):
                if tree.tree[j] == 0 and total_depth + 1 - num_one(j) == i:
                    temp_idx = start_index[j]
                    pt[j] = ctxt_zero.deep_copy()
                    for k in range(decomp_deg[j] + 1):
                        coeff = decomp_coeff[temp_idx]
                        if abs(coeff) > 1.0 / scale:
                            if T[k] is None:
                                raise ValueError(f"T[{k}] is None")
                            term = fhe.homo_mul_scalar_double(T[k], coeff, cryptoContext)
                            pt[j] = fhe.homo_add(pt[j], term, cryptoContext)
                        temp_idx += 1
                    pt[j] = fhe.homo_rescale(pt[j], 1, cryptoContext)

            # Evaluate inner intersections (avoid redundant recomputation via ancestry check)
            seen = set()
            for j in range(1, 2 ** (tree.depth + 1)):
                if tree.tree[j] > 0 and total_depth + 1 - num_one(j) == i:
                    temp = j
                    while temp not in seen and temp > 1:
                        if temp % 2 == 0:
                            temp //= 2
                        else:
                            break
                    if temp in seen:
                        continue

                    seen.add(j)
                    k = j
                    if T[tree.tree[k]] is None or pt[2 * k + 1] is None:
                        raise ValueError("Missing operands in tree node evaluation")

                    pt[j] = fhe.homo_mul(T[tree.tree[k]], pt[2 * k + 1], cryptoContext)
                    k *= 2
                    while tree.tree[k] > 0:
                        if T[tree.tree[k]] is None or pt[2 * k + 1] is None:
                            raise ValueError("Missing recursive operands")
                        term = fhe.homo_mul(T[tree.tree[k]], pt[2 * k + 1], cryptoContext)
                        pt[j] = fhe.homo_add(pt[j], term, cryptoContext)
                        k *= 2
                    pt[j] = fhe.homo_rescale(pt[j], 1, cryptoContext)
                    pt[j] = fhe.homo_add(pt[j], pt[k], cryptoContext)

            # Evaluate needed T_g terms
            for g in range(2, tree.b + 1):
                if 2 ** (i - 1) < g <= 2 ** i:
                    if g % 2 == 0:
                        T[g] = evalT(T[g // 2], T[g // 2], T[0], cryptoContext)
                    else:
                        T[g] = evalT(T[g // 2], T[(g + 1) // 2], T[1], cryptoContext)

            for j in range(1, tree.m):
                g = (2 ** j) * tree.b
                if 2 ** (i - 1) < g <= 2 ** i:
                    if g % 2 == 0:
                        T[g] = evalT(T[g // 2], T[g // 2], T[0], cryptoContext)
                    else:
                        T[g] = evalT(T[g // 2], T[(g + 1) // 2], T[1], cryptoContext)

        return pt[1]

    return None


def coeff_number(deg: int, tree) -> int:
    """
    Compute the number of coefficients needed for evaluating a polynomial of degree `deg`
    over the decomposition tree `tree`.

    Args:
        deg: Degree of the polynomial.
        tree: A Tree object representing the evaluation structure.

    Returns:
        Total number of coefficients needed.
    """
    from MinicompFunc import pow2

    num = 0
    size = pow2(tree.depth + 1)
    decomp_deg = [0] * size
    decomp_deg[1] = deg

    for i in range(1, tree.depth + 1):
        for j in range(pow2(i), pow2(i + 1)):
            parent = j // 2
            if j % 2 == 0:
                decomp_deg[j] = tree.tree[parent] - 1
            else:
                decomp_deg[j] = decomp_deg[parent] - tree.tree[parent]

    for i in range(size):
        if tree.tree[i] == 0:
            num += decomp_deg[i] + 1

    return num


def show_failure_relu(cipher, ground_truth_vec, precision, cryptoContext):
    """
    Compare homomorphic ReLU result with ground truth.

    Args:
        cipher: Ciphertext after ReLU approximation.
        ground_truth_vec: Plain input vector (list or numpy array).
        precision: Bit precision for tolerance bound (e.g., 40 => 2^-40).
        cryptoContext: Crypto context with decryptor and encoder.

    Returns:
        Number of positions where |ReLU(x) - output[i]| > 2^-precision.
    """

    bound = 2 ** (-precision)
    output = cryptoContext.openfhe_context.decrypt(cipher).cpu().numpy().reshape(-1)
    failure = 0

    for i in range(len(ground_truth_vec)):
        gt_val = ground_truth_vec[i]
        gt_relu = gt_val if gt_val > 0 else 0.0
        if abs(gt_relu - output[i]) > bound:
            failure += 1

    print("-------------------------------------------------")
    print(f"Failure count: {failure}")
    print("-------------------------------------------------")
    return failure



def minimax_relu(comp_no, deg_list, alpha, tree_list, scaled_val, cipher_in, cryptoContext):
    """
    Evaluate a minimax-based ReLU approximation on ciphertext `cipher_in`.
    Args:
        comp_no: number of component polynomials in the composite.
        deg_list: list of degrees for each component.
        alpha: index for coefficient file.
        tree_list: list of Tree objects for each component.
        scaled_val: final scaling adjustment (typically 1.0).
        cipher_in: input ciphertext.
        cryptoContext: context holding HE parameters and openfhe_context.
    Returns:
        Ciphertext representing approximate ReLU(x).
    """

    # Load coefficients
    decomp_coeff = []
    path = f"./coeffs/d{alpha}.txt"
    with open(path, "r") as f:
        for i in range(comp_no):
            coeffs = []
            coeff_count = coeff_number(deg_list[i], tree_list[i])
            for _ in range(coeff_count):
                coeffs.append(float(f.readline().strip()))
            decomp_coeff.append(coeffs)

    # Scale adjustments
    scale_val = [1.0] + [2.0] * (comp_no - 2) + [scaled_val]
    for i in range(comp_no - 1):
        decomp_coeff[i] = [c / scale_val[i + 1] for c in decomp_coeff[i]]
    decomp_coeff[-1] = [c * 0.5 for c in decomp_coeff[-1]]  # final stage scales by 1/2

    # Evaluate f(x) = composite polynomial (inplace on cipher_x)
    cipher_x = cipher_in.deep_copy()
    for i in range(comp_no):
        cipher_x = eval_polynomial_integrate(cipher_x, deg_list[i], decomp_coeff[i], tree_list[i], cryptoContext)

    # Compute (1 + f(x)) / 2
    # fixme: should check whether or not slots here can be hardcoded to Nh
    # slots = cipher_x.slots
    # half_vec = np.full(slots, 0.5, dtype=np.float64)
    # cipher_half = cryptoContext.openfhe_context.encrypt(half_vec, 1, cryptoContext.L - cipher_x.cur_limbs, slots)
    cipher_half = cryptoContext.cipher_half

    temp = fhe.homo_add(cipher_x, cipher_half, cryptoContext)
    result = fhe.homo_mul(temp, cipher_in, cryptoContext)
    result = fhe.homo_rescale(result, 1, cryptoContext)

    return result


