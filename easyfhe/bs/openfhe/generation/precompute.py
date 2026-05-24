import cmath
import math
from math import log2

import numpy as np

from .plan import get_bootstrap_approx_plan
from .rotations import (
    collapsed_fft_params,
    reduce_rotation,
    select_layers,
)


class BootstrapPrecompute:
    def __init__(
        self,
        N,
        logslot,
        correctionFactor,
        secret_key_dist,
    ):
        self.N = N
        self.M = N * 2
        self.Nh = N >> 1
        self.logslot = logslot
        self.correctionFactor = correctionFactor
        self.secret_key_dist = secret_key_dist
        self.m_U0hatTPreFFT = None
        self.m_U0PreFFT = None
        self.paramsDec = None
        self.paramsEnc = None

        self.m_U0Pre = None
        self.m_U0hatTPre = None

        approx_plan = get_bootstrap_approx_plan(secret_key_dist)
        self.coefficients = np.copy(approx_plan.coefficients)
        self.k = approx_plan.message_scaling_factor


    def eval_bootstrap_setup(self, context, level_budget, dim1, numslots, correction_factor, baby_step=None):

        M = context.M
        slots = M // 4 if numslots == 0 else numslots

        if correction_factor == 0:
            self.correctionFactor = 9
        else:
            self.correctionFactor = correction_factor

        self.m_slots = slots
        self.m_dim1 = dim1[0]

        log_slots = int(math.log2(slots))

        new_budget = [level_budget[0], level_budget[1]]
        for index, budget in enumerate(new_budget):
            if budget < 1 or budget > log_slots:
                label = "encoding" if index == 0 else "decoding"
                raise ValueError(
                    f"bootstrap {label} level budget must be between 1 and log2(slots)={log_slots}, "
                    f"got {budget}"
                )

        baby_step = (None, None) if baby_step is None else baby_step
        self.paramsEnc = collapsed_fft_params(slots, new_budget[0], dim1[0], baby_step=baby_step[0])
        self.paramsDec = collapsed_fft_params(slots, new_budget[1], dim1[1], baby_step=baby_step[1])


        K_SPARSE = 28
        q = context.moduliQ[0]
        q_double = float(q)
        factor = 1 << int(round(math.log2(q_double)))
        pre = q_double / factor
        k = K_SPARSE if context.secretKeyDist == "SPARSE_TERNARY" else 1.0
        scaleEnc = pre / k
        scaleDec = 1 / pre

        self.m_U0hatTPreFFT = self.eval_coeffs_to_slots_precompute(
            self.logslot,
            scaleEnc,
            context,
        )
        self.m_U0PreFFT = self.eval_slots_to_coeffs_precompute(
            self.logslot,
            scaleDec,
            context,
        )

    def coeff_encoding_one_level(self, pows, rot_group, flag_i):
        M_PI = 3.14159265358979323846

        dim = len(pows) - 1
        slots = len(rot_group)

        coeff = [[0.0j] * slots for _ in range(int(3 * math.log2(slots)))]

        m = slots
        while m > 1:
            s = int(log2(m)) - 1

            for k in range(0, slots, m):
                lenh = m >> 1
                lenq = m << 2

                for j in range(lenh):
                    j_twiddle = (lenq - (rot_group[j] % lenq)) * (dim // lenq)

                    if flag_i and (m == 2):
                        w = np.exp(-1j * M_PI / 2) * pows[j_twiddle]
                        coeff[s + int(log2(slots))][j + k] = np.exp(-1j * M_PI / 2)  # not shifted
                        coeff[s + 2 * int(log2(slots))][j + k] = np.exp(-1j * M_PI / 2)  # shifted left
                        coeff[s + int(log2(slots))][j + k + lenh] = -w  # not shifted
                        coeff[s][j + k + lenh] = w  # shifted right
                    else:
                        w = pows[j_twiddle]
                        coeff[s + int(log2(slots))][j + k] = 1  # not shifted
                        coeff[s + 2 * int(log2(slots))][j + k] = 1  # shifted left
                        coeff[s + int(log2(slots))][j + k + lenh] = -w  # not shifted
                        coeff[s][j + k + lenh] = w  # shifted right
            m >>= 1

        return coeff


    def coeff_encoding_collapse(self, pows, rot_group, level_budget, flag_i):
        slots = len(rot_group)
        log_slots = int(log2(slots))
        rotation_mask = slots - 1
        dims = select_layers(log_slots, level_budget)
        layers_collapse = dims[0]
        rem_collapse = dims[2]

        dim_collapse = level_budget
        stop = 0
        flag_rem = 0

        if rem_collapse == 0:
            stop = -1
            flag_rem = 0
        else:
            stop = 0
            flag_rem = 1

        num_rotations = (1 << (layers_collapse + 1)) - 1
        num_rotations_rem = (1 << (rem_collapse + 1)) - 1

        coeff1 = self.coeff_encoding_one_level(pows, rot_group, flag_i)

        coeff = []
        for i in range(dim_collapse):
            if flag_rem:
                if i >= 1:
                    coeff.append([[0j] * slots for _ in range(num_rotations)])
                else:
                    coeff.append([[0j] * slots for _ in range(num_rotations_rem)])
            else:
                coeff.append([[0j] * slots for _ in range(num_rotations)])

        for s in range(dim_collapse - 1, stop, -1):
            top = log_slots - (dim_collapse - 1 - s) * layers_collapse - 1

            for l in range(layers_collapse):
                if l == 0:
                    coeff[s][0] = coeff1[top]
                    coeff[s][1] = coeff1[top + log_slots]
                    coeff[s][2] = coeff1[top + 2 * log_slots]
                else:
                    temp = coeff[s]
                    zeros = [[0.0] * slots for _ in range(num_rotations)]
                    coeff[s] = zeros
                    t = 0
                    shift = 1 << (top - l)
                    coeff_shift_right = coeff1[top - l]
                    coeff_not_shifted = coeff1[top - l + log_slots]
                    coeff_shift_left = coeff1[top - l + 2 * log_slots]

                    for u in range((1 << (l + 1)) - 1):
                        temp_u = temp[u]
                        for k in range(slots):
                            coeff[s][u + t][k] += coeff_shift_right[k] * temp_u[(k - shift) & rotation_mask]
                            coeff[s][u + t + 1][k] += coeff_not_shifted[k] * temp_u[k]
                            coeff[s][u + t + 2][k] += coeff_shift_left[k] * temp_u[(k + shift) & rotation_mask]
                        t += 1

        if flag_rem:
            s = 0
            top = log_slots - (dim_collapse - 1 - s) * layers_collapse - 1

            for l in range(rem_collapse):
                if l == 0:
                    coeff[s][0] = coeff1[top]
                    coeff[s][1] = coeff1[top + log_slots]
                    coeff[s][2] = coeff1[top + 2 * log_slots]
                else:
                    temp = coeff[s]
                    zeros = [[0j] * slots for _ in range(num_rotations_rem)]
                    coeff[s] = zeros
                    t = 0
                    shift = 1 << (top - l)
                    coeff_shift_right = coeff1[top - l]
                    coeff_not_shifted = coeff1[top - l + log_slots]
                    coeff_shift_left = coeff1[top - l + 2 * log_slots]

                    for u in range((1 << (l + 1)) - 1):
                        temp_u = temp[u]
                        for k in range(slots):
                            coeff[s][u + t][k] += coeff_shift_right[k] * temp_u[(k - shift) & rotation_mask]
                            coeff[s][u + t + 1][k] += coeff_not_shifted[k] * temp_u[k]
                            coeff[s][u + t + 2][k] += coeff_shift_left[k] * temp_u[(k + shift) & rotation_mask]
                        t += 1

        return coeff


    def rotate(self, a, index):
        slots = len(a)
        result = np.zeros(slots, dtype=np.complex128)

        if index < 0 or index > slots:
            index = reduce_rotation(index, slots)

        if index == 0:
            result = np.array(a, dtype=np.complex128)
        else:
            result[:slots - index] = a[index:]
            result[slots - index:] = a[:index]

        return result

    def eval_coeffs_to_slots_precompute(self, log_bs_slots, scale, cryptoContext):
        slots = 1 << int(log_bs_slots)
        fivePows = 1

        encode_params_ksiPows = []
        encode_params_rotGroup = []

        m = 4 * slots
        for i in range(slots):
            encode_params_rotGroup.append(fivePows)
            fivePows = (fivePows * 5) % m

        for j in range((4*slots+1)):
            angle = 2.0 * math.pi * j / m
            encode_params_ksiPows.append(cmath.exp(1j * angle))
        encode_params_ksiPows.append(encode_params_ksiPows[0])

        encode_params_ksiPows = np.array(encode_params_ksiPows, dtype=np.complex128)
        encode_params_rotGroup = np.array(encode_params_rotGroup)

        flag_i = False

        level_budget = self.paramsEnc.level_budget
        layers_collapse = self.paramsEnc.layers_coll
        rem_collapse = self.paramsEnc.layers_rem
        num_rotations = self.paramsEnc.num_rotations
        b = self.paramsEnc.baby_step
        g = self.paramsEnc.giant_step
        num_rotations_rem = self.paramsEnc.num_rotations_rem
        b_rem = self.paramsEnc.baby_step_rem
        g_rem = self.paramsEnc.giant_step_rem

        stop = -1
        flag_rem = 0

        if rem_collapse != 0:
            stop = 0
            flag_rem = 1

        result = [[] for _ in range(level_budget)]
        for i in range(level_budget):
            if flag_rem == 1 and i == 0:
                result[i] = [None] * num_rotations_rem
            else:
                result[i] = [None] * num_rotations

        M = cryptoContext.M
        if slots == M // 4:
            coeff = self.coeff_encoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, flag_i)

            for s in range(level_budget - 1, stop, -1):
                for i in range(b):
                    for j in range(g):
                        if g * i + j != num_rotations:
                            rot = reduce_rotation(-g * i * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)), slots)
                            if flag_rem == 0 and s == stop + 1:
                                for k in range(slots):
                                    coeff[s][g * i + j][k] *= scale

                            rotate_temp = self.rotate(coeff[s][g * i + j], rot)
                            result[s][g * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)

            if flag_rem:
                for i in range(b_rem):
                    for j in range(g_rem):
                        if g_rem * i + j != num_rotations_rem:
                            rot = reduce_rotation(-g_rem * i, slots)
                            for k in range(slots):
                                coeff[stop][g_rem * i + j][k] *= scale

                            rotate_temp = self.rotate(coeff[stop][g_rem * i + j], rot)
                            result[stop][g_rem * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)

        else:
            coeff = self.coeff_encoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, False)
            coeffi = self.coeff_encoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, True)

            for s in range(level_budget - 1, stop, -1):
                for i in range(b):
                    for j in range(g):
                        if g * i + j != num_rotations:
                            rot = reduce_rotation(-g * i * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)), M // 4)
                            clear_temp = coeff[s][g * i + j] + coeffi[s][g * i + j]
                            if flag_rem == 0 and s == stop + 1:
                                for k in range(len(clear_temp)):
                                    clear_temp[k] *= scale

                            rotate_temp = self.rotate(clear_temp, rot)
                            result[s][g * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)

            if flag_rem:
                for i in range(b_rem):
                    for j in range(g_rem):
                        if g_rem * i + j != num_rotations_rem:
                            rot = reduce_rotation(-g_rem * i, M // 4)
                            clear_temp = coeff[stop][g_rem * i + j] + coeffi[stop][g_rem * i + j]
                            for k in range(len(clear_temp)):
                                clear_temp[k] *= scale

                            rotate_temp = self.rotate(clear_temp, rot)
                            result[stop][g_rem * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)
        return result


    def coeff_decoding_one_level(self, pows, rot_group, flag_i):
        M_PI = 3.14159265358979323846

        dim = len(pows) - 1
        slots = len(rot_group)

        coeff = [[0.0j] * slots for _ in range(int(3 * math.log2(slots)))]

        m = 2
        while m <= slots:
            s = int(log2(m)) - 1

            for k in range(0, slots, m):
                lenh = m >> 1
                lenq = m << 2

                for j in range(lenh):
                    j_twiddle = (rot_group[j] % lenq) * (dim // lenq)

                    if flag_i and (m == 2):
                        w = np.exp(M_PI / 2 * 1j) * pows[j_twiddle]
                        coeff[s + int(math.log2(slots))][j + k] = np.exp(M_PI / 2 * 1j)  # not shifted
                        coeff[s + 2 * int(math.log2(slots))][j + k] = w  # shifted left
                        coeff[s + int(math.log2(slots))][j + k + lenh] = -w  # not shifted
                        coeff[s][j + k + lenh] = np.exp(M_PI / 2 * 1j)  # shifted right
                    else:
                        w = pows[j_twiddle]
                        coeff[s + int(log2(slots))][j + k] = 1  # not shifted
                        coeff[s + 2 * int(log2(slots))][j + k] = w  # shifted left
                        coeff[s + int(log2(slots))][j + k + lenh] = -w  # not shifted
                        coeff[s][j + k + lenh] = 1  # shifted right
            m <<= 1

        return coeff


    def coeff_decoding_collapse(self, pows, rot_group, level_budget, flag_i):
        slots = len(rot_group)
        log_slots = int(log2(slots))

        dims = select_layers(log_slots, level_budget)
        layers_collapse = dims[0]
        rows_collapse = dims[1]
        rem_collapse = dims[2]

        dim_collapse = level_budget
        flag_rem = 0

        if rem_collapse == 0:
            flag_rem = 0
        else:
            flag_rem = 1

        num_rotations = (1 << (layers_collapse + 1)) - 1
        num_rotations_rem = (1 << (rem_collapse + 1)) - 1

        coeff1 = self.coeff_decoding_one_level(pows, rot_group, flag_i)

        coeff = []
        for i in range(dim_collapse):
            if flag_rem:
                if i < level_budget - 1:
                    coeff.append([[0j] * slots for _ in range(num_rotations)])
                else:
                    coeff.append([[0j] * slots for _ in range(num_rotations_rem)])
            else:
                coeff.append([[0j] * slots for _ in range(num_rotations)])

        for s in range(rows_collapse):
            for l in range(layers_collapse):
                if l == 0:
                    coeff[s][0] = coeff1[s * layers_collapse]
                    coeff[s][1] = coeff1[s * layers_collapse + log_slots]
                    coeff[s][2] = coeff1[s * layers_collapse + 2 * log_slots]
                else:
                    temp = coeff[s]
                    zeros = [[0.0] * slots for _ in range(num_rotations)]
                    coeff[s] = zeros
                    coeff_not_shifted = coeff1[s * layers_collapse + l]
                    coeff_shift_left = coeff1[s * layers_collapse + l + log_slots]
                    coeff_shift_right = coeff1[s * layers_collapse + l + 2 * log_slots]

                    for t in range(3):
                        for u in range((1 << (l + 1)) - 1):
                            temp_u = temp[u]
                            for k in range(slots):
                                if t == 0:
                                    coeff[s][u][k] += coeff_not_shifted[k] * temp_u[k]
                                elif t == 1:
                                    coeff[s][u + (1 << l)][k] += coeff_shift_left[k] * temp_u[k]
                                elif t == 2:
                                    coeff[s][u + (1 << (l + 1))][k] += \
                                    coeff_shift_right[k] * temp_u[k]

        if flag_rem:
            s = rows_collapse
            for l in range(rem_collapse):
                if l == 0:
                    coeff[s][0] = coeff1[s * layers_collapse]
                    coeff[s][1] = coeff1[s * layers_collapse + log_slots]
                    coeff[s][2] = coeff1[s * layers_collapse + 2 * log_slots]
                else:
                    temp = coeff[s]
                    zeros = [[0j] * slots for _ in range(num_rotations_rem)]
                    coeff[s] = zeros
                    coeff_not_shifted = coeff1[s * layers_collapse + l]
                    coeff_shift_left = coeff1[s * layers_collapse + l + log_slots]
                    coeff_shift_right = coeff1[s * layers_collapse + l + 2 * log_slots]

                    for t in range(3):
                        for u in range((1 << (l + 1)) - 1):
                            temp_u = temp[u]
                            for k in range(slots):
                                if t == 0:
                                    coeff[s][u][k] += coeff_not_shifted[k] * temp_u[k]
                                elif t == 1:
                                    coeff[s][u + (1 << l)][k] += coeff_shift_left[k] * temp_u[k]
                                elif t == 2:
                                    coeff[s][u + (1 << (l + 1))][k] += \
                                    coeff_shift_right[k] * temp_u[k]
        return coeff


    def eval_slots_to_coeffs_precompute(self, log_bs_slots, scale, cryptoContext):
        slots = 1 << int(log_bs_slots)
        fivePows = 1

        encode_params_ksiPows = []
        encode_params_rotGroup = []

        m = 4 * slots
        for i in range(slots):
            encode_params_rotGroup.append(fivePows)
            fivePows = (fivePows * 5) % m

        for j in range((4 * slots + 1)):
            angle = 2.0 * math.pi * j / m
            encode_params_ksiPows.append(cmath.exp(1j * angle))
        encode_params_ksiPows.append(encode_params_ksiPows[0])

        encode_params_ksiPows = np.array(encode_params_ksiPows, dtype=np.complex128)
        encode_params_rotGroup = np.array(encode_params_rotGroup)

        flag_i = False

        level_budget = self.paramsDec.level_budget
        layers_collapse = self.paramsDec.layers_coll
        rem_collapse = self.paramsDec.layers_rem
        num_rotations = self.paramsDec.num_rotations
        b = self.paramsDec.baby_step
        g = self.paramsDec.giant_step
        num_rotations_rem = self.paramsDec.num_rotations_rem
        b_rem = self.paramsDec.baby_step_rem
        g_rem = self.paramsDec.giant_step_rem

        flag_rem = 0

        if rem_collapse != 0:
            flag_rem = 1

        result = [[] for _ in range(level_budget)]
        for i in range(level_budget):
            if flag_rem == 1 and i == (level_budget - 1):
                result[i] = [None] * num_rotations_rem
            else:
                result[i] = [None] * num_rotations

        M = cryptoContext.M
        if slots == M // 4:
            coeff = self.coeff_decoding_collapse(
                encode_params_ksiPows,
                encode_params_rotGroup,
                level_budget,
                flag_i,
            )

            for s in range(level_budget - flag_rem):
                for i in range(b):
                    for j in range(g):
                        if g * i + j != num_rotations:
                            rot = reduce_rotation(-g * i * (1 << (s * layers_collapse)), slots)
                            if flag_rem == 0 and s == level_budget - flag_rem - 1:
                                for k in range(slots):
                                    coeff[s][g * i + j][k] *= scale

                            rotate_temp = self.rotate(coeff[s][g * i + j], rot)
                            result[s][g * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)

            if flag_rem:
                s = level_budget - flag_rem
                for i in range(b_rem):
                    for j in range(g_rem):
                        if g_rem * i + j != num_rotations_rem:
                            rot = reduce_rotation(-g_rem * i * (1 << (s * layers_collapse)), slots)
                            for k in range(slots):
                                coeff[s][g_rem * i + j][k] *= scale

                            rotate_temp = self.rotate(coeff[s][g_rem * i + j], rot)
                            result[s][g_rem * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)

        else:
            coeff = self.coeff_decoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, False)
            coeffi = self.coeff_decoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, True)

            for s in range(level_budget - flag_rem):
                for i in range(b):
                    for j in range(g):
                        if g * i + j != num_rotations:
                            rot = reduce_rotation(-g * i * (1 << (s * layers_collapse)), M // 4)
                            clear_temp = coeff[s][g * i + j] + coeffi[s][g * i + j]
                            if flag_rem == 0 and s == level_budget - flag_rem - 1:
                                for k in range(len(clear_temp)):
                                    clear_temp[k] *= scale

                            rotate_temp = self.rotate(clear_temp, rot)

                            result[s][g * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)

            if flag_rem:
                s = level_budget - flag_rem
                for i in range(b_rem):
                    for j in range(g_rem):
                        if g_rem * i + j != num_rotations_rem:
                            rot = reduce_rotation(-g_rem * i * (1 << (s * layers_collapse)), M // 4)
                            clear_temp = coeff[s][g_rem * i + j] + coeffi[s][g_rem * i + j]
                            for k in range(len(clear_temp)):
                                clear_temp[k] *= scale

                            rotate_temp = self.rotate(clear_temp, rot)
                            result[s][g_rem * i + j] = np.asarray(rotate_temp, dtype=np.complex128).reshape(-1)
        return result
