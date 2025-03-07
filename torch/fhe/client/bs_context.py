import numpy as np
import math
from ..ciphertext import Plaintext


class CKKS_Boot_Params:
    def __init__(
        self,
        level_budget,
        layers_coll,
        layers_rem,
        num_rotations,
        baby_step,
        giant_step,
        num_rotations_rem,
        baby_step_rem,
        giant_step_rem,
    ):
        self.level_budget = level_budget
        self.layers_coll = layers_coll
        self.layers_rem = layers_rem
        self.num_rotations = num_rotations
        self.baby_step = baby_step
        self.giant_step = giant_step
        self.num_rotations_rem = num_rotations_rem
        self.baby_step_rem = baby_step_rem
        self.giant_step_rem = giant_step_rem


class BsContext:
    def __init__(
        self, N, moduliQ_scalar, moduliP_scalar, q_mu, p_mu, correctionFactor, BOOT_CNST
    ):
        K = len(moduliP_scalar)
        self.M = N * 2
        self.correctionFactor = correctionFactor
        self.m_U0hatTPre = None
        self.m_U0hatTPreFFT = None
        self.m_U0Pre = None
        self.m_U0PreFFT = None
        self.paramsDec = None
        self.paramsEnc = None

        self.m_U0hatTPreFFT_mx = BOOT_CNST["C2S"]
        self.m_U0PreFFT_mx = BOOT_CNST["S2C"]
        self.m_U0hatTPreFFT_dim = BOOT_CNST["C2S_dim"]
        self.m_U0PreFFT_dim = BOOT_CNST["S2C_dim"]
        self.m_U0hatTPreFFT_limbs = BOOT_CNST["C2S_limbs"]
        self.m_U0PreFFT_limbs = BOOT_CNST["S2C_limbs"]

        coefficients = np.array(
            [
                -0.18646470117093214, 0.036680543700430925, -0.20323558926782626, 0.029327390306199311,
                -0.24346234149506416, 0.011710240188138248, -0.27023281815251715, -0.017621188001030602,
                -0.21383614034992021, -0.048567932060728937, -0.013982336571484519, -0.051097367628344978,
                0.24300487324019346, 0.0016547743046161035, 0.23316923792642233, 0.060707936480887646,
                -0.18317928363421143, 0.0076878773048247966, -0.24293447776635235, -0.071417413140564698,
                0.37747441314067182, 0.065154496937795681, -0.24810721693607704, -0.033588418808958603,
                0.10510660697380972, 0.012045222815124426, -0.032574751830745423, -0.0032761730196023873,
                0.0078689491066424744, 0.00070965574480802061, -0.0015405394287521192, -0.00012640521062948649,
                0.00025108496615830787, 0.000018944629154033562, -0.000034753284216308228, -2.4309868106111825e-6,
                4.1486274737866247e-6, 2.7079833113674568e-7, -4.3245388569898879e-7, -2.6482744214856919e-8,
                3.9770028771436554e-8, 2.2951153557906580e-9, -3.2556026220554990e-9, -1.7691071323926939e-10,
                2.5459052150406730e-10
            ],
            dtype=np.float64,
        )

        self.coefficients = np.copy(coefficients)
        self.k = 1.0

        self.QplusP_map = {}
        self.QmuplusPmu_map = {}
        for cur_limbs in range(len(moduliQ_scalar)):
            self.QplusP_map[cur_limbs] = np.array(
                np.concatenate((moduliQ_scalar[0:cur_limbs], moduliP_scalar[0:K])),
                dtype=np.uint64,
            )
            self.QmuplusPmu_map[cur_limbs] = np.array(
                np.concatenate((q_mu[0:cur_limbs], p_mu[:K])), dtype=np.uint64
            )

    def compute_C2S_rot(self, slots, M):
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

        rot_in = [[] for _ in range(level_budget)]
        for i in range(level_budget):
            if flag_rem == 1 and i == 0:
                rot_in[i] = [0] * (num_rotations_rem + 1)
            else:
                rot_in[i] = [0] * (num_rotations + 1)

        rot_out = [[] for _ in range(level_budget)]
        for i in range(level_budget):
            rot_out[i] = [0] * (b + b_rem)

        for s in range(level_budget - 1, stop, -1):
            for j in range(g):
                rot_in[s][j] = self.reduce_rotation(
                    (j - (num_rotations + 1) // 2 + 1)
                    * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)),
                    slots,
                )

            for i in range(b):
                rot_out[s][i] = self.reduce_rotation(
                    (g * i) * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)),
                    M // 4,
                )

        if flag_rem:
            for j in range(g_rem):
                rot_in[stop][j] = self.reduce_rotation(
                    (j - (num_rotations_rem + 1) // 2 + 1), slots
                )

            for i in range(b_rem):
                rot_out[stop][i] = self.reduce_rotation((g_rem * i), M // 4)

        self.C2S_rot_in = rot_in
        self.C2S_rot_out = rot_out

    def compute_S2C_rot(self, slots, M):
        level_budget = self.paramsDec.level_budget
        layers_collapse = self.paramsDec.layers_coll
        rem_collapse = self.paramsDec.layers_rem
        num_rotations = self.paramsDec.num_rotations
        b = self.paramsDec.baby_step
        g = self.paramsDec.giant_step
        num_rotations_rem = self.paramsDec.num_rotations_rem
        b_rem = self.paramsDec.baby_step_rem
        g_rem = self.paramsDec.giant_step_rem

        flag_rem = 1 if rem_collapse != 0 else 0

        rot_in = []
        rot_out = []

        for i in range(level_budget):
            if flag_rem == 1 and i == (level_budget - 1):
                rot_in.append([0] * (num_rotations_rem + 1))

            else:
                rot_in.append([0] * (num_rotations + 1))
        for i in range(level_budget):
            rot_out.append([0] * (b + b_rem))

        for s in range(level_budget - flag_rem):
            for j in range(g):
                rot_in[s][j] = self.reduce_rotation(
                    (j - ((num_rotations + 1) / 2) + 1) * (1 << (s * layers_collapse)),
                    M // 4,
                )

            for i in range(b):
                rot_out[s][i] = self.reduce_rotation(
                    (g * i) * (1 << (s * layers_collapse)), M // 4
                )

        if flag_rem:
            s = level_budget - flag_rem
            for j in range(g_rem):
                rot_in[s][j] = self.reduce_rotation(
                    (j - (num_rotations_rem + 1) // 2 + 1)
                    * (1 << (s * layers_collapse)),
                    M // 4,
                )

            for i in range(b_rem):
                rot_out[s][i] = self.reduce_rotation(
                    (g_rem * i) * (1 << (s * layers_collapse)), M // 4
                )

        self.S2C_rot_in = rot_in
        self.S2C_rot_out = rot_out

    def SelectLayers(self, logBsSlots, budget):
        layers = math.ceil(logBsSlots / budget)
        rows = logBsSlots // layers
        rem = logBsSlots % layers

        dim = rows
        if rem != 0:
            dim = rows + 1

        if dim < budget:
            layers -= 1
            rows = logBsSlots // layers
            rem = logBsSlots - rows * layers
            dim = rows

            if rem != 0:
                dim = rows + 1

            while dim != budget:
                rows -= 1
                rem = logBsSlots - rows * layers
                dim = rows
                if rem != 0:
                    dim = rows + 1

        return [layers, rows, rem]

    def GetCollapsedFFTParams(self, slots, levelBudget, dim1):
        dims = self.SelectLayers(int(math.log2(slots)), levelBudget)
        layersCollapse = dims[0]
        remCollapse = dims[2]

        flagRem = 1 if remCollapse != 0 else 0

        numRotations = (1 << (layersCollapse + 1)) - 1
        numRotationsRem = (1 << (remCollapse + 1)) - 1
        if dim1 == 0 or dim1 > numRotations:
            if numRotations > 7:
                g = 1 << (int(layersCollapse / 2) + 2)
            else:
                g = 1 << (int(layersCollapse / 2) + 1)
        else:
            g = dim1

        b = (numRotations + 1) // g
        bRem = 0
        gRem = 0

        if flagRem:
            if numRotationsRem > 7:
                gRem = 1 << (int(remCollapse / 2) + 2)
            else:
                gRem = 1 << (int(remCollapse / 2) + 1)
            bRem = (numRotationsRem + 1) // gRem

        return CKKS_Boot_Params(
            int(levelBudget),
            layersCollapse,
            remCollapse,
            int(numRotations),
            b,
            g,
            int(numRotationsRem),
            bRem,
            gRem,
        )

    def reduce_rotation(self, index, slots):
        islots = int(slots)
        index = int(index)

        if (int(slots) & int(slots - 1)) == 0:
            n = int(math.log2(slots))
            if index >= 0:
                return index - ((index >> n) << n)
            return index + islots + ((abs(index) >> n) << n)

        return (islots + index % islots) % islots

    def eval_bootstrap_setup(
        self, context, level_budget, dim1, numslots, correction_factor
    ):

        m_U0hatTPreFFT_dim1 = len(self.m_U0hatTPreFFT_dim)
        m_U0hatTPreFFT_dim2 = self.m_U0hatTPreFFT_dim
        m_U0hatTPreFFT_limbs = self.m_U0hatTPreFFT_limbs
        mx_len = context.N
        mx_slots = numslots
        m_U0PreFFT_dim1 = len(self.m_U0PreFFT_dim)
        m_U0PreFFT_dim2 = self.m_U0PreFFT_dim
        m_U0PreFFT_limbs = self.m_U0PreFFT_limbs

        M = context.M
        slots = M // 4 if numslots == 0 else numslots

        if correction_factor == 0:
            self.correctionFactor = 9
        else:
            self.correctionFactor = correction_factor

        self.m_slots = slots
        self.m_dim1 = dim1[0]

        log_slots = math.log2(slots)

        new_budget = [level_budget[0], level_budget[1]]

        if level_budget[0] > log_slots:
            print(
                f"\nWarning, the level budget for encoding cannot be this large. "
                f"The budget was changed to {int(log_slots)}"
            )
            new_budget[0] = int(log_slots)
        if level_budget[0] < 1:
            print(
                f"\nWarning, the level budget for encoding has to be at least 1. "
                f"The budget was changed to 1"
            )
            new_budget[0] = 1

        if level_budget[1] > log_slots:
            print(
                f"\nWarning, the level budget for decoding cannot be this large. "
                f"The budget was changed to {int(log_slots)}"
            )
            new_budget[1] = int(log_slots)
        if level_budget[1] < 1:
            print(
                f"\nWarning, the level budget for decoding has to be at least 1. "
                f"The budget was changed to 1"
            )
            new_budget[1] = 1

        self.paramsEnc = self.GetCollapsedFFTParams(slots, new_budget[0], dim1[0])
        self.paramsDec = self.GetCollapsedFFTParams(slots, new_budget[1], dim1[1])

        self.compute_C2S_rot(slots, self.M)
        self.compute_S2C_rot(slots, self.M)

        assert (
            not (m_U0hatTPreFFT_dim1 == 1 and m_U0PreFFT_dim1 == 1)
            and "Not Implemented"
        )

        RHScnt = 0
        cnt = 0
        sizeP = context.K
        self.m_U0hatTPreFFT = [[0] * i for i in m_U0hatTPreFFT_dim2]
        for i in range(0, m_U0hatTPreFFT_dim1):
            j_len = m_U0hatTPreFFT_dim2[i]
            limbs = m_U0hatTPreFFT_limbs[i]
            m_U0hatTPreFFT_len = mx_len * limbs
            for j in range(j_len):
                m_U0hatTPreFFT = self.m_U0hatTPreFFT_mx[
                    RHScnt : RHScnt + m_U0hatTPreFFT_len
                ].copy()
                RHScnt += m_U0hatTPreFFT_len
                self.m_U0hatTPreFFT[i][j] = Plaintext(
                    m_U0hatTPreFFT, limbs - sizeP, 1, mx_slots, True
                )
                cnt += 1
        self.m_U0hatTPreFFT_mx = None

        RHScnt = 0
        cnt = 0
        self.m_U0PreFFT = [[0] * i for i in m_U0PreFFT_dim2]
        for i in range(m_U0PreFFT_dim1):
            j_len = m_U0PreFFT_dim2[i]
            limbs = m_U0PreFFT_limbs[i]
            m_U0PreFFT_len = mx_len * limbs
            for j in range(j_len):
                m_U0PreFFT = self.m_U0PreFFT_mx[RHScnt : RHScnt + m_U0PreFFT_len].copy()
                RHScnt += m_U0PreFFT_len
                self.m_U0PreFFT[i][j] = Plaintext(
                    m_U0PreFFT, limbs - sizeP, 1, mx_slots, True
                )
                cnt += 1
        self.m_U0PreFFT_mx = None
