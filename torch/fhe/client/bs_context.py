import numpy as np
import math
from ..ciphertext import Plaintext

K_UNIFORM = 512

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
        self.level_budget = level_budget  # the level budget
        self.layers_coll = layers_coll  # the number of layers to collapse in one level
        self.layers_rem = layers_rem  # the number of layers remaining to be collapsed in one level to have exactly the number of levels specified in the level budget
        self.num_rotations = num_rotations  # the number of rotations in one level
        self.baby_step = baby_step  # the baby step in the baby-step giant-step strategy
        self.giant_step = (
            giant_step  # the giant step in the baby-step giant-step strategy
        )
        self.num_rotations_rem = (
            num_rotations_rem  # the number of rotations in the remaining level
        )
        self.baby_step_rem = baby_step_rem  # the baby step in the baby-step giant-step strategy for the remaining level
        self.giant_step_rem = giant_step_rem  # the giant step in the baby-step giant-step strategy for the remaining level
        self.total_elements = 9


def round_half_away_from_zero(number, ndigits=0):
    multiplier = 10 ** ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    elif number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    else:
        return 0.0

class BsContext:
    def __init__(
        self,
        N,
        moduliP_scalar,
        correctionFactor,
        secretKeyDist,
        BOOT_CNST
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
        self.m_U0hatTPreFFT_scaling_factor = BOOT_CNST["U0hatTPreFFTScalingFactor"]
        self.m_U0PreFFT_scaling_factor = BOOT_CNST["U0PreFFTScalingFactor"]

        coefficientsSparse = np.array(
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

        coefficientsUniform = np.array(
            [
                0.15421426400235561,
                -0.0037671538417132409,
                0.16032011744533031,
                -0.0034539657223742453,
                0.17711481926851286,
                -0.0027619720033372291,
                0.19949802549604084,
                -0.0015928034845171929,
                0.21756948616367638,
                0.00010729951647566607,
                0.21600427371240055,
                0.0022171399198851363,
                0.17647500259573556,
                0.0042856217194480991,
                0.086174491919472254,
                0.0054640252312780444,
                -0.046667988130649173,
                0.0047346914623733714,
                -0.17712686172280406,
                0.0016205080004247200,
                -0.22703114241338604,
                -0.0028145845916205865,
                -0.13123089730288540,
                -0.0056345646688793190,
                0.078818395388692147,
                -0.0037868875028868542,
                0.23226434602675575,
                0.0021116338645426574,
                0.13985510526186795,
                0.0059365649669377071,
                -0.13918475289368595,
                0.0018580676740836374,
                -0.23254376365752788,
                -0.0054103844866927788,
                0.056840618403875359,
                -0.0035227192748552472,
                0.25667909012207590,
                0.0055029673963982112,
                -0.073334392714092062,
                0.0027810273357488265,
                -0.24912792167850559,
                -0.0069524866497120566,
                0.21288810409948347,
                0.0017810057298691725,
                0.088760951809475269,
                0.0055957188940032095,
                -0.31937177676259115,
                -0.0087539416335935556,
                0.34748800245527145,
                0.0075378299617709235,
                -0.25116537379803394,
                -0.0047285674679876204,
                0.13970502851683486,
                0.0023672533925155220,
                -0.063649401080083698,
                -0.00098993213448982727,
                0.024597838934816905,
                0.00035553235917057483,
                -0.0082485030307578155,
                -0.00011176184313622549,
                0.0024390574829093264,
                0.000031180384864488629,
                -0.00064373524734389861,
                -7.8036008952377965e-6,
                0.00015310015145922058,
                1.7670804180220134e-6,
                -0.000033066844379476900,
                -3.6460909134279425e-7,
                6.5276969021754105e-6,
                6.8957843666189918e-8,
                -1.1842811187642386e-6,
                -1.2015133285307312e-8,
                1.9839339947648331e-7,
                1.9372045971100854e-9,
                -3.0815418032523593e-8,
                -2.9013806338735810e-10,
                4.4540904298173700e-9,
                4.0505136697916078e-11,
                -6.0104912807134771e-10,
                -5.2873323696828491e-12,
                7.5943206779351725e-11,
                6.4679566322060472e-13,
                -9.0081200925539902e-12,
                -7.4396949275292252e-14,
                1.0057423059167244e-12,
                8.1701187638005194e-15,
                -1.0611736208855373e-13,
                -8.9597492970451533e-16,
                1.1421575296031385e-14,
            ],
            dtype=np.float64,
        )

        # Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        if secretKeyDist == "SPARSE_TERNARY":
            self.coefficients = np.copy(coefficientsSparse)
            self.k = (
                1.0  # do not divide by k as we already did it during precomputation
            )
        else:
            self.coefficients = np.copy(coefficientsUniform)
            self.k = K_UNIFORM

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

    # Placeholder function for SelectLayers, which needs to be defined as per the logic in your system.
    def SelectLayers(self, logBsSlots, budget):
        layers = math.ceil(logBsSlots / budget)
        rows = logBsSlots // layers
        rem = logBsSlots % layers

        dim = rows
        if rem != 0:
            dim = rows + 1

        # The above choice ensures dim <= budget
        if dim < budget:
            layers -= 1
            rows = logBsSlots // layers
            rem = logBsSlots - rows * layers
            dim = rows

            if rem != 0:
                dim = rows + 1

            # The above choice ensures dim >= budget
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

        # Computing the baby-step b and the giant-step g for the collapsed layers for decoding.
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

        # If this return statement changes then CKKS_BOOT_PARAMS should be altered as well
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

    def eval_bootstrap_setup(self, context, level_budget, dim1, numslots, correction_factor):

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
        rescale_tech = context.rescaleTech

        # 设置 correction_factor
        if correction_factor == 0:
            if (
                rescale_tech == "FLEXIBLEAUTO"
                or rescale_tech == "FLEXIBLEAUTOEXT"
            ):
                # 实验结果得出的最佳精度对应的默认 correction factors
                tmp = round_half_away_from_zero(-0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
                if tmp < 7:
                    self.correctionFactor = 7
                elif tmp > 13:
                    self.correctionFactor = 13
                else:
                    self.correctionFactor = int(tmp)
            else:
                self.correctionFactor = 9
        else:
            self.correctionFactor = correction_factor

        self.m_slots = slots
        self.m_dim1 = dim1[0]

        log_slots = math.log2(slots)

        # 检查 level budget 并计算参数
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

        self.paramsEnc = self.GetCollapsedFFTParams(
            slots, new_budget[0], dim1[0]
        )
        self.paramsDec =self.GetCollapsedFFTParams(
            slots, new_budget[1], dim1[1]
        )

        self.compute_C2S_rot(slots, self.M)
        self.compute_S2C_rot(slots, self.M)

        assert not (m_U0hatTPreFFT_dim1 == 1 and m_U0PreFFT_dim1 == 1) and "Not Implemented"

        RHScnt = 0
        cnt = 0
        sizeP = context.K
        self.m_U0hatTPreFFT = [[0] * i for i in m_U0hatTPreFFT_dim2]
        for i in range(0, m_U0hatTPreFFT_dim1):
            j_len = m_U0hatTPreFFT_dim2[i]
            limbs = m_U0hatTPreFFT_limbs[i]
            m_U0hatTPreFFT_len = mx_len * limbs
            # print("m_U0hatTPreFFT_len", m_U0hatTPreFFT_len)
            # print("m_U0hatTPreFFT_scaling_factor", len()
            for j in range(j_len):
                m_U0hatTPreFFT = self.m_U0hatTPreFFT_mx[
                    RHScnt : RHScnt + m_U0hatTPreFFT_len
                ].copy()
                RHScnt += m_U0hatTPreFFT_len
                self.m_U0hatTPreFFT[i][j] = Plaintext(
                    m_U0hatTPreFFT,
                    limbs-sizeP,
                    self.m_U0hatTPreFFT_scaling_factor[cnt],
                    1,
                    mx_slots,
                    True
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
                    m_U0PreFFT,
                    limbs-sizeP,
                    self.m_U0PreFFT_scaling_factor[cnt],
                    1,
                    mx_slots,
                    True
                )
                cnt += 1
        self.m_U0PreFFT_mx = None
