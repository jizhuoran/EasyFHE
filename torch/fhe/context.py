from math import log, pi, cos, sin
from . import number_theory as nbtheory
import numpy as np
import math
import random
import warnings
import torch
from enum import Enum

K_UNIFORM = 512

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"

def reduce_rotation(index, slots):
    islots = int(slots)
    index = int(index)

    # if slots is a power of 2
    if (int(slots) & int(slots - 1)) == 0:
        n = int(math.log2(slots))
        if index >= 0:
            return index - ((index >> n) << n)
        return index + islots + ((abs(index) >> n) << n)

    return (islots + index % islots) % islots

warnings.formatwarning = custom_warning_format
class SecretKeyDist(Enum):
    GAUSSIAN = 0
    UNIFORM_TERNARY = 1
    SPARSE_TERNARY = 2

class ScalingTechnique(Enum):
    FIXEDMANUAL = 0
    FIXEDAUTO = 1
    FLEXIBLEAUTO = 2
    FLEXIBLEAUTOEXT = 3
    NORESCALE = 4
    INVALID_RS_TECHNIQUE = 5

class CKKS_Boot_Params:
    def __init__(self, level_budget, layers_coll, layers_rem, num_rotations, baby_step, giant_step, num_rotations_rem, baby_step_rem, giant_step_rem):
        self.level_budget = level_budget          # the level budget
        self.layers_coll =layers_coll           # the number of layers to collapse in one level
        self.layers_rem = layers_rem            # the number of layers remaining to be collapsed in one level to have exactly the number of levels specified in the level budget
        self.num_rotations = num_rotations         # the number of rotations in one level
        self.baby_step = baby_step             # the baby step in the baby-step giant-step strategy
        self.giant_step = giant_step            # the giant step in the baby-step giant-step strategy
        self.num_rotations_rem = num_rotations_rem     # the number of rotations in the remaining level
        self.baby_step_rem = baby_step_rem         # the baby step in the baby-step giant-step strategy for the remaining level
        self.giant_step_rem = giant_step_rem        # the giant step in the baby-step giant-step strategy for the remaining level
        self.total_elements = 9

class BsContext:
    def __init__(self, cryptoContext, levelBudget, dim1, numslots, correctionFactor, rescaleTech, secretKeyDist):
        self.M = cryptoContext.N *2
        slots = self.M // 4 if numslots == 0 else numslots
        self.correctionFactor = correctionFactor
        self.secretKeyDist = secretKeyDist
        self.m_U0hatTPre = None
        self.m_U0hatTPreFFT = None
        self.m_U0Pre = None
        self.m_U0PreFFT = None
        self.paramsDec = None
        self.paramsEnc = None
        self.rescaleTech = rescaleTech
        self.precompute_auto_map = {}

        # precom = scheme.precom
        if correctionFactor == 0:
            if rescaleTech == ScalingTechnique.FLEXIBLEAUTO or rescaleTech == ScalingTechnique.FLEXIBLEAUTOEXT:
                tmp = round(-0.265 * (2 * math.log2(self.M  / 2) + math.log2(slots)) + 19.1)
                if tmp < 7:
                    self.m_correctionFactor = 7
                elif tmp > 13:
                    self.m_correctionFactor = 13
                else:
                    self.m_correctionFactor = int(tmp)
            else:
                self.m_correctionFactor = 9
        else:
            self.m_correctionFactor = correctionFactor

        self.slots = slots
        self.dim1 = dim1[0]

        logSlots = math.log2(slots)
        newBudget = [levelBudget[0], levelBudget[1]]
        if levelBudget[0] > logSlots:
            print(f"\nWarning, the level budget for encoding cannot be this large. The budget was changed to {int(logSlots)}")
            newBudget[0] = int(logSlots)

        if levelBudget[0] < 1:
            print(f"\nWarning, the level budget for encoding has to be at least 1. The budget was changed to 1")
            newBudget[0] = 1

        if levelBudget[1] > logSlots:
            print(f"\nWarning, the level budget for decoding cannot be this large. The budget was changed to {int(logSlots)}")
            newBudget[1] = int(logSlots)

        if levelBudget[1] < 1:
            print(f"\nWarning, the level budget for decoding has to be at least 1. The budget was changed to 1")
            newBudget[1] = 1

        self.paramsEnc = self.GetCollapsedFFTParams(slots, newBudget[0], dim1[0])
        self.paramsDec = self.GetCollapsedFFTParams(slots, newBudget[1], dim1[1])

        coefficientsSparse = np.array([
            0, -0.0190665676962401, 0, -0.0181773905007824, 0, -0.0162862756167401, 0, -0.0131970301188482,
            0, -0.00869599648960049, 0, -0.00266512292674043, 0, 0.00475378458365385, 0, 0.0129619218183744,
            0, 0.0207345065018299, 0, 0.0261987740118010, 0, 0.0271237206149663, 0, 0.0216632442529301,
            0, 0.00952467756531695, 0, -0.00682586258643841, 0, -0.0217665193289893, 0, -0.0279850481505861,
            0, -0.0202671538394630, 0, -0.000311697041869291, 0, 0.0210206341691402, 0, 0.0282597848811002,
            0, 0.0130902946902468, 0, -0.0144903750619968, 0, -0.0292119597624053, 0, -0.0133436971840822,
            0, 0.0187762764821447, 0, 0.0284541504148807, 0, -0.000489726742355156, 0, -0.0298222811587479,
            0, -0.0127584877864399, 0, 0.0267192319192248, 0, 0.0186624682104780, 0, -0.0261495713329483,
            0, -0.0179030470013594, 0, 0.0303046477803535, 0, 0.00859965792435869, 0, -0.0352157135816712,
            0, 0.0127788627989003, 0, 0.0264211888837408, 0, -0.0374200640582086, 0, 0.0132393631154040,
            0, 0.0219435428661135, 0, -0.0444788687151216, 0, 0.0477866972698431, 0, -0.0383304915060382,
            0, 0.0252513113739573, 0, -0.0142806559093283, 0, 0.00711359650506429, 0, -0.00317433716746386,
            0, 0.00128436605459822, 0, -0.000475515283653384, 0, 0.000162257517416398, 0, -0.0000513272589524132,
            0, 0.0000151253840421986, 0, -4.16938339926456e-6, 0, 1.07891901728700e-6, 0, -2.62909460240295e-7,
            0, 6.04943494968095e-8, 0, -1.31757718513370e-8, 0, 2.72234854083432e-9, 0, -5.34663845707394e-10,
            0, 9.99938555825121e-11, 0, -1.78377633651571e-11, 0, 3.03978611829284e-12, 0, -4.95680040223255e-13,
            0, 7.73718537798400e-14, 0, -1.14402314781930e-14, 0, 1.69000615970718e-15, 0
        ], dtype=np.float64)

        coefficientsUniform = np.array([
            0.15421426400235561, -0.0037671538417132409, 0.16032011744533031, -0.0034539657223742453,
            0.17711481926851286, -0.0027619720033372291, 0.19949802549604084, -0.0015928034845171929,
            0.21756948616367638, 0.00010729951647566607, 0.21600427371240055, 0.0022171399198851363,
            0.17647500259573556, 0.0042856217194480991, 0.086174491919472254, 0.0054640252312780444,
            -0.046667988130649173, 0.0047346914623733714, -0.17712686172280406, 0.0016205080004247200,
            -0.22703114241338604, -0.0028145845916205865, -0.13123089730288540, -0.0056345646688793190,
            0.078818395388692147, -0.0037868875028868542, 0.23226434602675575, 0.0021116338645426574,
            0.13985510526186795, 0.0059365649669377071, -0.13918475289368595, 0.0018580676740836374,
            -0.23254376365752788, -0.0054103844866927788, 0.056840618403875359, -0.0035227192748552472,
            0.25667909012207590, 0.0055029673963982112, -0.073334392714092062, 0.0027810273357488265,
            -0.24912792167850559, -0.0069524866497120566, 0.21288810409948347, 0.0017810057298691725,
            0.088760951809475269, 0.0055957188940032095, -0.31937177676259115, -0.0087539416335935556,
            0.34748800245527145, 0.0075378299617709235, -0.25116537379803394, -0.0047285674679876204,
            0.13970502851683486, 0.0023672533925155220, -0.063649401080083698, -0.00098993213448982727,
            0.024597838934816905, 0.00035553235917057483, -0.0082485030307578155, -0.00011176184313622549,
            0.0024390574829093264, 0.000031180384864488629, -0.00064373524734389861, -7.8036008952377965e-6,
            0.00015310015145922058, 1.7670804180220134e-6, -0.000033066844379476900, -3.6460909134279425e-7,
            6.5276969021754105e-6, 6.8957843666189918e-8, -1.1842811187642386e-6, -1.2015133285307312e-8,
            1.9839339947648331e-7, 1.9372045971100854e-9, -3.0815418032523593e-8, -2.9013806338735810e-10,
            4.4540904298173700e-9, 4.0505136697916078e-11, -6.0104912807134771e-10, -5.2873323696828491e-12,
            7.5943206779351725e-11, 6.4679566322060472e-13, -9.0081200925539902e-12, -7.4396949275292252e-14,
            1.0057423059167244e-12, 8.1701187638005194e-15, -1.0611736208855373e-13, -8.9597492970451533e-16,
            1.1421575296031385e-14
        ], dtype=np.float64)


        # Chebyshev series coefficients for modular reduction
        if secretKeyDist == SecretKeyDist.SPARSE_TERNARY:
            self.coefficients = np.copy(coefficientsSparse)
            self.k = 1.0
        else:
            self.coefficients = np.copy(coefficientsUniform)
            self.k = K_UNIFORM

        self.compute_C2S_rot(cryptoContext)
        self.compute_S2C_rot(cryptoContext)

        for key, _ in cryptoContext.left_rot_key_map.items():
            self.precompute_auto_map[int(key)] = self.compute_auto_map(int(key), cryptoContext)

        self.QplusP_map = {}
        self.QmuplusPmu_map = {}
        for cur_limbs in range(len(cryptoContext.moduliQ)):
            self.QplusP_map[cur_limbs] = torch.tensor(np.concatenate((cryptoContext.moduliQ[0:cur_limbs], cryptoContext.moduliP[0:cryptoContext.K])), dtype=torch.uint64, device="cuda")
            self.QmuplusPmu_map[cur_limbs] = torch.tensor(np.concatenate((cryptoContext.q_mu[0:cur_limbs], cryptoContext.p_mu[:cryptoContext.K])), dtype=torch.uint64, device="cuda")

        #compute auto index map
        self.auto_index = {}
        self.auto_index[slots] = self.find_auto_index(slots, cryptoContext.N << 1)
        for step in range(int(math.log2(cryptoContext.N // (2 * slots)))):
            self.auto_index[(1 << step) * slots] = self.find_auto_index((1 << step) * slots, cryptoContext.N << 1)
        for i in self.C2S_rot_in + self.C2S_rot_out + self.S2C_rot_in + self.S2C_rot_out:
            for j in i:
                if j not in self.auto_index:
                    self.auto_index[j] = self.find_auto_index(j, cryptoContext.N << 1)

    def find_auto_index(self, i, m):
        def inv_mod(a, m):
            m0, x0, x1 = m, 0, 1
            if m == 1:
                return 0
            while a > 1:
                q = a // m
                m, a = a % m, m
                x0, x1 = x1 - q * x0, x0
            if x1 < 0:
                x1 += m0
            return x1

        if i == 0:
            return 1

        # Conjugation automorphism
        if i == m - 1:
            return i

        # Generator
        if i < 0:
            g0 = inv_mod(5, m)
            g0 = (g0 * 5) % m
        else:
            g0 = 5

        i_unsigned = abs(i)
        g = g0

        for j in range(1, int(i_unsigned)):
            g = (g * g0) % m

        return g


    def compute_auto_map(self, k, cryptoContext):

        def reverse_bits(num, num_bits):
            """Reverses the bits of a number."""
            rev = 0
            for i in range(num_bits):
                rev = (rev << 1) | (num & 1)
                num >>= 1
            return rev

        """computes the automorphism map"""
        n = cryptoContext.N
        m = n << 1  # cyclOrder
        logm = round(np.log2(m))
        logn = round(np.log2(n))
        res = np.zeros(n, dtype=np.int32)
        for j in range(n):
            j_tmp = (j << 1) + 1
            idx = ((j_tmp * k) - (((j_tmp * k) >> logm) << logm)) >> 1
            j_rev = reverse_bits(j, logn)
            idx_rev = reverse_bits(idx, logn)
            res[j_rev] = idx_rev

        return torch.from_numpy(np.array(res)).cuda()

    def compute_C2S_rot(self, cryptoContext):
        slots = self.slots

        N = cryptoContext.N
        M = cryptoContext.M
        K = cryptoContext.K
        logN = cryptoContext.logN
        special_limbs = K

        # precom = cryptoContext.BsContext
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
                rot_in[s][j] = reduce_rotation(
                    (j - (num_rotations + 1) // 2 + 1) * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)),
                    slots)

            for i in range(b):
                rot_out[s][i] = reduce_rotation((g * i) * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)), M // 4)

        if flag_rem:
            for j in range(g_rem):
                rot_in[stop][j] = reduce_rotation((j - (num_rotations_rem + 1) // 2 + 1), slots)

            for i in range(b_rem):
                rot_out[stop][i] = reduce_rotation((g_rem * i), M // 4)
        
        self.C2S_rot_in = rot_in
        self.C2S_rot_out = rot_out

    def compute_S2C_rot(self, cryptoContext):
        slots = self.slots

        N = cryptoContext.N
        M = cryptoContext.M
        K = cryptoContext.K
        logN = cryptoContext.logN
        special_limbs = K

        # precom = cryptoContext.BsContext
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
                rot_in.append(np.zeros(num_rotations_rem + 1))

            else:
                rot_in.append(np.zeros(num_rotations + 1))
        for i in range(level_budget):
            rot_out.append(np.zeros(b + b_rem))

        for s in range(level_budget - flag_rem):
            for j in range(g):
                rot_in[s][j] = reduce_rotation((j - ((num_rotations + 1) / 2) + 1) * (1 << (s * layers_collapse)), M // 4)

            for i in range(b):
                rot_out[s][i] = reduce_rotation((g * i) * (1 << (s * layers_collapse)), M // 4)

        if flag_rem:
            s = level_budget - flag_rem
            for j in range(g_rem):
                rot_in[s][j] = reduce_rotation((j - (num_rotations_rem + 1) // 2 + 1) * (1 << (s * layers_collapse)),
                                            M // 4)

            for i in range(b_rem):
                rot_out[s][i] = reduce_rotation((g_rem * i) * (1 << (s * layers_collapse)), M // 4)

        self.S2C_rot_in = rot_in
        self.S2C_rot_out = rot_out

    # Placeholder function for SelectLayers, which needs to be defined as per the logic in your system.

    def SelectLayers(self, logSlots, budget):
        layers = math.ceil(logSlots / budget)
        rows = logSlots // layers
        rem = logSlots % layers

        dim = rows
        if rem != 0:
            dim = rows + 1

        # The above choice ensures dim <= budget
        if dim < budget:
            layers -= 1
            rows = logSlots // layers
            rem = logSlots - rows * layers
            dim = rows

            if rem != 0:
                dim = rows + 1

            # The above choice ensures dim >= budget
            while dim != budget:
                rows -= 1
                rem = logSlots - rows * layers
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
        return CKKS_Boot_Params(int(levelBudget), layersCollapse, remCollapse, int(numRotations), b, g, int(numRotationsRem), bRem, gRem)
class Context:
    def __init__(
        self,
        logN,
        logq0,
        logqi,
        logp,
        L,
        K,
        moduliQ=None,
        moduliP=None,
        rootsQ=None,
        rootsP=None,
        MULT_SWK=None,
        ROT_SWK=None,
        BOOT_KEY=None,
        h=64,
        sigma=32
    ):
        self.BsContext = None
        self.logp = logp
        self.slots = None # 固定值
        self.qVec = None
        self.left_rot_key_map = {} #{index: [ax， bx]}
        self.key_map = None
        self.correctionFactor = 0

        self.logN = logN
        self.logqi = logqi
        self.L = int(L)
        self.K = int(K)
        self.dnum = math.ceil(L / K)
        self.h = h
        self.sigma = sigma
        self.N = int(1 << logN)
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1
        self.p = 1 << logqi

        self.moduliQ = [0] * L
        self.qrVec = [0] * L
        self.qTwok = [0] * L
        self.qkVec = [0] * L
        self.qdVec = [0] * L
        self.qInvVec = [0] * L
        self.qRoots = [0] * L
        self.qRootsInv = [0] * L
        self.qRootPows = [[] for _ in range(L)]
        self.qRootScalePows = [[] for _ in range(L)]
        self.qRootScalePowsOverq = [[] for _ in range(L)]
        self.qRootScalePowsInv = [[] for _ in range(L)]
        self.qRootPowsInv = [[] for _ in range(L)]
        self.NInvModq = [0] * L
        self.NScaleInvModq = [0] * L
        bnd = 1
        cnt = 1
        if moduliQ is None and rootsQ is None:
            while True:
                prime = (1 << logq0) + bnd * self.M + 1
                if nbtheory.is_prime(prime):
                    self.moduliQ[0] = prime
                    break
                bnd += 1
            # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.moduliQ[i])
            self.qRoots[0] = nbtheory.root_of_unity(
                order=self.M, modulus=self.moduliQ[0]
            )
            # print("moduliQ[0]", self.moduliQ[0])
            bnd = 1
            while cnt < L:
                prime1 = (1 << logqi) + bnd * self.M + 1
                if self.primeTest(prime1):
                    self.moduliQ[cnt] = prime1
                    cnt += 1
                prime2 = (1 << logqi) - bnd * self.M + 1
                if self.primeTest(prime2):
                    self.moduliQ[cnt] = prime2
                    # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.moduliQ[i])
                    self.qRoots[cnt] = nbtheory.root_of_unity(
                        order=self.M, modulus=self.moduliQ[cnt - 1]
                    )
                    cnt += 1
                bnd += 1

            if logqi - logN - 1 - math.ceil(math.log2(bnd)) < 10:
                print("ERROR: too small number of precision")
                print("TRY to use larger logqi or smaller depth")
        else:
            if moduliQ is None:
                print("moduliQ needs to be set!")
                return
            elif rootsQ is None:
                print("rootsQ needs to be set!")
                return
            for i in range(L):
                self.moduliQ[i] = moduliQ[i]
                self.qRoots[i] = rootsQ[i]

        for i in range(L):
            # print(i)
            self.qTwok[i] = 2 * (int(math.log2(self.moduliQ[i])) + 1)
            self.qrVec[i] = (1 << int(self.qTwok[i])) // int(self.moduliQ[i])
            self.qkVec[i] = (
                (nbtheory.mod_inv(1 << 62, int(self.moduliQ[i])) << 62) - 1
            ) // int(self.moduliQ[i])
            self.qRootsInv[i] = nbtheory.mod_inv(self.qRoots[i], int(self.moduliQ[i]))
            self.NInvModq[i] = nbtheory.mod_inv(self.N, int(self.moduliQ[i]))
            self.NScaleInvModq[i] = self.mulMod(
                int(self.NInvModq[i]), int(1 << 32), int(self.moduliQ[i])
            )
            self.NScaleInvModq[i] = self.mulMod(
                int(self.NScaleInvModq[i]), int(1 << 32), int(self.moduliQ[i])
            )
            self.qInvVec[i] = self.inv(self.moduliQ[i])
            self.qRootPows[i] = [0] * self.N
            self.qRootPowsInv[i] = [0] * self.N
            self.qRootScalePows[i] = [0] * self.N
            self.qRootScalePowsOverq[i] = [0] * self.N
            self.qRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                self.qRootPows[i][jprime] = int(power)
                # tmp = (power << 64)
                tmp = int(power) << 64
                self.qRootScalePowsOverq[i][jprime] = int(tmp // int(self.moduliQ[i]))
                self.qRootScalePows[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootPows[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                self.qRootScalePows[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootScalePows[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                self.qRootPowsInv[i][jprime] = int(powerInv)
                self.qRootScalePowsInv[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootPowsInv[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                self.qRootScalePowsInv[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootScalePowsInv[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                if j < self.N - 1:
                    power = self.mulMod(
                        int(power), int(self.qRoots[i]), int(self.moduliQ[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(self.qRootsInv[i]), int(self.moduliQ[i])
                    )
        q_mu = []  # for barret mul mod
        for mod in self.moduliQ:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)  # 取低64位
            high = x >> 64  # 取高64位
            q_mu.append([low, high])
        self.q_mu = np.array(q_mu, dtype=np.uint64)
        self.q_mu_cuda = torch.from_numpy(np.array(q_mu, dtype=np.uint64)).cuda()
        self.moduliQ_cuda = torch.from_numpy(np.array(self.moduliQ, dtype=np.uint64)).cuda()

        self.moduliP = [0] * self.K
        self.prVec = [0] * self.K
        self.pTwok = [0] * self.K
        self.pkVec = [0] * self.K
        self.pdVec = [0] * self.K
        self.pInvVec = [0] * self.K
        self.pRoots = [0] * self.K
        self.pRootsInv = [0] * self.K
        self.pRootPows = [[] for _ in range(self.K)]
        self.pRootPowsInv = [[] for _ in range(self.K)]
        self.pRootScalePows = [[] for _ in range(self.K)]
        self.pRootScalePowsOverp = [[] for _ in range(self.K)]
        self.pRootScalePowsInv = [[] for _ in range(self.K)]
        self.NInvModp = [0] * self.K
        self.NScaleInvModp = [0] * self.K

        if moduliP is None and rootsP is None:
            cnt = 0
            while cnt < self.K:
                prime1 = (1 << logp) + bnd * self.M + 1
                if self.primeTest(prime1):
                    self.moduliP[cnt] = prime1
                    self.pRoots[cnt] = nbtheory.root_of_unity(
                        order=self.M, modulus=self.moduliP[cnt]
                    )
                    cnt += 1
                if cnt == self.K:
                    break
                prime2 = (1 << logp) - bnd * self.M + 1
                if self.primeTest(prime2):
                    self.moduliP[cnt] = prime2
                    self.pRoots[cnt] = nbtheory.root_of_unity(
                        order=self.M, modulus=self.moduliP[cnt]
                    )
                    cnt += 1
                bnd += 1

        else:
            if moduliP is None:
                print("moduliP needs to be set")
                return
            elif rootsP is None:
                print("rootsP needs to be set")
                return
            for i in range(K):
                self.moduliP[i] = moduliP[i]
                self.pRoots[i] = rootsP[i]

        for i in range(K):
            # print(i)
            self.pTwok[i] = 2 * (int(math.log2(self.moduliP[i])) + 1)
            self.prVec[i] = (1 << int(self.pTwok[i])) // int(self.moduliP[i])
            self.pkVec[i] = (
                (nbtheory.mod_inv(1 << 62, int(self.moduliP[i])) << 62) - 1
            ) // int(self.moduliP[i])
            self.pRootsInv[i] = nbtheory.mod_inv(self.pRoots[i], int(self.moduliP[i]))
            self.NInvModp[i] = nbtheory.mod_inv(self.N, int(self.moduliP[i]))
            self.NScaleInvModp[i] = self.mulMod(
                int(self.NInvModp[i]), int(1 << 32), int(self.moduliP[i])
            )
            self.NScaleInvModp[i] = self.mulMod(
                int(self.NScaleInvModp[i]), int(1 << 32), int(self.moduliP[i])
            )
            self.pInvVec[i] = self.inv(self.moduliP[i])
            self.pRootPows[i] = [0] * self.N
            self.pRootPowsInv[i] = [0] * self.N
            self.pRootScalePows[i] = [0] * self.N
            self.pRootScalePowsOverp[i] = [0] * self.N
            self.pRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                self.pRootPows[i][jprime] = int(power)
                tmp = int(power) << 64
                self.pRootScalePowsOverp[i][jprime] = tmp // int(self.moduliP[i])
                self.pRootScalePows[i][jprime] = self.mulMod(
                    self.pRootPows[i][jprime], int(1 << 32), int(self.moduliP[i])
                )
                self.pRootScalePows[i][jprime] = self.mulMod(
                    self.pRootScalePows[i][jprime], int(1 << 32), int(self.moduliP[i])
                )
                self.pRootPowsInv[i][jprime] = powerInv
                self.pRootScalePowsInv[i][jprime] = self.mulMod(
                    self.pRootPowsInv[i][jprime], int(1 << 32), int(self.moduliP[i])
                )
                self.pRootScalePowsInv[i][jprime] = self.mulMod(
                    self.pRootScalePowsInv[i][jprime],
                    int(1 << 32),
                    int(self.moduliP[i]),
                )
                if j < self.N - 1:
                    power = self.mulMod(
                        power, int(self.pRoots[i]), int(self.moduliP[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(self.pRootsInv[i]), int(self.moduliP[i])
                    )

        p_mu = []  # for barret mul mod
        for mod in self.moduliP:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)  # 取低64位
            high = x >> 64  # 取高64位
            p_mu.append([low, high])
        self.p_mu = np.array(p_mu, dtype=np.uint64)

        moduliPartQ = [0] * self.dnum
        for j in range(self.dnum):
            moduliPartQ[j] = int(1)
            for i in range(K * j, K * (j + 1)):
                if i < L:
                    moduliPartQ[j] *= int(self.moduliQ[i])

        self.PartQlHatInvModq = [
            [[0 for _ in range(K)] for _ in range(K)] for _ in range(self.dnum)
        ]
        for k in range(self.dnum):
            sizePartQk = (L - (k * K)) if (k == self.dnum - 1) else K
            modulusPartQ = moduliPartQ[k]
            for l in range(sizePartQk):
                if l > 0:
                    modulusPartQ = int(
                        int(modulusPartQ) // int(self.moduliQ[k * K + sizePartQk - l])
                    )
                for i in range(sizePartQk - l):
                    moduli = int(self.moduliQ[k * K + i])
                    QHat = modulusPartQ // moduli
                    QHatInvModqi = int(self.invMod(QHat, moduli))
                    self.PartQlHatInvModq[k][sizePartQk - l - 1][i] = QHatInvModqi

        # 初始化 PartQlHatModp
        self.PartQlHatModp = [
            [
                [[0 for _ in range(self.dnum * K)] for _ in range(K)]
                for _ in range(self.dnum)
            ]
            for _ in range(L)
        ]
        for l in range(L):
            beta = math.ceil((l + 1) / K)
            for k in range(beta):
                partQ_size = (
                    (L - (beta - 1) * K) if (beta == self.dnum and k == beta - 1) else K
                )
                digitSize = K
                modulusPartQ = int(moduliPartQ[k])

                if k == beta - 1:
                    digitSize = l + 1 - k * K
                    for idx in range(digitSize, partQ_size):
                        modulusPartQ //= int(self.moduliQ[K * k + idx])

                for i in range(digitSize):
                    partQHat = modulusPartQ // int(self.moduliQ[K * k + i])

                    start_idx = k * K
                    end_idx = start_idx + digitSize
                    complBasis_vec = (
                        self.moduliQ[:start_idx]
                        + self.moduliQ[end_idx : l + 1]
                        + self.moduliP
                    )

                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp[l][k][i][j] = QHatModpj

        # 初始化 PartQlHatModp_pad
        self.PartQlHatModp_pad = [
            [
                [[0 for _ in range(self.dnum * K)] for _ in range(K)]
                for _ in range(self.dnum)
            ]
            for _ in range(L)
        ]
        for l in range(L):
            beta = math.ceil((l + 1) / K)
            ceil_curr_limbs = beta * K
            for k in range(beta):
                partQ_size = (
                    (L - (beta - 1) * K) if (beta == self.dnum and k == beta - 1) else K
                )
                digitSize = K
                modulusPartQ = int(moduliPartQ[k])

                if k == beta - 1:
                    digitSize = l + 1 - k * K
                    for idx in range(digitSize, partQ_size):
                        modulusPartQ //= int(self.moduliQ[K * k + idx])

                for i in range(digitSize):
                    partQHat = modulusPartQ // int(self.moduliQ[K * k + i])

                    start_idx = k * K
                    end_idx = start_idx + digitSize
                    complBasis_vec = (
                        self.moduliQ[:start_idx] + self.moduliQ[end_idx : l + 1]
                    )
                    offset = len(complBasis_vec)
                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp_pad[l][k][i][j] = QHatModpj

                    complBasis_vec = self.moduliQ[l + 1 : ceil_curr_limbs]
                    for j, mod in enumerate(complBasis_vec):
                        self.PartQlHatModp_pad[l][k][i][offset + j] = 0

                    complBasis_vec = self.moduliP
                    offset = ceil_curr_limbs - K
                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp_pad[l][k][i][offset + j] = QHatModpj

        self.pHatModp = [0] * K  # 初始化 pHatModp 列表
        self.pHatInvModp = [0] * K  # 初始化 pHatInvModp 列表
        # 计算 pHatModp
        for k in range(K):
            self.pHatModp[k] = int(1)
            for j in list(range(k)) + list(range(k + 1, K)):
                temp = int(self.moduliP[j] % self.moduliP[k])
                self.pHatModp[k] = (self.pHatModp[k] * temp) % int(self.moduliP[k])

        # 计算 pHatInvModp # [k] qhat_k^-1 mod q_k
        for k in range(K):
            self.pHatInvModp[k] = int(
                self.invMod(int(self.pHatModp[k]), self.moduliP[k])
            )

        # 初始化 pHatModq
        self.pHatModq = [[0] * L for _ in range(K)]
        for k in range(K):
            for i in range(L):
                self.pHatModq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, K)):
                    temp = int(self.moduliP[s]) % int(self.moduliQ[i])
                    self.pHatModq[k][i] = self.mulMod(
                        int(self.pHatModq[k][i]), temp, int(self.moduliQ[i])
                    )

        self.PModq = [0] * L  # 初始化 PModq

        # 计算 PModq
        for i in range(L):
            self.PModq[i] = int(1)
            for k in range(K):
                temp = self.moduliP[k] % self.moduliQ[i]
                self.PModq[i] = self.mulMod(
                    int(self.PModq[i]), int(temp), int(self.moduliQ[i])
                )

        self.PInvModq = [0] * L  # 初始化 PInvModq
        # 计算 PInvModq
        for i in range(L):
            self.PInvModq[i] = self.invMod(int(self.PModq[i]), int(self.moduliQ[i]))

        self.qInvModq = [[0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            for j in list(range(i)) + list(range(i + 1, L)):
                self.qInvModq[i][j] = self.invMod(
                    int(self.moduliQ[i]), int(self.moduliQ[j])
                )

        # rescale param
        # sizeQ in openFHE equals to L here.
        self.QlQlInvModqlDivqlModq = [[0] * (L - 1) for _ in range(L - 1)]
        # self.QlQlInvModqlDivqlModq = [None] * (L - 1)
        for k in range(L - 1):
            l = L - (k + 1)
            # self.QlQlInvModqlDivqlModq[k] = [0] * l

            for i in range(l):
                QlInvModql = int(1)

                for j in range(l):
                    temp = self.invMod(self.moduliQ[j], self.moduliQ[l])
                    QlInvModql = self.mulMod(
                        int(QlInvModql), int(temp), int(self.moduliQ[l])
                    )

                modulusQ = int(1)
                for j in range(l):
                    modulusQ *= int(self.moduliQ[j])

                result = int((int(QlInvModql) * modulusQ) // int(self.moduliQ[l]))
                result %= int(self.moduliQ[i])

                self.QlQlInvModqlDivqlModq[k][i] = np.uint64(result)

        self.mult_swk = [None, None]
        if MULT_SWK is None:
            warnings.warn(
                "\n------------------------\n"
                "MULT_SWK needs to be set"
                "\n------------------------\n",
                UserWarning,
            )
            # todo: set data in numpy array
        else:
            self.mult_swk[0] = MULT_SWK[0]
            self.mult_swk[1] = MULT_SWK[1]

        self.moduliQ = np.array(self.moduliQ, dtype=np.uint64)
        self.qrVec = np.array(self.qrVec, dtype=np.uint64)
        self.qTwok = np.array(self.qTwok, dtype=np.uint64)
        self.qkVec = np.array(self.qkVec, dtype=np.uint64)
        self.qdVec = np.array(self.qdVec, dtype=np.uint64)
        self.moduliP = np.array(self.moduliP, dtype=np.uint64)
        self.prVec = np.array(self.prVec, dtype=np.uint64)
        self.pTwok = np.array(self.pTwok, dtype=np.uint64)
        self.pkVec = np.array(self.pkVec, dtype=np.uint64)
        self.pdVec = np.array(self.pdVec, dtype=np.uint64)
        self.qRoots = np.array(self.qRoots, dtype=np.uint64)
        self.pRoots = np.array(self.pRoots, dtype=np.uint64)

        self.qInvVec = np.array(self.qInvVec, dtype=np.uint64)
        self.pInvVec = np.array(self.pInvVec, dtype=np.uint64)
        self.qRootScalePows = np.array(self.qRootScalePows, dtype=np.uint64)
        self.pRootScalePows = np.array(self.pRootScalePows, dtype=np.uint64)
        self.qRootScalePowsInv = np.array(self.qRootScalePowsInv, dtype=np.uint64)
        self.pRootScalePowsInv = np.array(self.pRootScalePowsInv, dtype=np.uint64)
        self.NInvModq = np.array(self.NInvModq, dtype=np.uint64)
        self.NInvModp = np.array(self.NInvModp, dtype=np.uint64)
        self.NScaleInvModq = np.array(self.NScaleInvModq, dtype=np.uint64)
        self.NScaleInvModp = np.array(self.NScaleInvModp, dtype=np.uint64)
        self.QHatInvModq = np.array(self.PartQlHatInvModq, dtype=np.uint64)
        self.QHatModp = np.array(self.PartQlHatModp, dtype=np.uint64)
        self.pHatInvModp = np.array(self.pHatInvModp, dtype=np.uint64)
        self.pHatModq = np.array(self.pHatModq, dtype=np.uint64)
        self.PInvModq = np.array(self.PInvModq, dtype=np.uint64)

        self.PartQlHatInvModq = np.array(self.PartQlHatInvModq, dtype=np.uint64)
        self.PartQlHatModp = np.array(self.PartQlHatModp, dtype=np.uint64)
        self.pHatModp = np.array(self.pHatModp, dtype=np.uint64)
        self.pHatInvModp = np.array(self.pHatInvModp, dtype=np.uint64)
        self.pHatModq = np.array(self.pHatModq, dtype=np.uint64)
        self.PModq = np.array(self.PModq, dtype=np.uint64)
        self.qInvModq = np.array(self.qInvModq, dtype=np.uint64)
        self.QlQlInvModqlDivqlModq = np.array(
            self.QlQlInvModqlDivqlModq, dtype=np.uint64
        )

        # for cuda context
        if torch.cuda.is_available():
            self.log_degree = logN
            self.degree = self.N
            self.level = self.L
            self.alpha = self.K
            self.max_num_moduli = self.L + self.K
            self.chain_length = self.L
            self.num_special_moduli = self.K
            self.primes = np.hstack((moduliQ, moduliP))

            self.power_of_roots = None
            self.power_of_roots_shoup = None
            self.inverse_power_of_roots_div_two = None
            self.inverse_scaled_power_of_roots_div_two = None
            self.power_of_roots_vec = []
            self.power_of_roots_shoup_vec = []
            self.inv_power_of_roots_vec = []
            self.inv_power_of_roots_shoup_vec = []
            self.barret_k = []
            self.barret_ratio = []

            # for modup
            self.num_moduli_after_modup = self.max_num_moduli
            self.hat_inverse_vec_modup = None
            self.hat_inverse_vec_shoup_modup = None
            self.prod_q_i_mod_q_j_modup = None

            # for moddown
            self.num_moduli_after_moddown = self.chain_length
            self.hat_inverse_vec_moddown = []
            self.hat_inverse_vec_shoup_moddown = []
            self.prod_q_i_mod_q_j_moddown = []
            self.prod_inv_moddown = []
            self.prod_inv_shoup_moddown = []

            # for drop_last_element_and_scale
            self.qlql_inv_mod_ql_div_ql_mod_q = None
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup = None
            self.q_inv_mod_q = None
            self.q_inv_mod_q_shoup = None

            self.swk_bx_cuda = torch.tensor(
                self.mult_swk[0].reshape(-1),dtype=torch.uint64, device="cuda")
            self.swk_ax_cuda = torch.tensor(
                self.mult_swk[1].reshape(-1),dtype=torch.uint64, device="cuda")
            
            # for output & workspace
            self.beta = (int)(self.level / self.alpha)
            self.inner_workspace = torch.tensor(
                [0] * (4 * self.num_moduli_after_modup * self.degree * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )
            self.inner_out = torch.tensor(
                [0] * (2 * self.num_moduli_after_modup * self.degree),
                dtype=torch.uint64,
                device="cuda",
            )
            self.moddown_out_ax = torch.tensor(
                [0] * (self.num_moduli_after_moddown * self.degree),
                dtype=torch.uint64,
                device="cuda",
            )
            self.moddown_out_bx = torch.tensor(
                [0] * (self.num_moduli_after_moddown * self.degree),
                dtype=torch.uint64,
                device="cuda",
            )
            self.modup_out = torch.tensor(
                [0] * (self.num_moduli_after_modup * self.degree * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )
            self.rescale_out = torch.tensor(
                [0] * ((self.L - 1) * self.degree),
                dtype=torch.uint64,
                device="cuda",
            )
            self.automorphism_transform_out = torch.tensor(
                [0] * (self.num_moduli_after_modup * self.degree * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )
            self.switch_modulus_out = torch.tensor(
                [0] * (self.num_moduli_after_modup * self.degree * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )

            power_of_roots = self.qRootPows + self.pRootPows
            inverse_power_of_roots = self.qRootPowsInv + self.pRootPowsInv
            # cal basic param
            for i, prime in enumerate(self.primes):
                barret = math.floor(math.log2(prime)) + 63
                self.barret_k.append(barret)

                temp = 1 << (barret - 64)
                temp <<= 64
                self.barret_ratio.append(int(temp) // int(prime))
                power_of_roots_shoup = self.shoup_each(power_of_roots[i], prime)
                inv_power_of_roots_div_two = self.div_two(
                    inverse_power_of_roots[i], prime
                )
                inv_power_of_roots_shoup = self.shoup_each(
                    inv_power_of_roots_div_two, prime
                )

                self.power_of_roots_vec.extend(power_of_roots[i])
                self.power_of_roots_shoup_vec.extend(power_of_roots_shoup)
                self.inv_power_of_roots_vec.extend(inv_power_of_roots_div_two)
                self.inv_power_of_roots_shoup_vec.extend(inv_power_of_roots_shoup)

            self.barret_k = torch.tensor(
                self.barret_k, dtype=torch.uint64, device="cuda"
            )
            self.barret_ratio = torch.tensor(
                self.barret_ratio, dtype=torch.uint64, device="cuda"
            )

            self.power_of_roots = torch.tensor(
                self.power_of_roots_vec, dtype=torch.uint64, device="cuda"
            )
            self.power_of_roots_shoup = torch.tensor(
                self.power_of_roots_shoup_vec, dtype=torch.uint64, device="cuda"
            )
            self.inverse_power_of_roots_div_two = torch.tensor(
                self.inv_power_of_roots_vec, dtype=torch.uint64, device="cuda"
            )
            self.inverse_scaled_power_of_roots_div_two = torch.tensor(
                self.inv_power_of_roots_shoup_vec, dtype=torch.uint64, device="cuda"
            )

            # cal modup param
            prod_q_i_mod_q_j_modup = []
            for l in range(self.L):
                prod_qi_mod_qj = []
                for dnum_idx in range(self.dnum):
                    prod_q_i_mod_q_j = self.PartQlHatModp[l][dnum_idx]
                    prod_q_i_mod_q_j = prod_q_i_mod_q_j.swapaxes(1, 0).flatten()
                    prod_qi_mod_qj.append(prod_q_i_mod_q_j)
                prod_q_i_mod_q_j_modup.append(prod_qi_mod_qj)
            self.prod_q_i_mod_q_j_modup = torch.tensor(
                np.array(prod_q_i_mod_q_j_modup, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            hat_inverse_vec_modup = []
            hat_inverse_vec_shoup_modup = []
            for dnum_idx in range(self.dnum):
                for k in range(self.K):
                    hat_inv_shoup = []
                    hat_inverse_vec = self.PartQlHatInvModq[dnum_idx][k]
                    hat_inverse_vec_modup.append(hat_inverse_vec)
                    for k_idx in range(self.K):
                        prime_idx = dnum_idx * self.K + k_idx
                        prime = self.primes[prime_idx]
                        shoup = self.shoup(int(hat_inverse_vec[k_idx]), prime)
                        hat_inv_shoup.append(shoup)
                    hat_inverse_vec_shoup_modup.append(hat_inv_shoup)
            self.hat_inverse_vec_modup = torch.tensor(
                np.array(hat_inverse_vec_modup, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.hat_inverse_vec_shoup_modup = torch.tensor(
                np.array(hat_inverse_vec_shoup_modup, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            # cal moddown param
            start_length = self.num_special_moduli
            end_length = self.chain_length
            start_begin = self.primes[end_length:]
            start_end = start_begin[start_length:]

            hat_inv_moddown = self.pHatInvModp
            hat_inv_shoup_moddown = []
            hat_inverse_vec_moddown = []
            hat_inverse_vec_shoup_moddown = []
            for k in range(self.K):
                prime = self.primes[self.L + k]
                shoup = self.shoup(int(hat_inv_moddown[k]), prime)
                hat_inv_shoup_moddown.append(shoup)
            hat_inverse_vec_moddown.append(hat_inv_moddown)
            self.hat_inverse_vec_moddown = torch.tensor(
                np.array(hat_inverse_vec_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            hat_inverse_vec_shoup_moddown.append(hat_inv_shoup_moddown)
            self.hat_inverse_vec_shoup_moddown = torch.tensor(
                np.array(hat_inverse_vec_shoup_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            prod_q_i_mod_q_j_moddown = []
            end_primes = self.set_difference(self.primes, start_begin)
            prod_q_i_mod_q_j_moddown.append(self.pHatModq.swapaxes(1, 0).flatten())
            self.prod_q_i_mod_q_j_moddown = torch.tensor(
                np.array(prod_q_i_mod_q_j_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            prod_inv = self.PInvModq
            prod_shoup = []

            for i, end_prime in enumerate(end_primes):
                inv = prod_inv[i]
                prod_shoup.append(self.shoup(int(inv), end_prime))

            prod_inv_moddown = []
            prod_inv_moddown.append(prod_inv)
            self.prod_inv_moddown = torch.tensor(
                np.array(prod_inv_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            prod_inv_shoup_moddown = []
            prod_inv_shoup_moddown.append(prod_shoup)
            self.prod_inv_shoup_moddown = torch.tensor(
                np.array(prod_shoup, dtype=np.uint64), dtype=torch.uint64, device="cuda"
            )

            # cal rescale param
            QlQlInvModqlDivqlModq = self.QlQlInvModqlDivqlModq.reshape(-1)
            qlql_inv_mod_ql_div_ql_mod_q_vec = []
            qlql_inv_mod_ql_div_ql_mod_q_shoup_vec = []
            for i in range(self.L - 1):
                for j in range(self.L - 1):
                    QlQlInvModqlDivqlModq_i = QlQlInvModqlDivqlModq[
                        i * (self.L - 1) + j
                    ]
                    prime = self.primes[j]
                    shoup = self.shoup(int(QlQlInvModqlDivqlModq_i), prime)
                    qlql_inv_mod_ql_div_ql_mod_q_vec.append(QlQlInvModqlDivqlModq_i)
                    qlql_inv_mod_ql_div_ql_mod_q_shoup_vec.append(shoup)
            self.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(
                np.array(qlql_inv_mod_ql_div_ql_mod_q_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(
                np.array(qlql_inv_mod_ql_div_ql_mod_q_shoup_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            qInvModq = self.qInvModq.reshape(-1)
            qInvModq_vec = []
            qInvModq_shoup_vec = []
            for i in range(self.L):
                for j in range(self.L):
                    qInvModq_i = qInvModq[i * self.L + j]
                    prime = self.primes[j]
                    shoup = self.shoup(int(qInvModq_i), prime)
                    qInvModq_vec.append(qInvModq_i)
                    qInvModq_shoup_vec.append(shoup)
            self.q_inv_mod_q = torch.tensor(
                np.array(qInvModq_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.q_inv_mod_q_shoup = torch.tensor(
                np.array(qInvModq_shoup_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.PModq_cuda = torch.tensor(self.PModq, dtype=torch.uint64, device="cuda")

            self.primes = torch.tensor(self.primes, dtype=torch.uint64, device="cuda")


        swk_bx = MULT_SWK[0].reshape(self.dnum, L + K, self.N)
        swk_ax = MULT_SWK[1].reshape(self.dnum, L + K, self.N)
        
        self.m_U0hatTPreFFT_mx = BOOT_KEY['C2S']
        self.m_U0PreFFT_mx = BOOT_KEY['S2C']
        self.m_U0hatTPreFFT_dim = BOOT_KEY['C2S_dim']
        self.m_U0PreFFT_dim = BOOT_KEY['S2C_dim']
        self.m_U0hatTPreFFT_limbs = BOOT_KEY['C2S_limbs']
        self.m_U0PreFFT_limbs = BOOT_KEY['S2C_limbs']

        key_map_ax_fixed = torch.tensor(swk_ax, dtype=torch.uint64, device="cuda")
        key_map_bx_fixed = torch.tensor(swk_bx, dtype=torch.uint64, device="cuda")
        self.key_map = [key_map_bx_fixed, key_map_ax_fixed]

        for i, bx, ax in ROT_SWK:
            self.left_rot_key_map[str(i)] = [torch.tensor(bx, dtype=torch.uint64, device="cuda").reshape(self.dnum, -1, self.N)
                                                    ,
                                                    torch.tensor(ax, dtype=torch.uint64, device="cuda").reshape(self.dnum, -1, self.N)]
            

    def shoup(self, in_value, prime):
        temp = in_value << 64
        return int(int(temp) // int(prime))

    def shoup_each(self, values, prime):
        return [self.shoup(value, prime) for value in values]

    def div_two(self, in_list, prime):
        two_inv = self.invMod(2, prime)
        out_list = [self.mulMod(int(x), int(two_inv), int(prime)) for x in in_list]
        return out_list

    def set_difference(self, begin, end):
        remove_set = set(end)
        return [item for item in begin if item not in remove_set]

    def negate(self, r, a):
        r = -a

    def addMod(self, r, a, b, m):
        r = (a + b) % m

    def subMod(self, r, a, b, m):
        r = b % m
        r = (a + m - r) % m

    def mulMod(self, a, b, m):
        mul = (a % m) * (b % m)
        mul %= m
        return int(mul)

    def mulModBarrett(self, r, a, b, p, pr, twok):
        mul = (a % p) * (b % p)
        self.modBarrett(r, mul, p, pr, twok)

    def modBarrett(self, r, a, m, mr, twok):
        tmp = (a * mr) >> twok
        tmp *= m
        tmp = a - tmp
        r = tmp
        if r < m:
            return
        else:
            r -= m
            return

    def invMod(self, x, m):
        temp = int(x) % int(m)
        if self.gcd(temp, m) != 1:
            raise ValueError("Inverse doesn't exist!!!")
        else:
            return self.powMod(int(temp), (int(m) - 2), int(m))

    def powMod(self, x, y, modulus):
        res = 1
        while y > 0:
            if y & 1:
                res = self.mulMod(res, x, modulus)
            y = y >> 1
            x = self.mulMod(x, x, modulus)
        return res

    def inv(self, x):
        UINT64_MAX = 0xFFFFFFFFFFFFFFFF
        return pow(int(x), UINT64_MAX, UINT64_MAX + 1)

    def pow(self, x, y):
        res = 1
        while y > 0:
            if y & 1:
                res *= x
            y = y >> 1
            x *= x
        return res

    def bitReverse(self, n, bit_size=32):
        reversed_bits = 0
        for i in range(bit_size):
            # 将 n 的最低有效位移到 reversed_bits 的适当位置
            reversed_bits <<= 1
            reversed_bits |= n & 1
            n >>= 1
        return reversed_bits

    def gcd(self, a, b):
        if a == 0:
            return b
        return self.gcd(int(b) % int(a), int(a))

    def findPrimeFactors(self, s, number):
        while number % 2 == 0:
            s.add(2)
            number //= 2
        for i in range(3, int(math.sqrt(number)) + 1):
            while number % i == 0:
                s.add(i)
                number //= i
        if number > 2:
            s.add(number)

    def findPrimitiveRoot(self, modulus):
        s = set()
        phi = modulus - 1
        self.findPrimeFactors(s, phi)
        for r in range(2, phi + 1):
            flag = False
            for prime in s:
                if self.powMod(r, phi // prime, modulus) == 1:
                    flag = True
                    break
            if not flag:
                return r
        return -1

    def findMthRootOfUnity(self, M, mod):
        res = self.findPrimitiveRoot(mod)
        if (mod - 1) % M == 0:
            factor = (mod - 1) // M
            res = self.powMod(res, factor, mod)
            return res
        else:
            return -1

    # Miller-Rabin Prime Test #
    def primeTest(self, p):
        if p < 2:
            return False
        if p != 2 and p % 2 == 0:
            return False
        s = p - 1
        while s % 2 == 0:
            s //= 2
        for _ in range(200):
            temp1 = random.getrandbits(64)
            temp1 = (temp1 << 32) | random.getrandbits(32)
            temp1 = temp1 % (p - 1) + 1
            temp2 = s
            mod = self.powMod(temp1, temp2, p)
            while temp2 != p - 1 and mod != 1 and mod != p - 1:
                mod = self.mulMod(mod, mod, p)
                temp2 *= 2
            if mod != p - 1 and temp2 % 2 == 0:
                return False
        return True

    def method(self):  # function to initialize variables
        pass
