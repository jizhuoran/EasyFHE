import cmath
import math
import warnings

import openfhe as openfhe
import torch
from .. import ciphertext as Cipher
import numpy as np

from ..ciphertext import Plaintext
from .. import homo_ops

MAX_BITS_IN_WORD = 61
MAX_64BIT_VALUE = (1 << 63) - (1 << 9) - 1 # openfhetodo: the var must be renamed
M_PI = 3.14159265358979323846

class PrecomputedValues:
    def __init__(self, m, nh):
        self.m_M = m
        self.m_Nh = nh

        # m_rotGroup stores powers of 5 mod m_M
        self.m_rotGroup = []
        fivePows = 1
        for i in range(self.m_Nh):
            self.m_rotGroup.append(fivePows)
            fivePows = (fivePows * 5) % self.m_M

        # m_ksiPows stores the complex roots of unity
        self.m_ksiPows = []
        for j in range(self.m_M):
            angle = 2.0 * M_PI * j / self.m_M
            self.m_ksiPows.append(cmath.exp(1j * angle))  # exp(j * angle) gives the complex number e^(j*theta)

        # m_ksiPows[m_M] is the same as m_ksiPows[0]
        self.m_ksiPows.append(self.m_ksiPows[0])

class OpenFHEContext:
    def __init__(self, content_map):
        openfhe.ClearEvalMultKeys()
        openfhe.ReleaseAllContexts()

        self.cc = openfhe.DeserializeCryptoContextString(content_map["cc"], openfhe.BINARY)
        self.publicKey = openfhe.DeserializePublicKeyString(content_map["publicKey"], openfhe.BINARY)
        self.secretKey = openfhe.DeserializePrivateKeyString(content_map["secretKey"], openfhe.BINARY)
        openfhe.DeserializeEvalAutomorphismKeyString(content_map["app_rot_key"], openfhe.BINARY)
        self.depth = content_map["depth"]

    def setup_for_debug(self, debug_keys, slots, level_budget):
        self.cc.EvalBootstrapSetup(level_budget, [0, 0], slots)
        openfhe.DeserializeEvalMultKeyString(debug_keys["mul_key"], openfhe.BINARY)
        openfhe.DeserializeEvalAutomorphismKeyString(debug_keys["rot_key"], openfhe.BINARY)


    def encode(self, x, scale_deg, level, slots):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            x = x.tolist()
        ptx = self.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
        ptx.Encode()
        data = ptx.GetVectorOfData()
        mv = [torch.tensor(data, device="cuda", dtype=torch.uint64)] #fixme: shall we set device = "cuda" directly?
        return Plaintext(mv, mv[0].shape[0], ptx.GetScalingFactor(), ptx.GetNoiseScaleDeg(), ptx.GetSlots(), False)

    def encrypt(self, x, scale_deg, level, slots):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            x= x.tolist()
        ptx = self.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        data = cipher.GetVectorOfData()
        cv = [torch.tensor(elem, device="cuda", dtype=torch.uint64) for elem in data] #fixme: shall we set device = "cuda" directly?
        gpufhe_cipher = Cipher.Cipher(cv, cv[0].shape[0], cipher.GetScalingFactor(), cipher.GetNoiseScaleDeg(), cipher.GetSlots(), is_ext=False)
        if self.config.PTX_TWIN:
            gpufhe_cipher.ptx_twin = x
        if self.config.mode == "debug":
            return gpufhe_cipher, cipher
        else:
            return gpufhe_cipher

    def decrypt(self, x):
        assert len(x.cv) == 2
        ptx = self.cc.MakeCKKSPackedPlaintext([0.0])
        cipher = self.cc.Encrypt(self.publicKey, ptx)
        cipher.SetNoiseScaleDeg(x.noise_deg)
        cipher.SetLevel(self.depth + 1 - x.cur_limbs)
        cipher.SetScalingFactor(x.scaling_factor)
        cipher.SetSlots(x.slots)

        data = [cv.tolist() for cv in x.cv]
        cipher.SetVectorOfData(data, x.cur_limbs)
        ptx = self.cc.Decrypt(cipher, self.secretKey)

        return torch.tensor(
            ptx.GetRealPackedValue(), device=x.cv[0].device, dtype=torch.float64
        )


    # def encode_gpu(self, cryptoContext, x, scale_deg=None, level=None, slots=None, use_gpu_fft=True):
    #     if not ((scale_deg is None and level is None and slots is None) or
    #             (scale_deg is not None and level is not None and slots is not None)):
    #         raise ValueError("Error: check if scale_deg, level, and slots are set correctly.")
    #
    #     slots = cryptoContext.Nh if slots is None else slots # note: default slots is N/2, which is Nh
    #     scale_deg = 1 if scale_deg is None else scale_deg
    #     cur_limb = cryptoContext.L if level is None else cryptoContext.L - level
    #     if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT" :
    #         scFact = cryptoContext.GetScalingFactorRealBig(cur_limb)
    #         # In FLEXIBLEAUTOEXT mode at level 0, we don't use the noiseScaleDeg
    #         # in our encoding function, so we set it to 1 to make sure it
    #         # has no effect on the encoding.
    #         scale_deg = 1
    #     else:
    #         scFact = cryptoContext.GetScalingFactorReal(cur_limb)
    #
    #     encoded_vector_dcrt_elements_cuda = (
    #         self.ptx_encode_cuda(x, cryptoContext, slots, 'IsDCRTPoly', scFact, cur_limb, scale_deg, use_gpu_fft))
    #
    #     mv = [encoded_vector_dcrt_elements_cuda]
    #     return Plaintext(mv, mv[0].shape[0], scFact, scale_deg, slots, False)
    #
    #
    def bit_reverse(self, vals):
        size = len(vals)
        vals = np.array(vals, dtype=np.complex128)  # 转为 numpy 复数数组
        j = 0
        for i in range(1, size):
            bit = size >> 1
            while j >= bit:
                j -= bit
                bit >>= 1
            j += bit
            if i < j:
                vals[i], vals[j] = vals[j], vals[i]  # 交换复数
        return vals

    def fft_special_inv(self, vals, M, rotGroup, ksiPows):

        # # 检查是否已为给定的cyclotomic order预计算了旋转因子
        # if cycl_order not in precomputed_values:
        #     raise ValueError(f"DiscreteFourierTransform::Initialize() must be called for cyclOrder = {cycl_order}")

        vals_size = len(vals)

        # FFT特定的操作
        len_size = vals_size
        while len_size >= 1:
            len_h = len_size >> 1
            len_q = len_size << 2
            gap = M // len_q  # 根据给定的m_M进行计算

            for i in range(0, vals_size, len_size):
                for j in range(len_h):
                    idx = (len_q - (rotGroup[j] % len_q)) * gap
                    u = vals[i + j] + vals[i + j + len_h]
                    v = vals[i + j] - vals[i + j + len_h]
                    v *= ksiPows[idx]  # 乘以预先计算的旋转因子
                    vals[i + j] = u
                    vals[i + j + len_h] = v
            len_size >>= 1

        vals = self.bit_reverse(vals)

        for i in range(vals_size):
            vals[i] /= vals_size
        return vals
    def fft_special(self, vals, cyclOrder, precomputed_values):
        # # check if the precomputed table exists for the given cyclotomic order
        # if cyclOrder not in precomputed_values:
        #     raise ValueError(f"DiscreteFourierTransform::Initialize() must be called for cyclOrder = {cyclOrder}")

        prepValues = precomputed_values

        # 比特逆序
        vals = self.bit_reverse(vals)

        size = len(vals)
        len2 = 2
        while len2 <= size:
            lenh = len2 // 2
            lenq = len2 * 4
            gap = prepValues.m_M // lenq

            for i in range(0, size, len2):
                for j in range(lenh):
                    idx = (prepValues.m_rotGroup[j] % lenq) * gap
                    u = vals[i + j]
                    v = vals[i + j + lenh] * prepValues.m_ksiPows[idx]
                    vals[i + j] = u + v
                    vals[i + j + lenh] = u - v

            len2 <<= 1
        return vals

    def fit_to_native_vector(self, vec, big_bound, native_vec, native_moduli, N):
        bigValueHf = big_bound >> 1
        modulus = int(native_moduli)
        diff = big_bound - modulus
        ringDim = N
        dslots = len(vec)
        gap = ringDim // dslots

        for i in range(dslots):
            n = vec[i]
            if n > bigValueHf:
                # n % modulus 是为了保证结果在模数范围内
                native_vec[gap * i] = (n - diff) % modulus
            else:
                native_vec[gap * i] = n % modulus
        return native_vec

    def ptx_encode_without_ntt(self, x, N, slots, type_flag, scaling_factor, moduliQ_scalar, L, M, Nh, noise_scale_deg=1):
        # /* Round X to nearest integral value, rounding halfway cases away from
        #    zero.  */
        def llround(x):
            # 对小数部分 >= 0.5 向上舍入，< 0.5 向下舍入
            if x - math.floor(x) > 0.5:
                return math.ceil(x)
            elif x - math.floor(x) == 0.5:
                if x<0:
                    return math.floor(x)
                elif x>0:
                    return math.ceil(x)
                elif x==0:
                    warnings.warn("The input value is zero, which is not expected.")
                    return 0
            return math.floor(x)
        ring_dim = N
        inverse = x

        if slots < len(inverse):
            raise ValueError(f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]")
        encoded_vector_dcrt_elements = np.zeros((L, ring_dim), dtype=np.uint64)
        # Clears all imaginary values as CKKS for complex numbers
        inverse = np.array([complex(v.real, 0.0) for v in inverse])

        # Resize the inverse to fit the slot size.
        # note that default: slots value should be greater than size of input data list x
        inverse = np.pad(inverse, pad_width=(0, slots-len(inverse)), mode='constant', constant_values=complex(0.0, 0.0))
        precomputed_values = PrecomputedValues(M, Nh)

        if type_flag == 'IsDCRTPoly':
            inverse = self.fft_special_inv(inverse, M, precomputed_values.m_rotGroup, precomputed_values.m_ksiPows)

            pow_p = scaling_factor
            logc = 0

            for i in range(slots):
                inverse[i] *= pow_p
                if inverse[i].real != 0:
                    logci = int(math.ceil(math.log2(abs(inverse[i].real))))
                    logc = max(logc, logci)
                if inverse[i].imag != 0:
                    logci = int(math.ceil(math.log2(abs(inverse[i].imag))))
                    logc = max(logc, logci)

            if logc < 0:
                raise ValueError("Too small scaling factor")

            log_valid = min(logc, MAX_BITS_IN_WORD)
            log_approx = logc - log_valid
            approx_factor = 2 ** log_approx

            temp = np.zeros(2 * slots, dtype=int)

            for i in range(slots):
                dre = inverse[i].real / approx_factor
                dim = inverse[i].imag / approx_factor
                re = llround(dre)
                im = llround(dim)

                temp[i] = (MAX_64BIT_VALUE + re) if (re<0) else re  # Handling negative overflow
                temp[i + slots] = (MAX_64BIT_VALUE + im) if (im < 0) else im

            for i in range(L):
                native_moduli = moduliQ_scalar[i]
                native_vec = np.zeros(ring_dim, dtype=np.uint64)
                native_vec = self.fit_to_native_vector(temp, MAX_64BIT_VALUE, native_vec, native_moduli, N)
                encoded_vector_dcrt_elements[i] = native_vec

            num_towers = L
            moduli = moduliQ_scalar[: num_towers]
            crt_pow_p = [llround(pow_p)] * num_towers
            curr_pow_p = crt_pow_p

            for i in range(2, noise_scale_deg):
                curr_pow_p = homo_ops.crt_mult(curr_pow_p, crt_pow_p, moduli)

            if noise_scale_deg > 1:
                for i in range(len(curr_pow_p)):
                    encoded_vector_dcrt_elements[i] = [(a * curr_pow_p[i]) % moduliQ_scalar[i] for a in encoded_vector_dcrt_elements[i]]

            # 反向缩放
            if log_approx > 0:
                max_log_step = 60
                log_step = log_approx if (log_approx <= max_log_step) else max_log_step
                int_step = 1 << log_step
                crt_approx = [int_step] * num_towers
                log_approx -= log_step

                while log_approx > 0:
                    log_step = log_approx if ( log_approx <= max_log_step) else max_log_step
                    int_step = 1 << log_step
                    crt_sf = [int_step] * num_towers
                    crt_approx = homo_ops.crt_mult(crt_approx, crt_sf, moduli)
                    log_approx -= log_step

                # mul_mod =  (a * b) % modulus
                for i in range(len(crt_approx)):
                    encoded_vector_dcrt_elements[i] = [(a * crt_approx[i]) % moduliQ_scalar[i] for a in encoded_vector_dcrt_elements[i]]
        # encoded_vector_dcrt = encoded_vector_dcrt_times(crt_approx)
        else:
            print("Only DCRTPoly is supported for CKKS.")

        return encoded_vector_dcrt_elements
    #
    # def ptx_encode_cuda(self, x, cryptocontext, slots, type_flag, scaling_factor, cur_limbs, noise_scale_deg=1, use_fft = False):
    #     inverse = x
    #     pt_encode = []
    #
    #     if slots < len(inverse):
    #         raise ValueError(f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]")
    #     # Clears all imaginary values as CKKS for complex numbers
    #     inverse = np.array([complex(v.real, 0.0) for v in inverse])
    #
    #     # Resize the inverse to fit the slot size.
    #     # note that default: slots value should be greater than size of input data list x
    #     inverse = np.pad(inverse, pad_width=(0, slots-len(inverse)), mode='constant', constant_values=complex(0.0, 0.0))
    #     if type_flag == 'IsDCRTPoly':
    #         if not use_fft:
    #             inverse = self.fft_special_inv(inverse, cryptocontext.M, cryptocontext.encode_params_rotGroup.cpu().numpy(), cryptocontext.encode_params_ksiPows)
    #
    #         #move precompute&inverse to cuda
    #         inverse_real = torch.tensor(inverse.real.astype(np.double), device="cuda")
    #         inverse_imag = torch.tensor(inverse.imag.astype(np.double),device="cuda")
    #
    #         pt_encode = torch.encode(cryptocontext.encode_out,
    #                                  inverse_real=inverse_real,
    #                                  inverse_imag=inverse_imag,
    #                                  temp=cryptocontext.encode_temp,
    #                                  primes=cryptocontext.primes,
    #                                  precompute_rotgroups=cryptocontext.encode_params_rotGroup,
    #                                  precompute_ksipows_real=cryptocontext.encode_params_ksiPows_real,
    #                                  precompute_ksipows_imag=cryptocontext.encode_params_ksiPows_imag,
    #                                  M=cryptocontext.M,
    #                                  N=cryptocontext.N,
    #                                  cur_limbs=cur_limbs,
    #                                  slots=slots,
    #                                  noise_scale_deg = noise_scale_deg,
    #                                  scaling_factor=scaling_factor,
    #                                  power_of_roots_shoup=cryptocontext.power_of_roots_shoup,
    #                                  power_of_roots=cryptocontext.power_of_roots,
    #                                  use_fft=use_fft)
    #     else:
    #         print("Only DCRTPoly is supported for CKKS.")
    #     return pt_encode


