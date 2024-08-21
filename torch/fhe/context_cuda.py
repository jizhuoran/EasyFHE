import math

import numpy as np
from functools import reduce

import torch
import warnings


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"


warnings.formatwarning = custom_warning_format


class Context_Cuda:
    def __init__(self, logN, L, dnum, moduliQ = None, moduliP = None, primes = None):
        self.log_degree = logN
        self.degree = 1 << logN
        self.dnum = dnum
        self.level = L
        self.alpha = L // dnum
        self.max_num_moduli = self.level + self.alpha
        self.chain_length = self.level
        self.num_special_moduli = self.alpha
        if (primes ==None):
            self.primes = np.hstack((moduliQ, moduliP))
        else:
            self.primes = primes
    
        self.num_moduli_after_modup = self.max_num_moduli
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
        self.hat_inverse_vec = []
        self.hat_inverse_vec_shoup = []
        self.prod_q_i_mod_q_j = []
        # for moddown
        self.hat_inverse_vec_moddown = []
        self.hat_inverse_vec_shoup_moddown = []
        self.prod_q_i_mod_q_j_moddown = []
        self.prod_inv_moddown = []
        self.prod_inv_shoup_moddown = []

        for prime in self.primes:
            barret = math.floor(math.log2(prime)) + 63
            self.barret_k.append(barret)

            temp = 1 << (barret - 64)
            temp <<= 64
            self.barret_ratio.append(temp // prime)

            root = self.find_primitive_root(prime)
            root = self.pow_mod(root, (prime - 1) // (2 * self.degree), prime)

            power_of_roots, inverse_power_of_roots = (
                self.gen_bit_reversed_twiddle_factors(root, prime, self.degree)
            )
            power_of_roots_shoup = self.shoup_each(power_of_roots, prime)
            inv_power_of_roots_div_two = self.div_two(inverse_power_of_roots, prime)
            inv_power_of_roots_shoup = self.shoup_each(
                inv_power_of_roots_div_two, prime
            )

            self.power_of_roots_vec.extend(power_of_roots)
            self.power_of_roots_shoup_vec.extend(power_of_roots_shoup)  # 需要展平
            self.inv_power_of_roots_vec.extend(inv_power_of_roots_div_two)
            self.inv_power_of_roots_shoup_vec.extend(inv_power_of_roots_shoup)

        self.barret_k = torch.tensor(self.barret_k, dtype=torch.uint64, device="cuda")
        self.barret_ratio = torch.tensor(self.barret_ratio, dtype=torch.uint64, device="cuda")

        self.power_of_roots = torch.tensor(self.power_of_roots_vec, dtype=torch.uint64, device="cuda")
        self.power_of_roots_shoup = torch.tensor(self.power_of_roots_shoup_vec, dtype=torch.uint64, device="cuda")
        self.inverse_power_of_roots_div_two = torch.tensor(self.inv_power_of_roots_vec, dtype=torch.uint64,
                                                           device="cuda")
        self.inverse_scaled_power_of_roots_div_two = torch.tensor(self.inv_power_of_roots_shoup_vec, dtype=torch.uint64,
                                                                  device="cuda")

        for dnum_idx in range(self.dnum):
            prime_begin = 0
            prime_end = len(self.primes)
            start_begin = prime_begin + dnum_idx * self.alpha
            start_end = start_begin + self.alpha
            primes_subset = self.primes[start_begin:start_end]
            hat_inv, hat_inv_shoup = self.compute_qi_mod_qj(primes_subset)
            self.hat_inverse_vec.append(torch.tensor(hat_inv, dtype=torch.uint64, device="cuda"))
            self.hat_inverse_vec_shoup.append(torch.tensor(hat_inv_shoup, dtype=torch.uint64, device="cuda"))
            end_primes = self.set_difference(self.primes, primes_subset)
            self.prod_q_i_mod_q_j.append(
                torch.tensor(self.compute_prod_qi_mod_qj(end_primes, primes_subset), dtype=torch.uint64, device="cuda"))

        for gap in range(self.chain_length):
            start_length = self.num_special_moduli + gap
            end_length = self.chain_length - gap

            start_begin = self.primes[end_length:]
            start_end = start_begin[start_length:]

            compute_qi_mod_qj_vec = start_begin[:start_length]
            hat_inv, hat_inv_shoup = self.compute_qi_mod_qj(compute_qi_mod_qj_vec)

            self.hat_inverse_vec_moddown.append(torch.tensor(hat_inv, dtype=torch.uint64, device="cuda"))
            self.hat_inverse_vec_shoup_moddown.append(torch.tensor(hat_inv_shoup, dtype=torch.uint64, device="cuda"))

            end_primes = self.set_difference(self.primes, start_begin)
            self.prod_q_i_mod_q_j_moddown.append(
                torch.tensor(self.compute_prod_qi_mod_qj(end_primes, start_begin), dtype=torch.uint64, device="cuda"))

            prod_inv = []
            prod_shoup = []

            pmodq = [1] * len(end_primes)
            moduli_p = self.set_difference(self.primes, self.primes[:end_length])

            for i, end_prime in enumerate(end_primes):
                for k in range(self.alpha):
                    temp = moduli_p[k] % end_prime
                    pmodq[i] = self.mul_mod(pmodq[i], temp, end_prime)

            for i, end_prime in enumerate(end_primes):
                inv = self.inv_mod(pmodq[i], end_prime)
                prod_inv.append(inv)
                prod_shoup.append(self.shoup(inv, end_prime))

            self.prod_inv_moddown.append(torch.tensor(prod_inv, dtype=torch.uint64, device="cuda"))
            self.prod_inv_shoup_moddown.append(torch.tensor(prod_shoup, dtype=torch.uint64, device="cuda"))
        self.primes = torch.tensor(self.primes, dtype=torch.uint64, device="cuda")

    def gcd(self, a, b):
        if a == 0:
            return b
        return self.gcd(int(b) % int(a), int(a))

    def inv_mod(self, x, m):
        temp = int(x) % int(m)
        if self.gcd(temp, m) != 1:
            raise ValueError("Inverse doesn't exist!!!")
        else:
            return self.pow_mod(temp, int(m - 2), int(m))

    def pow_mod(self, x, y, modulus):
        res = 1
        while y > 0:
            if y & 1:
                res = self.mul_mod(res, x, modulus)
            y >>= 1
            x = self.mul_mod(x, x, modulus)
        return res

    def find_prime_factors(self, number):
        factors = []
        while number % 2 == 0:
            factors.append(2)
            number //= 2
        for i in range(3, int(math.sqrt(number)), 1):
            while number % i == 0:
                factors.append(i)
                number //= i
        if number > 2:
            factors.append(number)
        return factors

    def find_primitive_root(self, modulus):
        phi = modulus - 1
        factors = self.find_prime_factors(phi)
        for r in range(2, phi + 1):
            flag = False
            for factor in factors:
                if self.pow_mod(r, phi // factor, modulus) == 1:
                    flag = True
                    break
            if flag == False:
                return r
        raise ValueError("Cannot find the primitive root of unity")

    def bit_reverse(self, vals):
        n = len(vals)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j >= bit:
                j -= bit
                bit >>= 1
            j += bit
            if i < j:
                vals[i], vals[j] = vals[j], vals[i]
        # def bit_reverse(self, x):
        #     x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1))
        #     x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2))
        #     x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4))
        #     x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8))
        #     return ((x >> 16) | (x << 16))

    def inverse(self, op, prime):
        if op > prime:
            tmp = op % prime
        else:
            tmp = op
        return self.pow_mod(tmp, prime - 2, prime)

    def mul_mod(self, a, b, m):
        # result = (a * b) % m
        # return result
        a = int(a)
        b = int(b)
        m = int(m)
        # mul = 4294967296*4294967296

        mul = int(a * b)
        # Modulo operation on the result
        result = mul % m
        return result

    def gen_bit_reversed_twiddle_factors(self, root, p, degree):
        pow_roots = [1] * degree
        inv_pow_roots = [1] * degree
        root_inverse = self.inverse(root, p)
        for i in range(1, degree):
            pow_roots[i] = self.mul_mod(pow_roots[i - 1], root, p)
            inv_pow_roots[i] = self.mul_mod(inv_pow_roots[i - 1], root_inverse, p)

        self.bit_reverse(pow_roots)
        self.bit_reverse(inv_pow_roots)

        return pow_roots, inv_pow_roots

    def shoup(self, in_value, prime):
        temp = in_value << 64
        return temp // prime

    def shoup_each(self, values, prime):
        return [self.shoup(value, prime) for value in values]

    def div_two(self, in_list, prime):
        two_inv = self.inverse(2, prime)
        out_list = [self.mul_mod(x, two_inv, prime) for x in in_list]
        return out_list

    def product_except(self, primes, except_val, modulus):
        return reduce(
            lambda accum, prime: accum if prime == except_val else self.mul_mod(accum, prime % modulus, modulus),
            primes,
            1
        )

    def compute_qi_hats(self, primes):
        q_i_hats = [
            self.product_except(primes, modulus, modulus)
            for modulus in primes
        ]
        return q_i_hats

    def compute_qi_mod_qj(self, primes):
        hat_inv_vec = []
        hat_inv_shoup_vec = []
        q_i_hats = self.compute_qi_hats(primes)
        hat_inv_vec = [self.inverse(q_i_hat, prime) for q_i_hat, prime in zip(q_i_hats, primes)]
        hat_inv_shoup_vec = [self.shoup(hat_inv, prime) for hat_inv, prime in zip(hat_inv_vec, primes)]
        return hat_inv_vec, hat_inv_shoup_vec

    def set_difference(self, begin, end):
        remove_set = set(end)
        return [item for item in begin if item not in remove_set]

    def compute_prod_qi_mod_qj(self, end_primes, start_primes):
        prod_q_i_mod_q_j = []
        for modulus in end_primes:
            for p in start_primes:
                prod_q_i_mod_q_j.append(self.product_except(start_primes, p, modulus))
        return prod_q_i_mod_q_j
