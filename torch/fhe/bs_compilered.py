from .bs_context import *
from . import functional as F
from . import homo_ops
from . import hybrid_keyswitch
from . import utils
import numpy as np

# @profile_python_function
def mod_raise(cipher, L0, cryptoContext):
    cv = [
        torch.mod_raise(
            cryptoContext.mod_raise_out,
            cv,
            primes=cryptoContext.primes,
            N=cryptoContext.N,
            L0=L0,
            logN=cryptoContext.logN,
            L=cryptoContext.L,
            inverse_power_of_roots_div_two=cryptoContext.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=cryptoContext.inverse_scaled_power_of_roots_div_two,
            power_of_roots_shoup=cryptoContext.power_of_roots_shoup,
            power_of_roots=cryptoContext.power_of_roots,
            barret_ratio=cryptoContext.barret_ratio,
            barret_k=cryptoContext.barret_k,
        ).reshape(-1, cryptoContext.N)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, L0)


def mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.cur_limbs, monomial_degree, cryptoContext)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.cur_limbs, monomial_degree, cryptoContext)
    return cipher

# note: EvalBootstrap in ckksrns-fhe.cpp
def eval_bootstrap(IN_NODE, L0, logBsSlots, cryptoContext):
    NODE1 = cryptoContext.BsContext.m_U0hatTPreFFT[0][0]
    NODE2 = cryptoContext.BsContext.m_U0hatTPreFFT[0][1]
    NODE3 = cryptoContext.BsContext.m_U0hatTPreFFT[0][2]
    NODE4 = cryptoContext.BsContext.m_U0hatTPreFFT[0][3]
    NODE5 = cryptoContext.BsContext.m_U0hatTPreFFT[0][4]
    NODE6 = cryptoContext.BsContext.m_U0hatTPreFFT[0][5]
    NODE7 = cryptoContext.BsContext.m_U0hatTPreFFT[0][6]
    NODE8 = cryptoContext.BsContext.m_U0hatTPreFFT[1][0]
    NODE9 = cryptoContext.BsContext.m_U0hatTPreFFT[1][1]
    NODE10 = cryptoContext.BsContext.m_U0hatTPreFFT[1][2]
    NODE11 = cryptoContext.BsContext.m_U0hatTPreFFT[1][3]
    NODE12 = cryptoContext.BsContext.m_U0hatTPreFFT[1][4]
    NODE13 = cryptoContext.BsContext.m_U0hatTPreFFT[1][5]
    NODE14 = cryptoContext.BsContext.m_U0hatTPreFFT[1][6]
    NODE15 = cryptoContext.BsContext.m_U0hatTPreFFT[2][0]
    NODE16 = cryptoContext.BsContext.m_U0hatTPreFFT[2][1]
    NODE17 = cryptoContext.BsContext.m_U0hatTPreFFT[2][2]
    NODE18 = cryptoContext.BsContext.m_U0hatTPreFFT[2][3]
    NODE19 = cryptoContext.BsContext.m_U0hatTPreFFT[2][4]
    NODE20 = cryptoContext.BsContext.m_U0hatTPreFFT[2][5]
    NODE21 = cryptoContext.BsContext.m_U0hatTPreFFT[2][6]
    NODE22 = cryptoContext.BsContext.m_U0hatTPreFFT[3][0]
    NODE23 = cryptoContext.BsContext.m_U0hatTPreFFT[3][1]
    NODE24 = cryptoContext.BsContext.m_U0hatTPreFFT[3][2]
    NODE25 = cryptoContext.BsContext.m_U0hatTPreFFT[3][3]
    NODE26 = cryptoContext.BsContext.m_U0hatTPreFFT[3][4]
    NODE27 = cryptoContext.BsContext.m_U0hatTPreFFT[3][5]
    NODE28 = cryptoContext.BsContext.m_U0hatTPreFFT[3][6]
    NODE29 = cryptoContext.BsContext.m_U0PreFFT[0][0]
    NODE30 = cryptoContext.BsContext.m_U0PreFFT[0][1]
    NODE31 = cryptoContext.BsContext.m_U0PreFFT[0][2]
    NODE32 = cryptoContext.BsContext.m_U0PreFFT[0][3]
    NODE33 = cryptoContext.BsContext.m_U0PreFFT[0][4]
    NODE34 = cryptoContext.BsContext.m_U0PreFFT[0][5]
    NODE35 = cryptoContext.BsContext.m_U0PreFFT[0][6]
    NODE36 = cryptoContext.BsContext.m_U0PreFFT[1][0]
    NODE37 = cryptoContext.BsContext.m_U0PreFFT[1][1]
    NODE38 = cryptoContext.BsContext.m_U0PreFFT[1][2]
    NODE39 = cryptoContext.BsContext.m_U0PreFFT[1][3]
    NODE40 = cryptoContext.BsContext.m_U0PreFFT[1][4]
    NODE41 = cryptoContext.BsContext.m_U0PreFFT[1][5]
    NODE42 = cryptoContext.BsContext.m_U0PreFFT[1][6]
    NODE43 = cryptoContext.BsContext.m_U0PreFFT[2][0]
    NODE44 = cryptoContext.BsContext.m_U0PreFFT[2][1]
    NODE45 = cryptoContext.BsContext.m_U0PreFFT[2][2]
    NODE46 = cryptoContext.BsContext.m_U0PreFFT[2][3]
    NODE47 = cryptoContext.BsContext.m_U0PreFFT[2][4]
    NODE48 = cryptoContext.BsContext.m_U0PreFFT[2][5]
    NODE49 = cryptoContext.BsContext.m_U0PreFFT[2][6]
    NODE50 = cryptoContext.BsContext.m_U0PreFFT[3][0]
    NODE51 = cryptoContext.BsContext.m_U0PreFFT[3][1]
    NODE52 = cryptoContext.BsContext.m_U0PreFFT[3][2]
    NODE53 = cryptoContext.BsContext.m_U0PreFFT[3][3]
    NODE54 = cryptoContext.BsContext.m_U0PreFFT[3][4]
    NODE55 = cryptoContext.BsContext.m_U0PreFFT[3][5]
    NODE56 = cryptoContext.BsContext.m_U0PreFFT[3][6]
    NODE57 = IN_NODE
    NODE58 = homo_ops.homo_rescale(NODE57, 0, cryptoContext) #out: limb=2, noise=1, in0: limb=2, noise=1
    NODE60 = homo_ops.homo_mul_scalar_double(NODE58, 0.0019531249999613642, cryptoContext) #out: limb=2, noise=2, in0: limb=2, noise=1
    NODE61 = homo_ops.homo_rescale(NODE60, 1, cryptoContext) #out: limb=1, noise=1, in0: limb=2, noise=2
    NODE62 = mod_raise(NODE61, 26, cryptoContext) #out: limb=26, noise=1, in0: limb=1, noise=1
    NODE63 = homo_ops.homo_mul_scalar_double(NODE62, 5.960464477539063e-08, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=1
    NODE64 = homo_ops.homo_rotate(NODE63, 256, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2
    NODE65 = homo_ops.homo_add(NODE63, NODE64, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2, in1: limb=26, noise=2
    NODE66 = homo_ops.homo_rotate(NODE65, 512, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2
    NODE67 = homo_ops.homo_add(NODE65, NODE66, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2, in1: limb=26, noise=2
    NODE68 = homo_ops.homo_rotate(NODE67, 1024, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2
    NODE69 = homo_ops.homo_add(NODE67, NODE68, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2, in1: limb=26, noise=2
    NODE70 = homo_ops.homo_rotate(NODE69, 2048, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2
    NODE71 = homo_ops.homo_add(NODE69, NODE70, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2, in1: limb=26, noise=2
    NODE72 = homo_ops.homo_rotate(NODE71, 4096, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2
    NODE73 = homo_ops.homo_add(NODE71, NODE72, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=2, in1: limb=26, noise=2
    NODE74 = homo_ops.homo_rescale(NODE73, 1, cryptoContext) #out: limb=25, noise=1, in0: limb=26, noise=2
    NODE75 = homo_ops.extract_cv(NODE74, 1) #out: limb=25, noise=1, in0: limb=25, noise=1
    NODE76 = hybrid_keyswitch.modup_to_ext(NODE75, cryptoContext) #out: limb=25, noise=1, in0: limb=25, noise=1
    NODE77 = homo_ops.eval_fast_rotate(NODE76, NODE74, 64, True, False, cryptoContext) #out: limb=25, noise=1, in0: limb=25, noise=1in1: limb=25, noise=1
    NODE78 = homo_ops.eval_fast_rotate(NODE76, NODE74, 128, True, False, cryptoContext) #out: limb=25, noise=1, in0: limb=25, noise=1in1: limb=25, noise=1
    NODE79 = homo_ops.eval_fast_rotate(NODE76, NODE74, 192, True, False, cryptoContext) #out: limb=25, noise=1, in0: limb=25, noise=1in1: limb=25, noise=1
    NODE80 = hybrid_keyswitch.key_switch_P_ext(NODE74, cryptoContext) #out: limb=25, noise=1, in0: limb=25, noise=1
    NODE81 = homo_ops.homo_mul_pt(NODE77, NODE22, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE82 = homo_ops.homo_mul_pt(NODE78, NODE23, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE83 = homo_ops.homo_add(NODE81, NODE82, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE84 = homo_ops.homo_mul_pt(NODE79, NODE24, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE85 = homo_ops.homo_add(NODE83, NODE84, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE86 = homo_ops.homo_mul_pt(NODE80, NODE25, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE87 = homo_ops.homo_add(NODE85, NODE86, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE88 = homo_ops.extract_cv(NODE87, 0) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE89 = hybrid_keyswitch.moddown_from_ext(NODE88, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE90 = homo_ops.extract_cv(NODE87, 1, append_zeros = True) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE91 = homo_ops.homo_mul_pt(NODE77, NODE26, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE92 = homo_ops.homo_mul_pt(NODE78, NODE27, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE93 = homo_ops.homo_add(NODE91, NODE92, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE94 = homo_ops.homo_mul_pt(NODE79, NODE28, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=1, in1: limb=25, noise=1
    NODE95 = homo_ops.homo_add(NODE93, NODE94, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE96 = hybrid_keyswitch.moddown_from_ext(NODE95, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE97 = homo_ops.extract_cv(NODE96, 0) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE98 = homo_ops.extract_cv(NODE96, 1) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE99 = homo_ops._cipher_automorphism(NODE97, 256, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE100 = homo_ops.homo_add(NODE89, NODE99, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE101 = hybrid_keyswitch.modup_to_ext(NODE98, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE102 = homo_ops.eval_fast_rotate(NODE101, None, 256, False, None, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE103 = homo_ops.homo_add(NODE90, NODE102, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE104 = hybrid_keyswitch.moddown_from_ext(NODE103, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE105 = homo_ops.extract_cv(NODE100, 0, append_zeros = True) #out: limb=25, noise=2, in0: limb=25, noise=2
    NODE106 = homo_ops.homo_add(NODE104, NODE105, cryptoContext) #out: limb=25, noise=2, in0: limb=25, noise=2, in1: limb=25, noise=2
    NODE107 = homo_ops.homo_rescale(NODE106, 1, cryptoContext) #out: limb=24, noise=1, in0: limb=25, noise=2
    NODE108 = homo_ops.extract_cv(NODE107, 1) #out: limb=24, noise=1, in0: limb=24, noise=1
    NODE109 = hybrid_keyswitch.modup_to_ext(NODE108, cryptoContext) #out: limb=24, noise=1, in0: limb=24, noise=1
    NODE110 = homo_ops.eval_fast_rotate(NODE109, NODE107, 208, True, False, cryptoContext) #out: limb=24, noise=1, in0: limb=24, noise=1in1: limb=24, noise=1
    NODE111 = homo_ops.eval_fast_rotate(NODE109, NODE107, 224, True, False, cryptoContext) #out: limb=24, noise=1, in0: limb=24, noise=1in1: limb=24, noise=1
    NODE112 = homo_ops.eval_fast_rotate(NODE109, NODE107, 240, True, False, cryptoContext) #out: limb=24, noise=1, in0: limb=24, noise=1in1: limb=24, noise=1
    NODE113 = hybrid_keyswitch.key_switch_P_ext(NODE107, cryptoContext) #out: limb=24, noise=1, in0: limb=24, noise=1
    NODE114 = homo_ops.homo_mul_pt(NODE110, NODE15, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE115 = homo_ops.homo_mul_pt(NODE111, NODE16, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE116 = homo_ops.homo_add(NODE114, NODE115, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE117 = homo_ops.homo_mul_pt(NODE112, NODE17, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE118 = homo_ops.homo_add(NODE116, NODE117, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE119 = homo_ops.homo_mul_pt(NODE113, NODE18, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE120 = homo_ops.homo_add(NODE118, NODE119, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE121 = homo_ops.extract_cv(NODE120, 0) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE122 = hybrid_keyswitch.moddown_from_ext(NODE121, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE123 = homo_ops.extract_cv(NODE120, 1, append_zeros = True) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE124 = homo_ops.homo_mul_pt(NODE110, NODE19, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE125 = homo_ops.homo_mul_pt(NODE111, NODE20, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE126 = homo_ops.homo_add(NODE124, NODE125, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE127 = homo_ops.homo_mul_pt(NODE112, NODE21, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=1, in1: limb=24, noise=1
    NODE128 = homo_ops.homo_add(NODE126, NODE127, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE129 = hybrid_keyswitch.moddown_from_ext(NODE128, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE130 = homo_ops.extract_cv(NODE129, 0) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE131 = homo_ops.extract_cv(NODE129, 1) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE132 = homo_ops._cipher_automorphism(NODE130, 64, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE133 = homo_ops.homo_add(NODE122, NODE132, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE134 = hybrid_keyswitch.modup_to_ext(NODE131, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE135 = homo_ops.eval_fast_rotate(NODE134, None, 64, False, None, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE136 = homo_ops.homo_add(NODE123, NODE135, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE137 = hybrid_keyswitch.moddown_from_ext(NODE136, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE138 = homo_ops.extract_cv(NODE133, 0, append_zeros = True) #out: limb=24, noise=2, in0: limb=24, noise=2
    NODE139 = homo_ops.homo_add(NODE137, NODE138, cryptoContext) #out: limb=24, noise=2, in0: limb=24, noise=2, in1: limb=24, noise=2
    NODE140 = homo_ops.homo_rescale(NODE139, 1, cryptoContext) #out: limb=23, noise=1, in0: limb=24, noise=2
    NODE141 = homo_ops.extract_cv(NODE140, 1) #out: limb=23, noise=1, in0: limb=23, noise=1
    NODE142 = hybrid_keyswitch.modup_to_ext(NODE141, cryptoContext) #out: limb=23, noise=1, in0: limb=23, noise=1
    NODE143 = homo_ops.eval_fast_rotate(NODE142, NODE140, 244, True, False, cryptoContext) #out: limb=23, noise=1, in0: limb=23, noise=1in1: limb=23, noise=1
    NODE144 = homo_ops.eval_fast_rotate(NODE142, NODE140, 248, True, False, cryptoContext) #out: limb=23, noise=1, in0: limb=23, noise=1in1: limb=23, noise=1
    NODE145 = homo_ops.eval_fast_rotate(NODE142, NODE140, 252, True, False, cryptoContext) #out: limb=23, noise=1, in0: limb=23, noise=1in1: limb=23, noise=1
    NODE146 = hybrid_keyswitch.key_switch_P_ext(NODE140, cryptoContext) #out: limb=23, noise=1, in0: limb=23, noise=1
    NODE147 = homo_ops.homo_mul_pt(NODE143, NODE8, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE148 = homo_ops.homo_mul_pt(NODE144, NODE9, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE149 = homo_ops.homo_add(NODE147, NODE148, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE150 = homo_ops.homo_mul_pt(NODE145, NODE10, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE151 = homo_ops.homo_add(NODE149, NODE150, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE152 = homo_ops.homo_mul_pt(NODE146, NODE11, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE153 = homo_ops.homo_add(NODE151, NODE152, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE154 = homo_ops.extract_cv(NODE153, 0) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE155 = hybrid_keyswitch.moddown_from_ext(NODE154, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE156 = homo_ops.extract_cv(NODE153, 1, append_zeros = True) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE157 = homo_ops.homo_mul_pt(NODE143, NODE12, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE158 = homo_ops.homo_mul_pt(NODE144, NODE13, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE159 = homo_ops.homo_add(NODE157, NODE158, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE160 = homo_ops.homo_mul_pt(NODE145, NODE14, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=1, in1: limb=23, noise=1
    NODE161 = homo_ops.homo_add(NODE159, NODE160, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE162 = hybrid_keyswitch.moddown_from_ext(NODE161, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE163 = homo_ops.extract_cv(NODE162, 0) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE164 = homo_ops.extract_cv(NODE162, 1) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE165 = homo_ops._cipher_automorphism(NODE163, 16, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE166 = homo_ops.homo_add(NODE155, NODE165, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE167 = hybrid_keyswitch.modup_to_ext(NODE164, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE168 = homo_ops.eval_fast_rotate(NODE167, None, 16, False, None, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE169 = homo_ops.homo_add(NODE156, NODE168, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE170 = hybrid_keyswitch.moddown_from_ext(NODE169, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE171 = homo_ops.extract_cv(NODE166, 0, append_zeros = True) #out: limb=23, noise=2, in0: limb=23, noise=2
    NODE172 = homo_ops.homo_add(NODE170, NODE171, cryptoContext) #out: limb=23, noise=2, in0: limb=23, noise=2, in1: limb=23, noise=2
    NODE173 = homo_ops.homo_rescale(NODE172, 1, cryptoContext) #out: limb=22, noise=1, in0: limb=23, noise=2
    NODE174 = homo_ops.extract_cv(NODE173, 1) #out: limb=22, noise=1, in0: limb=22, noise=1
    NODE175 = hybrid_keyswitch.modup_to_ext(NODE174, cryptoContext) #out: limb=22, noise=1, in0: limb=22, noise=1
    NODE176 = homo_ops.eval_fast_rotate(NODE175, NODE173, 253, True, False, cryptoContext) #out: limb=22, noise=1, in0: limb=22, noise=1in1: limb=22, noise=1
    NODE177 = homo_ops.eval_fast_rotate(NODE175, NODE173, 254, True, False, cryptoContext) #out: limb=22, noise=1, in0: limb=22, noise=1in1: limb=22, noise=1
    NODE178 = homo_ops.eval_fast_rotate(NODE175, NODE173, 255, True, False, cryptoContext) #out: limb=22, noise=1, in0: limb=22, noise=1in1: limb=22, noise=1
    NODE179 = hybrid_keyswitch.key_switch_P_ext(NODE173, cryptoContext) #out: limb=22, noise=1, in0: limb=22, noise=1
    NODE180 = homo_ops.homo_mul_pt(NODE176, NODE1, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE181 = homo_ops.homo_mul_pt(NODE177, NODE2, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE182 = homo_ops.homo_add(NODE180, NODE181, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE183 = homo_ops.homo_mul_pt(NODE178, NODE3, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE184 = homo_ops.homo_add(NODE182, NODE183, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE185 = homo_ops.homo_mul_pt(NODE179, NODE4, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE186 = homo_ops.homo_add(NODE184, NODE185, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE187 = homo_ops.extract_cv(NODE186, 0) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE188 = hybrid_keyswitch.moddown_from_ext(NODE187, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE189 = homo_ops.extract_cv(NODE186, 1, append_zeros = True) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE190 = homo_ops.homo_mul_pt(NODE176, NODE5, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE191 = homo_ops.homo_mul_pt(NODE177, NODE6, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE192 = homo_ops.homo_add(NODE190, NODE191, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE193 = homo_ops.homo_mul_pt(NODE178, NODE7, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=1, in1: limb=22, noise=1
    NODE194 = homo_ops.homo_add(NODE192, NODE193, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE195 = hybrid_keyswitch.moddown_from_ext(NODE194, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE196 = homo_ops.extract_cv(NODE195, 0) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE197 = homo_ops.extract_cv(NODE195, 1) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE198 = homo_ops._cipher_automorphism(NODE196, 4, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE199 = homo_ops.homo_add(NODE188, NODE198, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE200 = hybrid_keyswitch.modup_to_ext(NODE197, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE201 = homo_ops.eval_fast_rotate(NODE200, None, 4, False, None, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE202 = homo_ops.homo_add(NODE189, NODE201, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE203 = hybrid_keyswitch.moddown_from_ext(NODE202, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE204 = homo_ops.extract_cv(NODE199, 0, append_zeros = True) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE205 = homo_ops.homo_add(NODE203, NODE204, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE206 = homo_ops.homo_rotate(NODE205, 32767, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2
    NODE207 = homo_ops.homo_add(NODE205, NODE206, cryptoContext) #out: limb=22, noise=2, in0: limb=22, noise=2, in1: limb=22, noise=2
    NODE208 = homo_ops.homo_rescale(NODE207, 1, cryptoContext) #out: limb=21, noise=1, in0: limb=22, noise=2
    NODE209 = homo_ops.homo_mul(NODE208, NODE208, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=1, in1: limb=21, noise=1
    NODE210 = homo_ops.homo_add(NODE209, NODE209, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=2, in1: limb=21, noise=2
    NODE211 = homo_ops.homo_add_scalar_double(NODE210, -1.0, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=2
    NODE212 = homo_ops.homo_mul(NODE208, NODE211, cryptoContext) #out: limb=20, noise=2, in0: limb=21, noise=1, in1: limb=21, noise=2
    NODE213 = homo_ops.homo_add(NODE212, NODE212, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2, in1: limb=20, noise=2
    NODE214 = homo_ops.homo_sub(NODE213, NODE208, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2, in1: limb=21, noise=1
    NODE215 = homo_ops.homo_mul(NODE211, NODE211, cryptoContext) #out: limb=20, noise=2, in0: limb=21, noise=2, in1: limb=21, noise=2
    NODE216 = homo_ops.homo_add(NODE215, NODE215, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2, in1: limb=20, noise=2
    NODE217 = homo_ops.homo_add_scalar_double(NODE216, -1.0, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2
    NODE218 = homo_ops.homo_mul(NODE211, NODE214, cryptoContext) #out: limb=19, noise=2, in0: limb=21, noise=2, in1: limb=20, noise=2
    NODE219 = homo_ops.homo_add(NODE218, NODE218, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE220 = homo_ops.homo_sub(NODE219, NODE208, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=21, noise=1
    NODE221 = homo_ops.homo_mul(NODE214, NODE214, cryptoContext) #out: limb=19, noise=2, in0: limb=20, noise=2, in1: limb=20, noise=2
    NODE222 = homo_ops.homo_add(NODE221, NODE221, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE223 = homo_ops.homo_add_scalar_double(NODE222, -1.0, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE224, NODE225 = homo_ops.adjust_levels_and_depth(NODE208, NODE223, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=21, noise=1, in1: limb=19, noise=2
    NODE226, NODE227 = homo_ops.adjust_levels_and_depth(NODE211, NODE225, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=21, noise=2, in1: limb=19, noise=2
    NODE228, NODE229 = homo_ops.adjust_levels_and_depth(NODE214, NODE227, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=20, noise=2, in1: limb=19, noise=2
    NODE230, NODE231 = homo_ops.adjust_levels_and_depth(NODE217, NODE229, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=20, noise=2, in1: limb=19, noise=2
    NODE232, NODE233 = homo_ops.adjust_levels_and_depth(NODE220, NODE231, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE234 = homo_ops.homo_square(NODE233, cryptoContext) #out: limb=18, noise=2, in0: limb=19, noise=2
    NODE235 = homo_ops.homo_add(NODE234, NODE234, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE236 = homo_ops.homo_add_scalar_double(NODE235, -1.0, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE237 = homo_ops.homo_square(NODE236, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2
    NODE238 = homo_ops.homo_add(NODE237, NODE237, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE239 = homo_ops.homo_add_scalar_double(NODE238, -1.0, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE240 = homo_ops.homo_square(NODE239, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2
    NODE241 = homo_ops.homo_add(NODE240, NODE240, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE242 = homo_ops.homo_add_scalar_double(NODE241, -1.0, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2
    NODE243 = homo_ops.homo_mul(NODE233, NODE236, cryptoContext) #out: limb=17, noise=2, in0: limb=19, noise=2, in1: limb=18, noise=2
    NODE244 = homo_ops.homo_add(NODE243, NODE243, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE245 = homo_ops.homo_sub(NODE244, NODE233, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=19, noise=2
    NODE246 = homo_ops.homo_mul(NODE245, NODE239, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE247 = homo_ops.homo_add(NODE246, NODE246, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE248 = homo_ops.homo_sub(NODE247, NODE233, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=19, noise=2
    NODE249 = homo_ops.homo_mul(NODE248, NODE242, cryptoContext) #out: limb=15, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE250 = homo_ops.homo_add(NODE249, NODE249, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=15, noise=2
    NODE251 = homo_ops.homo_sub(NODE250, NODE233, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=19, noise=2
    NODE252, NODE253 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE254, NODE255 = homo_ops.adjust_levels_and_depth(NODE226, NODE253, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE256, NODE257 = homo_ops.adjust_levels_and_depth(NODE228, NODE255, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE258, NODE259 = homo_ops.adjust_levels_and_depth(NODE230, NODE257, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE260 = homo_ops.homo_rescale(NODE252, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE261 = homo_ops.homo_rescale(NODE254, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE262 = homo_ops.homo_rescale(NODE256, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE263 = homo_ops.homo_rescale(NODE258, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE264 = homo_ops.homo_rescale(NODE259, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE265 = homo_ops.homo_mul_scalar_double(NODE260, np.float64(-0.0005862476626482575), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE266 = homo_ops.homo_mul_scalar_double(NODE261, np.float64(-0.05094407670735883), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE267 = homo_ops.homo_add(NODE265, NODE266, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE268 = homo_ops.homo_mul_scalar_double(NODE262, np.float64(0.010324286361991016), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE269 = homo_ops.homo_add(NODE267, NODE268, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE270 = homo_ops.homo_mul_scalar_double(NODE263, np.float64(-0.06820640296455721), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE271 = homo_ops.homo_add(NODE269, NODE270, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE272 = homo_ops.homo_mul_scalar_double(NODE264, np.float64(-0.01629177159536448), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE273 = homo_ops.homo_add(NODE271, NODE272, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE274 = homo_ops.homo_add_scalar_double(NODE273, np.float64(-0.3617312474102197), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE275, NODE276 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE277, NODE278 = homo_ops.adjust_levels_and_depth(NODE226, NODE276, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE279, NODE280 = homo_ops.adjust_levels_and_depth(NODE228, NODE278, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE281, NODE282 = homo_ops.adjust_levels_and_depth(NODE230, NODE280, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE283 = homo_ops.homo_rescale(NODE275, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE284 = homo_ops.homo_rescale(NODE277, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE285 = homo_ops.homo_rescale(NODE279, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE286 = homo_ops.homo_rescale(NODE281, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE287 = homo_ops.homo_rescale(NODE282, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE288 = homo_ops.homo_mul_scalar_double(NODE283, np.float64(-3.646495964794955e-07), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE289 = homo_ops.homo_mul_scalar_double(NODE284, np.float64(6.523242811745607e-06), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE290 = homo_ops.homo_add(NODE288, NODE289, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE291 = homo_ops.homo_mul_scalar_double(NODE285, np.float64(6.924798172957744e-08), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE292 = homo_ops.homo_add(NODE290, NODE291, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE293 = homo_ops.homo_mul_scalar_double(NODE286, np.float64(-1.153465700731715e-06), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE294 = homo_ops.homo_add(NODE292, NODE293, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE295 = homo_ops.homo_mul_scalar_double(NODE287, np.float64(-1.3952337882417398e-08), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE296 = homo_ops.homo_add(NODE294, NODE295, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE297 = homo_ops.homo_add_scalar_double(NODE296, np.float64(-0.25001653312166516), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE298, NODE299 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE300, NODE301 = homo_ops.adjust_levels_and_depth(NODE226, NODE299, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE302, NODE303 = homo_ops.adjust_levels_and_depth(NODE228, NODE301, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE304, NODE305 = homo_ops.adjust_levels_and_depth(NODE230, NODE303, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE306 = homo_ops.homo_rescale(NODE298, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE307 = homo_ops.homo_rescale(NODE300, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE308 = homo_ops.homo_rescale(NODE302, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE309 = homo_ops.homo_rescale(NODE304, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE310 = homo_ops.homo_rescale(NODE305, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE311 = homo_ops.homo_mul_scalar_double(NODE306, np.float64(-5.287332369682873e-12), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE312 = homo_ops.homo_mul_scalar_double(NODE307, np.float64(7.59317852040558e-11), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE313 = homo_ops.homo_add(NODE311, NODE312, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE314 = homo_ops.homo_mul_scalar_double(NODE308, np.float64(6.476916381503101e-13), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE315 = homo_ops.homo_add(NODE313, NODE314, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE316 = homo_ops.homo_mul_scalar_double(NODE309, np.float64(-8.902002730465436e-12), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE317 = homo_ops.homo_add(NODE315, NODE316, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE318 = homo_ops.homo_mul_scalar_double(NODE310, np.float64(-8.256706803909277e-14), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE319 = homo_ops.homo_add(NODE317, NODE318, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE320 = homo_ops.homo_add_scalar_double(NODE319, np.float64(-0.6250000003005246), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE321, NODE322 = homo_ops.adjust_levels_and_depth(NODE224, NODE230, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE323, NODE324 = homo_ops.adjust_levels_and_depth(NODE226, NODE322, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE325, NODE326 = homo_ops.adjust_levels_and_depth(NODE228, NODE324, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE327 = homo_ops.homo_rescale(NODE321, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE328 = homo_ops.homo_rescale(NODE323, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE329 = homo_ops.homo_rescale(NODE325, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE330 = homo_ops.homo_rescale(NODE326, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE331 = homo_ops.homo_mul_scalar_double(NODE327, np.float64(6.536095011040416e-14), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE332 = homo_ops.homo_mul_scalar_double(NODE328, np.float64(-8.489388967084298e-13), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE333 = homo_ops.homo_add(NODE331, NODE332, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE334 = homo_ops.homo_mul_scalar_double(NODE329, np.float64(-7.167799437636123e-15), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE335 = homo_ops.homo_add(NODE333, NODE334, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE336 = homo_ops.homo_mul_scalar_double(NODE330, np.float64(9.137260236825108e-14), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE337 = homo_ops.homo_add(NODE335, NODE336, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE338 = homo_ops.homo_mul_scalar_int(NODE233, 8, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE339 = homo_ops.homo_add(NODE337, NODE338, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE340 = homo_ops.homo_add_scalar_double(NODE339, np.float64(4.022969223666898e-12), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE341, NODE342 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE343, NODE344 = homo_ops.adjust_levels_and_depth(NODE226, NODE342, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE345, NODE346 = homo_ops.adjust_levels_and_depth(NODE228, NODE344, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE347, NODE348 = homo_ops.adjust_levels_and_depth(NODE230, NODE346, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE349 = homo_ops.homo_rescale(NODE341, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE350 = homo_ops.homo_rescale(NODE343, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE351 = homo_ops.homo_rescale(NODE345, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE352 = homo_ops.homo_rescale(NODE347, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE353 = homo_ops.homo_rescale(NODE348, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE354 = homo_ops.homo_mul_scalar_double(NODE349, np.float64(7.749189507306355e-09), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE355 = homo_ops.homo_mul_scalar_double(NODE350, np.float64(-1.2322659470598354e-07), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE356 = homo_ops.homo_add(NODE354, NODE355, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE357 = homo_ops.homo_mul_scalar_double(NODE351, np.float64(-1.1631474999766912e-09), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE358 = homo_ops.homo_add(NODE356, NODE357, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE359 = homo_ops.homo_mul_scalar_double(NODE352, np.float64(1.7512691686329833e-08), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE360 = homo_ops.homo_add(NODE358, NODE359, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE361 = homo_ops.homo_mul_scalar_double(NODE353, np.float64(1.8316987627039723e-10), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE362 = homo_ops.homo_add(NODE360, NODE361, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE363 = homo_ops.homo_add(NODE362, NODE233, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE364 = homo_ops.homo_add_scalar_double(NODE363, np.float64(3.9678931330873265e-07), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE365 = homo_ops.homo_add(NODE236, NODE320, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE366 = homo_ops.homo_mul(NODE365, NODE340, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE367 = homo_ops.homo_add(NODE366, NODE364, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE368, NODE369 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE370, NODE371 = homo_ops.adjust_levels_and_depth(NODE226, NODE369, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE372, NODE373 = homo_ops.adjust_levels_and_depth(NODE228, NODE371, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE374, NODE375 = homo_ops.adjust_levels_and_depth(NODE230, NODE373, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE376 = homo_ops.homo_rescale(NODE368, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE377 = homo_ops.homo_rescale(NODE370, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE378 = homo_ops.homo_rescale(NODE372, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE379 = homo_ops.homo_rescale(NODE374, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE380 = homo_ops.homo_rescale(NODE375, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE381 = homo_ops.homo_mul_scalar_double(NODE376, np.float64(-0.001985156334269243), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE382 = homo_ops.homo_mul_scalar_double(NODE377, np.float64(0.04890697608593539), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE383 = homo_ops.homo_add(NODE381, NODE382, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE384 = homo_ops.homo_mul_scalar_double(NODE378, np.float64(0.000726959089533452), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE385 = homo_ops.homo_add(NODE383, NODE384, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE386 = homo_ops.homo_mul_scalar_double(NODE379, np.float64(-0.015211841305959308), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE387 = homo_ops.homo_add(NODE385, NODE386, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE388 = homo_ops.homo_mul_scalar_double(NODE380, np.float64(-0.00028591236092496), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE389 = homo_ops.homo_add(NODE387, NODE388, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE390 = homo_ops.homo_add_scalar_double(NODE389, np.float64(-2.0636663259883834), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE391, NODE392 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE393, NODE394 = homo_ops.adjust_levels_and_depth(NODE226, NODE392, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE395, NODE396 = homo_ops.adjust_levels_and_depth(NODE228, NODE394, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE397, NODE398 = homo_ops.adjust_levels_and_depth(NODE230, NODE396, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE399 = homo_ops.homo_rescale(NODE391, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE400 = homo_ops.homo_rescale(NODE393, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE401 = homo_ops.homo_rescale(NODE395, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE402 = homo_ops.homo_rescale(NODE397, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE403 = homo_ops.homo_rescale(NODE398, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE404 = homo_ops.homo_mul_scalar_double(NODE399, np.float64(0.00012477734912342626), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE405 = homo_ops.homo_mul_scalar_double(NODE400, np.float64(-0.002570327090752504), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE406 = homo_ops.homo_add(NODE404, NODE405, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE407 = homo_ops.homo_mul_scalar_double(NODE401, np.float64(-3.149139809684569e-05), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE408 = homo_ops.homo_add(NODE406, NODE407, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE409 = homo_ops.homo_mul_scalar_double(NODE402, np.float64(0.0005863073308399249), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE410 = homo_ops.homo_add(NODE408, NODE409, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE411 = homo_ops.homo_mul_scalar_double(NODE403, np.float64(8.526941207335678e-06), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE412 = homo_ops.homo_add(NODE410, NODE411, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE413 = homo_ops.homo_mul_scalar_int(NODE233, 2, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE414 = homo_ops.homo_add(NODE412, NODE413, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE415 = homo_ops.homo_add_scalar_double(NODE414, np.float64(0.004878114964813271), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE416, NODE417 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE418, NODE419 = homo_ops.adjust_levels_and_depth(NODE226, NODE417, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE420, NODE421 = homo_ops.adjust_levels_and_depth(NODE228, NODE419, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE422, NODE423 = homo_ops.adjust_levels_and_depth(NODE230, NODE421, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE424 = homo_ops.homo_rescale(NODE416, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE425 = homo_ops.homo_rescale(NODE418, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE426 = homo_ops.homo_rescale(NODE420, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE427 = homo_ops.homo_rescale(NODE422, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE428 = homo_ops.homo_rescale(NODE423, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE429 = homo_ops.homo_mul_scalar_double(NODE424, np.float64(0.015624554739935346), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE430 = homo_ops.homo_mul_scalar_double(NODE425, np.float64(-0.4926956557510557), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE431 = homo_ops.homo_add(NODE429, NODE430, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE432 = homo_ops.homo_mul_scalar_double(NODE426, np.float64(-0.010257275690652224), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE433 = homo_ops.homo_add(NODE431, NODE432, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE434 = homo_ops.homo_mul_scalar_double(NODE427, np.float64(0.231850036361479), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE435 = homo_ops.homo_add(NODE433, NODE434, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE436 = homo_ops.homo_mul_scalar_double(NODE428, np.float64(0.006741889972948954), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE437 = homo_ops.homo_add(NODE435, NODE436, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE438 = homo_ops.homo_add(NODE437, NODE233, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE439 = homo_ops.homo_add_scalar_double(NODE438, np.float64(0.3576223526326776), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE440 = homo_ops.homo_add(NODE236, NODE390, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE441 = homo_ops.homo_mul(NODE440, NODE415, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE442 = homo_ops.homo_add(NODE441, NODE439, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE443 = homo_ops.homo_add(NODE239, NODE297, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE444 = homo_ops.homo_mul(NODE443, NODE367, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE445 = homo_ops.homo_add(NODE444, NODE442, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=17, noise=2
    NODE446, NODE447 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE448, NODE449 = homo_ops.adjust_levels_and_depth(NODE226, NODE447, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE450, NODE451 = homo_ops.adjust_levels_and_depth(NODE228, NODE449, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE452, NODE453 = homo_ops.adjust_levels_and_depth(NODE230, NODE451, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE454 = homo_ops.homo_rescale(NODE446, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE455 = homo_ops.homo_rescale(NODE448, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE456 = homo_ops.homo_rescale(NODE450, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE457 = homo_ops.homo_rescale(NODE452, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE458 = homo_ops.homo_rescale(NODE453, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE459 = homo_ops.homo_mul_scalar_double(NODE454, np.float64(-0.011226693751242361), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE460 = homo_ops.homo_mul_scalar_double(NODE455, np.float64(-0.44790338425378234), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE461 = homo_ops.homo_add(NODE459, NODE460, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE462 = homo_ops.homo_mul_scalar_double(NODE456, np.float64(-0.007289276288252454), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE463 = homo_ops.homo_add(NODE461, NODE462, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE464 = homo_ops.homo_mul_scalar_double(NODE457, np.float64(-0.3635089821481836), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE465 = homo_ops.homo_add(NODE463, NODE464, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE466 = homo_ops.homo_mul_scalar_double(NODE458, np.float64(-0.0018484514839724984), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE467 = homo_ops.homo_add(NODE465, NODE466, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE468 = homo_ops.homo_add_scalar_double(NODE467, np.float64(-0.5683810697947218), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE469, NODE470 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE471, NODE472 = homo_ops.adjust_levels_and_depth(NODE226, NODE470, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE473, NODE474 = homo_ops.adjust_levels_and_depth(NODE228, NODE472, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE475, NODE476 = homo_ops.adjust_levels_and_depth(NODE230, NODE474, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE477 = homo_ops.homo_rescale(NODE469, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE478 = homo_ops.homo_rescale(NODE471, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE479 = homo_ops.homo_rescale(NODE473, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE480 = homo_ops.homo_rescale(NODE475, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE481 = homo_ops.homo_rescale(NODE476, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE482 = homo_ops.homo_mul_scalar_double(NODE477, np.float64(0.007108279652990469), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE483 = homo_ops.homo_mul_scalar_double(NODE478, np.float64(0.0019489797083740077), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE484 = homo_ops.homo_add(NODE482, NODE483, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE485 = homo_ops.homo_mul_scalar_double(NODE479, np.float64(-0.0028843595536510837), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE486 = homo_ops.homo_add(NODE484, NODE485, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE487 = homo_ops.homo_mul_scalar_double(NODE480, np.float64(0.05422695740202513), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE488 = homo_ops.homo_add(NODE486, NODE487, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE489 = homo_ops.homo_mul_scalar_double(NODE481, np.float64(-0.025460400506956273), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE490 = homo_ops.homo_add(NODE488, NODE489, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE491 = homo_ops.homo_add_scalar_double(NODE490, np.float64(-0.819156762828709), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE492, NODE493 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE494, NODE495 = homo_ops.adjust_levels_and_depth(NODE226, NODE493, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE496, NODE497 = homo_ops.adjust_levels_and_depth(NODE228, NODE495, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE498, NODE499 = homo_ops.adjust_levels_and_depth(NODE230, NODE497, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE500 = homo_ops.homo_rescale(NODE492, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE501 = homo_ops.homo_rescale(NODE494, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE502 = homo_ops.homo_rescale(NODE496, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE503 = homo_ops.homo_rescale(NODE498, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE504 = homo_ops.homo_rescale(NODE499, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE505 = homo_ops.homo_mul_scalar_double(NODE500, np.float64(0.08762600334510307), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE506 = homo_ops.homo_mul_scalar_double(NODE501, np.float64(0.01248205312237963), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE507 = homo_ops.homo_add(NODE505, NODE506, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE508 = homo_ops.homo_mul_scalar_double(NODE502, np.float64(-0.03159516554230107), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE509 = homo_ops.homo_add(NODE507, NODE508, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE510 = homo_ops.homo_mul_scalar_double(NODE503, np.float64(-0.891126735623574), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE511 = homo_ops.homo_add(NODE509, NODE510, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE512 = homo_ops.homo_mul_scalar_double(NODE504, np.float64(-0.021505227410223895), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE513 = homo_ops.homo_add(NODE511, NODE512, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE514 = homo_ops.homo_mul_scalar_int(NODE233, 4, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE515 = homo_ops.homo_add(NODE513, NODE514, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE516 = homo_ops.homo_add_scalar_double(NODE515, np.float64(0.5084800652890352), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE517, NODE518 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE519, NODE520 = homo_ops.adjust_levels_and_depth(NODE226, NODE518, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE521, NODE522 = homo_ops.adjust_levels_and_depth(NODE228, NODE520, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE523, NODE524 = homo_ops.adjust_levels_and_depth(NODE230, NODE522, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE525 = homo_ops.homo_rescale(NODE517, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE526 = homo_ops.homo_rescale(NODE519, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE527 = homo_ops.homo_rescale(NODE521, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE528 = homo_ops.homo_rescale(NODE523, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE529 = homo_ops.homo_rescale(NODE524, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE530 = homo_ops.homo_mul_scalar_double(NODE525, np.float64(0.100241699512197), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE531 = homo_ops.homo_mul_scalar_double(NODE526, np.float64(0.36533466956500815), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE532 = homo_ops.homo_add(NODE530, NODE531, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE533 = homo_ops.homo_mul_scalar_double(NODE527, np.float64(-0.013581504535905302), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE534 = homo_ops.homo_add(NODE532, NODE533, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE535 = homo_ops.homo_mul_scalar_double(NODE528, np.float64(-0.48032816165746395), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE536 = homo_ops.homo_add(NODE534, NODE535, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE537 = homo_ops.homo_mul_scalar_double(NODE529, np.float64(-0.006172605335918508), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE538 = homo_ops.homo_add(NODE536, NODE537, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE539 = homo_ops.homo_add(NODE538, NODE233, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE540 = homo_ops.homo_add_scalar_double(NODE539, np.float64(0.5188622170523641), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE541 = homo_ops.homo_add(NODE236, NODE491, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE542 = homo_ops.homo_mul(NODE541, NODE516, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE543 = homo_ops.homo_add(NODE542, NODE540, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE544, NODE545 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE546, NODE547 = homo_ops.adjust_levels_and_depth(NODE226, NODE545, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE548, NODE549 = homo_ops.adjust_levels_and_depth(NODE228, NODE547, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE550, NODE551 = homo_ops.adjust_levels_and_depth(NODE230, NODE549, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE552 = homo_ops.homo_rescale(NODE544, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE553 = homo_ops.homo_rescale(NODE546, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE554 = homo_ops.homo_rescale(NODE548, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE555 = homo_ops.homo_rescale(NODE550, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE556 = homo_ops.homo_rescale(NODE551, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE557 = homo_ops.homo_mul_scalar_double(NODE552, np.float64(-0.00043842621075034687), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE558 = homo_ops.homo_mul_scalar_double(NODE553, np.float64(-0.0731687237383307), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE559 = homo_ops.homo_add(NODE557, NODE558, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE560 = homo_ops.homo_mul_scalar_double(NODE554, np.float64(0.023004825216687307), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE561 = homo_ops.homo_add(NODE559, NODE560, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE562 = homo_ops.homo_mul_scalar_double(NODE555, np.float64(-0.1998683402180191), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE563 = homo_ops.homo_add(NODE561, NODE562, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE564 = homo_ops.homo_mul_scalar_double(NODE556, np.float64(-0.046113339586327906), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE565 = homo_ops.homo_add(NODE563, NODE564, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE566 = homo_ops.homo_add_scalar_double(NODE565, np.float64(-1.914253203669351), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE567, NODE568 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE569, NODE570 = homo_ops.adjust_levels_and_depth(NODE226, NODE568, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE571, NODE572 = homo_ops.adjust_levels_and_depth(NODE228, NODE570, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE573, NODE574 = homo_ops.adjust_levels_and_depth(NODE230, NODE572, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE575 = homo_ops.homo_rescale(NODE567, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE576 = homo_ops.homo_rescale(NODE569, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE577 = homo_ops.homo_rescale(NODE571, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE578 = homo_ops.homo_rescale(NODE573, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE579 = homo_ops.homo_rescale(NODE574, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE580 = homo_ops.homo_mul_scalar_double(NODE575, np.float64(0.05464497109317642), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE581 = homo_ops.homo_mul_scalar_double(NODE576, np.float64(0.8598467539571557), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE582 = homo_ops.homo_add(NODE580, NODE581, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE583 = homo_ops.homo_mul_scalar_double(NODE577, np.float64(0.029222571731159365), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE584 = homo_ops.homo_add(NODE582, NODE583, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE585 = homo_ops.homo_mul_scalar_double(NODE578, np.float64(0.949487448098409), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE586 = homo_ops.homo_add(NODE584, NODE585, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE587 = homo_ops.homo_mul_scalar_double(NODE579, np.float64(0.020919545017172893), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE588 = homo_ops.homo_add(NODE586, NODE587, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE589 = homo_ops.homo_mul_scalar_int(NODE233, 2, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE590 = homo_ops.homo_add(NODE588, NODE589, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE591 = homo_ops.homo_add_scalar_double(NODE590, np.float64(0.07590256636785217), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE592, NODE593 = homo_ops.adjust_levels_and_depth(NODE224, NODE232, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE594, NODE595 = homo_ops.adjust_levels_and_depth(NODE226, NODE593, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE596, NODE597 = homo_ops.adjust_levels_and_depth(NODE228, NODE595, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE598, NODE599 = homo_ops.adjust_levels_and_depth(NODE230, NODE597, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE600 = homo_ops.homo_rescale(NODE592, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE601 = homo_ops.homo_rescale(NODE594, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE602 = homo_ops.homo_rescale(NODE596, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE603 = homo_ops.homo_rescale(NODE598, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE604 = homo_ops.homo_rescale(NODE599, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE605 = homo_ops.homo_mul_scalar_double(NODE600, np.float64(0.1682903223534578), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE606 = homo_ops.homo_mul_scalar_double(NODE601, np.float64(2.349462982852268), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE607 = homo_ops.homo_add(NODE605, NODE606, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE608 = homo_ops.homo_mul_scalar_double(NODE602, np.float64(0.05342369715514261), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE609 = homo_ops.homo_add(NODE607, NODE608, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE610 = homo_ops.homo_mul_scalar_double(NODE603, np.float64(2.370388179536648), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE611 = homo_ops.homo_add(NODE609, NODE610, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE612 = homo_ops.homo_mul_scalar_double(NODE604, np.float64(0.0533822343175047), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE613 = homo_ops.homo_add(NODE611, NODE612, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE614 = homo_ops.homo_add(NODE613, NODE233, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE615 = homo_ops.homo_add_scalar_double(NODE614, np.float64(0.6710793101298114), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE616 = homo_ops.homo_add(NODE236, NODE566, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE617 = homo_ops.homo_mul(NODE616, NODE591, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE618 = homo_ops.homo_add(NODE617, NODE615, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE619 = homo_ops.homo_add(NODE239, NODE468, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE620 = homo_ops.homo_mul(NODE619, NODE543, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE621 = homo_ops.homo_add(NODE620, NODE618, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=17, noise=2
    NODE622 = homo_ops.homo_add(NODE242, NODE274, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=18, noise=2
    NODE623 = homo_ops.homo_mul(NODE622, NODE445, cryptoContext) #out: limb=15, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE624 = homo_ops.homo_add(NODE623, NODE621, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=16, noise=2
    NODE625 = homo_ops.homo_sub(NODE624, NODE251, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=15, noise=2
    NODE626 = homo_ops.homo_rescale(NODE625, 1, cryptoContext) #out: limb=14, noise=1, in0: limb=15, noise=2
    NODE627 = homo_ops.homo_square(NODE626, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=1
    NODE628 = homo_ops.homo_add(NODE627, NODE627, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=2, in1: limb=14, noise=2
    NODE629 = homo_ops.homo_add_scalar_double(NODE628, -0.9441845270914478, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=2
    NODE630 = homo_ops.homo_square(NODE629, cryptoContext) #out: limb=13, noise=2, in0: limb=14, noise=2
    NODE631 = homo_ops.homo_add(NODE630, NODE630, cryptoContext) #out: limb=13, noise=2, in0: limb=13, noise=2, in1: limb=13, noise=2
    NODE632 = homo_ops.homo_add_scalar_double(NODE631, -0.891484421198901, cryptoContext) #out: limb=13, noise=2, in0: limb=13, noise=2
    NODE633 = homo_ops.homo_square(NODE632, cryptoContext) #out: limb=12, noise=2, in0: limb=13, noise=2
    NODE634 = homo_ops.homo_add(NODE633, NODE633, cryptoContext) #out: limb=12, noise=2, in0: limb=12, noise=2, in1: limb=12, noise=2
    NODE635 = homo_ops.homo_add_scalar_double(NODE634, -0.7947444732403395, cryptoContext) #out: limb=12, noise=2, in0: limb=12, noise=2
    NODE636 = homo_ops.homo_square(NODE635, cryptoContext) #out: limb=11, noise=2, in0: limb=12, noise=2
    NODE637 = homo_ops.homo_add(NODE636, NODE636, cryptoContext) #out: limb=11, noise=2, in0: limb=11, noise=2, in1: limb=11, noise=2
    NODE638 = homo_ops.homo_add_scalar_double(NODE637, -0.6316187777460647, cryptoContext) #out: limb=11, noise=2, in0: limb=11, noise=2
    NODE639 = homo_ops.homo_square(NODE638, cryptoContext) #out: limb=10, noise=2, in0: limb=11, noise=2
    NODE640 = homo_ops.homo_add(NODE639, NODE639, cryptoContext) #out: limb=10, noise=2, in0: limb=10, noise=2, in1: limb=10, noise=2
    NODE641 = homo_ops.homo_add_scalar_double(NODE640, -0.3989422804014327, cryptoContext) #out: limb=10, noise=2, in0: limb=10, noise=2
    NODE642 = homo_ops.homo_square(NODE641, cryptoContext) #out: limb=9, noise=2, in0: limb=10, noise=2
    NODE643 = homo_ops.homo_add(NODE642, NODE642, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2, in1: limb=9, noise=2
    NODE644 = homo_ops.homo_add_scalar_double(NODE643, -0.15915494309189535, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2
    NODE645 = homo_ops.homo_mul_scalar_int(NODE644, 2, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2
    NODE646 = homo_ops.homo_rescale(NODE645, 1, cryptoContext) #out: limb=8, noise=1, in0: limb=9, noise=2
    NODE647 = homo_ops.extract_cv(NODE646, 1) #out: limb=8, noise=1, in0: limb=8, noise=1
    NODE648 = hybrid_keyswitch.modup_to_ext(NODE647, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1
    NODE649 = homo_ops.eval_fast_rotate(NODE648, NODE646, 8189, True, False, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1in1: limb=8, noise=1
    NODE650 = homo_ops.eval_fast_rotate(NODE648, NODE646, 8190, True, False, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1in1: limb=8, noise=1
    NODE651 = homo_ops.eval_fast_rotate(NODE648, NODE646, 8191, True, False, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1in1: limb=8, noise=1
    NODE652 = hybrid_keyswitch.key_switch_P_ext(NODE646, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1
    NODE653 = homo_ops.homo_mul_pt(NODE649, NODE29, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE654 = homo_ops.homo_mul_pt(NODE650, NODE30, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE655 = homo_ops.homo_add(NODE653, NODE654, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE656 = homo_ops.homo_mul_pt(NODE651, NODE31, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE657 = homo_ops.homo_add(NODE655, NODE656, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE658 = homo_ops.homo_mul_pt(NODE652, NODE32, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE659 = homo_ops.homo_add(NODE657, NODE658, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE660 = homo_ops.extract_cv(NODE659, 0) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE661 = hybrid_keyswitch.moddown_from_ext(NODE660, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE662 = homo_ops.extract_cv(NODE659, 1, append_zeros = True) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE663 = homo_ops.homo_mul_pt(NODE649, NODE33, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE664 = homo_ops.homo_mul_pt(NODE650, NODE34, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE665 = homo_ops.homo_add(NODE663, NODE664, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE666 = homo_ops.homo_mul_pt(NODE651, NODE35, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE667 = homo_ops.homo_add(NODE665, NODE666, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE668 = hybrid_keyswitch.moddown_from_ext(NODE667, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE669 = homo_ops.extract_cv(NODE668, 0) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE670 = homo_ops.extract_cv(NODE668, 1) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE671 = homo_ops._cipher_automorphism(NODE669, 4, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE672 = homo_ops.homo_add(NODE661, NODE671, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE673 = hybrid_keyswitch.modup_to_ext(NODE670, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE674 = homo_ops.eval_fast_rotate(NODE673, None, 4, False, None, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE675 = homo_ops.homo_add(NODE662, NODE674, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE676 = hybrid_keyswitch.moddown_from_ext(NODE675, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE677 = homo_ops.extract_cv(NODE672, 0, append_zeros = True) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE678 = homo_ops.homo_add(NODE676, NODE677, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE679 = homo_ops.homo_rescale(NODE678, 1, cryptoContext) #out: limb=7, noise=1, in0: limb=8, noise=2
    NODE680 = homo_ops.extract_cv(NODE679, 1) #out: limb=7, noise=1, in0: limb=7, noise=1
    NODE681 = hybrid_keyswitch.modup_to_ext(NODE680, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1
    NODE682 = homo_ops.eval_fast_rotate(NODE681, NODE679, 8180, True, False, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1in1: limb=7, noise=1
    NODE683 = homo_ops.eval_fast_rotate(NODE681, NODE679, 8184, True, False, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1in1: limb=7, noise=1
    NODE684 = homo_ops.eval_fast_rotate(NODE681, NODE679, 8188, True, False, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1in1: limb=7, noise=1
    NODE685 = hybrid_keyswitch.key_switch_P_ext(NODE679, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1
    NODE686 = homo_ops.homo_mul_pt(NODE682, NODE36, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE687 = homo_ops.homo_mul_pt(NODE683, NODE37, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE688 = homo_ops.homo_add(NODE686, NODE687, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE689 = homo_ops.homo_mul_pt(NODE684, NODE38, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE690 = homo_ops.homo_add(NODE688, NODE689, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE691 = homo_ops.homo_mul_pt(NODE685, NODE39, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE692 = homo_ops.homo_add(NODE690, NODE691, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE693 = homo_ops.extract_cv(NODE692, 0) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE694 = hybrid_keyswitch.moddown_from_ext(NODE693, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE695 = homo_ops.extract_cv(NODE692, 1, append_zeros = True) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE696 = homo_ops.homo_mul_pt(NODE682, NODE40, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE697 = homo_ops.homo_mul_pt(NODE683, NODE41, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE698 = homo_ops.homo_add(NODE696, NODE697, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE699 = homo_ops.homo_mul_pt(NODE684, NODE42, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE700 = homo_ops.homo_add(NODE698, NODE699, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE701 = hybrid_keyswitch.moddown_from_ext(NODE700, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE702 = homo_ops.extract_cv(NODE701, 0) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE703 = homo_ops.extract_cv(NODE701, 1) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE704 = homo_ops._cipher_automorphism(NODE702, 16, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE705 = homo_ops.homo_add(NODE694, NODE704, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE706 = hybrid_keyswitch.modup_to_ext(NODE703, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE707 = homo_ops.eval_fast_rotate(NODE706, None, 16, False, None, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE708 = homo_ops.homo_add(NODE695, NODE707, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE709 = hybrid_keyswitch.moddown_from_ext(NODE708, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE710 = homo_ops.extract_cv(NODE705, 0, append_zeros = True) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE711 = homo_ops.homo_add(NODE709, NODE710, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE712 = homo_ops.homo_rescale(NODE711, 1, cryptoContext) #out: limb=6, noise=1, in0: limb=7, noise=2
    NODE713 = homo_ops.extract_cv(NODE712, 1) #out: limb=6, noise=1, in0: limb=6, noise=1
    NODE714 = hybrid_keyswitch.modup_to_ext(NODE713, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1
    NODE715 = homo_ops.eval_fast_rotate(NODE714, NODE712, 8144, True, False, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1in1: limb=6, noise=1
    NODE716 = homo_ops.eval_fast_rotate(NODE714, NODE712, 8160, True, False, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1in1: limb=6, noise=1
    NODE717 = homo_ops.eval_fast_rotate(NODE714, NODE712, 8176, True, False, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1in1: limb=6, noise=1
    NODE718 = hybrid_keyswitch.key_switch_P_ext(NODE712, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1
    NODE719 = homo_ops.homo_mul_pt(NODE715, NODE43, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE720 = homo_ops.homo_mul_pt(NODE716, NODE44, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE721 = homo_ops.homo_add(NODE719, NODE720, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE722 = homo_ops.homo_mul_pt(NODE717, NODE45, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE723 = homo_ops.homo_add(NODE721, NODE722, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE724 = homo_ops.homo_mul_pt(NODE718, NODE46, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE725 = homo_ops.homo_add(NODE723, NODE724, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE726 = homo_ops.extract_cv(NODE725, 0) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE727 = hybrid_keyswitch.moddown_from_ext(NODE726, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE728 = homo_ops.extract_cv(NODE725, 1, append_zeros = True) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE729 = homo_ops.homo_mul_pt(NODE715, NODE47, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE730 = homo_ops.homo_mul_pt(NODE716, NODE48, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE731 = homo_ops.homo_add(NODE729, NODE730, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE732 = homo_ops.homo_mul_pt(NODE717, NODE49, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE733 = homo_ops.homo_add(NODE731, NODE732, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE734 = hybrid_keyswitch.moddown_from_ext(NODE733, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE735 = homo_ops.extract_cv(NODE734, 0) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE736 = homo_ops.extract_cv(NODE734, 1) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE737 = homo_ops._cipher_automorphism(NODE735, 64, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE738 = homo_ops.homo_add(NODE727, NODE737, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE739 = hybrid_keyswitch.modup_to_ext(NODE736, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE740 = homo_ops.eval_fast_rotate(NODE739, None, 64, False, None, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE741 = homo_ops.homo_add(NODE728, NODE740, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE742 = hybrid_keyswitch.moddown_from_ext(NODE741, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE743 = homo_ops.extract_cv(NODE738, 0, append_zeros = True) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE744 = homo_ops.homo_add(NODE742, NODE743, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE745 = homo_ops.homo_rescale(NODE744, 1, cryptoContext) #out: limb=5, noise=1, in0: limb=6, noise=2
    NODE746 = homo_ops.extract_cv(NODE745, 1) #out: limb=5, noise=1, in0: limb=5, noise=1
    NODE747 = hybrid_keyswitch.modup_to_ext(NODE746, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1
    NODE748 = homo_ops.eval_fast_rotate(NODE747, NODE745, 8000, True, False, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1in1: limb=5, noise=1
    NODE749 = homo_ops.eval_fast_rotate(NODE747, NODE745, 8064, True, False, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1in1: limb=5, noise=1
    NODE750 = homo_ops.eval_fast_rotate(NODE747, NODE745, 8128, True, False, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1in1: limb=5, noise=1
    NODE751 = hybrid_keyswitch.key_switch_P_ext(NODE745, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1
    NODE752 = homo_ops.homo_mul_pt(NODE748, NODE50, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE753 = homo_ops.homo_mul_pt(NODE749, NODE51, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE754 = homo_ops.homo_add(NODE752, NODE753, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE755 = homo_ops.homo_mul_pt(NODE750, NODE52, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE756 = homo_ops.homo_add(NODE754, NODE755, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE757 = homo_ops.homo_mul_pt(NODE751, NODE53, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE758 = homo_ops.homo_add(NODE756, NODE757, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE759 = homo_ops.extract_cv(NODE758, 0) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE760 = hybrid_keyswitch.moddown_from_ext(NODE759, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE761 = homo_ops.extract_cv(NODE758, 1, append_zeros = True) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE762 = homo_ops.homo_mul_pt(NODE748, NODE54, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE763 = homo_ops.homo_mul_pt(NODE749, NODE55, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE764 = homo_ops.homo_add(NODE762, NODE763, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE765 = homo_ops.homo_mul_pt(NODE750, NODE56, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE766 = homo_ops.homo_add(NODE764, NODE765, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE767 = hybrid_keyswitch.moddown_from_ext(NODE766, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE768 = homo_ops.extract_cv(NODE767, 0) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE769 = homo_ops.extract_cv(NODE767, 1) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE770 = homo_ops._cipher_automorphism(NODE768, 256, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE771 = homo_ops.homo_add(NODE760, NODE770, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE772 = hybrid_keyswitch.modup_to_ext(NODE769, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE773 = homo_ops.eval_fast_rotate(NODE772, None, 256, False, None, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE774 = homo_ops.homo_add(NODE761, NODE773, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE775 = hybrid_keyswitch.moddown_from_ext(NODE774, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE776 = homo_ops.extract_cv(NODE771, 0, append_zeros = True) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE777 = homo_ops.homo_add(NODE775, NODE776, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE778 = homo_ops.homo_rotate(NODE777, 256, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE779 = homo_ops.homo_add(NODE777, NODE778, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE780 = homo_ops.homo_mul_scalar_int(NODE779, 512, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2


    return NODE780

def homo_bootstrap(cipher, L0, logBsSlots, cryptoContext):

    if cryptoContext.autoLoadAndSetConfig == True:
        cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]

    result = eval_bootstrap(cipher, L0, logBsSlots, cryptoContext)

    if (
        cryptoContext.rescaleTech == "FIXEDMANUAL"
    ):  # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
        result = homo_ops.homo_rescale(result, result.noise_deg - 1, cryptoContext)

    return result
