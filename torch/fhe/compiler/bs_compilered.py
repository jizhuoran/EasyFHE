from ..bs_context import *
from .. import functional as F
from .. import homo_ops
from .. import hybrid_keyswitch
from .. import utils
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
    NODE1 = cryptoContext.BsContext.m_U0hatTPreFFT[0][0] # limb=22, noise=1
    NODE2 = cryptoContext.BsContext.m_U0hatTPreFFT[0][1] # limb=22, noise=1
    NODE3 = cryptoContext.BsContext.m_U0hatTPreFFT[0][2] # limb=22, noise=1
    NODE4 = cryptoContext.BsContext.m_U0hatTPreFFT[0][3] # limb=22, noise=1
    NODE5 = cryptoContext.BsContext.m_U0hatTPreFFT[0][4] # limb=22, noise=1
    NODE6 = cryptoContext.BsContext.m_U0hatTPreFFT[0][5] # limb=22, noise=1
    NODE7 = cryptoContext.BsContext.m_U0hatTPreFFT[0][6] # limb=22, noise=1
    NODE8 = cryptoContext.BsContext.m_U0hatTPreFFT[1][0] # limb=23, noise=1
    NODE9 = cryptoContext.BsContext.m_U0hatTPreFFT[1][1] # limb=23, noise=1
    NODE10 = cryptoContext.BsContext.m_U0hatTPreFFT[1][2] # limb=23, noise=1
    NODE11 = cryptoContext.BsContext.m_U0hatTPreFFT[1][3] # limb=23, noise=1
    NODE12 = cryptoContext.BsContext.m_U0hatTPreFFT[1][4] # limb=23, noise=1
    NODE13 = cryptoContext.BsContext.m_U0hatTPreFFT[1][5] # limb=23, noise=1
    NODE14 = cryptoContext.BsContext.m_U0hatTPreFFT[1][6] # limb=23, noise=1
    NODE15 = cryptoContext.BsContext.m_U0hatTPreFFT[2][0] # limb=24, noise=1
    NODE16 = cryptoContext.BsContext.m_U0hatTPreFFT[2][1] # limb=24, noise=1
    NODE17 = cryptoContext.BsContext.m_U0hatTPreFFT[2][2] # limb=24, noise=1
    NODE18 = cryptoContext.BsContext.m_U0hatTPreFFT[2][3] # limb=24, noise=1
    NODE19 = cryptoContext.BsContext.m_U0hatTPreFFT[2][4] # limb=24, noise=1
    NODE20 = cryptoContext.BsContext.m_U0hatTPreFFT[2][5] # limb=24, noise=1
    NODE21 = cryptoContext.BsContext.m_U0hatTPreFFT[2][6] # limb=24, noise=1
    NODE22 = cryptoContext.BsContext.m_U0hatTPreFFT[3][0] # limb=25, noise=1
    NODE23 = cryptoContext.BsContext.m_U0hatTPreFFT[3][1] # limb=25, noise=1
    NODE24 = cryptoContext.BsContext.m_U0hatTPreFFT[3][2] # limb=25, noise=1
    NODE25 = cryptoContext.BsContext.m_U0hatTPreFFT[3][3] # limb=25, noise=1
    NODE26 = cryptoContext.BsContext.m_U0hatTPreFFT[3][4] # limb=25, noise=1
    NODE27 = cryptoContext.BsContext.m_U0hatTPreFFT[3][5] # limb=25, noise=1
    NODE28 = cryptoContext.BsContext.m_U0hatTPreFFT[3][6] # limb=25, noise=1
    NODE29 = cryptoContext.BsContext.m_U0PreFFT[0][0] # limb=8, noise=1
    NODE30 = cryptoContext.BsContext.m_U0PreFFT[0][1] # limb=8, noise=1
    NODE31 = cryptoContext.BsContext.m_U0PreFFT[0][2] # limb=8, noise=1
    NODE32 = cryptoContext.BsContext.m_U0PreFFT[0][3] # limb=8, noise=1
    NODE33 = cryptoContext.BsContext.m_U0PreFFT[0][4] # limb=8, noise=1
    NODE34 = cryptoContext.BsContext.m_U0PreFFT[0][5] # limb=8, noise=1
    NODE35 = cryptoContext.BsContext.m_U0PreFFT[0][6] # limb=8, noise=1
    NODE36 = cryptoContext.BsContext.m_U0PreFFT[1][0] # limb=7, noise=1
    NODE37 = cryptoContext.BsContext.m_U0PreFFT[1][1] # limb=7, noise=1
    NODE38 = cryptoContext.BsContext.m_U0PreFFT[1][2] # limb=7, noise=1
    NODE39 = cryptoContext.BsContext.m_U0PreFFT[1][3] # limb=7, noise=1
    NODE40 = cryptoContext.BsContext.m_U0PreFFT[1][4] # limb=7, noise=1
    NODE41 = cryptoContext.BsContext.m_U0PreFFT[1][5] # limb=7, noise=1
    NODE42 = cryptoContext.BsContext.m_U0PreFFT[1][6] # limb=7, noise=1
    NODE43 = cryptoContext.BsContext.m_U0PreFFT[2][0] # limb=6, noise=1
    NODE44 = cryptoContext.BsContext.m_U0PreFFT[2][1] # limb=6, noise=1
    NODE45 = cryptoContext.BsContext.m_U0PreFFT[2][2] # limb=6, noise=1
    NODE46 = cryptoContext.BsContext.m_U0PreFFT[2][3] # limb=6, noise=1
    NODE47 = cryptoContext.BsContext.m_U0PreFFT[2][4] # limb=6, noise=1
    NODE48 = cryptoContext.BsContext.m_U0PreFFT[2][5] # limb=6, noise=1
    NODE49 = cryptoContext.BsContext.m_U0PreFFT[2][6] # limb=6, noise=1
    NODE50 = cryptoContext.BsContext.m_U0PreFFT[3][0] # limb=5, noise=1
    NODE51 = cryptoContext.BsContext.m_U0PreFFT[3][1] # limb=5, noise=1
    NODE52 = cryptoContext.BsContext.m_U0PreFFT[3][2] # limb=5, noise=1
    NODE53 = cryptoContext.BsContext.m_U0PreFFT[3][3] # limb=5, noise=1
    NODE54 = cryptoContext.BsContext.m_U0PreFFT[3][4] # limb=5, noise=1
    NODE55 = cryptoContext.BsContext.m_U0PreFFT[3][5] # limb=5, noise=1
    NODE56 = cryptoContext.BsContext.m_U0PreFFT[3][6] # limb=5, noise=1
    NODE57 = IN_NODE
    NODE58 = homo_ops.homo_rescale_internal(NODE57, 0, cryptoContext) #out: limb=2, noise=1, in0: limb=2, noise=1
    NODE60 = homo_ops.homo_mul_scalar_double(NODE58, 0.015625000014665602, cryptoContext) #out: limb=2, noise=2, in0: limb=2, noise=1
    NODE61 = homo_ops.homo_rescale_internal(NODE60, 1, cryptoContext) #out: limb=1, noise=1, in0: limb=2, noise=2
    NODE61.scaling_factor = 4503599627763713.0
    NODE62 = mod_raise(NODE61, 26, cryptoContext) #out: limb=26, noise=1, in0: limb=1, noise=1
    NODE63 = homo_ops.homo_mul_scalar_double(NODE62, 7.450580596923828e-09, cryptoContext) #out: limb=26, noise=2, in0: limb=26, noise=1
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
    NODE74 = homo_ops.homo_rescale_internal(NODE73, 1, cryptoContext) #out: limb=25, noise=1, in0: limb=26, noise=2
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
    NODE107 = homo_ops.homo_rescale_internal(NODE106, 1, cryptoContext) #out: limb=24, noise=1, in0: limb=25, noise=2
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
    NODE140 = homo_ops.homo_rescale_internal(NODE139, 1, cryptoContext) #out: limb=23, noise=1, in0: limb=24, noise=2
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
    NODE173 = homo_ops.homo_rescale_internal(NODE172, 1, cryptoContext) #out: limb=22, noise=1, in0: limb=23, noise=2
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
    NODE208 = homo_ops.homo_rescale_internal(NODE207, 1, cryptoContext) #out: limb=21, noise=1, in0: limb=22, noise=2
    NODE209 = homo_ops.homo_mul(NODE208, NODE208, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=1, in1: limb=21, noise=1
    NODE210 = homo_ops.homo_add(NODE209, NODE209, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=2, in1: limb=21, noise=2
    NODE211 = homo_ops.homo_rescale(NODE210, 1, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=2
    NODE213 = homo_ops.homo_add_scalar_double(NODE211, -1.0, cryptoContext) #out: limb=21, noise=2, in0: limb=21, noise=2
    NODE214 = homo_ops.homo_mul(NODE208, NODE213, cryptoContext) #out: limb=20, noise=2, in0: limb=21, noise=1, in1: limb=21, noise=2
    NODE215 = homo_ops.homo_add(NODE214, NODE214, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2, in1: limb=20, noise=2
    NODE216 = homo_ops.homo_rescale(NODE215, 1, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2
    NODE218 = homo_ops.homo_sub(NODE216, NODE208, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2, in1: limb=21, noise=1
    NODE219 = homo_ops.homo_mul(NODE213, NODE213, cryptoContext) #out: limb=20, noise=2, in0: limb=21, noise=2, in1: limb=21, noise=2
    NODE220 = homo_ops.homo_add(NODE219, NODE219, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2, in1: limb=20, noise=2
    NODE221 = homo_ops.homo_rescale(NODE220, 1, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2
    NODE223 = homo_ops.homo_add_scalar_double(NODE221, -1.0, cryptoContext) #out: limb=20, noise=2, in0: limb=20, noise=2
    NODE224 = homo_ops.homo_mul(NODE213, NODE218, cryptoContext) #out: limb=19, noise=2, in0: limb=21, noise=2, in1: limb=20, noise=2
    NODE225 = homo_ops.homo_add(NODE224, NODE224, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE226 = homo_ops.homo_rescale(NODE225, 1, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE228 = homo_ops.homo_sub(NODE226, NODE208, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=21, noise=1
    NODE229 = homo_ops.homo_mul(NODE218, NODE218, cryptoContext) #out: limb=19, noise=2, in0: limb=20, noise=2, in1: limb=20, noise=2
    NODE230 = homo_ops.homo_add(NODE229, NODE229, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE231 = homo_ops.homo_rescale(NODE230, 1, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE233 = homo_ops.homo_add_scalar_double(NODE231, -1.0, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE234, NODE235 = homo_ops.adjust_levels_and_depth(NODE208, NODE233, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=21, noise=1, in1: limb=19, noise=2
    NODE236, NODE237 = homo_ops.adjust_levels_and_depth(NODE213, NODE235, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=21, noise=2, in1: limb=19, noise=2
    NODE238, NODE239 = homo_ops.adjust_levels_and_depth(NODE218, NODE237, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=20, noise=2, in1: limb=19, noise=2
    NODE240, NODE241 = homo_ops.adjust_levels_and_depth(NODE223, NODE239, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=20, noise=2, in1: limb=19, noise=2
    NODE242, NODE243 = homo_ops.adjust_levels_and_depth(NODE228, NODE241, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE244 = homo_ops.homo_square(NODE243, cryptoContext) #out: limb=18, noise=2, in0: limb=19, noise=2
    NODE245 = homo_ops.homo_add(NODE244, NODE244, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE246 = homo_ops.homo_rescale(NODE245, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE248 = homo_ops.homo_add_scalar_double(NODE246, -1.0, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE249 = homo_ops.homo_square(NODE248, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2
    NODE250 = homo_ops.homo_add(NODE249, NODE249, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE251 = homo_ops.homo_rescale(NODE250, 1, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE253 = homo_ops.homo_add_scalar_double(NODE251, -1.0, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE254 = homo_ops.homo_square(NODE253, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2
    NODE255 = homo_ops.homo_add(NODE254, NODE254, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE256 = homo_ops.homo_rescale(NODE255, 1, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2
    NODE258 = homo_ops.homo_add_scalar_double(NODE256, -1.0, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2
    NODE259 = homo_ops.homo_mul(NODE243, NODE248, cryptoContext) #out: limb=17, noise=2, in0: limb=19, noise=2, in1: limb=18, noise=2
    NODE260 = homo_ops.homo_add(NODE259, NODE259, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE261 = homo_ops.homo_rescale(NODE260, 1, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE263 = homo_ops.homo_sub(NODE261, NODE243, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=19, noise=2
    NODE264 = homo_ops.homo_mul(NODE263, NODE253, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE265 = homo_ops.homo_add(NODE264, NODE264, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE266 = homo_ops.homo_rescale(NODE265, 1, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2
    NODE268 = homo_ops.homo_sub(NODE266, NODE243, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=19, noise=2
    NODE269 = homo_ops.homo_mul(NODE268, NODE258, cryptoContext) #out: limb=15, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE270 = homo_ops.homo_add(NODE269, NODE269, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=15, noise=2
    NODE271 = homo_ops.homo_rescale(NODE270, 1, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2
    NODE273 = homo_ops.homo_sub(NODE271, NODE243, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=19, noise=2
    NODE274, NODE275 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE276, NODE277 = homo_ops.adjust_levels_and_depth(NODE236, NODE275, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE278, NODE279 = homo_ops.adjust_levels_and_depth(NODE238, NODE277, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE280, NODE281 = homo_ops.adjust_levels_and_depth(NODE240, NODE279, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE282 = homo_ops.homo_rescale_internal(NODE274, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE283 = homo_ops.homo_rescale_internal(NODE276, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE284 = homo_ops.homo_rescale_internal(NODE278, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE285 = homo_ops.homo_rescale_internal(NODE280, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE286 = homo_ops.homo_rescale_internal(NODE281, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE287 = homo_ops.homo_mul_scalar_double(NODE282, np.float64(-0.0005862476626482575), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE288 = homo_ops.homo_mul_scalar_double(NODE283, np.float64(-0.05094407670735883), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE289 = homo_ops.homo_add(NODE287, NODE288, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE290 = homo_ops.homo_mul_scalar_double(NODE284, np.float64(0.010324286361991016), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE291 = homo_ops.homo_add(NODE289, NODE290, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE292 = homo_ops.homo_mul_scalar_double(NODE285, np.float64(-0.06820640296455721), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE293 = homo_ops.homo_add(NODE291, NODE292, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE294 = homo_ops.homo_mul_scalar_double(NODE286, np.float64(-0.01629177159536448), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE295 = homo_ops.homo_add(NODE293, NODE294, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE296 = homo_ops.homo_rescale(NODE295, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE298 = homo_ops.homo_add_scalar_double(NODE296, np.float64(-0.3617312474102197), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE299, NODE300 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE301, NODE302 = homo_ops.adjust_levels_and_depth(NODE236, NODE300, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE303, NODE304 = homo_ops.adjust_levels_and_depth(NODE238, NODE302, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE305, NODE306 = homo_ops.adjust_levels_and_depth(NODE240, NODE304, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE307 = homo_ops.homo_rescale_internal(NODE299, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE308 = homo_ops.homo_rescale_internal(NODE301, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE309 = homo_ops.homo_rescale_internal(NODE303, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE310 = homo_ops.homo_rescale_internal(NODE305, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE311 = homo_ops.homo_rescale_internal(NODE306, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE312 = homo_ops.homo_mul_scalar_double(NODE307, np.float64(-3.646495964794955e-07), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE313 = homo_ops.homo_mul_scalar_double(NODE308, np.float64(6.523242811745607e-06), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE314 = homo_ops.homo_add(NODE312, NODE313, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE315 = homo_ops.homo_mul_scalar_double(NODE309, np.float64(6.924798172957744e-08), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE316 = homo_ops.homo_add(NODE314, NODE315, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE317 = homo_ops.homo_mul_scalar_double(NODE310, np.float64(-1.153465700731715e-06), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE318 = homo_ops.homo_add(NODE316, NODE317, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE319 = homo_ops.homo_mul_scalar_double(NODE311, np.float64(-1.3952337882417398e-08), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE320 = homo_ops.homo_add(NODE318, NODE319, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE321 = homo_ops.homo_rescale(NODE320, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE323 = homo_ops.homo_add_scalar_double(NODE321, np.float64(-0.25001653312166516), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE324, NODE325 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE326, NODE327 = homo_ops.adjust_levels_and_depth(NODE236, NODE325, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE328, NODE329 = homo_ops.adjust_levels_and_depth(NODE238, NODE327, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE330, NODE331 = homo_ops.adjust_levels_and_depth(NODE240, NODE329, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE332 = homo_ops.homo_rescale_internal(NODE324, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE333 = homo_ops.homo_rescale_internal(NODE326, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE334 = homo_ops.homo_rescale_internal(NODE328, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE335 = homo_ops.homo_rescale_internal(NODE330, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE336 = homo_ops.homo_rescale_internal(NODE331, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE337 = homo_ops.homo_mul_scalar_double(NODE332, np.float64(-5.287332369682873e-12), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE338 = homo_ops.homo_mul_scalar_double(NODE333, np.float64(7.59317852040558e-11), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE339 = homo_ops.homo_add(NODE337, NODE338, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE340 = homo_ops.homo_mul_scalar_double(NODE334, np.float64(6.476916381503101e-13), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE341 = homo_ops.homo_add(NODE339, NODE340, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE342 = homo_ops.homo_mul_scalar_double(NODE335, np.float64(-8.902002730465436e-12), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE343 = homo_ops.homo_add(NODE341, NODE342, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE344 = homo_ops.homo_mul_scalar_double(NODE336, np.float64(-8.256706803909277e-14), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE345 = homo_ops.homo_add(NODE343, NODE344, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE346 = homo_ops.homo_rescale(NODE345, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE348 = homo_ops.homo_add_scalar_double(NODE346, np.float64(-0.6250000003005246), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE349, NODE350 = homo_ops.adjust_levels_and_depth(NODE234, NODE240, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE351, NODE352 = homo_ops.adjust_levels_and_depth(NODE236, NODE350, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE353, NODE354 = homo_ops.adjust_levels_and_depth(NODE238, NODE352, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE355 = homo_ops.homo_rescale_internal(NODE349, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE356 = homo_ops.homo_rescale_internal(NODE351, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE357 = homo_ops.homo_rescale_internal(NODE353, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE358 = homo_ops.homo_rescale_internal(NODE354, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE359 = homo_ops.homo_mul_scalar_double(NODE355, np.float64(6.536095011040416e-14), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE360 = homo_ops.homo_mul_scalar_double(NODE356, np.float64(-8.489388967084298e-13), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE361 = homo_ops.homo_add(NODE359, NODE360, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE362 = homo_ops.homo_mul_scalar_double(NODE357, np.float64(-7.167799437636123e-15), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE363 = homo_ops.homo_add(NODE361, NODE362, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE364 = homo_ops.homo_mul_scalar_double(NODE358, np.float64(9.137260236825108e-14), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE365 = homo_ops.homo_add(NODE363, NODE364, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE366 = homo_ops.homo_rescale(NODE365, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE368 = homo_ops.homo_mul_scalar_int(NODE243, 8, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE369 = homo_ops.homo_add(NODE366, NODE368, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE370 = homo_ops.homo_add_scalar_double(NODE369, np.float64(4.022969223666898e-12), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE371, NODE372 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE373, NODE374 = homo_ops.adjust_levels_and_depth(NODE236, NODE372, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE375, NODE376 = homo_ops.adjust_levels_and_depth(NODE238, NODE374, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE377, NODE378 = homo_ops.adjust_levels_and_depth(NODE240, NODE376, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE379 = homo_ops.homo_rescale_internal(NODE371, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE380 = homo_ops.homo_rescale_internal(NODE373, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE381 = homo_ops.homo_rescale_internal(NODE375, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE382 = homo_ops.homo_rescale_internal(NODE377, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE383 = homo_ops.homo_rescale_internal(NODE378, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE384 = homo_ops.homo_mul_scalar_double(NODE379, np.float64(7.749189507306355e-09), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE385 = homo_ops.homo_mul_scalar_double(NODE380, np.float64(-1.2322659470598354e-07), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE386 = homo_ops.homo_add(NODE384, NODE385, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE387 = homo_ops.homo_mul_scalar_double(NODE381, np.float64(-1.1631474999766912e-09), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE388 = homo_ops.homo_add(NODE386, NODE387, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE389 = homo_ops.homo_mul_scalar_double(NODE382, np.float64(1.7512691686329833e-08), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE390 = homo_ops.homo_add(NODE388, NODE389, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE391 = homo_ops.homo_mul_scalar_double(NODE383, np.float64(1.8316987627039723e-10), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE392 = homo_ops.homo_add(NODE390, NODE391, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE393 = homo_ops.homo_rescale(NODE392, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE395 = homo_ops.homo_add(NODE393, NODE243, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE396 = homo_ops.homo_add_scalar_double(NODE395, np.float64(3.9678931330873265e-07), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE397 = homo_ops.homo_add(NODE248, NODE348, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE398 = homo_ops.homo_mul(NODE397, NODE370, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE399 = homo_ops.homo_rescale(NODE398, 1, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE401 = homo_ops.homo_add(NODE399, NODE396, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE402, NODE403 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE404, NODE405 = homo_ops.adjust_levels_and_depth(NODE236, NODE403, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE406, NODE407 = homo_ops.adjust_levels_and_depth(NODE238, NODE405, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE408, NODE409 = homo_ops.adjust_levels_and_depth(NODE240, NODE407, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE410 = homo_ops.homo_rescale_internal(NODE402, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE411 = homo_ops.homo_rescale_internal(NODE404, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE412 = homo_ops.homo_rescale_internal(NODE406, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE413 = homo_ops.homo_rescale_internal(NODE408, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE414 = homo_ops.homo_rescale_internal(NODE409, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE415 = homo_ops.homo_mul_scalar_double(NODE410, np.float64(-0.001985156334269243), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE416 = homo_ops.homo_mul_scalar_double(NODE411, np.float64(0.04890697608593539), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE417 = homo_ops.homo_add(NODE415, NODE416, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE418 = homo_ops.homo_mul_scalar_double(NODE412, np.float64(0.000726959089533452), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE419 = homo_ops.homo_add(NODE417, NODE418, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE420 = homo_ops.homo_mul_scalar_double(NODE413, np.float64(-0.015211841305959308), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE421 = homo_ops.homo_add(NODE419, NODE420, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE422 = homo_ops.homo_mul_scalar_double(NODE414, np.float64(-0.00028591236092496), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE423 = homo_ops.homo_add(NODE421, NODE422, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE424 = homo_ops.homo_rescale(NODE423, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE426 = homo_ops.homo_add_scalar_double(NODE424, np.float64(-2.0636663259883834), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE427, NODE428 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE429, NODE430 = homo_ops.adjust_levels_and_depth(NODE236, NODE428, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE431, NODE432 = homo_ops.adjust_levels_and_depth(NODE238, NODE430, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE433, NODE434 = homo_ops.adjust_levels_and_depth(NODE240, NODE432, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE435 = homo_ops.homo_rescale_internal(NODE427, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE436 = homo_ops.homo_rescale_internal(NODE429, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE437 = homo_ops.homo_rescale_internal(NODE431, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE438 = homo_ops.homo_rescale_internal(NODE433, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE439 = homo_ops.homo_rescale_internal(NODE434, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE440 = homo_ops.homo_mul_scalar_double(NODE435, np.float64(0.00012477734912342626), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE441 = homo_ops.homo_mul_scalar_double(NODE436, np.float64(-0.002570327090752504), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE442 = homo_ops.homo_add(NODE440, NODE441, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE443 = homo_ops.homo_mul_scalar_double(NODE437, np.float64(-3.149139809684569e-05), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE444 = homo_ops.homo_add(NODE442, NODE443, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE445 = homo_ops.homo_mul_scalar_double(NODE438, np.float64(0.0005863073308399249), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE446 = homo_ops.homo_add(NODE444, NODE445, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE447 = homo_ops.homo_mul_scalar_double(NODE439, np.float64(8.526941207335678e-06), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE448 = homo_ops.homo_add(NODE446, NODE447, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE449 = homo_ops.homo_rescale(NODE448, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE451 = homo_ops.homo_mul_scalar_int(NODE243, 2, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE452 = homo_ops.homo_add(NODE449, NODE451, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE453 = homo_ops.homo_add_scalar_double(NODE452, np.float64(0.004878114964813271), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE454, NODE455 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE456, NODE457 = homo_ops.adjust_levels_and_depth(NODE236, NODE455, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE458, NODE459 = homo_ops.adjust_levels_and_depth(NODE238, NODE457, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE460, NODE461 = homo_ops.adjust_levels_and_depth(NODE240, NODE459, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE462 = homo_ops.homo_rescale_internal(NODE454, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE463 = homo_ops.homo_rescale_internal(NODE456, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE464 = homo_ops.homo_rescale_internal(NODE458, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE465 = homo_ops.homo_rescale_internal(NODE460, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE466 = homo_ops.homo_rescale_internal(NODE461, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE467 = homo_ops.homo_mul_scalar_double(NODE462, np.float64(0.015624554739935346), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE468 = homo_ops.homo_mul_scalar_double(NODE463, np.float64(-0.4926956557510557), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE469 = homo_ops.homo_add(NODE467, NODE468, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE470 = homo_ops.homo_mul_scalar_double(NODE464, np.float64(-0.010257275690652224), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE471 = homo_ops.homo_add(NODE469, NODE470, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE472 = homo_ops.homo_mul_scalar_double(NODE465, np.float64(0.231850036361479), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE473 = homo_ops.homo_add(NODE471, NODE472, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE474 = homo_ops.homo_mul_scalar_double(NODE466, np.float64(0.006741889972948954), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE475 = homo_ops.homo_add(NODE473, NODE474, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE476 = homo_ops.homo_rescale(NODE475, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE478 = homo_ops.homo_add(NODE476, NODE243, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE479 = homo_ops.homo_add_scalar_double(NODE478, np.float64(0.3576223526326776), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE480 = homo_ops.homo_add(NODE248, NODE426, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE481 = homo_ops.homo_mul(NODE480, NODE453, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE482 = homo_ops.homo_rescale(NODE481, 1, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE484 = homo_ops.homo_add(NODE482, NODE479, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE485 = homo_ops.homo_add(NODE253, NODE323, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE486 = homo_ops.homo_mul(NODE485, NODE401, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE487 = homo_ops.homo_rescale(NODE486, 1, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2
    NODE489 = homo_ops.homo_add(NODE487, NODE484, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=17, noise=2
    NODE490, NODE491 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE492, NODE493 = homo_ops.adjust_levels_and_depth(NODE236, NODE491, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE494, NODE495 = homo_ops.adjust_levels_and_depth(NODE238, NODE493, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE496, NODE497 = homo_ops.adjust_levels_and_depth(NODE240, NODE495, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE498 = homo_ops.homo_rescale_internal(NODE490, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE499 = homo_ops.homo_rescale_internal(NODE492, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE500 = homo_ops.homo_rescale_internal(NODE494, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE501 = homo_ops.homo_rescale_internal(NODE496, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE502 = homo_ops.homo_rescale_internal(NODE497, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE503 = homo_ops.homo_mul_scalar_double(NODE498, np.float64(-0.011226693751242361), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE504 = homo_ops.homo_mul_scalar_double(NODE499, np.float64(-0.44790338425378234), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE505 = homo_ops.homo_add(NODE503, NODE504, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE506 = homo_ops.homo_mul_scalar_double(NODE500, np.float64(-0.007289276288252454), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE507 = homo_ops.homo_add(NODE505, NODE506, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE508 = homo_ops.homo_mul_scalar_double(NODE501, np.float64(-0.3635089821481836), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE509 = homo_ops.homo_add(NODE507, NODE508, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE510 = homo_ops.homo_mul_scalar_double(NODE502, np.float64(-0.0018484514839724984), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE511 = homo_ops.homo_add(NODE509, NODE510, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE512 = homo_ops.homo_rescale(NODE511, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE514 = homo_ops.homo_add_scalar_double(NODE512, np.float64(-0.5683810697947218), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE515, NODE516 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE517, NODE518 = homo_ops.adjust_levels_and_depth(NODE236, NODE516, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE519, NODE520 = homo_ops.adjust_levels_and_depth(NODE238, NODE518, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE521, NODE522 = homo_ops.adjust_levels_and_depth(NODE240, NODE520, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE523 = homo_ops.homo_rescale_internal(NODE515, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE524 = homo_ops.homo_rescale_internal(NODE517, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE525 = homo_ops.homo_rescale_internal(NODE519, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE526 = homo_ops.homo_rescale_internal(NODE521, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE527 = homo_ops.homo_rescale_internal(NODE522, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE528 = homo_ops.homo_mul_scalar_double(NODE523, np.float64(0.007108279652990469), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE529 = homo_ops.homo_mul_scalar_double(NODE524, np.float64(0.0019489797083740077), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE530 = homo_ops.homo_add(NODE528, NODE529, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE531 = homo_ops.homo_mul_scalar_double(NODE525, np.float64(-0.0028843595536510837), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE532 = homo_ops.homo_add(NODE530, NODE531, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE533 = homo_ops.homo_mul_scalar_double(NODE526, np.float64(0.05422695740202513), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE534 = homo_ops.homo_add(NODE532, NODE533, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE535 = homo_ops.homo_mul_scalar_double(NODE527, np.float64(-0.025460400506956273), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE536 = homo_ops.homo_add(NODE534, NODE535, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE537 = homo_ops.homo_rescale(NODE536, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE539 = homo_ops.homo_add_scalar_double(NODE537, np.float64(-0.819156762828709), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE540, NODE541 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE542, NODE543 = homo_ops.adjust_levels_and_depth(NODE236, NODE541, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE544, NODE545 = homo_ops.adjust_levels_and_depth(NODE238, NODE543, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE546, NODE547 = homo_ops.adjust_levels_and_depth(NODE240, NODE545, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE548 = homo_ops.homo_rescale_internal(NODE540, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE549 = homo_ops.homo_rescale_internal(NODE542, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE550 = homo_ops.homo_rescale_internal(NODE544, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE551 = homo_ops.homo_rescale_internal(NODE546, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE552 = homo_ops.homo_rescale_internal(NODE547, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE553 = homo_ops.homo_mul_scalar_double(NODE548, np.float64(0.08762600334510307), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE554 = homo_ops.homo_mul_scalar_double(NODE549, np.float64(0.01248205312237963), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE555 = homo_ops.homo_add(NODE553, NODE554, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE556 = homo_ops.homo_mul_scalar_double(NODE550, np.float64(-0.03159516554230107), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE557 = homo_ops.homo_add(NODE555, NODE556, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE558 = homo_ops.homo_mul_scalar_double(NODE551, np.float64(-0.891126735623574), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE559 = homo_ops.homo_add(NODE557, NODE558, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE560 = homo_ops.homo_mul_scalar_double(NODE552, np.float64(-0.021505227410223895), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE561 = homo_ops.homo_add(NODE559, NODE560, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE562 = homo_ops.homo_rescale(NODE561, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE564 = homo_ops.homo_mul_scalar_int(NODE243, 4, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE565 = homo_ops.homo_add(NODE562, NODE564, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE566 = homo_ops.homo_add_scalar_double(NODE565, np.float64(0.5084800652890352), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE567, NODE568 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE569, NODE570 = homo_ops.adjust_levels_and_depth(NODE236, NODE568, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE571, NODE572 = homo_ops.adjust_levels_and_depth(NODE238, NODE570, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE573, NODE574 = homo_ops.adjust_levels_and_depth(NODE240, NODE572, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE575 = homo_ops.homo_rescale_internal(NODE567, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE576 = homo_ops.homo_rescale_internal(NODE569, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE577 = homo_ops.homo_rescale_internal(NODE571, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE578 = homo_ops.homo_rescale_internal(NODE573, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE579 = homo_ops.homo_rescale_internal(NODE574, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE580 = homo_ops.homo_mul_scalar_double(NODE575, np.float64(0.100241699512197), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE581 = homo_ops.homo_mul_scalar_double(NODE576, np.float64(0.36533466956500815), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE582 = homo_ops.homo_add(NODE580, NODE581, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE583 = homo_ops.homo_mul_scalar_double(NODE577, np.float64(-0.013581504535905302), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE584 = homo_ops.homo_add(NODE582, NODE583, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE585 = homo_ops.homo_mul_scalar_double(NODE578, np.float64(-0.48032816165746395), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE586 = homo_ops.homo_add(NODE584, NODE585, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE587 = homo_ops.homo_mul_scalar_double(NODE579, np.float64(-0.006172605335918508), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE588 = homo_ops.homo_add(NODE586, NODE587, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE589 = homo_ops.homo_rescale(NODE588, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE591 = homo_ops.homo_add(NODE589, NODE243, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE592 = homo_ops.homo_add_scalar_double(NODE591, np.float64(0.5188622170523641), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE593 = homo_ops.homo_add(NODE248, NODE539, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE594 = homo_ops.homo_mul(NODE593, NODE566, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE595 = homo_ops.homo_rescale(NODE594, 1, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE597 = homo_ops.homo_add(NODE595, NODE592, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE598, NODE599 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE600, NODE601 = homo_ops.adjust_levels_and_depth(NODE236, NODE599, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE602, NODE603 = homo_ops.adjust_levels_and_depth(NODE238, NODE601, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE604, NODE605 = homo_ops.adjust_levels_and_depth(NODE240, NODE603, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE606 = homo_ops.homo_rescale_internal(NODE598, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE607 = homo_ops.homo_rescale_internal(NODE600, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE608 = homo_ops.homo_rescale_internal(NODE602, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE609 = homo_ops.homo_rescale_internal(NODE604, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE610 = homo_ops.homo_rescale_internal(NODE605, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE611 = homo_ops.homo_mul_scalar_double(NODE606, np.float64(-0.00043842621075034687), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE612 = homo_ops.homo_mul_scalar_double(NODE607, np.float64(-0.0731687237383307), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE613 = homo_ops.homo_add(NODE611, NODE612, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE614 = homo_ops.homo_mul_scalar_double(NODE608, np.float64(0.023004825216687307), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE615 = homo_ops.homo_add(NODE613, NODE614, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE616 = homo_ops.homo_mul_scalar_double(NODE609, np.float64(-0.1998683402180191), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE617 = homo_ops.homo_add(NODE615, NODE616, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE618 = homo_ops.homo_mul_scalar_double(NODE610, np.float64(-0.046113339586327906), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE619 = homo_ops.homo_add(NODE617, NODE618, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE620 = homo_ops.homo_rescale(NODE619, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE622 = homo_ops.homo_add_scalar_double(NODE620, np.float64(-1.914253203669351), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE623, NODE624 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE625, NODE626 = homo_ops.adjust_levels_and_depth(NODE236, NODE624, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE627, NODE628 = homo_ops.adjust_levels_and_depth(NODE238, NODE626, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE629, NODE630 = homo_ops.adjust_levels_and_depth(NODE240, NODE628, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE631 = homo_ops.homo_rescale_internal(NODE623, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE632 = homo_ops.homo_rescale_internal(NODE625, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE633 = homo_ops.homo_rescale_internal(NODE627, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE634 = homo_ops.homo_rescale_internal(NODE629, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE635 = homo_ops.homo_rescale_internal(NODE630, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE636 = homo_ops.homo_mul_scalar_double(NODE631, np.float64(0.05464497109317642), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE637 = homo_ops.homo_mul_scalar_double(NODE632, np.float64(0.8598467539571557), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE638 = homo_ops.homo_add(NODE636, NODE637, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE639 = homo_ops.homo_mul_scalar_double(NODE633, np.float64(0.029222571731159365), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE640 = homo_ops.homo_add(NODE638, NODE639, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE641 = homo_ops.homo_mul_scalar_double(NODE634, np.float64(0.949487448098409), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE642 = homo_ops.homo_add(NODE640, NODE641, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE643 = homo_ops.homo_mul_scalar_double(NODE635, np.float64(0.020919545017172893), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE644 = homo_ops.homo_add(NODE642, NODE643, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE645 = homo_ops.homo_rescale(NODE644, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE647 = homo_ops.homo_mul_scalar_int(NODE243, 2, cryptoContext) #out: limb=19, noise=2, in0: limb=19, noise=2
    NODE648 = homo_ops.homo_add(NODE645, NODE647, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE649 = homo_ops.homo_add_scalar_double(NODE648, np.float64(0.07590256636785217), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE650, NODE651 = homo_ops.adjust_levels_and_depth(NODE234, NODE242, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE652, NODE653 = homo_ops.adjust_levels_and_depth(NODE236, NODE651, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE654, NODE655 = homo_ops.adjust_levels_and_depth(NODE238, NODE653, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE656, NODE657 = homo_ops.adjust_levels_and_depth(NODE240, NODE655, cryptoContext) #out0: limb=19, noise=2, #out1: limb=19, noise=2, in0: limb=19, noise=2, in1: limb=19, noise=2
    NODE658 = homo_ops.homo_rescale_internal(NODE650, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE659 = homo_ops.homo_rescale_internal(NODE652, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE660 = homo_ops.homo_rescale_internal(NODE654, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE661 = homo_ops.homo_rescale_internal(NODE656, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE662 = homo_ops.homo_rescale_internal(NODE657, 1, cryptoContext) #out: limb=18, noise=1, in0: limb=19, noise=2
    NODE663 = homo_ops.homo_mul_scalar_double(NODE658, np.float64(0.1682903223534578), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE664 = homo_ops.homo_mul_scalar_double(NODE659, np.float64(2.349462982852268), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE665 = homo_ops.homo_add(NODE663, NODE664, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE666 = homo_ops.homo_mul_scalar_double(NODE660, np.float64(0.05342369715514261), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE667 = homo_ops.homo_add(NODE665, NODE666, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE668 = homo_ops.homo_mul_scalar_double(NODE661, np.float64(2.370388179536648), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE669 = homo_ops.homo_add(NODE667, NODE668, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE670 = homo_ops.homo_mul_scalar_double(NODE662, np.float64(0.0533822343175047), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=1
    NODE671 = homo_ops.homo_add(NODE669, NODE670, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE672 = homo_ops.homo_rescale(NODE671, 1, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE674 = homo_ops.homo_add(NODE672, NODE243, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=19, noise=2
    NODE675 = homo_ops.homo_add_scalar_double(NODE674, np.float64(0.6710793101298114), cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2
    NODE676 = homo_ops.homo_add(NODE248, NODE622, cryptoContext) #out: limb=18, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE677 = homo_ops.homo_mul(NODE676, NODE649, cryptoContext) #out: limb=17, noise=2, in0: limb=18, noise=2, in1: limb=18, noise=2
    NODE678 = homo_ops.homo_rescale(NODE677, 1, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2
    NODE680 = homo_ops.homo_add(NODE678, NODE675, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE681 = homo_ops.homo_add(NODE253, NODE514, cryptoContext) #out: limb=17, noise=2, in0: limb=17, noise=2, in1: limb=18, noise=2
    NODE682 = homo_ops.homo_mul(NODE681, NODE597, cryptoContext) #out: limb=16, noise=2, in0: limb=17, noise=2, in1: limb=17, noise=2
    NODE683 = homo_ops.homo_rescale(NODE682, 1, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2
    NODE685 = homo_ops.homo_add(NODE683, NODE680, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=17, noise=2
    NODE686 = homo_ops.homo_add(NODE258, NODE298, cryptoContext) #out: limb=16, noise=2, in0: limb=16, noise=2, in1: limb=18, noise=2
    NODE687 = homo_ops.homo_mul(NODE686, NODE489, cryptoContext) #out: limb=15, noise=2, in0: limb=16, noise=2, in1: limb=16, noise=2
    NODE688 = homo_ops.homo_rescale(NODE687, 1, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2
    NODE690 = homo_ops.homo_add(NODE688, NODE685, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=16, noise=2
    NODE691 = homo_ops.homo_sub(NODE690, NODE273, cryptoContext) #out: limb=15, noise=2, in0: limb=15, noise=2, in1: limb=15, noise=2
    NODE692 = homo_ops.homo_rescale_internal(NODE691, 1, cryptoContext) #out: limb=14, noise=1, in0: limb=15, noise=2
    NODE693 = homo_ops.homo_square(NODE692, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=1
    NODE694 = homo_ops.homo_add(NODE693, NODE693, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=2, in1: limb=14, noise=2
    NODE695 = homo_ops.homo_add_scalar_double(NODE694, -0.9441845270914478, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=2
    NODE696 = homo_ops.homo_rescale(NODE695, 1, cryptoContext) #out: limb=14, noise=2, in0: limb=14, noise=2
    NODE698 = homo_ops.homo_square(NODE696, cryptoContext) #out: limb=13, noise=2, in0: limb=14, noise=2
    NODE699 = homo_ops.homo_add(NODE698, NODE698, cryptoContext) #out: limb=13, noise=2, in0: limb=13, noise=2, in1: limb=13, noise=2
    NODE700 = homo_ops.homo_add_scalar_double(NODE699, -0.891484421198901, cryptoContext) #out: limb=13, noise=2, in0: limb=13, noise=2
    NODE701 = homo_ops.homo_rescale(NODE700, 1, cryptoContext) #out: limb=13, noise=2, in0: limb=13, noise=2
    NODE703 = homo_ops.homo_square(NODE701, cryptoContext) #out: limb=12, noise=2, in0: limb=13, noise=2
    NODE704 = homo_ops.homo_add(NODE703, NODE703, cryptoContext) #out: limb=12, noise=2, in0: limb=12, noise=2, in1: limb=12, noise=2
    NODE705 = homo_ops.homo_add_scalar_double(NODE704, -0.7947444732403395, cryptoContext) #out: limb=12, noise=2, in0: limb=12, noise=2
    NODE706 = homo_ops.homo_rescale(NODE705, 1, cryptoContext) #out: limb=12, noise=2, in0: limb=12, noise=2
    NODE708 = homo_ops.homo_square(NODE706, cryptoContext) #out: limb=11, noise=2, in0: limb=12, noise=2
    NODE709 = homo_ops.homo_add(NODE708, NODE708, cryptoContext) #out: limb=11, noise=2, in0: limb=11, noise=2, in1: limb=11, noise=2
    NODE710 = homo_ops.homo_add_scalar_double(NODE709, -0.6316187777460647, cryptoContext) #out: limb=11, noise=2, in0: limb=11, noise=2
    NODE711 = homo_ops.homo_rescale(NODE710, 1, cryptoContext) #out: limb=11, noise=2, in0: limb=11, noise=2
    NODE713 = homo_ops.homo_square(NODE711, cryptoContext) #out: limb=10, noise=2, in0: limb=11, noise=2
    NODE714 = homo_ops.homo_add(NODE713, NODE713, cryptoContext) #out: limb=10, noise=2, in0: limb=10, noise=2, in1: limb=10, noise=2
    NODE715 = homo_ops.homo_add_scalar_double(NODE714, -0.3989422804014327, cryptoContext) #out: limb=10, noise=2, in0: limb=10, noise=2
    NODE716 = homo_ops.homo_rescale(NODE715, 1, cryptoContext) #out: limb=10, noise=2, in0: limb=10, noise=2
    NODE718 = homo_ops.homo_square(NODE716, cryptoContext) #out: limb=9, noise=2, in0: limb=10, noise=2
    NODE719 = homo_ops.homo_add(NODE718, NODE718, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2, in1: limb=9, noise=2
    NODE720 = homo_ops.homo_add_scalar_double(NODE719, -0.15915494309189535, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2
    NODE721 = homo_ops.homo_rescale(NODE720, 1, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2
    NODE723 = homo_ops.homo_mul_scalar_int(NODE721, 16, cryptoContext) #out: limb=9, noise=2, in0: limb=9, noise=2
    NODE724 = homo_ops.homo_rescale_internal(NODE723, 1, cryptoContext) #out: limb=8, noise=1, in0: limb=9, noise=2
    NODE725 = homo_ops.extract_cv(NODE724, 1) #out: limb=8, noise=1, in0: limb=8, noise=1
    NODE726 = hybrid_keyswitch.modup_to_ext(NODE725, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1
    NODE727 = homo_ops.eval_fast_rotate(NODE726, NODE724, 8189, True, False, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1in1: limb=8, noise=1
    NODE728 = homo_ops.eval_fast_rotate(NODE726, NODE724, 8190, True, False, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1in1: limb=8, noise=1
    NODE729 = homo_ops.eval_fast_rotate(NODE726, NODE724, 8191, True, False, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1in1: limb=8, noise=1
    NODE730 = hybrid_keyswitch.key_switch_P_ext(NODE724, cryptoContext) #out: limb=8, noise=1, in0: limb=8, noise=1
    NODE731 = homo_ops.homo_mul_pt(NODE727, NODE29, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE732 = homo_ops.homo_mul_pt(NODE728, NODE30, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE733 = homo_ops.homo_add(NODE731, NODE732, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE734 = homo_ops.homo_mul_pt(NODE729, NODE31, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE735 = homo_ops.homo_add(NODE733, NODE734, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE736 = homo_ops.homo_mul_pt(NODE730, NODE32, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE737 = homo_ops.homo_add(NODE735, NODE736, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE738 = homo_ops.extract_cv(NODE737, 0) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE739 = hybrid_keyswitch.moddown_from_ext(NODE738, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE740 = homo_ops.extract_cv(NODE737, 1, append_zeros = True) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE741 = homo_ops.homo_mul_pt(NODE727, NODE33, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE742 = homo_ops.homo_mul_pt(NODE728, NODE34, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE743 = homo_ops.homo_add(NODE741, NODE742, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE744 = homo_ops.homo_mul_pt(NODE729, NODE35, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=1, in1: limb=8, noise=1
    NODE745 = homo_ops.homo_add(NODE743, NODE744, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE746 = hybrid_keyswitch.moddown_from_ext(NODE745, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE747 = homo_ops.extract_cv(NODE746, 0) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE748 = homo_ops.extract_cv(NODE746, 1) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE749 = homo_ops._cipher_automorphism(NODE747, 4, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE750 = homo_ops.homo_add(NODE739, NODE749, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE751 = hybrid_keyswitch.modup_to_ext(NODE748, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE752 = homo_ops.eval_fast_rotate(NODE751, None, 4, False, None, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE753 = homo_ops.homo_add(NODE740, NODE752, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE754 = hybrid_keyswitch.moddown_from_ext(NODE753, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE755 = homo_ops.extract_cv(NODE750, 0, append_zeros = True) #out: limb=8, noise=2, in0: limb=8, noise=2
    NODE756 = homo_ops.homo_add(NODE754, NODE755, cryptoContext) #out: limb=8, noise=2, in0: limb=8, noise=2, in1: limb=8, noise=2
    NODE757 = homo_ops.homo_rescale_internal(NODE756, 1, cryptoContext) #out: limb=7, noise=1, in0: limb=8, noise=2
    NODE758 = homo_ops.extract_cv(NODE757, 1) #out: limb=7, noise=1, in0: limb=7, noise=1
    NODE759 = hybrid_keyswitch.modup_to_ext(NODE758, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1
    NODE760 = homo_ops.eval_fast_rotate(NODE759, NODE757, 8180, True, False, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1in1: limb=7, noise=1
    NODE761 = homo_ops.eval_fast_rotate(NODE759, NODE757, 8184, True, False, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1in1: limb=7, noise=1
    NODE762 = homo_ops.eval_fast_rotate(NODE759, NODE757, 8188, True, False, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1in1: limb=7, noise=1
    NODE763 = hybrid_keyswitch.key_switch_P_ext(NODE757, cryptoContext) #out: limb=7, noise=1, in0: limb=7, noise=1
    NODE764 = homo_ops.homo_mul_pt(NODE760, NODE36, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE765 = homo_ops.homo_mul_pt(NODE761, NODE37, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE766 = homo_ops.homo_add(NODE764, NODE765, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE767 = homo_ops.homo_mul_pt(NODE762, NODE38, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE768 = homo_ops.homo_add(NODE766, NODE767, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE769 = homo_ops.homo_mul_pt(NODE763, NODE39, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE770 = homo_ops.homo_add(NODE768, NODE769, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE771 = homo_ops.extract_cv(NODE770, 0) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE772 = hybrid_keyswitch.moddown_from_ext(NODE771, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE773 = homo_ops.extract_cv(NODE770, 1, append_zeros = True) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE774 = homo_ops.homo_mul_pt(NODE760, NODE40, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE775 = homo_ops.homo_mul_pt(NODE761, NODE41, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE776 = homo_ops.homo_add(NODE774, NODE775, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE777 = homo_ops.homo_mul_pt(NODE762, NODE42, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=1, in1: limb=7, noise=1
    NODE778 = homo_ops.homo_add(NODE776, NODE777, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE779 = hybrid_keyswitch.moddown_from_ext(NODE778, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE780 = homo_ops.extract_cv(NODE779, 0) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE781 = homo_ops.extract_cv(NODE779, 1) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE782 = homo_ops._cipher_automorphism(NODE780, 16, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE783 = homo_ops.homo_add(NODE772, NODE782, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE784 = hybrid_keyswitch.modup_to_ext(NODE781, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE785 = homo_ops.eval_fast_rotate(NODE784, None, 16, False, None, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE786 = homo_ops.homo_add(NODE773, NODE785, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE787 = hybrid_keyswitch.moddown_from_ext(NODE786, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE788 = homo_ops.extract_cv(NODE783, 0, append_zeros = True) #out: limb=7, noise=2, in0: limb=7, noise=2
    NODE789 = homo_ops.homo_add(NODE787, NODE788, cryptoContext) #out: limb=7, noise=2, in0: limb=7, noise=2, in1: limb=7, noise=2
    NODE790 = homo_ops.homo_rescale_internal(NODE789, 1, cryptoContext) #out: limb=6, noise=1, in0: limb=7, noise=2
    NODE791 = homo_ops.extract_cv(NODE790, 1) #out: limb=6, noise=1, in0: limb=6, noise=1
    NODE792 = hybrid_keyswitch.modup_to_ext(NODE791, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1
    NODE793 = homo_ops.eval_fast_rotate(NODE792, NODE790, 8144, True, False, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1in1: limb=6, noise=1
    NODE794 = homo_ops.eval_fast_rotate(NODE792, NODE790, 8160, True, False, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1in1: limb=6, noise=1
    NODE795 = homo_ops.eval_fast_rotate(NODE792, NODE790, 8176, True, False, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1in1: limb=6, noise=1
    NODE796 = hybrid_keyswitch.key_switch_P_ext(NODE790, cryptoContext) #out: limb=6, noise=1, in0: limb=6, noise=1
    NODE797 = homo_ops.homo_mul_pt(NODE793, NODE43, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE798 = homo_ops.homo_mul_pt(NODE794, NODE44, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE799 = homo_ops.homo_add(NODE797, NODE798, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE800 = homo_ops.homo_mul_pt(NODE795, NODE45, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE801 = homo_ops.homo_add(NODE799, NODE800, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE802 = homo_ops.homo_mul_pt(NODE796, NODE46, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE803 = homo_ops.homo_add(NODE801, NODE802, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE804 = homo_ops.extract_cv(NODE803, 0) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE805 = hybrid_keyswitch.moddown_from_ext(NODE804, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE806 = homo_ops.extract_cv(NODE803, 1, append_zeros = True) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE807 = homo_ops.homo_mul_pt(NODE793, NODE47, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE808 = homo_ops.homo_mul_pt(NODE794, NODE48, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE809 = homo_ops.homo_add(NODE807, NODE808, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE810 = homo_ops.homo_mul_pt(NODE795, NODE49, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=1, in1: limb=6, noise=1
    NODE811 = homo_ops.homo_add(NODE809, NODE810, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE812 = hybrid_keyswitch.moddown_from_ext(NODE811, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE813 = homo_ops.extract_cv(NODE812, 0) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE814 = homo_ops.extract_cv(NODE812, 1) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE815 = homo_ops._cipher_automorphism(NODE813, 64, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE816 = homo_ops.homo_add(NODE805, NODE815, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE817 = hybrid_keyswitch.modup_to_ext(NODE814, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE818 = homo_ops.eval_fast_rotate(NODE817, None, 64, False, None, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE819 = homo_ops.homo_add(NODE806, NODE818, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE820 = hybrid_keyswitch.moddown_from_ext(NODE819, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE821 = homo_ops.extract_cv(NODE816, 0, append_zeros = True) #out: limb=6, noise=2, in0: limb=6, noise=2
    NODE822 = homo_ops.homo_add(NODE820, NODE821, cryptoContext) #out: limb=6, noise=2, in0: limb=6, noise=2, in1: limb=6, noise=2
    NODE823 = homo_ops.homo_rescale_internal(NODE822, 1, cryptoContext) #out: limb=5, noise=1, in0: limb=6, noise=2
    NODE824 = homo_ops.extract_cv(NODE823, 1) #out: limb=5, noise=1, in0: limb=5, noise=1
    NODE825 = hybrid_keyswitch.modup_to_ext(NODE824, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1
    NODE826 = homo_ops.eval_fast_rotate(NODE825, NODE823, 8000, True, False, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1in1: limb=5, noise=1
    NODE827 = homo_ops.eval_fast_rotate(NODE825, NODE823, 8064, True, False, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1in1: limb=5, noise=1
    NODE828 = homo_ops.eval_fast_rotate(NODE825, NODE823, 8128, True, False, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1in1: limb=5, noise=1
    NODE829 = hybrid_keyswitch.key_switch_P_ext(NODE823, cryptoContext) #out: limb=5, noise=1, in0: limb=5, noise=1
    NODE830 = homo_ops.homo_mul_pt(NODE826, NODE50, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE831 = homo_ops.homo_mul_pt(NODE827, NODE51, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE832 = homo_ops.homo_add(NODE830, NODE831, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE833 = homo_ops.homo_mul_pt(NODE828, NODE52, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE834 = homo_ops.homo_add(NODE832, NODE833, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE835 = homo_ops.homo_mul_pt(NODE829, NODE53, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE836 = homo_ops.homo_add(NODE834, NODE835, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE837 = homo_ops.extract_cv(NODE836, 0) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE838 = hybrid_keyswitch.moddown_from_ext(NODE837, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE839 = homo_ops.extract_cv(NODE836, 1, append_zeros = True) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE840 = homo_ops.homo_mul_pt(NODE826, NODE54, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE841 = homo_ops.homo_mul_pt(NODE827, NODE55, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE842 = homo_ops.homo_add(NODE840, NODE841, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE843 = homo_ops.homo_mul_pt(NODE828, NODE56, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=1, in1: limb=5, noise=1
    NODE844 = homo_ops.homo_add(NODE842, NODE843, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE845 = hybrid_keyswitch.moddown_from_ext(NODE844, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE846 = homo_ops.extract_cv(NODE845, 0) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE847 = homo_ops.extract_cv(NODE845, 1) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE848 = homo_ops._cipher_automorphism(NODE846, 256, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE849 = homo_ops.homo_add(NODE838, NODE848, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE850 = hybrid_keyswitch.modup_to_ext(NODE847, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE851 = homo_ops.eval_fast_rotate(NODE850, None, 256, False, None, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE852 = homo_ops.homo_add(NODE839, NODE851, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE853 = hybrid_keyswitch.moddown_from_ext(NODE852, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE854 = homo_ops.extract_cv(NODE849, 0, append_zeros = True) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE855 = homo_ops.homo_add(NODE853, NODE854, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE856 = homo_ops.homo_rotate(NODE855, 256, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2
    NODE857 = homo_ops.homo_add(NODE855, NODE856, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2, in1: limb=5, noise=2
    NODE858 = homo_ops.homo_mul_scalar_int(NODE857, 64, cryptoContext) #out: limb=5, noise=2, in0: limb=5, noise=2

    return NODE858

def homo_bootstrap(cipher, L0, logBsSlots, cryptoContext):

    if cryptoContext.autoLoadAndSetConfig == True:
        cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]

    result = eval_bootstrap(cipher, L0, logBsSlots, cryptoContext)

    if (
        cryptoContext.rescaleTech == "FIXEDMANUAL"
    ):  # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
        result = homo_ops.homo_rescale(result, result.noise_deg - 1, cryptoContext)

    return result
