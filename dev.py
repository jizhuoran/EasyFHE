import numpy as np
import time
from enum import Enum

import torch
import torch.fhe.functional as F
import torch.fhe.test as T
from torch.fhe.Ciphertext import Ciphertext
from torch.fhe.context import Context
from torch.fhe.context_cuda import Context_Cuda

# T.test_homo_add()
# T.test_HMult_and_rescale_1()
# T.test_SwitchModulus()
T.test_ApproxMod()
T.test_KS3_ct()
T.test_logN17()
# a = torch.tensor([6] * (2**15), dtype=torch.uint64, device='cuda')
# b = torch.tensor([4] * (2**15), dtype=torch.uint64, device='cuda')
#
# mu = torch.tensor([14347467612885206812, 2049638230412172401], dtype=torch.uint64, device='cuda')

# c = F.vec_mul_mod(a, 7, 9, mu)

class ParamType(Enum):
    TYPE_PRECISION = 1
    TYPE_A = 2
    TYPE_B = 3

# Function to configure parameters based on param type
def configure_parameters(param: ParamType):
    if param == ParamType.TYPE_PRECISION:
        log_degree= 11
        level= 3+1
        dnum= 4
        primes= [
            9007199254781953, 4503599627366401, 4503599627481089, 4503599627542529, 4503599627554817
            ]
        return Context_Cuda(logN=log_degree, L=level, dnum=dnum, primes=primes)
    elif param == ParamType.TYPE_A:
        
        log_degree=16
        level =34+1
        dnum= 5
        primes= [
            2305843009218281473, 2251799661248513, 2251799661641729, 2251799665180673, 2251799682088961,
            2251799678943233, 2251799717609473, 2251799710138369, 2251799708827649, 2251799707385857,
            2251799713677313, 2251799712366593, 2251799716691969, 2251799714856961, 2251799726522369,
            2251799726129153, 2251799747493889, 2251799741857793, 2251799740416001, 2251799746707457,
            2251799756013569, 2251799775805441, 2251799763091457, 2251799767154689, 2251799765975041,
            2251799770562561, 2251799769776129, 2251799772266497, 2251799775281153, 2251799774887937,
            2251799797432321, 2251799787995137, 2251799787601921, 2251799791403009, 2251799789568001,
            2251799795466241, 2251799807131649, 2251799806345217, 2251799805165569, 2305843009218936833,
            2305843009220116481, 2305843009221820417
        ]
        return Context_Cuda(logN=log_degree, L=level, dnum=dnum, primes=primes)
    else:  # Default case
        
        log_degree= 17
        level=29 +1
        dnum=3
        primes=[
            2305843009146585089, 2251799756013569, 2251799787995137, 2251800352915457, 2251799780917249,
            2251799666884609, 2251799678943233, 2251799696244737, 2251800082382849, 2251799776198657,
            2251799929028609, 2251799774887937, 2251799849336833, 2251799883153409, 2251799777771521,
            2251799879483393, 2251799772266497, 2251799763091457, 2251799844093953, 2251799823384577,
            2251799851958273, 2251799789568001, 2251799797432321, 2251799799267329, 2251799836753921,
            2251799806345217, 2251799807131649, 2251799818928129, 2251799816568833, 2251799815520257,
            2305843009221820417, 2305843009224179713, 2305843009229946881, 2305843009255636993,
            2305843009350008833, 2305843009448574977, 2305843009746370561, 2305843009751089153,
            2305843010287697921, 2305843010288484353
        ]
        return Context_Cuda(logN=log_degree, L=level, dnum=dnum, primes=primes)

# ##################################modup core###########################################
context_cuda = configure_parameters(ParamType.TYPE_PRECISION)
input = torch.tensor(
    [5956358506108845] * (2048 * context_cuda.level), dtype=torch.uint64, device="cuda"
)
num_moduli_after_modup = 5
beta = (int)(context_cuda.level / context_cuda.alpha)
out = torch.tensor(
    [0] * (num_moduli_after_modup * context_cuda.degree * beta), dtype=torch.uint64, device="cuda"
)

modup = F.modup_core(
    input,
    context_cuda.hat_inverse_vec,
    context_cuda.hat_inverse_vec_shoup,
    context_cuda.prod_q_i_mod_q_j,
    context_cuda.primes,
    context_cuda.barret_ratio,
    context_cuda.barret_k,
    beta,
    context_cuda.degree,
    context_cuda.alpha,
    num_moduli_after_modup,
    out,
)
modup = modup.cpu()


print("modup res:", modup)

########################################## moddown core##################################
input_moddown = torch.tensor(
    [5956358506108845] * (2**11 * context_cuda.level), dtype=torch.uint64, device="cuda"
)
target_chain_idx = context_cuda.level
param_chain_length = context_cuda.level
param_max_num_moduli = context_cuda.level + context_cuda.alpha
param_degree = context_cuda.degree
param_log_degree = context_cuda.log_degree
to = torch.tensor(
    [0] * (param_chain_length * context_cuda.degree), dtype=torch.uint64, device="cuda"
)

moddown = F.moddown_core(
    input_moddown,
    target_chain_idx,
    param_chain_length,
    param_max_num_moduli,
    param_degree,
    param_log_degree,
    context_cuda.hat_inverse_vec_moddown,
    context_cuda.hat_inverse_vec_shoup_moddown,
    context_cuda.prod_q_i_mod_q_j_moddown,
    context_cuda.prod_inv_moddown,
    context_cuda.prod_inv_shoup_moddown,
    context_cuda.primes,
    context_cuda.barret_ratio,
    context_cuda.barret_k,
    to,
)
print("moddown res:", moddown)

########################NTT##########################################
start_prime_idx = 0
batch = target_chain_idx
ntt = F.NTT(
    input,
    start_prime_idx,
    batch,
    context_cuda.degree,
    context_cuda.power_of_roots_shoup,
    context_cuda.primes,
    context_cuda.power_of_roots,
)
print("ntt_res:", ntt)

########################iNTT##########################################
start_prime_idx = 0
batch = target_chain_idx
intt = F.iNTT(
    input,
    start_prime_idx,
    batch,
    param_degree,
    context_cuda.inverse_power_of_roots_div_two,
    context_cuda.primes,
    context_cuda.inverse_scaled_power_of_roots_div_two,
)
print("intt_res:", intt)


#################################KeySwitch################################
context_cuda = configure_parameters(ParamType.TYPE_PRECISION)
# start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
# end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

# inner_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
# inner_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

# moddown_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
# moddown_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

num_moduli_after_modup = context_cuda.level + context_cuda.alpha
beta = (int)(context_cuda.level / context_cuda.alpha)
target_chain_idx = context_cuda.level
param_chain_length = context_cuda.level
param_max_num_moduli = context_cuda.level + context_cuda.alpha

# np_input_ks = np.load("/home/yons/wwz/project/GPU-FHE/script/pydata/modup_intput.npy")
# np_ax = np.load("/home/yons/wwz/project/GPU-FHE/script/pydata/key_ax_inner.npy")
# np_bx = np.load("/home/yons/wwz/project/GPU-FHE/script/pydata/key_bx_inner.npy")
# input_ks = torch.from_numpy(np_input_ks).cuda()
# ax = torch.from_numpy(np_ax).cuda()
# bx = torch.from_numpy(np_bx).cuda()
input_ks = torch.tensor(
        [5956358506108845] * (context_cuda.degree * context_cuda.level), dtype=torch.uint64, device="cuda"
    )

ax = torch.tensor(
    [0] * (num_moduli_after_modup * context_cuda.degree * beta),
    dtype=torch.uint64,
    device="cuda",
)
bx = torch.tensor(
    [0] * (num_moduli_after_modup * context_cuda.degree * beta),
    dtype=torch.uint64,
    device="cuda",
)
modup_out = torch.tensor(
    [0] * (num_moduli_after_modup * context_cuda.degree * beta), dtype=torch.uint64, device="cuda"
)
inner_out = torch.tensor(
    [0] * (2*context_cuda.max_num_moduli * context_cuda.degree),
    dtype=torch.uint64,
    device="cuda",
)
moddown_out = torch.tensor(
    [0] * (param_chain_length * context_cuda.degree), dtype=torch.uint64, device="cuda"
)
workspace = torch.tensor([0] * (4*num_moduli_after_modup * context_cuda.degree * beta), dtype=torch.uint64, device="cuda")

for i in range(10):
    # start_events[i].record()
    modup_start = time.time()
    modup = F.modup(
        input_ks,
        context_cuda.hat_inverse_vec,
        context_cuda.hat_inverse_vec_shoup,
        context_cuda.prod_q_i_mod_q_j,
        context_cuda.primes,
        context_cuda.barret_ratio,
        context_cuda.barret_k,
        beta,
        context_cuda.degree,
        context_cuda.alpha,
        num_moduli_after_modup,
        context_cuda.power_of_roots_shoup,
        context_cuda.power_of_roots,
        context_cuda.inverse_power_of_roots_div_two,
        context_cuda.inverse_scaled_power_of_roots_div_two,
        modup_out,
        inplace=False
    )
    modup_end = time.time()
    print("modup total time:", (modup_end-modup_start)*1e3,"\n")
    # end_events[i].record()
    
    # inner_start_events[i].record()
    start = time.time()
    inner_product = F.innerproduct(
        modup,
        ax,
        bx,
        context_cuda.degree,context_cuda.max_num_moduli, 
        context_cuda.primes,
        context_cuda.barret_ratio,
        context_cuda.barret_k,
        workspace,
        inner_out,
        inplace = False
    )
    end = time.time()
    # inner_end_events[i].record()
    print("inner total time:", (end-start)*1e3,"\n")
    
    sumMult_ax = inner_product[:context_cuda.max_num_moduli * context_cuda.degree]
    sumMult_bx = inner_product[context_cuda.max_num_moduli * context_cuda.degree:]

    # moddown_start_events[i].record()
    moddown_start = time.time()
    moddown_ax = F.moddown(
        sumMult_ax,
        target_chain_idx,
        param_chain_length,
        param_max_num_moduli,
        context_cuda.degree,
        context_cuda.log_degree,
        context_cuda.hat_inverse_vec_moddown,
        context_cuda.hat_inverse_vec_shoup_moddown,
        context_cuda.prod_q_i_mod_q_j_moddown,
        context_cuda.prod_inv_moddown,
        context_cuda.prod_inv_shoup_moddown,
        context_cuda.primes,
        context_cuda.barret_ratio,
        context_cuda.barret_k,
        context_cuda.power_of_roots_shoup,
        context_cuda.power_of_roots,
        context_cuda.inverse_power_of_roots_div_two,
        context_cuda.inverse_scaled_power_of_roots_div_two,
        moddown_out,
        inplace=False
    )
    moddown_end = time.time()
    print("moddown total time:", (moddown_end-moddown_start)*1e3,"\n")
    # moddown_end_events[i].record()
    torch.cuda.synchronize()

    # res_0 = np.load("/home/yons/wwz/project/GPU-FHE/script/pydata/moddown_output_ax.npy")
    # np_moddown_ax = moddown_ax.cpu().detach().numpy()
    # for i in range (param_chain_length*context_cuda.degree):
    #     if(np_moddown_ax[i] != res_0[i]):
    #         print("i=", i, ", res_0=", res_0[i], ",moddown_ax=", moddown_ax[i], "\n")
    #         break

    moddown_bx = F.moddown(
        sumMult_bx,
        target_chain_idx,
        param_chain_length,
        param_max_num_moduli,
        context_cuda.degree,
        context_cuda.log_degree,
        context_cuda.hat_inverse_vec_moddown,
        context_cuda.hat_inverse_vec_shoup_moddown,
        context_cuda.prod_q_i_mod_q_j_moddown,
        context_cuda.prod_inv_moddown,
        context_cuda.prod_inv_shoup_moddown,
        context_cuda.primes,
        context_cuda.barret_ratio,
        context_cuda.barret_k,
        context_cuda.power_of_roots_shoup,
        context_cuda.power_of_roots,
        context_cuda.inverse_power_of_roots_div_two,
        context_cuda.inverse_scaled_power_of_roots_div_two,
        moddown_out
    )

# torch.cuda.synchronize()
# times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
# print("modup cuda time:", times)

# inner_times = [s.elapsed_time(e) for s, e in zip(inner_start_events, inner_end_events)]
# print("inner cuda time:", inner_times)

# moddown_times = [s.elapsed_time(e) for s, e in zip(moddown_start_events, moddown_end_events)]
# print("moddown cuda time:", moddown_times)