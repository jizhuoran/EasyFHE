import easyfhe as torch
import easyfhe.fhe as fhe
import easyfhe.fhe.ciphertext as Cipher

def _encrypt(pk0, pk1, ptx, device, cryptoContext):
    logn = cryptoContext.logN
    cur_limbs = ptx.cur_limbs
    l = cryptoContext.L
    nh = cryptoContext.N // 2
    moduliP_scalar = cryptoContext.moduliP_scalar
    moduliQ_scalar = cryptoContext.moduliQ_scalar
    moduliP_scalar = torch.from_numpy(moduliP_scalar).to(device)
    moduliQ_scalar = torch.from_numpy(moduliQ_scalar).to(device)
    cv = torch.encrypt(ptx=ptx.cv[0], pk0=pk0, pk1=pk1, l=l, logn=logn, nh=nh, moduliP_scalar=moduliP_scalar,
                       moduliQ_scalar=moduliQ_scalar, primes=cryptoContext.QplusP_map[cur_limbs],
                       max_int_diffs=cryptoContext.QmaxdiffplusPmaxdiff_map[cur_limbs],
                       barret_ratio=cryptoContext.QbarretRatioplusPbarretRatio_map[cur_limbs],
                       barret_k=cryptoContext.QbarretKplusPbarretK_map[cur_limbs],
                       power_of_roots_shoup=cryptoContext.power_of_roots_shoup,
                       power_of_roots=cryptoContext.power_of_roots)
    bx, ax = cv
    n = 1 << logn
    if bx.numel() == l * n:
        bx_reshaped = bx.view(l, n)
        ax_reshaped = ax.view(l, n)
    cv_reshaped = [bx_reshaped, ax_reshaped]
    return Cipher.Cipher(cv_reshaped, cur_limbs, ptx.scaling_factor, ptx.noise_deg, ptx.slots, is_ext=False)


def encrypt(x, device, scale_deg, level, slots, openfheContext, cryptoContext):
    pk=openfheContext.publicKey.GetKeyValue()
    [pk0,pk1] = [torch.tensor(elem, device=device, dtype=torch.uint64) for elem in pk]
    ptx = fhe.encode(x, "encrypt", level, slots, False, cryptoContext)
    cipher = _encrypt(pk0, pk1, ptx, device, cryptoContext)

    if cryptoContext.config.COMPARE_WITH_OPENFHE:
        openfheptx = openfheContext.cc.MakeCKKSPackedPlaintext(x, scale_deg, level, None, slots)
        openfhe_cipher = openfheContext.cc.Encrypt(openfheContext.publicKey, openfheptx)
        return cipher, openfhe_cipher
    else:
        return cipher

