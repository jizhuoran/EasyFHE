import numpy as np
import pytest

import easyfhe as torch
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe


_MILLER_RABIN_BASES_64 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_probable_prime_u64(value):
    value = int(value)
    if value < 2:
        return False
    for prime in _MILLER_RABIN_BASES_64:
        if value == prime:
            return True
        if value % prime == 0:
            return False

    d = value - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for base in _MILLER_RABIN_BASES_64:
        if base >= value:
            continue
        x = pow(base, d, value)
        if x == 1 or x == value - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % value
            if x == value - 1:
                break
        else:
            return False
    return True


def _prime_congruent_near(target, cycl_order, used=()):
    target = int(round(float(target)))
    cycl_order = int(cycl_order)
    used = {int(prime) for prime in used}
    center = target + ((1 - target) % cycl_order)
    if center < 3:
        center += cycl_order

    for offset in range(20000):
        candidates = (center,) if offset == 0 else (center - offset * cycl_order, center + offset * cycl_order)
        for candidate in candidates:
            if candidate > 2 and candidate not in used and _is_probable_prime_u64(candidate):
                return int(candidate)
    raise RuntimeError(f"could not find a prime near {target}")


def _exact_u64_q_primes(log_n, depth, *, scale_bits=59, rel_delta=0.0):
    cycl_order = 2 << int(log_n)
    first = _prime_congruent_near(2.0**60, cycl_order)
    used = {first}
    ordinary_target = (2.0**int(scale_bits)) * (1.0 + float(rel_delta))
    ordinary = []
    for _ in range(int(depth)):
        prime = _prime_congruent_near(ordinary_target, cycl_order, used)
        ordinary.append(prime)
        used.add(prime)
    return (first, *ordinary)


def test_u64_exact_q_primes_are_used_by_native_sampler():
    exact_q_primes = _exact_u64_q_primes(log_n=6, depth=3, scale_bits=45, rel_delta=3.6e-5)

    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=3,
            log_n=6,
            dnum=1,
            dcrt_bits=45,
            first_mod=60,
            exact_q_primes=exact_q_primes,
        ),
        device="cpu",
    )

    assert tuple(int(prime) for prime in ctx.moduliQ_scalar) == exact_q_primes
    assert tuple(ctx.context_generation_config["exactQPrimes"]) == exact_q_primes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FHE kernels required")
@pytest.mark.parametrize("rel_delta", [0.0, 3.6e-5, -3.6e-5, 9.5e-5, -9.5e-5])
def test_u64_flexible_bootstrap_precision_with_exact_prime_scale_mismatch_cuda(rel_delta):
    log_n = 16
    log_bs_slots = 14
    level_budget = (4, 4)
    bs_depth = bs.depth(log_bs_slots=log_bs_slots, level_budget=level_budget)
    post_bootstrap_levels = 2
    total_depth = bs_depth + post_bootstrap_levels
    rotations = bs.plan_rot_keys(log_n=log_n, log_bs_slots=log_bs_slots, level_budget=level_budget)
    exact_q_primes = _exact_u64_q_primes(
        log_n=log_n,
        depth=total_depth,
        scale_bits=59,
        rel_delta=rel_delta,
    )
    client, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=total_depth,
            log_n=log_n,
            dnum=1,
            dcrt_bits=59,
            first_mod=60,
            exact_q_primes=exact_q_primes,
            rotations=tuple(rotations),
            auto_load_keys=True,
            scale_mode="flexible",
        ),
        device="cuda",
    )
    constants, plan = bs.generate(
        ctx,
        log_bs_slots=log_bs_slots,
        level_budget=level_budget,
        post_bootstrap_levels=post_bootstrap_levels,
    )
    values = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float64)

    cipher = client.encrypt(
        values,
        device="cuda",
        slots=1 << log_bs_slots,
        scaling_factor=ctx.scale_at(ctx.L),
    )
    result = bs.bootstrap(cipher, ctx, constants, plan, L0=ctx.L, bootstrap_mode="modraise_first")
    decoded = client.decrypt(result)[: values.size].cpu().numpy()
    max_error = float(np.max(np.abs(decoded - values)))

    assert max_error < (2e-3 if rel_delta == 0.0 else 2e-2)
