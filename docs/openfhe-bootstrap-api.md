# OpenFHE Bootstrap API

This page documents the public API exported by:

```python
import easyfhe.bs.openfhe as bs
```

The API builds and runs OpenFHE-compatible CKKS bootstrapping. Runtime internals
under `easyfhe.bs.openfhe.runtime` and generation internals under
`easyfhe.bs.openfhe.generation` are implementation details unless re-exported
from `easyfhe.bs.openfhe`.

## Typical Flow

```python
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

log_bs_slots = 14
level_budget = (4, 4)
post_bootstrap_levels = 12

bootstrap_depth = bs.depth(
    log_bs_slots=(log_bs_slots,),
    level_budget=(level_budget,),
    secret_key_dist="SPARSE_TERNARY",
)

bootstrap_rotations = bs.plan_rot_keys(
    log_n=16,
    log_bs_slots=(log_bs_slots,),
    level_budget=(level_budget,),
    strategy="double_hoist",
)

client, ctx = fhe.generate_client_context(
    fhe.CKKSContextSpec(
        depth=post_bootstrap_levels + bootstrap_depth,
        log_n=16,
        dnum=3,
        dcrt_bits=59,
        first_mod=60,
        secret_key_dist="SPARSE_TERNARY",
        scale_mode="fixed",
        rescale_policy="manual",
        rotations=bootstrap_rotations,
    ),
    device="cuda",
)

constants, plan = bs.generate(
    ctx,
    log_bs_slots=log_bs_slots,
    level_budget=level_budget,
    post_bootstrap_levels=post_bootstrap_levels,
    strategy="double_hoist",
)

boot = bs.bootstrap(
    cipher,
    ctx,
    constants,
    plan,
    L0=cipher.state.cur_limbs,
    bootstrap_mode="modraise_first",
)
```

For one context/constant set that can run all three runtime strategies, generate
rotation keys with `strategy="double_hoist"`. `normal_giant` and `normal_bsgs`
use subsets of those keys.

## Public Symbols

```python
bs.depth(...)
bs.plan_rot_keys(...)
bs.generate(...)
bs.bootstrap(...)
bs.describe_plan(...)
bs.BootstrapPlan
```

## `bs.depth`

```python
bs.depth(*, log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY") -> int
```

Returns the extra CKKS depth needed to support the bootstrap transform and
EvalMod approximation. Add this to the number of post-bootstrap levels the
application needs when creating `fhe.CKKSContextSpec`.

Parameters:

- `log_bs_slots`: `int` or sequence of `int`.
  Base-2 log of each bootstrap slot count. For one bootstrap with 16384 slots,
  pass `14`. For multiple supported slot counts, pass a sequence such as
  `(14, 12)`.
- `level_budget`: pair `(c2s_budget, s2c_budget)` or sequence of pairs.
  The number of levels assigned to the collapsed FFT stages for
  CoeffsToSlots and SlotsToCoeffs. Each entry must be greater than `1`;
  the linear-transform route is not currently supported.
- `secret_key_dist`: `"SPARSE_TERNARY"` or `"UNIFORM_TERNARY"`.
  Selects the EvalMod polynomial/depth profile.

Returns:

- `int`: the maximum required bootstrap depth across all parameter sets.

Notes:

- `depth(...)` is independent of `bootstrap_mode`. `modraise_first` and
  `stc_first` use the same context depth and produce the same
  `post_bootstrap_levels`.
- `stc_first` has an input-state requirement at runtime: the input ciphertext
  must have enough limbs to run the initial SlotsToCoeffs step.

## `bs.plan_rot_keys`

```python
bs.plan_rot_keys(
    *,
    log_n,
    log_bs_slots,
    level_budget,
    strategy="double_hoist",
    dim1=None,
    baby_step=None,
) -> tuple[int, ...]
```

Returns the rotation-key indices needed by the selected bootstrap parameter
sets.

Parameters:

- `log_n`: `int`.
  Base-2 log of the CKKS ring dimension.
- `log_bs_slots`: `int` or sequence of `int`.
  Bootstrap slot counts, as in `bs.depth`.
- `level_budget`: pair or sequence of pairs.
  C2S/S2C level budgets, as in `bs.depth`.
- `strategy`: one of `"double_hoist"`, `"normal_giant"`, or `"normal_bsgs"`.
  Controls which giant-rotation keys are required.
- `dim1`: optional pair or sequence of pairs.
  OpenFHE-style BSGS dimension override for C2S and S2C. `None` uses
  OpenFHE-style defaults.
  For one bootstrap parameter set, pass `(c2s_dim1, s2c_dim1)`. For multiple
  parameter sets, pass a sequence of such pairs.
- `baby_step`: optional int, pair, or sequence of pairs.
  Actual BSGS baby-step count override. Do not pass both `dim1` and
  `baby_step`.

Returns:

- `tuple[int, ...]`: unique rotation indices, preserving first-use order.

Strategy guidance:

- `double_hoist` generally requires the most keys.
- `normal_giant` and `normal_bsgs` can run with a subset of the double-hoist
  keys.
- If one context should test or serve all three strategies, request keys with
  `strategy="double_hoist"` and switch the plan's `strategy` field for runtime
  experiments.

## `bs.generate`

```python
bs.generate(
    crypto_context,
    *,
    log_bs_slots,
    level_budget,
    post_bootstrap_levels=None,
    max_levels_remaining=None,
    dim1=None,
    baby_step=None,
    strategy="double_hoist",
) -> tuple[ConstantBundle, BootstrapPlan]
```

Generates reusable bootstrap constants and a `BootstrapPlan` for one bootstrap
slot count and one level-budget pair.

Parameters:

- `crypto_context`: EasyFHE CKKS context returned by
  `fhe.generate_client_context`.
- `log_bs_slots`: `int`.
  Base-2 log of the bootstrap slot count.
- `level_budget`: pair `(c2s_budget, s2c_budget)`.
  Collapsed FFT level budget.
- `post_bootstrap_levels`: `int`.
  Number of levels to keep available after a successful bootstrap. The runtime
  aligns the output to `post_bootstrap_levels + 1` limbs with `noise_deg == 1`.
- `max_levels_remaining`: optional `int`.
  Backward-compatible alias for `post_bootstrap_levels`.
- `dim1`: optional pair `(c2s_dim1, s2c_dim1)`.
  OpenFHE-style BSGS dimension override for the generated C2S and S2C linear
  transforms. `0` means use the default for that direction.
- `baby_step`: optional int or pair `(c2s_baby_step, s2c_baby_step)`.
  Actual BSGS baby-step count override. Do not pass both `dim1` and
  `baby_step`.
- `strategy`: one of `"double_hoist"`, `"normal_giant"`, or `"normal_bsgs"`.
  Stored on the returned plan as the default runtime strategy.

Returns:

- `constants`: an `easyfhe.fhe.ConstantBundle` containing generated scalar and
  vector constants.
- `plan`: a `BootstrapPlan` describing C2S, S2C, EvalMod, output-level, and
  rotation requirements.

Notes:

- `generate(...)` does not choose between `modraise_first` and `stc_first`.
  Choose the bootstrap route when calling `bs.bootstrap(...)`.
- The same constants and plan materials can be used by both bootstrap routes.
- The same constants can be used with all three runtime strategies as long as
  the context has the required rotation keys.
- `dim1`/`baby_step` changes the generated linear-transform schedule, so use
  the same value for `bs.plan_rot_keys(...)` and `bs.generate(...)`.

### Baby-Step Override

Use `baby_step` to control the actual BSGS baby-step count for the collapsed
FFT linear transforms:

```python
baby_step = (8, 8)

rotations = bs.plan_rot_keys(
    log_n=16,
    log_bs_slots=14,
    level_budget=(4, 4),
    strategy="double_hoist",
    baby_step=baby_step,
)

constants, plan = bs.generate(
    ctx,
    log_bs_slots=14,
    level_budget=(4, 4),
    post_bootstrap_levels=12,
    strategy="double_hoist",
    baby_step=baby_step,
)
```

Use `0` for either entry to keep the default for that direction. `dim1` is
still available when you want the lower-level OpenFHE split parameter:

```python
constants, plan = bs.generate(..., dim1=(0, 8))
```

Changing `baby_step`/`dim1` can change:

- the baby/giant grouping inside C2S and S2C;
- the rotation-key set returned by `bs.plan_rot_keys(...)`;
- the generated plaintext vector batches stored in `constants`;
- runtime memory/cache shape and speed.

It does not change the high-level bootstrap mode. The same generated plan can
still run both `modraise_first` and `stc_first`.

## `bs.bootstrap`

```python
bs.bootstrap(
    cipher,
    crypto_context,
    constants,
    plan,
    *,
    L0,
    bootstrap_mode="modraise_first",
) -> Cipher
```

Runs a CKKS bootstrap.

Parameters:

- `cipher`: ciphertext to bootstrap.
- `crypto_context`: EasyFHE CKKS context.
- `constants`: `ConstantBundle` returned by `bs.generate(...)`.
- `plan`: `BootstrapPlan` returned by `bs.generate(...)`.
- `L0`: explicit modulus-raise target limb count. In most application code this
  is `cipher.state.cur_limbs`; ResNet AESPA sometimes passes `ctx.L` when
  bootstrapping at a known full-limb point.
- `bootstrap_mode`: route selector. Accepted values:
  - `"modraise_first"` or `"classic"`:
    `ModRaise -> CoeffsToSlots -> EvalMod -> SlotsToCoeffs`
  - `"stc_first"`, `"slots_first"`, or `"s2c_first"`:
    `SlotsToCoeffs -> ModRaise -> CoeffsToSlots -> EvalMod`

Returns:

- `Cipher`: bootstrapped ciphertext with the original slot count, `noise_deg`
  reduced to `1`, and output limbs aligned to `plan.post_bootstrap_levels + 1`
  when possible.

The u64 bootstrap has one canonical output state. It does not expose H/L output
rails: each transform/rescale step drops one physical Q prime and the final
ciphertext is normalized to `noise_deg == 1`.

Runtime requirements:

- The context must contain all rotation keys required by the plan and runtime
  strategy.
- The input ciphertext must have `noise_deg == 1` or be reducible to it.
- `stc_first` needs enough input limbs to run the initial SlotsToCoeffs stage;
  the runtime checks this and raises `ValueError` if the input is too depleted.

## `bs.describe_plan`

```python
bs.describe_plan(plan) -> str
```

Returns a human-readable summary of the EvalMod polynomial/evaluation plan and
bootstrap scalar layout. This is useful for debugging and profiling.

Parameters:

- `plan`: `BootstrapPlan`.

Returns:

- `str`: multi-line textual summary.

## `BootstrapPlan`

`BootstrapPlan` is an immutable dataclass returned by `bs.generate(...)`.
Application code usually passes it back to `bs.bootstrap(...)`.

Important fields:

- `log_bs_slots: int`
  Base-2 log of bootstrap slots.
- `slots: int`
  Property equal to `1 << log_bs_slots`.
- `level_budget: tuple[int, int]`
  `(c2s_budget, s2c_budget)`.
- `dim1: tuple[int, int]`
  OpenFHE-style BSGS dimension settings for C2S and S2C.
- `baby_step: tuple[int, int]`
  Requested actual BSGS baby-step settings, or `(0, 0)` for defaults.
- `strategy: str`
  Default runtime strategy: `"double_hoist"`, `"normal_giant"`, or
  `"normal_bsgs"`.
- `post_bootstrap_levels: int`
  Requested levels remaining after bootstrap.
- `required_rotations: tuple[int, ...]`
  Rotation indices required for the plan's current strategy.
- `c2s_plan`, `s2c_plan`
  Runtime linear-transform plans.

Compatibility:

- `max_levels_remaining` is a read-only compatibility property that returns
  `post_bootstrap_levels`.

Advanced strategy switching:

```python
from dataclasses import replace

plan_for_bsgs = replace(plan, strategy="normal_bsgs")
out = bs.bootstrap(cipher, ctx, constants, plan_for_bsgs, L0=cipher.state.cur_limbs)
```

Only use this when the context has the rotation keys required by the selected
strategy. A context generated with `strategy="double_hoist"` keys can run all
three current strategies.

## Mode And Strategy Matrix

`bootstrap_mode` and `strategy` are independent:

- `bootstrap_mode` chooses the high-level bootstrap route:
  `modraise_first` versus `stc_first`.
- `strategy` chooses the BSGS/hoisting implementation used inside the C2S/S2C
  linear transforms.

The supported matrix is:

| bootstrap mode | double_hoist | normal_giant | normal_bsgs |
| --- | --- | --- | --- |
| modraise_first | supported | supported | supported |
| stc_first | supported | supported | supported |

The same generated constants can serve the full matrix. For one reusable
context, generate rotation keys with `strategy="double_hoist"`.

## Errors

Common errors:

- `ValueError("generate requires post_bootstrap_levels ...")`
  `bs.generate(...)` needs to know the requested output level budget.
- `NotImplementedError("... linear-transform route ...")`
  One of the level-budget entries is `1`. This route is not currently
  implemented.
- `ValueError("bootstrap L0 must be explicit")`
  Pass `L0=...` to `bs.bootstrap(...)`.
- `ValueError("stc_first bootstrap needs at least ... limbs ...")`
  The input ciphertext is too depleted for the initial SlotsToCoeffs step.
