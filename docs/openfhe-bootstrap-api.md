# OpenFHE Bootstrap API

Import the public u64 CKKS bootstrap API from the package root:

```python
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe
```

Generation and runtime modules below `easyfhe.bs.openfhe` are implementation
details. Applications describe a bootstrap once, derive its requirements, bind
it to a context, and execute the resulting program.

## Typical flow

```python
bootstrap_spec = bs.BootstrapSpec(
    log_slots=14,
    level_budget=(4, 4),
    output_levels=12,
    strategy="double_hoist",
    mode="modraise_first",
)

requirements = bs.requirements(
    bootstrap_spec,
    log_n=16,
    secret_key_dist="SPARSE_TERNARY",
)

client, ctx = fhe.generate_client_context(
    fhe.CKKSContextSpec(
        depth=requirements.context_depth,
        log_n=16,
        dnum=3,
        dcrt_bits=59,
        first_mod=60,
        secret_key_dist="SPARSE_TERNARY",
        scale_mode="flexible",
        rotations=requirements.rotations,
    ),
    device="cuda",
)

program = bs.generate(ctx, bootstrap_spec)
cipher = client.encrypt(values, slots=bootstrap_spec.slots)
boot = bs.bootstrap(cipher, ctx, program)
```

The public symbols are `BootstrapSpec`, `BootstrapRequirements`,
`BootstrapProgram`, `requirements`, `generate`, `bootstrap`, and
`describe_plan`.

## `BootstrapSpec`

```python
bs.BootstrapSpec(
    log_slots: int,
    level_budget: tuple[int, int],
    output_levels: int,
    strategy: str = "double_hoist",
    mode: str = "modraise_first",
    dim1: tuple[int, int] | None = None,
    baby_step: tuple[int, int] | None = None,
    raise_to_limbs: int | None = None,
)
```

- `log_slots` is the base-2 logarithm of the bootstrap slot count.
- `level_budget` assigns collapsed-FFT levels to CoeffsToSlots and
  SlotsToCoeffs. Both entries must be greater than one.
- `output_levels` is the number of usable multiplication levels after the
  bootstrap. The output has `output_levels + 1` u64 Q primes and
  `noise_deg == 1`.
- `strategy` is exactly `double_hoist`, `normal_giant`, or `normal_bsgs`.
- `mode` is exactly `modraise_first` or `stc_first`. Historical aliases are not
  accepted.
- `dim1` and `baby_step` are alternative C2S/S2C BSGS overrides. Do not set
  both.
- `raise_to_limbs` optionally fixes the modulus-raise target. When omitted,
  `generate` binds it to `ctx.max_limbs`.

The derived `slots` property is `1 << log_slots`.

## `requirements`

```python
bs.requirements(
    spec_or_specs,
    *,
    log_n,
    secret_key_dist="SPARSE_TERNARY",
) -> BootstrapRequirements
```

For one spec, the result contains:

- `bootstrap_depth`: depth consumed by C2S, EvalMod, and S2C;
- `context_depth`: minimum `CKKSContextSpec.depth`, including requested output
  levels and an explicit raise target when present;
- `rotations`: unique rotation indices required by the selected strategy.

A sequence of specs is also accepted. Requirements are merged by taking the
maximum needed depth and the ordered union of rotation keys. Generate one
`BootstrapProgram` per spec after creating the shared context.

## `generate`

```python
program = bs.generate(ctx, bootstrap_spec)
```

`generate` resolves the raise target, validates context capacity and rotation
keys, generates constants, and returns an immutable `BootstrapProgram`.
Constants and the low-level execution plan are intentionally bundled so they
cannot be paired incorrectly.

Useful program fields are:

- `program.spec`;
- `program.raise_to_limbs`;
- `program.output_state`.

`program.output_state` is the exact state promised by `bootstrap`: its limb
count is `spec.output_levels + 1`, its noise degree is one, and its scale is
bound to the generated raise target.

## `bootstrap`

```python
boot = bs.bootstrap(cipher, ctx, program)
```

The program owns the runtime mode and raise target. Runtime calls therefore do
not accept `L0`, constants, a separate plan, or mode overrides. The call checks
that:

- the program was generated for this context;
- the input limb count does not exceed `program.raise_to_limbs`;
- the input slot count does not exceed `program.spec.slots`.

`modraise_first` runs:

```text
ModRaise -> CoeffsToSlots -> EvalMod -> SlotsToCoeffs
```

`stc_first` runs:

```text
SlotsToCoeffs -> ModRaise -> CoeffsToSlots -> EvalMod
```

The latter requires enough input limbs for its initial SlotsToCoeffs stage.

## Lower raise targets

A lower target is useful when a larger context also serves computation before
the bootstrap. Bind it in the spec, not at runtime:

```python
spec = bs.BootstrapSpec(
    log_slots=14,
    level_budget=(4, 4),
    output_levels=2,
    raise_to_limbs=21,
)
requirements = bs.requirements(spec, log_n=16)
program = bs.generate(ctx, spec)
```

The target must leave enough distance for the bootstrap depth and requested
output levels. Constants are generated specifically for that target.

## Plan inspection

```python
print(bs.describe_plan(program))
```

This renders the internal EvalMod schedule for diagnostics. Application code
should not depend on the private runtime-plan fields.

## u64 level semantics

The current frontend uses one physical u64 Q prime per limb. Every rescale
drops exactly one prime. Bootstrap never exposes H/L rails, composite limbs,
drop counts, or backend-width switches in its public API.
