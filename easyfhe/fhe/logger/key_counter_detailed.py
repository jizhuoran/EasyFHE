import functools


# Registry to keep track of function call counts
import pprint
from collections import defaultdict, OrderedDict
from typing import DefaultDict, OrderedDict, Mapping, Literal, Any

# ---------------------------------------------------------------------------
# 1.  The registry itself:  index → limb → Count
# ---------------------------------------------------------------------------
# index → limb → Count
rotKey_registry_detailed: DefaultDict[str, DefaultDict[int, DefaultDict[int, int]]]
rotKey_registry_detailed = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

def key_counter_detailed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ == "homo_rotate":
            limbs = args[0].cur_limbs
            index = args[1]
        elif func.__name__ == "fast_rotate":
            limbs = args[0].cur_limbs
            index = args[1]
        elif func.__name__ == "eval_fast_rotate_and_add_batch":
            limbs = args[0].cur_limbs
            index = args[1]
        else:
            return func(*args, **kwargs)

        cryptoContext = args[-1]


        if isinstance(index, list):
            for ind in index:
                norm_index = cryptoContext.norm_rot_index(ind)
                rotKey_registry_detailed["total"][norm_index][limbs] += 1

                cat = _infer_category(cryptoContext)
                rotKey_registry_detailed[cat][norm_index][limbs] += 1
        else:
            norm_index = cryptoContext.norm_rot_index(index)
            rotKey_registry_detailed["total"][norm_index][limbs] += 1

            cat = _infer_category(cryptoContext)
            rotKey_registry_detailed[cat][norm_index][limbs] += 1

        return func(*args, **kwargs)
    return wrapper




Categories = Literal["C2S", "EvalMod", "S2C", "app"]
def _infer_category(ctx: Any) -> Categories:
    """根据 cryptoContext 判定日志类别"""
    if getattr(ctx, "inBS", False):
        if getattr(ctx, "inC2S", False):
            return "C2S"
        if getattr(ctx, "inEvalMod", False):
            return "EvalMod"
        if getattr(ctx, "inS2C", False):
            return "S2C"
    else:
        # 其余 BS 内步骤暂时也归为 app，可按需扩展
        return "app"



"""
Pretty-print the three-level `call_registry` (Stage → OpName → ComputeAmt → Count)
as a *plain* dict literal that you can copy-paste into another file.

Key features
------------
* Top-level stage order:  total → C2S → EvalMod → S2C
* OpName order:           by total workload  Σ (compute_amt * count)  (descending)
* ComputeAmt order:       configurable
      - "amt"   → ascending by compute amount     (default)
      - "count" → descending by invocation count
"""

def reset_key_counter_detailed() -> None:
    rotKey_registry_detailed.clear()


# ---------------------------------------------------------------------------
# 2.  Helper: convert OrderedDict (and nested mappings) → plain dict
# ---------------------------------------------------------------------------
def _to_plain_dict(obj: Any) -> Any:
    """Recursively turn OrderedDict → dict so that the repr is a literal dict."""
    if isinstance(obj, OrderedDict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, Mapping):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# 3.  prt  —  pretty-print the registry as a Python literal
# ---------------------------------------------------------------------------
def prt_dict_keys_counter_detailed(
    dict_name: str,
    *,
    registry=rotKey_registry_detailed,
    stage_order=("total", "C2S", "EvalMod", "S2C", "app"),
    sort_compute_by: str = "amt",  # "amt" | "count"
) -> None:
    """
    Parameters
    ----------
    dict_name       Variable name that will prefix the printed literal.
    registry        Source mapping; defaults to global `call_registry`.
    stage_order     Desired order of the top-level stages.
    sort_compute_by "amt"   → ComputeAmt ascending
                    "count" → ComputeAmt sorted by count descending
    """

    # ----- assemble snapshot with deterministic order -----
    snapshot = OrderedDict()
    for stage in stage_order:
        if stage in registry:
            stage_dict = OrderedDict()
            for index in sorted(registry[stage]):            # index ascending
                inner = registry[stage][index]
                stage_dict[index] = OrderedDict(sorted(inner.items(), reverse=True)) # limb descending

            snapshot[stage] = OrderedDict(sorted(registry[stage].items()))

    # ----- convert to plain dict & pretty-print -----

    plain = _to_plain_dict(snapshot)
    # first prt name
    print(f"{dict_name} = {{")
    # then prt body, without the first "{" printed above
    body = pprint.pformat(plain, sort_dicts=False, width=120, compact=False)
    print(body[1:])


def prt_MAX_RNS_LIMBS_BY_ROT_EVK(
    dict_name: str,
    *,
    registry=rotKey_registry_detailed,
):
    """
    Pretty-print the maximum RNS limbs by rotation EVK.

    Parameters
    ----------
    dict_name       Variable name that will prefix the printed literal.
    registry        Source mapping; defaults to global `rotKey_registry_detailed`.
    """
    print(f"{dict_name} = {{")
    for index, limbs in sorted(registry["total"].items()):
        print(f"  {index}: {max(limbs)},")
    print("}")
