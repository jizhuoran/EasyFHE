from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:
    from .paths import ensure_repo_on_path
except ImportError:
    from paths import ensure_repo_on_path

ensure_repo_on_path()

import easyfhe.fhe as fhe
import easyfhe.bs.openfhe as bs
from easyfhe.fhe.ops.encoding import encode_stage1, encode_stage2


KEYSWITCH_LIKE_OPS = {"mul", "square", "rotate", "fast_rotate", "bootstrap"}
CONTEXT_CACHE_MODES = {"persistent", "transient", "memory"}

MIN_SUPPORTED_CUR_LIMBS = {
    "encode": 1,
    "add": 1,
    "add_pt": 1,
    "mul_pt": 2,
    "mul": 2,
    "rescale": 2,
    "force_rescale": 2,
    "drop_last_elements": 2,
    "modup_to_ext": 1,
    "moddown_from_ext": 1,
    "square": 2,
    "rotate": 1,
    "fast_rotate": 1,
    "bootstrap": 2,
}


@dataclass
class BenchmarkCase:
    op: str
    cur_limbs: int
    dnum: int
    slots: int
    ct_size: int
    extra: Dict[str, Any]


def parse_ops(values: Iterable[str]) -> List[str]:
    ops: List[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                ops.append(token)
    return ops


def case_id(case: BenchmarkCase) -> str:
    extra = json.dumps(case.extra, sort_keys=True, separators=(",", ":"))
    return (
        f"{case.op}|cur={int(case.cur_limbs)}|dnum={int(case.dnum)}|"
        f"slots={int(case.slots)}|ct={int(case.ct_size)}|extra={extra}"
    )


def case_manifest_row(case: BenchmarkCase) -> Dict[str, Any]:
    return {
        "case_id": case_id(case),
        "op": case.op,
        "cur_limbs": int(case.cur_limbs),
        "dnum": int(case.dnum),
        "slots": int(case.slots),
        "ct_size": int(case.ct_size),
        "extra": dict(case.extra),
    }


def result_is_complete(result: Dict[str, Any], *, rerun_failed: bool) -> bool:
    status = str(result.get("status", ""))
    if status in {"ok", "unsupported"}:
        return True
    if status == "failed" and not rerun_failed:
        return True
    return False


def load_completed_results(output_json: Path, *, rerun_failed: bool) -> Dict[str, Dict[str, Any]]:
    if not output_json.exists():
        return {}
    try:
        data = json.loads(output_json.read_text())
    except Exception:
        return {}

    completed: Dict[str, Dict[str, Any]] = {}
    for result in data.get("results", []):
        result_case_id = result.get("case_id")
        if not result_case_id:
            continue
        if result_is_complete(result, rerun_failed=rerun_failed):
            completed[str(result_case_id)] = result
    return completed


def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"ok": 0, "unsupported": 0, "failed": 0}
    for result in results:
        status = str(result.get("status", "failed"))
        if status not in summary:
            status = "failed"
        summary[status] += 1
    return summary


def vector_values(slots: int) -> List[float]:
    base = [0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0]
    return [base[index % len(base)] for index in range(slots)]


class ContextCache:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._cache: Dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._cache_mode = str(getattr(args, "context_cache_mode", "persistent"))
        if self._cache_mode not in CONTEXT_CACHE_MODES:
            raise ValueError(
                f"Unsupported context cache mode: {self._cache_mode}. "
                f"Expected one of {sorted(CONTEXT_CACHE_MODES)}"
            )
        self._temp_cache_root: tempfile.TemporaryDirectory[str] | None = None
        if self._cache_mode == "memory":
            self._temp_cache_root = tempfile.TemporaryDirectory(prefix="easyfhe_ctx_")

    def _base_cache_root(self) -> Path:
        if self._cache_mode == "persistent":
            return Path(self.args.save_dir)
        if self._cache_mode == "transient":
            root = getattr(self.args, "context_cache_root", None)
            if root:
                return Path(root)
            output_json = getattr(self.args, "output_json", None)
            if output_json:
                return Path(output_json).resolve().parent / ".context_cache"
            return Path(self.args.save_dir) / ".context_cache"
        assert self._temp_cache_root is not None
        return Path(self._temp_cache_root.name)

    def get(
        self,
        dnum: int,
        need_bootstrap: bool = False,
        app_rot_indices: List[int] | None = None,
        log_bs_slots_list: List[int] | None = None,
        level_budget_list: List[List[int]] | None = None,
    ):
        app_rot_indices = [] if app_rot_indices is None else [int(value) for value in app_rot_indices]
        log_bs_slots_list = [] if log_bs_slots_list is None else [int(value) for value in log_bs_slots_list]
        level_budget_list = [] if level_budget_list is None else [[int(item) for item in value] for value in level_budget_list]
        key = (
            int(dnum),
            bool(need_bootstrap),
            tuple(app_rot_indices),
            tuple(log_bs_slots_list),
            tuple(tuple(item) for item in level_budget_list),
        )
        if key in self._cache:
            return self._cache[key]

        save_dir = self._base_cache_root()
        if app_rot_indices:
            rot_sig = "_".join(str(value) for value in sorted(set(app_rot_indices)))
            save_dir = save_dir / f"with_rotkeys_{rot_sig}"
        save_dir.mkdir(parents=True, exist_ok=True)

        options = fhe.RuntimeOptions(
            auto_load_keys=bool(need_bootstrap or app_rot_indices),
            rotation_random_mode=str(getattr(self.args, "rotation_random_mode", "reuse_by_shape")),
            rotation_key_limb_limits=getattr(self.args, "rotation_key_limb_limits", {}) or {},
        )
        bootstrap_depth = bs.depth(
            log_bs_slots=log_bs_slots_list,
            level_budget=level_budget_list,
            secret_key_dist=str(self.args.secret_key_dist),
        )
        bootstrap_rotations = (
            bs.plan_rot_keys(
                log_n=int(self.args.logN),
                log_bs_slots=log_bs_slots_list,
                level_budget=level_budget_list,
                secret_key_dist=str(self.args.secret_key_dist),
            )
            if need_bootstrap
            else ()
        )
        rotations = tuple(dict.fromkeys([*app_rot_indices, *bootstrap_rotations]))
        client, crypto_context = fhe.generate_client_context(
            fhe.CKKSContextSpec(
                depth=int(self.args.max_levels_remaining) + bootstrap_depth,
                log_n=int(self.args.logN),
                dnum=int(dnum),
                dcrt_bits=int(self.args.dcrt_bits),
                first_mod=int(self.args.first_mod),
                secret_key_dist=str(self.args.secret_key_dist),
                scale_mode=str(self.args.scale_mode),
                rescale_policy=str(self.args.rescale_policy),
                rotations=rotations,
            ),
            device=str(self.args.device),
            options=options,
        )
        bootstrap_constants = {}
        for log_bs_slots, level_budget in zip(log_bs_slots_list, level_budget_list):
            constants, plan = bs.generate(
                crypto_context,
                log_bs_slots=int(log_bs_slots),
                level_budget=[int(x) for x in level_budget],
                max_levels_remaining=int(self.args.max_levels_remaining),
            )
            bootstrap_constants[(int(log_bs_slots), tuple(int(x) for x in level_budget))] = (
                constants,
                plan,
            )
        crypto_context.benchmark_bootstrap_constants = bootstrap_constants
        crypto_context.maxLevelsRemaining = int(self.args.max_levels_remaining)
        crypto_context.DIRECT_LOAD = False
        crypto_context.LOAD_CHECKPOINT = False
        crypto_context.weight_path = ""
        crypto_context.pre_encode_type = None
        self._cache[key] = (client, crypto_context)
        return self._cache[key]


def level_for_cur_limbs(crypto_context: Any, cur_limbs: int) -> int:
    return int(crypto_context.L) - int(cur_limbs)


def mark_unsupported(case: BenchmarkCase, reason: str) -> Dict[str, Any]:
    return {
        "op": case.op,
        "status": "unsupported",
        "reason": reason,
        "cur_limbs": case.cur_limbs,
        "dnum": case.dnum,
        "slots": case.slots,
        "ct_size": case.ct_size,
        "extra": case.extra,
    }


def validate_case_support(case: BenchmarkCase) -> str | None:
    min_cur_limbs = int(MIN_SUPPORTED_CUR_LIMBS.get(case.op, 1))
    if case.cur_limbs < min_cur_limbs:
        return f"{case.op} requires cur_limbs >= {min_cur_limbs}"
    return None


def make_cipher(openfhe_context: Any, crypto_context: Any, values: List[float], cur_limbs: int, slots: int):
    level = level_for_cur_limbs(crypto_context, cur_limbs)
    if level < 0:
        raise ValueError(f"cur_limbs={cur_limbs} exceeds context L={crypto_context.L}")
    return openfhe_context.encrypt(values, device=crypto_context.device, scale_deg=1, level=level, slots=slots)


def make_plain_from_middle(crypto_context: Any, values: List[float], cur_limbs: int, slots: int, name: str):
    level = level_for_cur_limbs(crypto_context, cur_limbs)
    if level < 0:
        raise ValueError(f"cur_limbs={cur_limbs} exceeds context L={crypto_context.L}")
    middle_value = encode_stage1(np.asarray(values, dtype=np.float64), slots, crypto_context.N)
    return encode_stage2(
        middle_value,
        level=level,
        slots=slots,
        is_ext=False,
        cryptoContext=crypto_context,
    )


def case_context_kwargs(case: BenchmarkCase) -> Dict[str, Any]:
    if case.op == "bootstrap":
        return {
            "need_bootstrap": True,
            "app_rot_indices": [],
            "log_bs_slots_list": [int(case.extra["logBsSlots"])],
            "level_budget_list": [[int(x) for x in case.extra["level_budget"]]],
        }
    if case.op in {"rotate", "fast_rotate"}:
        return {
            "need_bootstrap": False,
            "app_rot_indices": [int(case.extra["step"])],
            "log_bs_slots_list": [],
            "level_budget_list": [],
        }
    return {
        "need_bootstrap": False,
        "app_rot_indices": [],
        "log_bs_slots_list": [],
        "level_budget_list": [],
    }
