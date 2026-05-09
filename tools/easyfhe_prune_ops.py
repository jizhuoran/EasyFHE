#!/usr/bin/env python3
"""Prune native_functions.yaml and derivatives.yaml for EasyFHE.

The default mode is a dry run. Pass --apply to write the pruned YAML files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE_YAML = ROOT / "aten/src/ATen/native/native_functions.yaml"
TS_NATIVE_YAML = ROOT / "aten/src/ATen/native/ts_native_functions.yaml"
DERIVATIVES_YAML = ROOT / "tools/autograd/derivatives.yaml"
DEFAULT_DELETE_LIST = ROOT / "tools/delete_ops.txt"

FHE_OPS = {
    "add_mod",
    "add_scalar_mod",
    "add_pt_broadcast",
    "add_pt_pairwise",
    "automorphism_transform",
    "batched_pairwise_mac",
    "cpmul_broadcast_pt",
    "drop_last_element_and_scale",
    "encode",
    "encrypt",
    "extend_ciphertext",
    "hash_tensor",
    "innerproduct",
    "innerproduct_broadcast_cipher",
    "mod_raise",
    "moddown",
    "modup",
    "mul_mod",
    "pre_encode",
    "sub_mod",
    "sub_scalar_mod",
}


@dataclass(frozen=True)
class Block:
    start: int
    end: int
    name: str
    base: str


def base_name(name: str) -> str:
    return name.split(".", 1)[0]


def extract_native_name(line: str) -> str | None:
    match = re.match(r"\s*- func:\s+([^\(\s]+)", line)
    return match.group(1) if match else None


def extract_derivative_name(line: str) -> str | None:
    match = re.match(r"\s*- name:\s+([^\(\s]+)", line)
    return match.group(1) if match else None


def extract_ts_list_name(line: str) -> str | None:
    match = re.match(r"\s*-\s+([^#\s]+)", line)
    return match.group(1) if match else None


def entry_blocks(lines: list[str], extractor) -> list[Block]:
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        name = extractor(line)
        if name is not None:
            starts.append((index, name))

    blocks: list[Block] = []
    for pos, (start, name) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        blocks.append(Block(start=start, end=end, name=name, base=base_name(name)))
    return blocks


def should_delete(block: Block, delete_specs: set[str]) -> bool:
    return block.name in delete_specs or block.base in delete_specs


def prune_lines(lines: list[str], blocks: list[Block], delete_specs: set[str]) -> tuple[list[str], list[Block]]:
    deleted: list[Block] = []
    output: list[str] = []
    cursor = 0

    for block in blocks:
        if not should_delete(block, delete_specs):
            continue
        output.extend(lines[cursor:block.start])
        while output and output[-1].strip() == "":
            output.pop()
        while output and output[-1].lstrip().startswith("#"):
            output.pop()
        cursor = block.end
        deleted.append(block)

    output.extend(lines[cursor:])
    return output, deleted


def load_delete_list(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def validate_derivatives(native_blocks: list[Block], derivative_blocks: list[Block], delete_specs: set[str]) -> list[str]:
    kept_native = {block.base for block in native_blocks if not should_delete(block, delete_specs)}
    problems = []
    for block in derivative_blocks:
        if not should_delete(block, delete_specs) and block.base not in kept_native:
            problems.append(block.name)
    return sorted(set(problems))


def ts_non_native_bases(lines: list[str]) -> set[str]:
    bases: set[str] = set()
    in_non_native = False
    for line in lines:
        if re.match(r"\w", line):
            in_non_native = line.startswith("non_native:")
        if not in_non_native:
            continue
        match = re.match(r"\s*- func:\s+([^\(\s]+)", line)
        if match:
            bases.add(base_name(match.group(1)))
    return bases


def prune_ts_lines(
    lines: list[str],
    kept_native_bases: set[str],
    delete_specs: set[str],
) -> tuple[list[str], list[str]]:
    non_native_bases = ts_non_native_bases(lines)
    output: list[str] = []
    deleted: list[str] = []
    in_non_native = False

    for line in lines:
        if re.match(r"\w", line):
            in_non_native = line.startswith("non_native:")
        name = None if in_non_native else extract_ts_list_name(line)
        if name is None:
            output.append(line)
            continue
        base = base_name(name)
        if name in delete_specs or base in delete_specs or (base not in kept_native_bases and base not in non_native_bases):
            deleted.append(name)
            continue
        output.append(line)

    return output, deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete-list", type=Path, default=DEFAULT_DELETE_LIST)
    parser.add_argument("--native-yaml", type=Path, default=NATIVE_YAML)
    parser.add_argument("--ts-native-yaml", type=Path, default=TS_NATIVE_YAML)
    parser.add_argument("--derivatives-yaml", type=Path, default=DERIVATIVES_YAML)
    parser.add_argument("--apply", action="store_true", help="write pruned YAML files")
    parser.add_argument("--allow-delete-fhe", action="store_true")
    args = parser.parse_args()

    delete_specs = load_delete_list(args.delete_list)
    fhe_deletes = sorted(delete_specs & FHE_OPS)
    if fhe_deletes and not args.allow_delete_fhe:
        raise SystemExit(f"Refusing to delete FHE ops: {', '.join(fhe_deletes)}")

    native_lines = args.native_yaml.read_text().splitlines(keepends=True)
    ts_native_lines = args.ts_native_yaml.read_text().splitlines(keepends=True)
    derivative_lines = args.derivatives_yaml.read_text().splitlines(keepends=True)
    native_blocks = entry_blocks(native_lines, extract_native_name)
    derivative_blocks = entry_blocks(derivative_lines, extract_derivative_name)

    dangling = validate_derivatives(native_blocks, derivative_blocks, delete_specs)
    if dangling:
        raise SystemExit(
            "derivatives.yaml has entries whose native op would be missing: "
            + ", ".join(dangling[:40])
            + (" ..." if len(dangling) > 40 else "")
        )

    pruned_native, deleted_native = prune_lines(native_lines, native_blocks, delete_specs)
    pruned_derivatives, deleted_derivatives = prune_lines(derivative_lines, derivative_blocks, delete_specs)
    kept_native_bases = {block.base for block in native_blocks if not should_delete(block, delete_specs)}
    pruned_ts_native, deleted_ts_native = prune_ts_lines(ts_native_lines, kept_native_bases, delete_specs)

    print(f"delete list entries: {len(delete_specs)}")
    print(f"native schemas: {len(native_blocks)} -> {len(native_blocks) - len(deleted_native)}")
    print(f"native deleted schemas: {len(deleted_native)}")
    print(f"ts native entries deleted: {len(deleted_ts_native)}")
    print(f"derivative entries: {len(derivative_blocks)} -> {len(derivative_blocks) - len(deleted_derivatives)}")
    print(f"derivative deleted entries: {len(deleted_derivatives)}")

    deleted_bases = sorted({block.base for block in deleted_native})
    if deleted_bases:
        print("deleted native bases:")
        for name in deleted_bases:
            print(f"  {name}")

    if args.apply:
        args.native_yaml.write_text("".join(pruned_native))
        args.ts_native_yaml.write_text("".join(pruned_ts_native))
        args.derivatives_yaml.write_text("".join(pruned_derivatives))
        print("wrote pruned YAML files")
    else:
        print("dry run only; pass --apply to write files")


if __name__ == "__main__":
    main()
