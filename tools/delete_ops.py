#!/usr/bin/env python3
"""Delete ops from native_functions.yaml and derivatives.yaml based on delete_ops.txt."""

import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
ROOT_DIR = TOOLS_DIR.parent
NATIVE_FUNCS = ROOT_DIR / "aten/src/ATen/native/native_functions.yaml"
DERIVATIVES = ROOT_DIR / "tools/autograd/derivatives.yaml"
DELETE_OPS_FILE = TOOLS_DIR / "delete_ops.txt"


def load_delete_ops() -> set[str]:
    with open(DELETE_OPS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def extract_func_name(line: str) -> str | None:
    m = re.match(r"- func:\s+(\w+)", line.strip())
    if m:
        return m.group(1)
    return None


def extract_deriv_name(line: str) -> str | None:
    m = re.match(r"- name:\s+(\w+)", line.strip())
    if m:
        return m.group(1)
    return None


def delete_from_native_functions(delete_ops: set[str]):
    """Remove op entries from native_functions.yaml."""
    with open(NATIVE_FUNCS) as f:
        lines = f.readlines()

    output = []
    i = 0
    deleted_count = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("- func:"):
            name = extract_func_name(stripped)
            if name and name in delete_ops:
                # Skip this entire op entry (until next "- func:" or end of file)
                # Also skip preceding comment lines that belong to this op
                # Remove trailing blank lines from output that were before this op
                while output and output[-1].strip() == "":
                    output.pop()
                # Also remove preceding comment block
                while output and output[-1].strip().startswith("#"):
                    output.pop()

                i += 1
                # Skip all continuation lines of this entry
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    # New entry starts with "- func:" or is a blank line followed by "- func:"
                    if next_stripped.startswith("- func:"):
                        break
                    if next_stripped == "" and i + 1 < len(lines) and lines[i + 1].strip().startswith("- func:"):
                        break
                    if next_stripped.startswith("- func:"):
                        break
                    # A non-indented non-empty line that's not a continuation
                    if next_stripped and not next_stripped.startswith("#") and not next_line.startswith(" ") and not next_line.startswith("\t"):
                        if next_stripped.startswith("- func:"):
                            break
                    i += 1
                deleted_count += 1
                continue
            else:
                output.append(line)
                i += 1
        else:
            output.append(line)
            i += 1

    with open(NATIVE_FUNCS, "w") as f:
        f.writelines(output)

    print(f"native_functions.yaml: deleted {deleted_count} ops")


def delete_from_derivatives(delete_ops: set[str]):
    """Remove entries from derivatives.yaml."""
    with open(DERIVATIVES) as f:
        lines = f.readlines()

    output = []
    i = 0
    deleted_count = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("- name:"):
            name = extract_deriv_name(stripped)
            if name and name in delete_ops:
                # Remove preceding blank/comment lines
                while output and output[-1].strip() == "":
                    output.pop()
                while output and output[-1].strip().startswith("#"):
                    output.pop()

                i += 1
                # Skip continuation lines
                while i < len(lines):
                    next_stripped = lines[i].strip()
                    if next_stripped.startswith("- name:"):
                        break
                    if next_stripped == "" and i + 1 < len(lines) and lines[i + 1].strip().startswith("- name:"):
                        break
                    i += 1
                deleted_count += 1
                continue
            else:
                output.append(line)
                i += 1
        else:
            output.append(line)
            i += 1

    with open(DERIVATIVES, "w") as f:
        f.writelines(output)

    print(f"derivatives.yaml: deleted {deleted_count} entries")


def main():
    delete_ops = load_delete_ops()
    print(f"Loaded {len(delete_ops)} ops to delete")

    delete_from_native_functions(delete_ops)
    delete_from_derivatives(delete_ops)
    print("Done.")


if __name__ == "__main__":
    main()
