#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import glob
import hashlib
import io
import os
from pathlib import Path
import tempfile
import zipfile


PYTHON_SUFFIXES = {".py", ".pyi"}


def _hash_record(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}"


def _record_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _version_from_env(repo: Path, cuda_flavor: str) -> str:
    version = os.environ.get("PYTORCH_BUILD_VERSION")
    if version:
        return version
    return f"{(repo / 'version.txt').read_text(encoding='utf-8').strip()}+{cuda_flavor}"


def _find_wheel(wheelhouse: Path, version: str, python_tag: str, plat: str) -> Path:
    pattern = wheelhouse / f"easyfhe-{version}-{python_tag}-{plat}.whl"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(f"no existing wheel matched {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple wheels matched {pattern}: {matches}")
    return Path(matches[0])


def _replaceable_repo_file(repo: Path, wheel_name: str) -> Path | None:
    path = Path(wheel_name)
    if path.suffix not in PYTHON_SUFFIXES:
        return None
    if not path.parts or path.parts[0] != "easyfhe":
        return None
    repo_path = repo / path
    if repo_path.is_file():
        return repo_path
    return None


def _copy_info(info: zipfile.ZipInfo, filename: str | None = None) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(filename or info.filename, date_time=info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.compress_type = info.compress_type
    return copied


def repack_python(wheel: Path, repo: Path) -> tuple[int, Path]:
    replaced = 0
    records: list[tuple[str, str, str]] = []
    original_stat = wheel.stat()
    fd, tmp_name = tempfile.mkstemp(prefix=wheel.name + ".", suffix=".tmp", dir=str(wheel.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with zipfile.ZipFile(wheel, "r") as zin, zipfile.ZipFile(tmp_path, "w") as zout:
            infos = zin.infolist()
            record_name = next((info.filename for info in infos if info.filename.endswith(".dist-info/RECORD")), None)
            if record_name is None:
                raise RuntimeError(f"{wheel} does not contain a dist-info RECORD")

            for info in infos:
                if info.filename == record_name:
                    continue

                repo_file = _replaceable_repo_file(repo, info.filename)
                if repo_file is None:
                    data = zin.read(info.filename)
                    out_info = _copy_info(info)
                else:
                    data = repo_file.read_bytes()
                    out_info = _copy_info(info)
                    replaced += 1

                zout.writestr(out_info, data)
                if not info.is_dir():
                    records.append((info.filename, _hash_record(data), str(len(data))))

            records.append((record_name, "", ""))
            record_data = _record_bytes(records)
            record_info = zipfile.ZipInfo(record_name)
            record_info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(record_info, record_data)

        os.chmod(tmp_path, original_stat.st_mode & 0o7777)
        try:
            os.chown(tmp_path, original_stat.st_uid, original_stat.st_gid)
        except PermissionError:
            pass
        except AttributeError:
            pass
        os.replace(tmp_path, wheel)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return replaced, wheel


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Python sources inside an existing EasyFHE wheel.")
    parser.add_argument("--repo", default="/io")
    parser.add_argument("--wheelhouse", default="/wheelhouse")
    parser.add_argument("--python-tag", required=True)
    parser.add_argument("--cuda-flavor", required=True)
    parser.add_argument("--plat", required=True)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    repo = Path(args.repo)
    wheelhouse = Path(args.wheelhouse)
    version = args.version or _version_from_env(repo, args.cuda_flavor)
    wheel = _find_wheel(wheelhouse, version, args.python_tag, args.plat)
    replaced, wheel = repack_python(wheel, repo)
    print(f"Repacked Python sources in {wheel}")
    print(f"Replaced files: {replaced}")


if __name__ == "__main__":
    main()
