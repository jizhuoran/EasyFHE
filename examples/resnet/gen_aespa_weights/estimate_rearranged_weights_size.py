#!/usr/bin/env python3
import os, sys, numpy as np, mmap, re

# ---------- 可调参数 ----------
# 判断一个字节是否属于典型的数值字符
_NUM_CHARS = set(b"0123456789+-.eE \n\r\t")

def looks_like_ascii(path, sample_bytes=4096):
    """粗判文件前几 KB 是否基本都是数值 ASCII 字符。"""
    with open(path, "rb") as f:
        head = f.read(sample_bytes)
    return all(b in _NUM_CHARS for b in head)

def load_numbers(path, dtype):
    """把 ASCII 数字转成 numpy 数组（float32/float16）。"""
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        txt = f.read()
    # np.fromstring 速度快、内存占用小
    return np.fromstring(txt, sep=' ', dtype=dtype)

def main():

    path = "/home/yhh/PNP/GPU-FHE/examples/resnet/weights_aespa_18"
    data_type = "float32"


    directory = os.path.expanduser(path)
    dtype_str = data_type
    dtype = {"float32": np.float32, "float16": np.float16}.get(dtype_str, np.float32)

    total_orig = total_bin = 0
    fmt = "{:<40s} {:>10s} → {:>10s}  节省 {:>6.1f}%"

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".bin"):
            continue
        path = os.path.join(directory, fname)
        orig_size = os.path.getsize(path)
        total_orig += orig_size

        if looks_like_ascii(path):
            arr = load_numbers(path, dtype)
            bin_size = arr.nbytes
            total_bin += bin_size
            if orig_size < 0.001:
                print(fmt.format(
                    fname,
                    f"{orig_size / 1024:.1f}K",
                    f"{bin_size / 1024:.1f}K",
                    0.0
                ))
            else:
                print(fmt.format(
                    fname,
                    f"{orig_size/1024:.1f}K",
                    f"{bin_size/1024:.1f}K",
                    100*(1 - bin_size/orig_size)
                ))
        else:
            print(f"{fname:<40s} 已是二进制格式，跳过。")

    if total_bin:
        print("\n=== 汇总 ===")
        print(fmt.format(
            "TOTAL",
            f"{total_orig/1024/1024:.2f}M",
            f"{total_bin/1024/1024:.2f}M",
            100*(1 - total_bin/total_orig)
        ))

if __name__ == "__main__":
    main()
