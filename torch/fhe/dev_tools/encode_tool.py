import functools, os, pickle, atexit
import numpy as np
from datetime import datetime
from ..ciphertext import PreEncodeValues

DATA_DIR = os.environ["DATA_DIR"]


def _fft_special_inv(vals, M, rotGroup, ksiPows):

    def _bit_reverse(vals):
        size = len(vals)
        vals = np.array(vals, dtype=np.complex128)  # 转为 numpy 复数数组
        j = 0
        for i in range(1, size):
            bit = size >> 1
            while j >= bit:
                j -= bit
                bit >>= 1
            j += bit
            if i < j:
                vals[i], vals[j] = vals[j], vals[i]  # 交换复数
        return vals

    vals_size = len(vals)

    # FFT特定的操作
    len_size = vals_size
    while len_size >= 1:
        len_h = len_size >> 1
        len_q = len_size << 2
        gap = M // len_q

        for i in range(0, vals_size, len_size):
            for j in range(len_h):
                idx = (len_q - (rotGroup[j] % len_q)) * gap
                u = vals[i + j] + vals[i + j + len_h]
                v = vals[i + j] - vals[i + j + len_h]
                v *= ksiPows[idx]
                vals[i + j] = u
                vals[i + j + len_h] = v
        len_size >>= 1

    vals = _bit_reverse(vals)

    for i in range(vals_size):
        vals[i] /= vals_size
    return vals

import cmath

N = 1 << 16
M = N << 1
Nh = N >> 1

# compute encode params
M_PI = 3.14159265358979323846
fivePows = 1
encode_params_ksiPows = []
encode_params_rotGroup = []
for i in range(Nh):
    encode_params_rotGroup.append(fivePows)
    fivePows = (fivePows * 5) % M

# m_ksiPows stores the complex roots of unity
for j in range(M):
    angle = 2.0 * M_PI * j / M
    encode_params_ksiPows.append(cmath.exp(1j * angle))
encode_params_ksiPows.append(encode_params_ksiPows[0])

encode_params_ksiPows = np.array(encode_params_ksiPows, dtype=np.complex128).view(np.float64).tolist()
encode_params_rotGroup = np.array(encode_params_rotGroup)

def pre_encode(x, slots):

    N = 1 << 16
    M = N << 1
    Nh = N >> 1

    inverse = x

    if slots < len(inverse):
        raise ValueError(f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]")

    # Clears all imaginary values as CKKS for complex numbers
    inverse_complex = np.array([complex(v.real, 0.0) for v in inverse])

    # Resize the inverse to fit the slot size.
    # note that default: slots value should be greater than size of input data list x
    inverse_complex = np.pad(
        inverse_complex,
        pad_width=(0, slots - len(inverse)),
        mode="constant",
        constant_values=complex(0.0, 0.0),
    )
    arr = np.array(encode_params_ksiPows, dtype=np.float64)
    complex_arr = arr[0::2] + arr[1::2] * 1j
    inverse_complex = _fft_special_inv(
        inverse_complex,
        M,
        np.array(encode_params_rotGroup, dtype=np.int32),
        complex_arr,
    )
    inverse_array = np.array(inverse_complex, dtype=np.complex128).view(np.float64)
    max_encoded_value = np.max(np.abs(inverse_array))

    encoded_val = PreEncodeValues(
        np.pad(
            x,
            pad_width=(0, slots - len(x)),
            mode="constant",
            constant_values=0.0,
        ),
        slots,
        inverse_array,
        max_encoded_value,
    )
    return encoded_val

middle_encoded_vals = {}
end_encoded_vals = {}

def save_middle_encode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ == "encode":
            input, name, _, slots, _ = args
            assert isinstance(input, list) or isinstance(input, np.ndarray)
            encoded_val = pre_encode(input, slots)
            middle_encoded_vals[name] = encoded_val
        return func(*args, **kwargs)
    return wrapper

def save_end_encode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if func.__name__ == "encode":
            input, name, _, slots, _ = args
            full_encode = res.deep_copy()
            full_encode.cv = [full_encode.cv[0].cpu().numpy()]
            end_encoded_vals[name] = full_encode
        return res
    return wrapper

@atexit.register
def save_encoded_vals():
    if len(middle_encoded_vals) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = DATA_DIR + f"/encode_{timestamp}.pkl"
        print("saving pre-encoded vals to {}".format(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(middle_encoded_vals, f)
    
    if len(end_encoded_vals) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = DATA_DIR + f"/full_encode_{timestamp}.pkl"
        print("saving pre-encoded vals to {}".format(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(end_encoded_vals, f)





