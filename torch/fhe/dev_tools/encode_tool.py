import functools, os, pickle, atexit, math
import numpy as np
from datetime import datetime
from ..ciphertext import PreEncodeValues
import torch


DATA_DIR = os.environ["DATA_DIR"]


# def _fft_special_inv(vals, M, rotGroup, ksiPows, bitrev):
#     vals_size = len(vals)
#     len_size = vals_size
#     while len_size >= 1:
#         len_h = len_size >> 1
#         len_q = len_size << 2
#         gap = M // len_q
#
#         for i in range(0, vals_size, len_size):
#             for j in range(len_h):
#                 idx = (len_q - (rotGroup[j] % len_q)) * gap
#                 u = vals[i + j] + vals[i + j + len_h]
#                 v = vals[i + j] - vals[i + j + len_h]
#                 v *= ksiPows[idx]
#                 vals[i + j] = u
#                 vals[i + j + len_h] = v
#         len_size >>= 1
#
#     vals[:] = vals[bitrev]
#     vals /= vals_size
#     return vals


def pre_encode(x, slots, cryptoContext):

    inverse = x

    if slots < len(inverse):
        raise ValueError(f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]")

    inverse_complex = torch.pre_encode(
        torch.tensor(inverse, dtype=torch.double),
        slots,
        cryptoContext.M,
        cryptoContext.encode_params_rotGroup,
        cryptoContext.encode_params_ksiPows,
        cryptoContext.encode_bitrev_indices[int(math.log2(slots))]
    )

    inverse_array = np.array(inverse_complex, dtype=np.complex128).view(np.float64)
    inverse_array = inverse_array.reshape(1, -1)
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
            input, name, _, slots, _, cryptoContext = args
            if isinstance(input, list) or isinstance(input, np.ndarray):
                encoded_val = pre_encode(input, slots, cryptoContext)
                middle_encoded_vals[name] = encoded_val
            elif isinstance(input, PreEncodeValues):
                # If input is already a PreEncodeValues, we can skip pre-encoding
                save_val = input.deep_copy()
                save_val.encoded_values = save_val.encoded_values.cpu().numpy()
                middle_encoded_vals[name] = save_val
                # print(input.encoded_values.shape)
            else:
                raise TypeError(f"Unsupported input type: {type(input)}. Expected list, numpy.ndarray, or PreEncodeValues.")
            # assert isinstance(input, list) or isinstance(input, np.ndarray), \
            #     f"Assertion failed: input is of type {type(input)}. " \
            #     "input should be either a list or a numpy.ndarray."
            # encoded_val = pre_encode(input, slots, cryptoContext)
            # middle_encoded_vals[name] = encoded_val
        return func(*args, **kwargs)
    return wrapper

def save_end_encode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ == "encode":
            _, name, _, _, _, _ = args
            if name in end_encoded_vals:
                return end_encoded_vals[name]
        res = func(*args, **kwargs)
        if func.__name__ == "encode":
            input, name, _, slots, _, _ = args
            full_encode = res.deep_copy()
            # full_encode.cv = [full_encode.cv[0].cpu().numpy()]
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
        for key, val in end_encoded_vals.items():
            val.cv = [val.cv[0].cpu().numpy()]
        with open(file_path, "wb") as f:
            pickle.dump(end_encoded_vals, f)





