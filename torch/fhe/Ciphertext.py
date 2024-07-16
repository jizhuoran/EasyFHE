import numpy as np

class Ciphertext:
    def __init__(self, cv, curr_limbs):  # cv for ciphertext vector
        # 确保polynomial是一个numpy数组
        if not isinstance(cv, np.ndarray):
            raise ValueError("Polynomial must be a numpy array")
        # 确保index是一个uint64类型
        if not isinstance(curr_limbs, (int, np.uint64)):
            raise ValueError("curr_limbs must be an integer or uint64")
        if curr_limbs < 0 or curr_limbs > np.iinfo(np.uint64).max:
            raise ValueError("curr_limbs must be in the range of uint64")

        self.cv = cv
        self.curr_limbs = int(curr_limbs)

    def __repr__(self):
        return f"Ciphertext(cv={self.cv}, curr_limbs={self.curr_limbs})"
