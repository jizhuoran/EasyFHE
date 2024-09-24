import unittest

import torch
import numpy as np
from torch.fhe.Ciphertext import Cipher
from torch.fhe.context import Context
from torch.fhe import homo_ops


class TestFHEModule(unittest.TestCase):

    def test_homo_cipher_add(self):
        """Test cipher add."""
        L, K = 2, 1

        ax = torch.tensor(
            [[1, 2, 3, 4], [4, 5, 6, 4]], dtype=torch.uint64, device="cuda"
        )
        bx = torch.tensor(
            [[7, 8, 9, 4], [0, 2, 0, 4]], dtype=torch.uint64, device="cuda"
        )
        a_cipher = Cipher(ax, bx, L)
        
        ax = torch.tensor(
            [[4, 5, 6, 4], [4, 5, 6, 4]], dtype=torch.uint64, device="cuda"
        )
        bx = torch.tensor(
            [[4, 5, 6, 4], [4, 5, 6, 4]], dtype=torch.uint64, device="cuda"
        )
        b_cipher = Cipher(ax, bx, L)

        cryptoContext = Context(2, 53, 52, 52, L, K)
        cryptoContext.moduliQ = torch.tensor(
            [10, 11], dtype=torch.uint64, device="cuda"
        )
        c_cipher = homo_ops.cipher_add(a_cipher, b_cipher, cryptoContext)

        ax_ground_truth = torch.tensor(
            [[5, 7, 9, 8], [8, 10, 1, 8]], dtype=torch.uint64, device="cuda"
        )
        bx_ground_truth = torch.tensor(
            [[1, 3, 5, 8], [4, 7, 6, 8]], dtype=torch.uint64, device="cuda"
        )
        ground_truth = Cipher(ax_ground_truth, bx_ground_truth, L)

        self.assertEqual(c_cipher, ground_truth)


if __name__ == "__main__":
    unittest.main()
