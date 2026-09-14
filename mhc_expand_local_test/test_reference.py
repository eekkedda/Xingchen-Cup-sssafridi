import unittest

import torch

from reference import backward, forward


class MhcExpandReferenceTest(unittest.TestCase):
    def test_forward_supported_shapes_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for s, d, m in ((1, 1, 2), (3, 17, 4), (64, 256, 8)):
                with self.subTest(dtype=dtype, s=s, d=d, m=m):
                    x = torch.randn(s, d, dtype=dtype)
                    actual = forward(x, m)
                    self.assertEqual(tuple(actual.shape), (s, m, d))
                    for replica in range(m):
                        self.assertTrue(torch.equal(actual[:, replica, :], x))

    def test_backward_supported_shapes_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for s, d, m in ((1, 1, 2), (3, 17, 4), (64, 256, 8)):
                with self.subTest(dtype=dtype, s=s, d=d, m=m):
                    o_grad = torch.randn(s, m, d, dtype=dtype)
                    actual = backward(o_grad, m)
                    expected = o_grad.float().sum(dim=1).to(dtype)
                    self.assertTrue(torch.equal(actual, expected))

    def test_invalid_rank_or_multiplier(self):
        with self.assertRaises(ValueError):
            forward(torch.zeros(2, 3, 4), 2)
        with self.assertRaises(ValueError):
            forward(torch.zeros(2, 3), 0)
        with self.assertRaises(ValueError):
            backward(torch.zeros(2, 3), 2)
        with self.assertRaises(ValueError):
            backward(torch.zeros(2, 3, 4), 2)


if __name__ == "__main__":
    unittest.main()
