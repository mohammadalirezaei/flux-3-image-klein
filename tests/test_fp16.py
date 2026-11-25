import unittest

import torch

from flux3.dit import Block, packed_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MixedFP16Test(unittest.TestCase):
    @torch.inference_mode()
    def test_mlp_projection_returns_finite_fp32(self):
        projection = Block(8, 1, 3.0).mlp_out.to(device="cuda", dtype=torch.float16)
        projection.weight.fill_(64)
        inputs = torch.full((8, 24), 64, device="cuda", dtype=torch.float16)

        result = projection(inputs)

        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full_like(result, 98304))

    @torch.inference_mode()
    def test_packed_attention_accepts_fp16(self):
        inputs = torch.ones((1, 2, 32, 128), device="cuda", dtype=torch.float16)

        result = packed_attention(inputs, inputs, inputs, [(0, 16), (16, 32)])

        torch.testing.assert_close(result, torch.ones((32, 256), device="cuda", dtype=torch.float16))
