# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Known-code and block-boundary checks for the calibration source decoder."""

import pytest
import torch
from source import decode_mxfp4


def test_codes_and_block_scales():
    # Two blocks, each cycling every positive and negative code twice.
    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 4], dtype=torch.uint8
    )
    scales = torch.tensor([[127, 128]], dtype=torch.uint8)
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    expected = torch.cat((values.repeat(2), values.repeat(2) * 2))[None, :]
    torch.testing.assert_close(decode_mxfp4(packed, scales), expected, rtol=0, atol=0)


def test_smallest_scale_and_reserved_nan():
    packed = torch.full((1, 16), 0x22, dtype=torch.uint8)
    result = decode_mxfp4(packed, torch.zeros((1, 1), dtype=torch.uint8))
    assert torch.all(result == 2.0**-127)
    with pytest.raises(ValueError, match="NaN"):
        decode_mxfp4(packed, torch.full((1, 1), 255, dtype=torch.uint8))


def test_reject_misaligned_scale_shape():
    with pytest.raises(ValueError, match="shape"):
        decode_mxfp4(
            torch.zeros((2, 32), dtype=torch.uint8),
            torch.zeros((2, 1), dtype=torch.uint8),
        )
