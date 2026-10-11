# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


def _assert_cumsum_matches_torch(actual, expected):
    actual = ttnn.to_torch(actual)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))

    inf_mask = torch.isinf(expected)
    if inf_mask.any():
        assert torch.equal(torch.signbit(actual[inf_mask]), torch.signbit(expected[inf_mask]))

    finite_mask = torch.isfinite(expected)
    if finite_mask.any():
        assert torch.allclose(actual[finite_mask], expected[finite_mask], rtol=0, atol=0)


@pytest.mark.parametrize("scan_len", [4, 8, 33, 129])
@pytest.mark.parametrize("dim", [0, -2])
@pytest.mark.parametrize("disable_compensated_sum", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [1.0, float("inf"), 1.0, 1.0],
        [1.0, float("-inf"), 1.0, 1.0],
        [float("inf"), float("-inf"), 1.0, 1.0],
        [float("-inf"), float("inf"), 1.0, 1.0],
        [2.0e38, 2.0e38, -2.0e38, 1.0],
        [2.0e38, 2.0e38, -2.0e38, -2.0e38],
        [1.0, float("nan"), 1.0, 1.0],
    ],
    ids=["pos_inf", "neg_inf", "pos_neg_inf", "neg_pos_inf", "overflow_recovery", "overflow_cancel", "nan"],
)
def test_cumsum_fp32_special_value_transitions_followup(
    scan_len, dim, disable_compensated_sum, values, device
):
    """Exercise special-value transitions, not just a single non-finite insertion."""
    shape = [32, 32]
    shape[dim] = scan_len
    torch_input = torch.ones(shape, dtype=torch.float32)

    index = [slice(None)] * len(shape)
    for offset, value in enumerate(values):
        index[dim] = offset
        torch_input[tuple(index)] = value

    expected = torch.cumsum(torch_input, dim=dim)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, -17.0)
    actual = ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated_sum)

    _assert_cumsum_matches_torch(actual, expected)
