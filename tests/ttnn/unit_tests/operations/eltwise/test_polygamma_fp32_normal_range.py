# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


pytestmark = pytest.mark.use_module_device


# These points sit just inside the normal-fp32 result range for the higher
# orders that were previously untested.  On the unfixed compensated-scale
# formulation the intermediate sum is n! times smaller than the answer and
# flushes to zero on Tensix, even though the final answer is still normal.
@pytest.mark.parametrize(
    "order,x",
    [
        (7, 200_000.0),
        (8, 50_000.0),
        (9, 15_000.0),
        (10, 6_000.0),
        (11, 2_500.0),
    ],
)
def test_polygamma_fp32_preserves_normal_results_at_large_x(device, order, x):
    torch_x = torch.tensor([[x]], dtype=torch.float32)
    reference = torch.polygamma(order, torch_x.double()).float()

    tt_x = ttnn.from_torch(
        torch_x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    actual = ttnn.to_torch(ttnn.polygamma(tt_x, order)).float()

    expected = reference.item()
    observed = actual.item()

    # The acceptance boundary is deliberately about representability first:
    # a normal FP32 reference must never collapse to zero on the device.
    assert abs(expected) >= torch.finfo(torch.float32).tiny
    assert observed != 0.0, (
        f"polygamma(order={order}, x={x}) flushed a normal FP32 result to zero: "
        f"reference={expected:.9e}"
    )
    assert torch.isfinite(actual).all()

    # The sign is part of the factorial-scaled definition and catches a
    # partial fix that only restores magnitude.
    assert torch.sign(actual).item() == torch.sign(reference).item()

    # Keep this regression independent of a single exact polynomial fit while
    # still requiring the repaired result to be numerically useful.
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=1e-38)
