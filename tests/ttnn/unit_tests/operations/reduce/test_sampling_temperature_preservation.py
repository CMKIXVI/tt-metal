# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for sampling temperature semantics.

The random-threshold precision fix in issue #59732 must not change the existing
meaning of ``temp``.  In particular, the threshold should sample from the
softmax distribution after temperature scaling rather than effectively using
the unscaled logits.
"""

import os

import pytest
import torch
import ttnn

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("TT_METAL_SIMULATOR")),
    reason="distribution-level sampling checks need hardware",
)

USERS = 32
DRAWS = int(os.environ.get("SAMPLING_TEMPERATURE_DRAWS", "2000"))


def _sample_two_token(device, temperature):
    values = torch.full((1, 1, USERS, 128), -30.0)
    values[..., 0] = 0.0
    values[..., 1] = -1.0
    values_tt = ttnn.from_torch(values.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    indices_tt = ttnn.from_torch(
        torch.arange(128, dtype=torch.int32).expand(1, 1, USERS, 128),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    k_tt = ttnn.from_torch(
        torch.full((USERS,), 2, dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    p_tt = ttnn.from_torch(
        torch.ones(USERS), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    temp_tt = ttnn.from_torch(
        torch.full((USERS,), temperature), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    seeds = ttnn.from_torch(
        torch.arange(1, USERS + 1, dtype=torch.int32) * 104729,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    user_ids = ttnn.from_torch(
        torch.arange(USERS, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)

    second = 0
    for _ in range(DRAWS):
        out = ttnn.to_torch(ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt)).reshape(-1)[:USERS]
        second += int((out == 1).sum().item())
    return second / (USERS * DRAWS)


def test_sampling_temperature_changes_distribution(device):
    """Temperature scaling must survive the random-threshold precision change.

    For logits [0, -1], the reference probabilities are about 0.119 at T=0.5
    and 0.378 at T=2.0.  The intervals are deliberately wide enough to avoid
    depending on exact device softmax rounding while still detecting a missing
    or inverted temperature factor.
    """
    low_temp = _sample_two_token(device, 0.5)
    high_temp = _sample_two_token(device, 2.0)

    assert 0.08 < low_temp < 0.16, f"T=0.5 produced P(second)={low_temp:.4f}, expected about 0.119"
    assert 0.33 < high_temp < 0.43, f"T=2.0 produced P(second)={high_temp:.4f}, expected about 0.378"
    assert high_temp > low_temp * 2.0, (
        f"temperature scaling is not reflected in sampling: T=0.5={low_temp:.4f}, T=2.0={high_temp:.4f}"
    )
