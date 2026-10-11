# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for distributed norm 2D post-all-gather row addressing.

The 2D post-all-gather path must advance by a complete global row when a core owns
multiple row tiles.  A shape with tiles_per_core_x > 1 is required to exercise the
boundary; single-row-per-core shapes can pass even with a flat row walk.
"""

import pytest
import torch

import ttnn


pytestmark = pytest.mark.use_module_device


def _reference_rmsnorm(x, weight, eps):
    return torch.nn.functional.rms_norm(
        x.float(),
        (x.shape[-1],),
        weight=weight.float(),
        eps=eps,
    ).to(torch.bfloat16)


def test_rms_norm_2d_post_all_gather_multi_row_stride(device):
    """Use a multi-row shape so post-all-gather crosses tiles_per_core_x > 1."""
    torch.manual_seed(56908)
    seq_len = 512
    hidden_dim = 1024
    eps = 1e-5

    torch_input = torch.arange(seq_len * hidden_dim, dtype=torch.float32).reshape(1, 1, seq_len, hidden_dim)
    torch_input = ((torch_input % 257) - 128).to(torch.bfloat16) / 64
    torch_weight = torch.linspace(0.5, 1.5, hidden_dim, dtype=torch.float32).to(torch.bfloat16)

    grid = device.compute_with_storage_grid_size()
    num_tile_rows = seq_len // 32
    cores_x = min(grid.y, num_tile_rows)
    while num_tile_rows % cores_x != 0 and cores_x > 1:
        cores_x -= 1
    tiles_per_core_x = num_tile_rows // cores_x
    assert tiles_per_core_x > 1, (
        f"shape does not exercise the regression on this device: grid={grid}, "
        f"seq_len={seq_len}, tiles_per_core_x={tiles_per_core_x}"
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weight = ttnn.from_torch(
        torch_weight.reshape(1, 1, 1, hidden_dim),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    stats = ttnn.rms_norm_pre_all_gather(
        tt_input,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        use_2d_core_grid=True,
    )
    output = ttnn.rms_norm_post_all_gather(
        tt_input,
        stats,
        epsilon=eps,
        weight=tt_weight,
        compute_kernel_config=compute_kernel_config,
        use_2d_core_grid=True,
    )

    actual = ttnn.to_torch(output)
    expected = _reference_rmsnorm(torch_input, torch_weight, eps)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
