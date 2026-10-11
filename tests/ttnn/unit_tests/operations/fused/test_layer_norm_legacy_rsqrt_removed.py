# SPDX-License-Identifier: Apache-2.0

"""Regression guard for removal of the legacy LayerNorm rsqrt compatibility API."""

import ttnn


def test_layer_norm_program_configs_have_no_legacy_rsqrt_api():
    assert not hasattr(ttnn.LayerNormDefaultProgramConfig, "legacy_rsqrt")
    assert not hasattr(ttnn.LayerNormShardedMultiCoreProgramConfig, "legacy_rsqrt")
