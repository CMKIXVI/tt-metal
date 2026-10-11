import pytest
import ttnn


@pytest.mark.parametrize("config_type", [ttnn.LayerNormDefaultProgramConfig])
def test_legacy_rsqrt_is_removed_from_layer_norm_config(config_type):
    config = config_type()
    assert not hasattr(config, "legacy_rsqrt")
    with pytest.raises(TypeError):
        config_type(legacy_rsqrt=False)
