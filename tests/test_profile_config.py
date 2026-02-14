import pytest

from capitalmarket.capitalselector.config import ProfileAConfig, ProfileBConfig


def test_profile_configs_are_closed():
    ProfileAConfig().validate_closed()
    ProfileBConfig().validate_closed()


def test_profile_config_missing_field_raises():
    cfg = ProfileAConfig()
    object.__setattr__(cfg, "score_mode", None)
    with pytest.raises(ValueError):
        cfg.validate_closed()
