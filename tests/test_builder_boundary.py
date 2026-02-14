from __future__ import annotations

from collections.abc import Iterable

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.config import ProfileAConfig, ProfileBConfig


def _contains_profile_obj(obj) -> bool:
    seen = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, (ProfileAConfig, ProfileBConfig)):
            return True
        if isinstance(cur, dict):
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple, set, frozenset)):
            stack.extend(cur)
            continue
        if hasattr(cur, "__dict__"):
            stack.extend(cur.__dict__.values())
    return False


def test_kernel_has_no_profile_identity():
    selector = CapitalSelectorBuilder.from_profile(ProfileAConfig()).with_K(0).build()
    assert not hasattr(selector, "profile")
    assert not _contains_profile_obj(selector)


def test_kernel_has_no_profile_identity_profile_b():
    selector = CapitalSelectorBuilder.from_profile(ProfileBConfig()).with_K(0).build()
    assert not hasattr(selector, "profile")
    assert not _contains_profile_obj(selector)
