"""Placeholder test for Issue #1 - proves the package installs and imports cleanly.

Actual encoding/decoding logic is out of scope for this issue and is covered
by the Phase 1-4 issues instead.
"""

import hex8
import hex8.common
import hex8.encoder
import hex8.decoder
import hex8.camera


def test_package_version_is_defined():
    assert hex8.__version__ == "0.1.0"


def test_subpackages_import():
    assert hex8.common is not None
    assert hex8.encoder is not None
    assert hex8.decoder is not None
    assert hex8.camera is not None
