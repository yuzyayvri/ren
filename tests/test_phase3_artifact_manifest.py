import json

import pytest

from scripts.phase3_artifact_manifest import (
    atomic_manifest,
    mark_test_exposed,
    require_unexposed,
)


def test_manifest_and_exposure_are_atomic_and_single_use(tmp_path):
    manifest = atomic_manifest(tmp_path / "manifest.json", stage="aml", config={"seed": 20260909}, inputs={"source": "x"})
    assert len(manifest["manifest_sha256"]) == 64
    marker = tmp_path / "test_exposed.json"
    require_unexposed(marker)
    mark_test_exposed(marker, manifest["manifest_sha256"])
    with pytest.raises(RuntimeError):
        require_unexposed(marker)
    assert json.loads(marker.read_text())["exposed"] is True
