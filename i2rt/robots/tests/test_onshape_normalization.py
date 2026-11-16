"""Regression tests for the ONShape URDF normalization scripts."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_normalizer(skill_root: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    script_dir = REPO_ROOT / skill_root / "skills/transform-onshape-urdf/scripts"
    script_path = script_dir / "normalize_onshape_urdf.py"
    monkeypatch.syspath_prepend(str(script_dir))
    spec = importlib.util.spec_from_file_location(f"{skill_root[1:]}_normalize_onshape_urdf", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("skill_root", [".agents", ".claude"])
@pytest.mark.parametrize("original_mesh_exists", [False, True])
def test_fix_meshes_applies_suffixed_product_base_rename_before_disk_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skill_root: str,
    original_mesh_exists: bool,
) -> None:
    """A suffixed product base resolves to base.stl even if the raw STL also exists."""
    normalizer = _load_normalizer(skill_root, monkeypatch)
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "base.stl").touch()
    if original_mesh_exists:
        (assets_dir / "ultra_base_1.stl").touch()
    robot = ET.fromstring(
        '<robot><link name="base"><visual><geometry><mesh filename="package://yam/meshes/ultra_base_1.stl"/>'
        "</geometry></visual></link></robot>"
    )

    normalizer.step_fix_meshes(robot, str(assets_dir), "assets", {"ultra_base_1": "base"})

    assert robot.find(".//mesh").get("filename") == "assets/base.stl"
