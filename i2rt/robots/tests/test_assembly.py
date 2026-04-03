"""Tests for robot assembly: arm + gripper XML combination (pure XML-level)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_ARM_GRIPPER_COMBOS = [(arm, gripper) for arm in ArmType for gripper in GripperType if arm != ArmType.NO_ARM]


def _combo_id(val: object) -> str:
    """Readable pytest ID for (ArmType, GripperType) tuples."""
    if isinstance(val, (ArmType, GripperType)):
        return val.value
    return str(val)


def _count_joints(root: ET.Element) -> int:
    """Count <joint> elements inside worldbody (recursively)."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        return 0
    return sum(1 for _ in worldbody.iter("joint"))


def _get_body_inertials(root: ET.Element) -> dict:
    """Return dict mapping body name -> inertial attributes or None.

    For each <body> with a name, if it has an <inertial> child the value is
    a dict with keys: mass, pos, quat, diaginertia (strings as found in XML).
    If the body has no <inertial> child the value is None.
    """
    result = {}
    for body in root.iter("body"):
        name = body.get("name")
        if name is None:
            continue
        inertial = body.find("inertial")
        if inertial is None:
            result[name] = None
        else:
            result[name] = {
                "mass": inertial.get("mass"),
                "pos": inertial.get("pos"),
                "ipos": inertial.get("ipos"),
                "quat": inertial.get("quat"),
                "diaginertia": inertial.get("diaginertia"),
            }
    return result


def _find_gripper(root: ET.Element) -> ET.Element | None:
    """Find the gripper body element."""
    return root.find(".//body[@name='gripper']")


# ---------------------------------------------------------------------------
# Test 1: combine_xml produces valid, parseable XML with gripper body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,gripper", ALL_ARM_GRIPPER_COMBOS, ids=_combo_id)
def test_combine_xml_produces_valid_xml(arm: ArmType, gripper: GripperType) -> None:
    """combine_arm_and_gripper_xml should produce valid XML for every combo."""
    out_path = combine_arm_and_gripper_xml(arm, gripper)

    assert isinstance(out_path, str)
    assert out_path.endswith(".xml")

    tree = ET.parse(out_path)
    root = tree.getroot()

    assert _find_gripper(root) is not None, "Combined XML must contain a body named 'gripper'"


# ---------------------------------------------------------------------------
# Test 2: combined XML preserves DOFs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,gripper", ALL_ARM_GRIPPER_COMBOS, ids=_combo_id)
def test_combined_xml_preserves_dofs(arm: ArmType, gripper: GripperType) -> None:
    """Joint count in combined XML should equal arm joints (gripper body's joint comes from gripper)."""
    arm_root = ET.parse(arm.get_xml_path()).getroot()
    gripper_root = ET.parse(gripper.get_xml_path()).getroot()
    combined_root = ET.parse(combine_arm_and_gripper_xml(arm, gripper)).getroot()

    arm_joints = _count_joints(arm_root)
    gripper_joints = _count_joints(gripper_root)
    combined_joints = _count_joints(combined_root)

    # The arm defines only its own joints (no gripper placeholder).
    # The gripper body (with its joints) is appended at runtime.
    # So combined = arm_joints + gripper_joints.
    expected = arm_joints + gripper_joints
    assert combined_joints == expected, (
        f"Expected {expected} joints (arm={arm_joints} + gripper={gripper_joints}), got {combined_joints}"
    )


# ---------------------------------------------------------------------------
# Test 3: combined XML preserves link inertial properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,gripper", ALL_ARM_GRIPPER_COMBOS, ids=_combo_id)
def test_combined_xml_preserves_link_properties(arm: ArmType, gripper: GripperType) -> None:
    """Bodies other than gripper should keep arm inertials; gripper body gets gripper's."""
    arm_root = ET.parse(arm.get_xml_path()).getroot()
    gripper_root = ET.parse(gripper.get_xml_path()).getroot()
    combined_root = ET.parse(combine_arm_and_gripper_xml(arm, gripper)).getroot()

    arm_inertials = _get_body_inertials(arm_root)
    gripper_inertials = _get_body_inertials(gripper_root)
    combined_inertials = _get_body_inertials(combined_root)

    gripper_name = "gripper"

    # Non-gripper bodies: must match arm values
    for name, arm_val in arm_inertials.items():
        if name == gripper_name:
            continue
        assert name in combined_inertials, f"Body '{name}' missing from combined XML"
        assert combined_inertials[name] == arm_val, (
            f"Inertial mismatch for body '{name}': arm={arm_val}, combined={combined_inertials[name]}"
        )

    # gripper body: should match gripper's definition (if gripper defines one)
    if gripper_name in gripper_inertials and gripper_inertials[gripper_name] is not None:
        assert gripper_name in combined_inertials, f"Body '{gripper_name}' missing from combined XML"
        grip_val = gripper_inertials[gripper_name]
        comb_val = combined_inertials[gripper_name]
        assert comb_val is not None, "Combined XML gripper inertial is None but gripper defines one"
        assert comb_val["mass"] == grip_val["mass"], (
            f"gripper mass mismatch: gripper={grip_val['mass']}, combined={comb_val['mass']}"
        )
        assert comb_val["diaginertia"] == grip_val["diaginertia"], (
            f"gripper diaginertia mismatch: gripper={grip_val['diaginertia']}, combined={comb_val['diaginertia']}"
        )


# ---------------------------------------------------------------------------
# Test 4: custom ee_mass / ee_inertia override gripper body properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,gripper", ALL_ARM_GRIPPER_COMBOS, ids=_combo_id)
def test_combine_xml_with_custom_ee_mass_inertia(arm: ArmType, gripper: GripperType) -> None:
    """Passing ee_mass and ee_inertia should override gripper inertial attributes."""
    rng = np.random.default_rng(42)
    ee_mass = rng.uniform(0.1, 2.0)
    ee_inertia = rng.uniform(0.001, 1.0, size=10)  # 3 pos + 4 quat + 3 diaginertia

    out_path = combine_arm_and_gripper_xml(
        arm,
        gripper,
        ee_mass=ee_mass,
        ee_inertia=ee_inertia,
    )
    combined_root = ET.parse(out_path).getroot()

    gripper_body = _find_gripper(combined_root)
    assert gripper_body is not None
    inertial = gripper_body.find("inertial")
    assert inertial is not None, "gripper body should have an <inertial> element after override"

    # Check mass
    assert float(inertial.get("mass")) == pytest.approx(ee_mass), (
        f"mass mismatch: expected {ee_mass}, got {inertial.get('mass')}"
    )

    # Check ipos (first 3 elements)
    ipos_vals = [float(x) for x in inertial.get("ipos").split()]
    np.testing.assert_allclose(ipos_vals, ee_inertia[:3], atol=1e-10)

    # Check quat (elements 3-7)
    quat_vals = [float(x) for x in inertial.get("quat").split()]
    np.testing.assert_allclose(quat_vals, ee_inertia[3:7], atol=1e-10)

    # Check diaginertia (last 3 elements)
    diag_vals = [float(x) for x in inertial.get("diaginertia").split()]
    np.testing.assert_allclose(diag_vals, ee_inertia[-3:], atol=1e-10)

    # All other bodies should remain unchanged from arm
    arm_root = ET.parse(arm.get_xml_path()).getroot()
    arm_inertials = _get_body_inertials(arm_root)
    combined_inertials = _get_body_inertials(combined_root)

    for name, arm_val in arm_inertials.items():
        if name == "gripper":
            continue
        assert combined_inertials[name] == arm_val, f"Body '{name}' inertial changed unexpectedly after ee override"
