"""The teaching handle's encoder reads the opposite way round from every gripper.

``PassiveEncoderInfo.position`` is the raw sensor value: 0 = trigger at rest (gripper
open), 1 = fully squeezed (closed).  Every gripper command in this repo is the other way
round (0 = closed, 1 = open), so ``gripper_pos`` carries the converted value and the
trigger hinge -- named ``gripper``, present only on the teaching handle -- is driven from
it.  These tests pin both the conversion and the resulting joint angle.
"""

from types import SimpleNamespace
from typing import Optional

import mujoco
import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import PassiveEncoderInfo
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.mujoco_control_interface import MujocoControlInterface

# Trigger travel from the ONShape export (dof_joint7), radians.
CLOSED_RAD = -0.680678


def _encoder(position: float) -> PassiveEncoderInfo:
    return PassiveEncoderInfo(id=1, position=position, velocity=0.0, io_inputs=[False, False])


@pytest.mark.parametrize("position", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_gripper_pos_is_inverted_position(position: float) -> None:
    """gripper_pos flips the encoder into the arm's 0=closed, 1=open convention."""
    assert _encoder(position).gripper_pos == pytest.approx(1.0 - position)


@pytest.mark.parametrize(
    "gripper_type,expected",
    [
        (GripperType.YAM_TEACHING_HANDLE_V1, True),
        (GripperType.YAM_TEACHING_HANDLE_V2, True),
        (GripperType.LINEAR_4310, False),
        (GripperType.LINEAR_3507, False),
        (GripperType.CRANK_4310, False),
        (GripperType.FLEXIBLE_4310, False),
        (GripperType.NO_GRIPPER, False),
    ],
)
def test_only_teaching_handle_has_a_joint_named_gripper(gripper_type: GripperType, expected: bool) -> None:
    """The trigger hook keys off this name, so it must be unique to the handle."""
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, gripper_type))
    found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper") != -1
    assert found is expected


class _FakeChain:
    """Minimal stand-in for DMChainCanInterface's same-bus encoder snapshot."""

    def __init__(self, state: Optional[object]) -> None:
        self.same_bus_device_driver = object()
        self._state = state

    def get_same_bus_device_states(self) -> Optional[list]:
        return None if self._state is None else [self._state]


def _interface_with_encoder(state: Optional[object]) -> MujocoControlInterface:
    """A real interface object whose encoder hook is fed by a fake chain."""
    iface = MujocoControlInterface.__new__(MujocoControlInterface)
    xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.YAM_TEACHING_HANDLE_V1)
    iface._model = mujoco.MjModel.from_xml_path(xml)
    iface._data = mujoco.MjData(iface._model)
    iface._is_sim = False
    iface._robot = SimpleNamespace(motor_chain=_FakeChain(state))
    iface._handle_joint_id = mujoco.mj_name2id(iface._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    return iface


@pytest.mark.parametrize(
    "position,expected_rad",
    [
        (0.0, 0.0),  # at rest       -> URDF 0 rad
        (0.5, CLOSED_RAD / 2),  # half squeeze
        (1.0, CLOSED_RAD),  # full squeeze  -> URDF -0.680678 rad
    ],
)
def test_trigger_tracks_encoder(position: float, expected_rad: float) -> None:
    """The rendered trigger angle follows the raw encoder reading."""
    iface = _interface_with_encoder(_encoder(position))
    iface._mirror_trigger_from_encoder()
    adr = iface._model.jnt_qposadr[iface._handle_joint_id]
    assert iface._data.qpos[adr] == pytest.approx(expected_rad, abs=1e-12)


@pytest.mark.parametrize("bad", [-0.3, 1.4])
def test_trigger_clamps_out_of_range_encoder(bad: float) -> None:
    """An out-of-range reading is clamped, never driven past the joint limit."""
    iface = _interface_with_encoder(_encoder(bad))
    iface._mirror_trigger_from_encoder()
    lo, hi = iface._model.jnt_range[iface._handle_joint_id]
    adr = iface._model.jnt_qposadr[iface._handle_joint_id]
    assert lo - 1e-12 <= iface._data.qpos[adr] <= hi + 1e-12


@pytest.mark.parametrize(
    "state",
    [
        None,  # no encoder snapshot yet
        SimpleNamespace(io_inputs=[False, False]),  # proxy chain reporting buttons only
    ],
)
def test_trigger_left_alone_without_an_encoder_reading(state: Optional[object]) -> None:
    """Without a reading the viewer's own Joint slider keeps control of the trigger."""
    iface = _interface_with_encoder(state)
    adr = iface._model.jnt_qposadr[iface._handle_joint_id]
    iface._data.qpos[adr] = -0.3  # as if set by hand in the viewer
    iface._mirror_trigger_from_encoder()
    assert iface._data.qpos[adr] == pytest.approx(-0.3)


@pytest.mark.parametrize(
    "trigger_rad,expected",
    [
        (0.0, 1.0),  # at rest      -> OPEN
        (CLOSED_RAD / 2, 0.5),  # half squeeze
        (CLOSED_RAD, 0.0),  # full squeeze -> CLOSED
    ],
)
def test_sim_publishes_normalized_gripper_pos(trigger_rad: float, expected: float) -> None:
    """The handle publishes gripper_pos as 0..1, never the trigger's raw angle."""
    robot = get_yam_robot(sim=True, arm_type=ArmType.YAM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1)
    try:
        adr = robot._model.jnt_qposadr[robot._handle_joint_id]
        robot._data.qpos[adr] = trigger_rad
        obs = robot.get_observations()
        assert "gripper_pos" in obs
        assert obs["gripper_pos"].shape == (1,)
        assert obs["gripper_pos"][0] == pytest.approx(expected, abs=1e-9)
    finally:
        robot.close()


def test_no_gripper_publishes_no_gripper_pos() -> None:
    """Only the handle gains the extra observation; no_gripper must stay arm-only."""
    robot = get_yam_robot(sim=True, arm_type=ArmType.YAM, gripper_type=GripperType.NO_GRIPPER)
    try:
        assert "gripper_pos" not in robot.get_observations()
    finally:
        robot.close()


def test_trigger_does_not_move_the_tcp_site() -> None:
    """tcp_site sits on the handle, not the lever, so squeezing must not shift it."""
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.YAM_TEACHING_HANDLE_V1))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]

    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    at_rest = data.site_xpos[site].copy()

    data.qpos[adr] = CLOSED_RAD
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[site], at_rest, atol=1e-12)
