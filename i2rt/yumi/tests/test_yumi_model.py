"""Tests for the yumi device model (i2rt/robot_models/yumi/yumi.urdf): its meshes and the conventions of its frames.

The `camera` and `vive` frames are what place a live image and a tracked pose on the model, and a wrong sign in
either still renders a plausible-looking device. These pin the conventions the README states against the model
itself: the fingertips sit below the image centre in front of the camera, and the tracker body sits on the far
side of its mounting face from the device.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh

from i2rt.robot_models import YUMI_URDF_PATH

_ROBOT = ET.parse(YUMI_URDF_PATH).getroot()


def _link_in_base(link: str) -> np.ndarray:
    """``link``'s pose in ``base``, from its (fixed) joint's ``<origin>``; rpy is R = Rz(yaw) Ry(pitch) Rx(roll)."""
    (joint,) = [j for j in _ROBOT.iter("joint") if j.find("child").get("link") == link]
    assert joint.find("parent").get("link") == "base"
    origin = joint.find("origin")
    roll, pitch, yaw = (float(v) for v in origin.get("rpy").split())
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = [float(v) for v in origin.get("xyz").split()]
    return transform


def test_every_mesh_exists() -> None:
    meshes = [m.get("filename") for m in _ROBOT.iter("mesh")]
    assert len(meshes) == 4
    for filename in meshes:
        assert (Path(YUMI_URDF_PATH).parent / filename).is_file(), filename


@pytest.mark.parametrize("frame", ["tcp", "camera", "vive"])
def test_frame_is_fixed_to_base(frame: str) -> None:
    (joint,) = [j for j in _ROBOT.iter("joint") if j.find("child").get("link") == frame]
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "base"


def test_tcp_is_below_the_image_centre() -> None:
    """OpenCV (+Y down): the fingertips are in front of the camera, straight below the image centre."""
    tcp_in_camera = np.linalg.inv(_link_in_base("camera")) @ _link_in_base("tcp")
    x, y, z = tcp_in_camera[:3, 3]
    assert z > 0.05
    assert y > 0.02
    assert abs(x) < 1e-3
    # The camera and the tcp share +X (the device's right), so the tcp's +Z (forward) is the camera's, pitched down.
    np.testing.assert_allclose(tcp_in_camera[:3, 0], [1, 0, 0], atol=1e-9)


def test_tracker_body_is_behind_its_mounting_face() -> None:
    """+Z points out of the mounting face away from the dome, so the tracker body is at z <= 0 in `vive`."""
    tracker = trimesh.load(Path(YUMI_URDF_PATH).parent / "assets" / "vive_tracker.stl")
    assert tracker.bounds[1, 2] < 1e-4
    assert tracker.bounds[0, 2] < -0.04
    # ... and the dome faces up (base +Z), with the frame's +Z into the seat.
    assert _link_in_base("vive")[2, 2] < -0.9
