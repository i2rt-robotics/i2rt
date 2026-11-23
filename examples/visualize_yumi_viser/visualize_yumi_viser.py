"""Visualize the yumi handheld device live in viser: its tracker pose, its camera stream, its gripper and buttons.

The scene is OpenVR's standing universe (metres, +Y up) with a floor grid. The yumi model
(i2rt/robot_models/yumi/yumi.urdf) follows the Vive tracker's pose, hung off the tracker by the model's `vive`
frame, and hides while the tracker has no optical lock. Its fingers follow the gripper opening. A camera frustum
sits at the model's `camera` frame and shows the live MF camera stream, raw (fisheye, not rectified). The panel shows the gripper opening (0 closed ... 1 open), the encoder's two buttons and each
stream's health. Quit with Ctrl+C in the terminal.

Usage:
    # The first camera plugged in and the first tracker SteamVR reports:
    uv run examples/visualize_yumi_viser/visualize_yumi_viser.py

    # Or name them:
    uv run examples/visualize_yumi_viser/visualize_yumi_viser.py \
        --tracker-serial LHR-XXXXXXXX --camera-serial MF150_2026061109

Then open http://localhost:8080 in a browser.
"""

import logging
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import trimesh
import tyro
import viser
import viser.transforms as vtf

from i2rt.robot_models import YUMI_URDF_PATH
from i2rt.utils.utils import override_log_level
from i2rt.yumi.gripper import DEFAULT_RANGE_RAD
from i2rt.yumi.yumi import Yumi

# Scene refresh rate. A new camera frame (30 fps) is sent only when one has arrived.
_UPDATE_PERIOD_S = 1.0 / 60.0
# shortcut: a display-only frustum angle, since the driver carries no intrinsics; derive it from a K once one exists.
_FRUSTUM_FOV_RAD = np.deg2rad(70.0)


@dataclass
class Args:
    tracker_serial: str | None = None
    """Serial of the Vive tracker on the handle (python i2rt/yumi/vive_tracker.py lists them); defaults to the first
    tracker SteamVR reports."""
    camera_serial: str | None = None
    """USB short serial of the MF camera, e.g. MF150_2026061109; defaults to the first camera plugged in."""
    can_channel: str = "can0"
    """CAN channel of the gripper's handle encoder."""
    range_rad: float = DEFAULT_RANGE_RAD
    """Encoder angle (rad) from the full-open zero that reads as fully closed."""
    port: int = 8080
    """Viser server port."""


def urdf_origin(element: ET.Element | None) -> np.ndarray:
    """The 4x4 transform of a URDF ``<origin>`` (``rpy`` is R = Rz(yaw) Ry(pitch) Rx(roll)); identity if absent."""
    transform = np.eye(4)
    if element is not None:
        roll, pitch, yaw = (float(v) for v in element.get("rpy", "0 0 0").split())
        transform[:3, :3] = vtf.SO3.from_rpy_radians(roll, pitch, yaw).as_matrix()
        transform[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
    return transform


def set_pose(node: viser.SceneNodeHandle, transform: np.ndarray) -> None:
    node.position = transform[:3, 3]
    node.wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz


@dataclass
class Finger:
    frame: viser.FrameHandle
    closed_position: np.ndarray
    """The finger link's origin in ``base`` with the finger closed (gripper 0)."""
    axis: np.ndarray
    travel: float
    """Joint travel (m) from closed (gripper 0) to open (gripper 1)."""


def add_yumi_model(
    server: viser.ViserServer, root: str
) -> tuple[viser.FrameHandle, dict[str, np.ndarray], list[Finger]]:
    """Add the yumi URDF's meshes under scene node ``root``, which stands for its ``base`` link.

    Every joint in the URDF hangs off ``base``, so each link is one frame directly under ``root``. Returns the root
    frame, every link's pose in ``base`` with the fingers closed, and the fingers.
    """
    robot = ET.parse(YUMI_URDF_PATH).getroot()
    joint_of = {joint.find("child").get("link"): joint for joint in robot.iter("joint")}
    root_frame = server.scene.add_frame(root, show_axes=False)
    link_in_base: dict[str, np.ndarray] = {}
    fingers = []
    for link in robot.iter("link"):
        name = link.get("name")
        joint = joint_of.get(name)
        if joint is not None and joint.find("parent").get("link") != "base":
            raise ValueError(f"{YUMI_URDF_PATH}: {name} hangs off {joint.find('parent').get('link')}, not base")
        pose = np.eye(4) if joint is None else urdf_origin(joint.find("origin"))
        link_in_base[name] = pose
        frame = server.scene.add_frame(f"{root}/{name}", show_axes=False)
        set_pose(frame, pose)
        for i, visual in enumerate(link.iter("visual")):
            mesh = trimesh.load(Path(YUMI_URDF_PATH).parent / visual.find("geometry/mesh").get("filename"))
            mesh.apply_transform(urdf_origin(visual.find("origin")))
            rgba = [float(v) for v in visual.find("material/color").get("rgba").split()]
            server.scene.add_mesh_simple(f"{root}/{name}/visual_{i}", mesh.vertices, mesh.faces, color=tuple(rgba[:3]))
        if joint is not None and joint.get("type") == "prismatic":
            axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
            lower, upper = (float(joint.find("limit").get(k)) for k in ("lower", "upper"))
            fingers.append(Finger(frame, pose[:3, 3] + lower * axis, axis, upper - lower))
    return root_frame, link_in_base, fingers


def main(args: Args) -> None:
    with Yumi(
        args.can_channel,
        args.tracker_serial,
        args.camera_serial,
        range_rad=args.range_rad,
    ) as yumi:
        width, height = yumi.camera.info["resolution"]

        server = viser.ViserServer(host="0.0.0.0", port=args.port, label="yumi")
        server.scene.set_up_direction("+y")  # OpenVR's standing universe is +Y up, with its origin on the floor.
        server.scene.add_grid("/floor", width=4.0, height=4.0, plane="xz")
        server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)
        tracker_frame = server.scene.add_frame("/tracker", axes_length=0.1, axes_radius=0.005)
        # The model and the frustum are children of /tracker, so they move and hide with it. The tracker's pose is
        # the model's `vive` frame, which places `base` at the inverse of `vive`'s pose in it.
        model_frame, link_in_base, fingers = add_yumi_model(server, "/tracker/yumi")
        base_in_tracker = np.linalg.inv(link_in_base["vive"])
        set_pose(model_frame, base_in_tracker)
        # viser's frustum looks along its +Z with +Y down, the same OpenCV convention as the model's `camera` frame.
        frustum = server.scene.add_camera_frustum(
            "/tracker/camera", fov=_FRUSTUM_FOV_RAD, aspect=width / height, scale=0.15
        )
        set_pose(frustum, base_in_tracker @ link_in_base["camera"])

        gripper_slider = server.gui.add_slider(
            "gripper (0 closed, 1 open)", min=0.0, max=1.0, step=0.001, initial_value=0.0, disabled=True
        )
        button_boxes = [
            server.gui.add_checkbox(f"button {i + 1}", initial_value=False, disabled=True) for i in range(2)
        ]
        status = server.gui.add_markdown("")

        print(
            f"[viser] tracker {yumi.tracker_serial}, camera {yumi.camera.info['serial']}: open "
            f"http://localhost:{args.port} (Ctrl+C to quit)"
        )
        last_frame_timestamp = None
        try:
            while True:
                started = time.monotonic()
                # Each stream is read on its own, so one stale stream is reported without freezing the others.
                try:
                    tracker = yumi.vive.latest()
                    tracker_frame.visible = tracker.valid
                    if tracker.valid:
                        set_pose(tracker_frame, tracker.pose)
                    lines = [f"**tracker** {'tracking' if tracker.valid else 'pose lost (no optical lock)'}"]
                except TimeoutError as exc:
                    tracker_frame.visible = False
                    lines = [f"**tracker** stale: {exc}"]
                try:
                    gripper = yumi.gripper.latest()
                    gripper_slider.value = gripper.position
                    for finger in fingers:  # assumes finger travel is linear in the handle encoder angle
                        finger.frame.position = finger.closed_position + gripper.position * finger.travel * finger.axis
                    for box, pressed in zip(button_boxes, gripper.buttons, strict=True):
                        box.value = pressed
                except TimeoutError as exc:
                    lines.append(f"**gripper** stale: {exc}")
                try:
                    frame = yumi.camera.latest()
                    if frame.timestamp != last_frame_timestamp:
                        last_frame_timestamp = frame.timestamp
                        frustum.image = frame.image
                except TimeoutError as exc:
                    lines.append(f"**camera** stale: {exc}")
                status.content = "  \n".join(lines)
                time.sleep(max(0.0, _UPDATE_PERIOD_S - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()


if __name__ == "__main__":
    override_log_level(logging.INFO)  # Yumi reports its startup (SteamVR, tracker wait) at INFO
    main(tyro.cli(Args))
