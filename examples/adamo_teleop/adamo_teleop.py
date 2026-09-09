"""Drive two YAM arms in task space from a pair of VR controllers, over Adamo.

Each arm ramps to a ready pose, then follows one hand: can0 the left controller,
can1 the right. A controller's grip is a clutch -- hold it to pin the hand to where
that arm's end-effector already is and move the arm one-for-one from there, release
to freeze. The trigger closes that arm's gripper.

The operator's view is the rig's two USB cameras, composited side by side into one
``stereo=True`` track. The whole capture -> decode -> composite -> encode graph runs
in the Rust runtime, so no frame crosses into Python.

    export ADAMO_API_KEY=ak_...
    python examples/adamo_teleop/adamo_teleop.py
"""

import os
import threading
import time

import adamo
import mink
import mujoco
import numpy as np
from adamo.xr import PoseStamped, XRJoy, subscribe_xr_control

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

NAME = "yam-san-mateo"
ARMS = (("can0", "left"), ("can1", "right"))
SITE = "grasp_site"
READY = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
DT = 0.02
MAX_JOINT_STEP = 0.05

TRACK = "zed"
# The rig's two USB cameras, in left/right order. Each is one of a pair of nodes on its
# own USB device (3-1 and 3-2); the odd-numbered twins carry no capture formats.
EYES = ("/dev/video6", "/dev/video4")
# 1080p per eye, a mode both cameras offer. The composite is what has to fit the
# 4096-pixel width an H.264 encoder (and a headset decoder) will take, so this is
# 3840 wide on the wire.
EYE_WIDTH, EYE_HEIGHT, FPS = 1920, 1080, 30
BITRATE_KBPS = 12000

# WebXR local-floor (+x right, +y up, -z forward) -> arm world (+x forward, +y left, +z up).
XR2W = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

lock = threading.Lock()
state = {side: {"pos": None, "rot": None, "clutch": False, "grip": None} for _, side in ARMS}


def quat_to_mat(q: object) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def on_sample(sample: object) -> None:
    parts = sample.topic.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "controller" or parts[1] not in state:
        return
    msg, side = sample.message, state[parts[1]]
    with lock:
        if len(parts) == 2 and isinstance(msg, PoseStamped):
            side["pos"] = XR2W @ np.array([msg.position.x, msg.position.y, msg.position.z])
            side["rot"] = XR2W @ quat_to_mat(msg.orientation) @ XR2W.T
        elif len(parts) == 3 and parts[2] == "joy" and isinstance(msg, XRJoy):
            side["clutch"] = len(msg.buttons) > 1 and bool(msg.buttons[1])
            trigger = (
                float(msg.axes[4])
                if len(msg.axes) > 4 and msg.axes[4] > 0
                else float(bool(msg.buttons and msg.buttons[0]))
            )
            if trigger > 0.05:  # a released trigger leaves the gripper where it was
                side["grip"] = 1.0 - trigger


def pose(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    target = np.eye(4)
    target[:3, :3] = rot
    target[:3, 3] = pos
    return target


def eye_branch(device: str, sink: str) -> str:
    """One camera, decoded and converted, into one compositor pad. The queue is one deep
    and leaky, so an eye that stalls drops its own frames rather than holding the other."""
    return (
        f"v4l2src device={device} do-timestamp=true"
        f" ! image/jpeg,width={EYE_WIDTH},height={EYE_HEIGHT},framerate={FPS}/1 ! jpegdec"
        " ! videoconvert ! video/x-raw,format=NV12"
        " ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
        f" ! {sink}"
    )


# The composite's caps are a named, quoted capsfilter: the parser reads a trailing bare
# caps greedily and would run straight past the space into the next chain.
PIPELINE = "  ".join(
    (
        f"compositor name=sbs background=black sink_0::xpos=0 sink_1::xpos={EYE_WIDTH}"
        f' ! capsfilter caps="video/x-raw,format=NV12,width={2 * EYE_WIDTH},'
        f'height={EYE_HEIGHT},framerate={FPS}/1"',
        eye_branch(EYES[0], "sbs.sink_0"),
        eye_branch(EYES[1], "sbs.sink_1"),
    )
)

xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
model = mujoco.MjModel.from_xml_path(xml_path)
kin = Kinematics(xml_path, SITE)
limits = [mink.ConfigurationLimit(model)]


class Arm:
    """One robot, the hand it follows, and where its end-effector is being asked to go."""

    def __init__(self, channel: str, side: str) -> None:
        self.side = side
        self.robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310)
        self.cmd = self.robot.get_joint_pos()
        self.q_full = np.zeros(model.nq)
        self.q_full[:6] = READY
        ready_pose = kin.fk(self.q_full, SITE)
        self.rot, self.target = ready_pose[:3, :3].copy(), ready_pose[:3, 3].copy()
        self.anchor = None

    def step(self) -> None:
        with lock:
            s = state[self.side]
            hand, hand_rot, clutch, grip = s["pos"], s["rot"], s["clutch"], s["grip"]
        if clutch and hand is not None:
            if self.anchor is None:
                self.anchor = (hand, hand_rot, self.target, self.rot)
                print(f"[teleop] {self.side} grip held, tracking from {np.round(self.target, 4)}", flush=True)
            hand0, hand_rot0, ee0, ee_rot0 = self.anchor
            candidate, candidate_rot = ee0 + (hand - hand0), hand_rot @ hand_rot0.T @ ee_rot0
            ok, solution = kin.ik(
                pose(candidate_rot, candidate), SITE, init_q=self.q_full, limits=limits, max_iters=40
            )
            if ok:  # a failed solve holds the target, so pushing into an unreachable pose stalls
                self.target, self.rot, self.q_full = candidate, candidate_rot, solution.copy()
            # Clamped so a solution that jumped across a singularity is walked, not snapped.
            self.cmd[:6] = np.clip(self.q_full[:6], self.cmd[:6] - MAX_JOINT_STEP, self.cmd[:6] + MAX_JOINT_STEP)
        elif self.anchor is not None:
            self.anchor = None
            print(f"[teleop] {self.side} grip released, holding {np.round(self.target, 4)}", flush=True)
        if grip is not None:
            self.cmd[6] = grip
        self.robot.command_joint_pos(self.cmd)


arms = [Arm(channel, side) for channel, side in ARMS]
adamo_robot = adamo.Robot(api_key=os.environ["ADAMO_API_KEY"], name=NAME)
# No width/height/pixel_format: a pipeline= track takes its size from the caps the graph
# negotiates, and this graph has already decoded to raw NV12 itself.
adamo_robot.attach_video(TRACK, pipeline=PIPELINE, fps=FPS, bitrate_kbps=BITRATE_KBPS, stereo=True)
subscribe_xr_control(adamo_robot.session, NAME, on_sample, max_age_seconds=0.25)
# attach_video only queues the graph; run() is what starts the Rust pipeline, and it
# blocks -- so it goes in a daemon thread and the control loop keeps the main one.
threading.Thread(target=adamo_robot.run, daemon=True).start()
print(f"[teleop] {NAME}: grip to move an arm, trigger for its gripper", flush=True)

try:
    # Ramp to the ready pose: the home pose sits on joint2/joint3's lower bound.
    starts = [arm.cmd[:6].copy() for arm in arms]
    for i in range(1, 201):
        for arm, q0 in zip(arms, starts, strict=True):
            arm.cmd[:6] = q0 + (READY - q0) * (i / 200)
            arm.robot.command_joint_pos(arm.cmd)
        time.sleep(DT)

    while True:
        for arm in arms:
            arm.step()
        time.sleep(DT)
except KeyboardInterrupt:
    print("\n[teleop] interrupted", flush=True)
finally:
    adamo_robot.close()
    for arm in arms:
        arm.robot.close()
