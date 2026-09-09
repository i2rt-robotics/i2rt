"""Drive two YAM arms in task space from a pair of VR controllers, over Adamo.

Each arm ramps to a ready pose, then follows one hand: can0 the left controller,
can1 the right. A controller's grip is a clutch -- hold it to pin the hand to where
that arm's end-effector already is and move the arm one-for-one from there, release
to freeze. The trigger closes that arm's gripper, and Y (B on the right controller)
walks that arm back to the pose it started in.

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
import jaxlie
import numpy as np
from adamo.xr import PoseStamped, XRJoy, subscribe_xr_control
from yam_ik import REST_POSE, VELOCITY_LIMITS, YamIK

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

NAME = "yam-san-mateo"
ARMS = (("can0", "left"), ("can1", "right"))
READY = REST_POSE  # the IK's posture bias, so the arm starts where the solver wants to sit
DT = 0.02
# Rehoming runs a sixth of the speed the IK clamps hand-tracking to: nobody is steering it,
# so it should look deliberate rather than as quick as the arm can manage.
HOME_SPEED = VELOCITY_LIMITS / 6.0

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
state = {side: {"pos": None, "rot": None, "clutch": False, "grip": None, "home": False} for _, side in ARMS}


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
            # Y on the left controller, B on the right: the same index in the Touch mapping.
            side["home"] = len(msg.buttons) > 5 and bool(msg.buttons[5])
            trigger = (
                float(msg.axes[4])
                if len(msg.axes) > 4 and msg.axes[4] > 0
                else float(bool(msg.buttons and msg.buttons[0]))
            )
            if trigger > 0.05:  # a released trigger leaves the gripper where it was
                side["grip"] = 1.0 - trigger


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

ik = YamIK()


class Arm:
    """One robot, the hand it follows, and where its end-effector is being asked to go."""

    def __init__(self, channel: str, side: str) -> None:
        self.side = side
        self.robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310)
        self.cmd = self.robot.get_joint_pos()
        self.q = READY.copy()
        position, wxyz = ik.tcp_pose(self.q)
        self.target, self.rot = position, np.asarray(jaxlie.SO3(wxyz).as_matrix())
        self.anchor = None
        self.homing = False

    def step(self) -> None:
        with lock:
            s = state[self.side]
            hand, hand_rot, clutch, grip, home = s["pos"], s["rot"], s["clutch"], s["grip"], s["home"]
        tracking = clutch and hand is not None
        if home and not tracking and not self.homing:  # a held clutch is never overridden
            self.homing = True
            print(f"[teleop] {self.side} rehoming to {np.round(READY, 3)}", flush=True)

        if tracking:
            self.homing = False
            if self.anchor is None:
                self.anchor = (hand, hand_rot, self.target, self.rot)
                print(f"[teleop] {self.side} grip held, tracking from {np.round(self.target, 4)}", flush=True)
            hand0, hand_rot0, ee0, ee_rot0 = self.anchor
            self.target, self.rot = ee0 + (hand - hand0), hand_rot @ hand_rot0.T @ ee_rot0
            # The solve is warm-started from the last configuration and rate-limited inside, so an
            # unreachable target degrades into a lagging arm rather than a jump, and recovers.
            wxyz = np.asarray(jaxlie.SO3.from_matrix(self.rot).wxyz)
            self.q, _, _ = ik.solve(self.target, wxyz, self.q, DT)
            self.cmd[:6] = self.q
        else:
            if self.anchor is not None:
                self.anchor = None
                print(f"[teleop] {self.side} grip released, holding {np.round(self.target, 4)}", flush=True)
            if self.homing:
                # Joint space, at HOME_SPEED. The target is carried along so a re-clutch
                # mid-rehome anchors to where the arm actually is.
                self.q = self.q + np.clip(READY - self.q, -HOME_SPEED * DT, HOME_SPEED * DT)
                position, wxyz = ik.tcp_pose(self.q)
                self.target, self.rot = position, np.asarray(jaxlie.SO3(wxyz).as_matrix())
                self.cmd[:6] = self.q
                if np.array_equal(self.q, READY):
                    self.homing = False
                    print(f"[teleop] {self.side} rehomed", flush=True)
        if grip is not None:
            self.cmd[6] = grip
        self.robot.command_joint_pos(self.cmd)


arms = [Arm(channel, side) for channel, side in ARMS]
adamo_robot = adamo.Robot(api_key=os.environ["ADAMO_API_KEY"], name=NAME)
# No width/height/pixel_format: a pipeline= track takes its size from the caps the graph
# negotiates, and this graph has already decoded to raw NV12 itself.
adamo_robot.attach_video(TRACK, pipeline=PIPELINE, fps=FPS, bitrate_kbps=BITRATE_KBPS, stereo=True)
subscribe_xr_control(adamo_robot.session, NAME, on_sample, max_age_seconds=0.25)
# Compile the solver before anything moves; the first solve takes seconds, every one after
# it a few milliseconds.
ik.solve(*ik.tcp_pose(READY), READY, DT)
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

    deadline = time.perf_counter()
    while True:
        for arm in arms:
            arm.step()
        # Sleep the remainder of the tick rather than a flat DT: the solve costs milliseconds,
        # and DT is what the IK's velocity clamp is scaled by.
        deadline += DT
        time.sleep(max(deadline - time.perf_counter(), 0.0))
except KeyboardInterrupt:
    print("\n[teleop] interrupted", flush=True)
finally:
    adamo_robot.close()
    for arm in arms:
        arm.robot.close()
