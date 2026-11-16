"""Zero a 7-DOF arm against its hard stops, test its joint directions, or send it home.

Ported from the bench zeroing tool. Run it on the arm's CAN channel with nothing else using the bus:

- ``zero``: push each joint to its positive hard stop under a force cap, back off by the travel from that stop to
  zero, then write every motor's position as its Flash zero. joint1 and joint3-7 go together, then joint2 alone,
  since it lifts the whole arm to +205 deg. The back-off is the model's upper joint limit, except on joint6, whose
  encoder reads the four-bar's input crank: there it is the top of the config's ``motor_range_deg``. The gripper
  then finds its stops (the detection the robot runs at startup), closes, and is zeroed there.
- ``dir-test``: push each joint gently toward + for 3 s and bring it back. Watch which way it turns: + is toward
  the joint's upper limit in the model (joint2: the +205 deg end). A joint that turns the other way is fixed in
  the config's ``directions``, never in the motor firmware.
- ``home``: ease every joint to 0 and release.

    python scripts/zero_7dof_arm.py --arm yam_7dof_right --channel can0 --mode dir-test
    python scripts/zero_7dof_arm.py --arm yam_7dof_right --channel can0 --mode zero
    python scripts/zero_7dof_arm.py --arm yam_7dof_right --channel can0 --mode home

Start ``zero`` with every joint within 60 deg of zero: a joint that stalls far short of its back-off is taken to be
pushing the wrong way, and the run backs out without writing anything. Keep the arm's space clear.
"""

import time
from typing import Literal

import mujoco
import numpy as np
import tyro

from i2rt.motor_config_tool import set_zero
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config, detect_gripper_limits

# Push force caps, Nm per arm joint, bench-tuned. A pushing joint's command leads its measured position by at most
# cap / kp (kp from the arm config). There is no gravity comp here, so joint1 and joint2, which lift the arm, get more.
ZERO_FORCE = np.array([12.0, 28.0, 6.4, 4.8, 3.2, 3.2, 3.2])
DIR_FORCE = np.minimum(ZERO_FORCE, 6.4)  # dir-test pushes gently
STALL_LAG, STALL_PROG, STALL_T = 0.06, 0.01, 0.15  # lagging > 0.06 rad with < 0.01 rad of progress in 0.15 s
ABORT_ERR = 0.6  # rad of tracking error on any motor aborts
SHORT = np.radians(60.0)  # stalling this far short of the back-off means the joint pushed the wrong way
OVERTRAVEL = np.radians(70.0)  # this far past the back-off without a stall aborts
STAGES = [[0, 2, 3, 4, 5, 6], [1]]
DT = 0.01


class Hold:
    """PD-holds every motor at `cmd`, and moves joints by moving `cmd`."""

    def __init__(self, chain: DMChainCanInterface, kp: np.ndarray, kd: np.ndarray) -> None:
        self.chain, self.kp, self.kd = chain, kp, kd
        self.cmd = self.read()
        self.tick()

    def read(self) -> np.ndarray:
        return np.array([s.pos for s in self.chain.read_states()])

    def tick(self, vel: np.ndarray | None = None) -> np.ndarray:
        # ponytail: time is counted in ticks, a little slow against the wall clock; fine for stall windows.
        if not self.chain.running:
            raise RuntimeError("motor chain thread stopped")
        n = len(self.cmd)
        vel = np.zeros(n) if vel is None else vel
        self.chain.set_commands(np.zeros(n), self.cmd, vel, self.kp, self.kd, get_state=False)
        time.sleep(DT)
        q = self.read()
        err = np.abs(self.cmd - q).max()
        if err > ABORT_ERR:
            raise RuntimeError(f"tracking error {np.degrees(err):.1f} deg, aborting")
        return q

    def push(
        self, js: list[int], speed: float, leash: np.ndarray, max_travel: np.ndarray, seconds: float = np.inf
    ) -> dict[int, tuple[float, bool]]:
        """Drive joints `js` toward + until each stalls or `seconds` runs out; {joint: (position, stalled)}."""
        q = self.tick()
        start = q.copy()
        ref = {j: (0, q[j]) for j in js}  # (tick, position) that progress is measured from
        out: dict[int, tuple[float, bool]] = {}
        k = 0
        while (moving := [j for j in js if j not in out]) and k * DT < seconds:
            k += 1
            vel = np.zeros(len(self.cmd))
            vel[moving] = speed
            q = self.tick(vel)
            for j in moving:
                self.cmd[j] = np.clip(self.cmd[j] + speed * DT, q[j] - leash[j], q[j] + leash[j])
                if abs(q[j] - start[j]) > max_travel[j]:
                    raise RuntimeError(f"joint{j + 1} went {np.degrees(q[j] - start[j]):+.0f} deg without a stall")
                if self.cmd[j] - q[j] < STALL_LAG:
                    ref[j] = (k, q[j])
                elif (k - ref[j][0]) * DT >= STALL_T:
                    if abs(q[j] - ref[j][1]) < STALL_PROG:
                        self.cmd[j] = q[j]  # stop pushing
                        out[j] = (float(q[j]), True)
                    else:
                        ref[j] = (k, q[j])
        for j in moving:  # ran out of time
            self.cmd[j] = q[j]
            out[j] = (float(q[j]), False)
        return out

    def move(self, targets: dict[int, float], speed: float) -> np.ndarray:
        """Cosine-ease joints to `targets` (rad), settle 0.5 s, and return the settled positions."""
        a = {j: self.cmd[j] for j in targets}
        n = max(int(max(abs(t - a[j]) for j, t in targets.items()) / speed / DT), 30)
        for k in range(1, n + 1):
            for j, t in targets.items():
                self.cmd[j] = a[j] + (t - a[j]) * (0.5 - 0.5 * np.cos(np.pi * k / n))
            self.tick()
        return np.mean([self.tick() for _ in range(50)][-20:], axis=0)


def zero_arm(hold: Hold, backoff: np.ndarray, speed: float = 0.4) -> None:
    """Push each stage to its + stops, then back off `backoff` (rad per arm joint) to where zero belongs."""
    for js in STAGES:
        start = {j: hold.cmd[j] for j in js}
        stops = hold.push(js, speed, ZERO_FORCE / hold.kp[: len(ZERO_FORCE)], backoff + OVERTRAVEL)
        short = [j for j in js if stops[j][0] - start[j] < backoff[j] - SHORT]
        if short:
            hold.move(start, speed)
            raise RuntimeError(
                f"joint{'/'.join(str(j + 1) for j in short)} stalled far short of its back-off: its direction is "
                "probably reversed (run --mode dir-test) or the arm did not start near zero. Nothing written."
            )
        settled = hold.move({j: stops[j][0] - backoff[j] for j in js}, speed)
        for j in js:
            print(f"joint{j + 1}: stop {np.degrees(stops[j][0]):+8.2f} deg, zero at {np.degrees(settled[j]):+8.2f}")


def main(
    arm: Literal["yam_7dof_left", "yam_7dof_right"],
    channel: str = "can0",
    mode: Literal["zero", "dir-test", "home"] = "zero",
    gripper: str = "linear_4310",
) -> None:
    """Zero the arm against its hard stops, test its joint directions, or send it home.

    Args:
        arm: Which 7-DOF arm config to read motors, directions and joint6's linkage from.
        channel: SocketCAN channel the arm is on.
        mode: zero, dir-test or home.
        gripper: Gripper on the arm; zeroed at its closed stop in `zero` mode. no_gripper leaves it out.
    """
    arm_type = ArmType.from_string_name(arm)
    hw = _load_arm_config(arm_type)
    n_arm = len(hw.motor_list)
    motor_list = [list(m) for m in hw.motor_list]
    directions, kp, kd = list(hw.directions), hw.kp, hw.kd
    g = GripperType.from_string_name(gripper)
    if g.get_motor_type(arm_type):  # a teaching handle or no_gripper has no motor
        motor_list.append([hw.gripper_can_id, g.get_motor_type(arm_type)])
        directions.append(g.get_motor_direction(arm_type))
        g_kp, g_kd = g.get_motor_kp_kd(arm_type)
        kp, kd = np.append(hw.kp, g_kp), np.append(hw.kd, g_kd)

    chain = DMChainCanInterface(
        motor_list,
        np.zeros(len(motor_list)),
        np.array(directions),
        channel,
        motor_chain_name="zero_7dof",
        receive_mode=ReceiveMode.p16,
        # Both startup checks, as get_yam_robot runs them: a wrong motor type or scaling mis-scales every push.
        check_motor_types=True,
        check_motor_config=True,
        expected_registers=hw.motor_registers,
    )
    zeroed = False
    try:
        hold = Hold(chain, kp, kd)
        if mode == "zero":
            backoff = mujoco.MjModel.from_xml_path(arm_type.get_xml_path()).jnt_range[:, 1].copy()
            backoff[5] = hw.transmissions[5].motor_range[1]  # joint6's encoder reads the four-bar's crank
            zero_arm(hold, backoff)
            if len(motor_list) > n_arm:
                closed = detect_gripper_limits(chain, n_arm, kp, kd)[0]
                hold.cmd[n_arm] = hold.read()[n_arm]
                hold.move({n_arm: closed}, 0.4)
            zeroed = True
        elif mode == "dir-test":
            for j in range(n_arm):
                start = hold.cmd[j]
                pos, stalled = hold.push([j], 0.1, DIR_FORCE / hw.kp, np.full(n_arm, np.inf), seconds=3.0)[j]
                hold.move({j: start}, 0.1)
                print(f"joint{j + 1}: moved {np.degrees(pos - start):+.1f} deg" + (" (stalled)" if stalled else ""))
        else:
            hold.move(dict.fromkeys(range(n_arm), 0.0), 0.3)
    finally:
        chain.set_commands(np.zeros(len(motor_list)), get_state=False)  # release
        time.sleep(0.05)
        chain.close()
    if zeroed:
        for motor_id, motor_type in motor_list:
            set_zero.main(motor_id=motor_id, channel=channel, motor_type=motor_type)


if __name__ == "__main__":
    tyro.cli(main)
