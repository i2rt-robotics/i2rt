import argparse
import curses
import time
from typing import Any

import numpy as np

from i2rt.motor_drivers.dm_driver import DMChainCanInterface


def main(stdscr: Any) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="can0")
    parser.add_argument("--motor_id", type=int, default=1)
    parser.add_argument("--motor_type", type=str, default="DM4340")
    parser.add_argument("--kp", type=float, default=80.0)
    parser.add_argument("--kd", type=float, default=3.0)
    parser.add_argument("--can_receive_mode", type=str, default="p16")
    parser.add_argument("--rate_hz", type=float, default=200.0)  # control loop rate
    parser.add_argument("--step", type=float, default=0.01)  # rad per key press
    args = parser.parse_args()

    motor_list = [[args.motor_id, args.motor_type]]
    motor_directions = [1]
    motor_chain = DMChainCanInterface(
        motor_list, [0], motor_directions, channel=args.channel, receive_mode=args.can_receive_mode
    )

    # Read current position and use it as initial target
    motor_states = motor_chain.read_states()
    current_pos = motor_states[0].pos
    target_pos = float(current_pos)

    # curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    dt = 1.0 / args.rate_hz
    step = float(args.step)
    running = True
    last_print = 0.0

    try:
        while running:
            t0 = time.monotonic()

            # Non-blocking key read
            key = stdscr.getch()
            if key != -1:
                if key in (curses.KEY_RIGHT, ord("l")):  # → / 'l'
                    target_pos += step
                elif key in (curses.KEY_LEFT, ord("h")):  # ← / 'h'
                    target_pos -= step
                elif key in (curses.KEY_UP, ord("k")):  # ↑ / 'k' : increase step
                    step *= 1.25
                elif key in (curses.KEY_DOWN, ord("j")):  # ↓ / 'j' : decrease step
                    step = max(step / 1.25, 1e-4)
                elif key == ord("r"):  # reset target to current
                    motor_states = motor_chain.read_states()
                    target_pos = float(motor_states[0].pos)
                elif key == ord(" "):  # hold (no-op)
                    pass
                elif key in (ord("q"), 27):  # 'q' or ESC
                    running = False

            # Send PD command: pos=target_pos, vel_ff=0, kp/kd from args
            motor_chain.set_commands(
                np.array([0.0]),  # velocity feedforward or torque_ff (keep 0.0 per your API)
                np.array([target_pos]),  # position target
                np.array([0.0]),  # torque/effort ff (0.0)
                np.array([args.kp]),
                np.array([args.kd]),
            )

            # # Periodic status
            # now = time.monotonic()
            # if now - last_print > 0.1:
            #     motor_states = motor_chain.read_states()
            #     pos = float(motor_states[0].pos)
            #     stdscr.erase()
            #     stdscr.addstr(0, 0, "Arrow-key PD teleop (q to quit)")
            #     stdscr.addstr(1, 0, f"Current pos: {pos:+.5f} rad")
            #     stdscr.addstr(2, 0, f"Target  pos: {target_pos:+.5f} rad")
            #     stdscr.addstr(3, 0, f"Step size : {step:.5f} rad   (↑ bigger / ↓ smaller)")
            #     stdscr.addstr(4, 0, f"KP={args.kp:.2f}  KD={args.kd:.2f}")
            #     stdscr.addstr(6, 0, "Controls: ←/→ move • r reset-to-current • SPACE hold • q quit")
            #     stdscr.refresh()
            #     last_print = now

            # Periodic status
            now = time.monotonic()
            if now - last_print > 0.1:
                motor_states = motor_chain.read_states()
                state = motor_states[0]  # since you only have one motor in this script
                pos = float(state.pos)
                vel = float(state.vel)
                torque = float(state.eff)
                temp_rotor = state.temp_rotor
                temp_mos = state.temp_mos
                err = state.error_code

                stdscr.erase()
                stdscr.addstr(0, 0, "Arrow-key PD teleop (q to quit)")
                stdscr.addstr(1, 0, f"Current pos : {pos:+.5f} rad")
                stdscr.addstr(2, 0, f"Target  pos : {target_pos:+.5f} rad")
                stdscr.addstr(3, 0, f"Velocity    : {vel:+.5f} rad/s")
                stdscr.addstr(4, 0, f"Torque      : {torque:+.5f} Nm (or driver units)")
                stdscr.addstr(5, 0, f"Temp rotor  : {temp_rotor:.1f} °C   Temp MOS: {temp_mos:.1f} °C")
                stdscr.addstr(6, 0, f"Error code  : {err}")
                stdscr.addstr(8, 0, f"Step size   : {step:.5f} rad   (↑ bigger / ↓ smaller)")
                stdscr.addstr(9, 0, f"KP={args.kp:.2f}  KD={args.kd:.2f}")
                stdscr.addstr(11, 0, "Controls: ←/→ move • r reset-to-current • SPACE hold • q quit")
                stdscr.refresh()
                last_print = now

            # Sleep to maintain loop rate
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        # Send a final hold-at-current-position command for safety
        try:
            motor_states = motor_chain.read_states()
            hold_pos = float(motor_states[0].pos)
            motor_chain.set_commands(
                np.array([0.0]),
                np.array([hold_pos]),
                np.array([0.0]),
                np.array([args.kp]),
                np.array([args.kd]),
            )
        except Exception:
            pass


if __name__ == "__main__":
    curses.wrapper(main)
