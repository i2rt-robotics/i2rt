"""Passive motor torque monitor.

Sniffs feedback frames already on the CAN bus using candump (receive only). This process
opens no CAN socket and transmits nothing, so it is safe to run during a live teleop or
recording session. It only shows data while another process is polling the motors.

The MIT feedback frame already carries estimated torque: a 12-bit field packed into
bytes 4-5 (same frame as position, velocity, and the two temperature bytes). i2rt
decodes it in dm_driver.parse_feedback but then stores it as MotorInfo.eff, so it is
easy to miss. This tool reads that field directly off the wire.

Default target is the linear-rail lift motor (id 9, DM8009 on can_linearbot).
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque

from i2rt.motor_drivers.utils import MotorErrorCode, MotorType, ReceiveMode, uint_to_float

RECEIVE_MODE = ReceiveMode.p16

ARM_MOTOR_TYPES = {
    1: MotorType.DM4340,
    2: MotorType.DM4340,
    3: MotorType.DM4340,
    4: MotorType.DM4310,
    5: MotorType.DM4310,
    6: MotorType.DM4310,
    7: MotorType.DM4310,
}

# flow base / linear rail bus: four (steering, drive) wheel pairs then the lift motor
BASE_MOTOR_TYPES = {
    1: MotorType.DM4310V,
    2: MotorType.DM_FLOW_WHEEL,
    3: MotorType.DM4310V,
    4: MotorType.DM_FLOW_WHEEL,
    5: MotorType.DM4310V,
    6: MotorType.DM_FLOW_WHEEL,
    7: MotorType.DM4310V,
    8: MotorType.DM_FLOW_WHEEL,
    9: MotorType.DM8009,
}

BASE_LABELS = {
    1: "steer1",
    2: "drive1",
    3: "steer2",
    4: "drive2",
    5: "steer3",
    6: "drive3",
    7: "steer4",
    8: "drive4",
    9: "RAIL",
}

ARM_LABELS = {i: f"j{i}" for i in range(1, 8)}

# continuous ratings used as the gauge full-scale (encode range is much wider)
ARM_TORQUE_LIMITS = {MotorType.DM4340: 20.0, MotorType.DM4310: 3.5}
BASE_TORQUE_LIMITS = {
    MotorType.DM4310V: 3.5,
    MotorType.DM_FLOW_WHEEL: 3.5,
    MotorType.DM8009: 9.0,
}

PROFILES = {
    "arm": {"types": ARM_MOTOR_TYPES, "limits": ARM_TORQUE_LIMITS, "labels": ARM_LABELS, "channel": "can_follower_l"},
    "base": {"types": BASE_MOTOR_TYPES, "limits": BASE_TORQUE_LIMITS, "labels": BASE_LABELS, "channel": "can_linearbot"},
}

MOTOR_TYPES = BASE_MOTOR_TYPES
TORQUE_LIMITS = BASE_TORQUE_LIMITS
MOTOR_LABELS = BASE_LABELS

# rail bus is driven by flow_base even when arms are idle
SESSION_PATTERNS = [
    "flow_base_controller.py",
    "minimum_gello.py",
    "lerobot-record",
    "lerobot_record",
]

CLEAR = "\033[H\033[J"
WARM_FRAC = 0.50
HOT_FRAC = 0.75
OVER_FRAC = 1.00

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "grey": "\033[90m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "blue": "\033[34m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "alarm": "\033[1;97;41m",
}


def parse_ids(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def decode_feedback(motor_id, data):
    const = MotorType.get_motor_constants(MOTOR_TYPES.get(motor_id, MotorType.DM4310))
    error_int = (data[0] & 0xF0) >> 4
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | (data[4] >> 4)
    t_int = ((data[4] & 0xF) << 8) | data[5]
    return {
        "payload_id": data[0] & 0x0F,
        "error_code": error_int,
        "error_message": MotorErrorCode.get_error_message(error_int),
        "position": uint_to_float(p_int, const.POSITION_MIN, const.POSITION_MAX, 16),
        "velocity": uint_to_float(v_int, const.VELOCITY_MIN, const.VELOCITY_MAX, 12),
        "torque": uint_to_float(t_int, const.TORQUE_MIN, const.TORQUE_MAX, 12),
        "temp_mos": float(data[6]),
        "temp_rotor": float(data[7]),
        "timestamp": time.time(),
    }


def split_log_line(line):
    """Parse one candump -L line: (timestamp) channel ID#DATA."""
    parts = line.split()
    if len(parts) < 3:
        return None
    id_str, sep, data_str = parts[-1].partition("#")
    if not sep or len(data_str) < 16:
        return None
    try:
        return int(id_str, 16), bytes.fromhex(data_str[:16])
    except ValueError:
        return None


def session_procs():
    procs = {}
    for pattern in SESSION_PATTERNS:
        try:
            result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        except FileNotFoundError:
            return procs
        for line in result.stdout.strip().splitlines():
            pid, _, cmd = line.partition(" ")
            procs[pid] = cmd
    return procs


def torque_limit(motor_id, override=None):
    if override is not None:
        return override
    mtype = MOTOR_TYPES.get(motor_id)
    return TORQUE_LIMITS.get(mtype, 9.0)


def motor_label(motor_id):
    return MOTOR_LABELS.get(motor_id, MOTOR_TYPES.get(motor_id, f"id{motor_id}"))


class PassiveSniffer:
    """Reads candump output in a background thread. Never sends a CAN frame."""

    def __init__(self, channel, motor_ids, history_s, use_filter=True):
        self.channel = channel
        self.motor_ids = motor_ids
        self.id_map = {RECEIVE_MODE.get_receive_id(m): m for m in motor_ids}
        self.use_filter = use_filter
        self.history_s = history_s
        self.proc = None
        self.running = False
        self.lock = threading.Lock()
        self.states = {}
        self.frame_count = 0
        self.ids_seen = Counter()
        self.errors = []
        self.peak_abs = {}  # motor id -> (abs torque, signed torque, when)
        self.peak_pos = {}  # motor id -> (torque, when)
        self.peak_neg = {}
        self.latched = {}
        self.history = defaultdict(deque)  # motor id -> deque[(t, torque)]
        self.started = time.time()

    def _cmd(self):
        spec = self.channel
        if self.use_filter:
            spec = f"{self.channel},010:7F0"
        cmd = ["candump", "-L", spec]
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL"] + cmd
        return cmd

    def start(self):
        if shutil.which("candump") is None:
            raise RuntimeError("candump not found, install can-utils: sudo apt install can-utils")
        self.proc = subprocess.Popen(
            self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self.running = True
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _trim_history(self, motor_id, now):
        cutoff = now - self.history_s
        hist = self.history[motor_id]
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _read_stdout(self):
        for line in self.proc.stdout:
            if not self.running:
                break
            parsed = split_log_line(line)
            if parsed is None:
                continue
            arb_id, data = parsed
            with self.lock:
                self.frame_count += 1
                self.ids_seen[arb_id] += 1
                motor_id = self.id_map.get(arb_id)
                if motor_id is None:
                    continue
                state = decode_feedback(motor_id, data)
                self.states[motor_id] = state
                tq, ts = state["torque"], state["timestamp"]
                self.history[motor_id].append((ts, tq))
                self._trim_history(motor_id, ts)
                abs_tq = abs(tq)
                peak = self.peak_abs.get(motor_id)
                if peak is None or abs_tq > peak[0]:
                    self.peak_abs[motor_id] = (abs_tq, tq, ts)
                pos = self.peak_pos.get(motor_id)
                if pos is None or tq > pos[0]:
                    self.peak_pos[motor_id] = (tq, ts)
                neg = self.peak_neg.get(motor_id)
                if neg is None or tq < neg[0]:
                    self.peak_neg[motor_id] = (tq, ts)
                if state["error_code"] != MotorErrorCode.normal:
                    self.latched[motor_id] = (state["error_code"], ts)

    def _read_stderr(self):
        for line in self.proc.stderr:
            line = line.strip()
            if line:
                with self.lock:
                    self.errors.append(line)

    def snapshot(self):
        with self.lock:
            return {
                "states": dict(self.states),
                "count": self.frame_count,
                "ids_seen": Counter(self.ids_seen),
                "errors": list(self.errors),
                "peak_abs": dict(self.peak_abs),
                "peak_pos": dict(self.peak_pos),
                "peak_neg": dict(self.peak_neg),
                "latched": dict(self.latched),
                "history": {k: list(v) for k, v in self.history.items()},
                "started": self.started,
            }

    def stop(self):
        self.running = False
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def want_colour(mode, stream):
    if mode == "never":
        return False
    if mode == "always":
        return True
    return "NO_COLOR" not in os.environ and hasattr(stream, "isatty") and stream.isatty()


def want_unicode(stream):
    try:
        "\u2588\u2591\u2502\u2500\u2584\u2580".encode(getattr(stream, "encoding", None) or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def paint(text, code, colour):
    return f"{code}{text}{ANSI['reset']}" if colour and code else text


def torque_colour(value, limit):
    frac = abs(value) / limit if limit > 0 else 0.0
    if frac >= OVER_FRAC:
        return ANSI["alarm"]
    if frac >= HOT_FRAC:
        return ANSI["red"]
    if frac >= WARM_FRAC:
        return ANSI["yellow"]
    if value < 0:
        return ANSI["cyan"]
    return ANSI["green"]


def signed_bar(value, peak_pos, peak_neg, limit, width, colour, unicode_ok):
    """Centered bar: left is negative, right is positive. Peak ticks on both sides."""
    fill_ch, empty_ch, mark_ch, zero_ch = (
        ("\u2588", "\u2591", "\u2502", "\u2502") if unicode_ok else ("#", ".", "|", "|")
    )
    if width < 9:
        width = 9
    if width % 2 == 0:
        width += 1
    half = width // 2

    def cells_from_zero(v):
        return int(round(min(abs(v), limit) / max(limit, 1e-6) * half))

    body = [empty_ch] * width
    codes = [ANSI["grey"]] * width
    body[half] = zero_ch
    codes[half] = ANSI["bold"] + ANSI["grey"]
    n = cells_from_zero(value)
    if value >= 0:
        for i in range(half + 1, min(half + 1 + n, width)):
            body[i] = fill_ch
            codes[i] = torque_colour(value, limit)
    else:
        for i in range(max(half - n, 0), half):
            body[i] = fill_ch
            codes[i] = torque_colour(value, limit)

    def mark(v):
        if v is None:
            return
        offset = cells_from_zero(v)
        if v >= 0:
            i = min(half + offset, width - 1)
        else:
            i = max(half - offset, 0)
        body[i] = mark_ch
        codes[i] = ANSI["bold"] + torque_colour(v, limit)

    mark(peak_pos)
    mark(peak_neg)
    out = []
    i = 0
    while i < width:
        j = i
        while j < width and codes[j] == codes[i]:
            j += 1
        out.append(paint("".join(body[i:j]), codes[i], colour))
        i = j
    return "[" + "".join(out) + "]"


def bucket_history(samples, width, now, window_s):
    """Downsample (t, value) samples into `width` latest-value buckets."""
    if width <= 0:
        return []
    buckets = [None] * width
    if not samples:
        return buckets
    start = now - window_s
    span = max(window_s, 1e-6)
    for t, v in samples:
        if t < start:
            continue
        i = int((t - start) / span * width)
        if i < 0:
            i = 0
        elif i >= width:
            i = width - 1
        buckets[i] = v
    last = None
    filled = []
    for v in buckets:
        if v is not None:
            last = v
        filled.append(last)
    return filled


def plot_history(values, limit, height, colour, unicode_ok):
    """Signed ASCII/unicode time-series. `values` is oldest-first, one per column."""
    if not values or height < 3:
        return []
    width = len(values)
    zero_row = height // 2
    block = "\u2588" if unicode_ok else "#"
    empty = " "
    axis = "\u2500" if unicode_ok else "-"
    rows = [[empty] * width for _ in range(height)]
    row_codes = [[ANSI["grey"]] * width for _ in range(height)]
    for c, v in enumerate(values):
        rows[zero_row][c] = axis
        row_codes[zero_row][c] = ANSI["dim"] + ANSI["grey"]
        if v is None:
            continue
        frac = max(-1.0, min(1.0, v / max(limit, 1e-6)))
        target = zero_row - int(round(frac * zero_row))
        target = max(0, min(height - 1, target))
        lo, hi = (target, zero_row) if target <= zero_row else (zero_row, target)
        code = torque_colour(v, limit)
        for r in range(lo, hi + 1):
            rows[r][c] = block
            row_codes[r][c] = code
        rows[target][c] = block
        row_codes[target][c] = ANSI["bold"] + code

    lines = []
    for r in range(height):
        if r == 0:
            label = f"+{limit:4.1f}"
        elif r == zero_row:
            label = "  0.0"
        elif r == height - 1:
            label = f"-{limit:4.1f}"
        else:
            label = "     "
        parts = []
        c = 0
        while c < width:
            d = c
            while d < width and row_codes[r][d] == row_codes[r][c] and rows[r][d] == rows[r][c]:
                d += 1
            parts.append(paint(rows[r][c] * (d - c), row_codes[r][c], colour))
            c = d
        prefix = paint(label, ANSI["grey"], colour)
        lines.append(f" {prefix} {''.join(parts)}")
    return lines


def sparkline(values, limit, colour, unicode_ok):
    glyphs = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588" if unicode_ok else " .:-=+*#"
    n = len(glyphs) - 1
    out = []
    for v in values:
        if v is None:
            out.append(paint(" ", ANSI["grey"], colour))
            continue
        frac = min(abs(v) / max(limit, 1e-6), 1.0)
        idx = max(1, int(round(frac * n))) if abs(v) > 1e-6 else 0
        out.append(paint(glyphs[idx], torque_colour(v, limit), colour))
    return "".join(out)


def render(args, sniffer, motor_ids, rate, session, colour, unicode_ok):
    snap = sniffer.snapshot()
    states = snap["states"]
    now = time.time()
    columns = shutil.get_terminal_size((100, 24)).columns
    name_w = max(6, max((len(motor_label(m)) for m in motor_ids), default=6))
    bar_w = max(17, min(41, columns - 62 - name_w))
    if bar_w % 2 == 0:
        bar_w += 1
    bus = "ACTIVE" if rate > 0 else "SILENT (nothing is polling the motors)"
    out = []
    out.append(f"passive torque monitor   channel={args.channel}   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(
        f"frames={snap['count']}  rate={rate:6.0f}/s  bus={bus}  "
        f"history={args.history:.0f}s  session={session}"
    )
    out.append(
        "gauge \u00b1continuous rating   "
        + paint(f"green/cyan<{WARM_FRAC * 100:.0f}%", ANSI["green"], colour)
        + "  "
        + paint(f"yellow<{HOT_FRAC * 100:.0f}%", ANSI["yellow"], colour)
        + "  "
        + paint(f"red>={HOT_FRAC * 100:.0f}%", ANSI["red"], colour)
        + "  "
        + paint("OVER", ANSI["alarm"], colour)
        + paint("   | peak", ANSI["grey"], colour)
    )

    faults = [(m, s) for m, s in sorted(states.items()) if s["error_code"] != MotorErrorCode.normal]
    if faults:
        out.append("")
        for motor_id, state in faults:
            out.append(
                paint(
                    f"  !!  FAULT  {motor_label(motor_id)} id {motor_id}: {state['error_message']} "
                    f"(0x{state['error_code']:X})  torque {state['torque']:+.2f}Nm  !!",
                    ANSI["alarm"],
                    colour,
                )
            )

    out.append("")
    out.append(
        f"{'ID':>3}  {'NAME':<{name_w}}  {'TORQUE':>8}  "
        f"{'GAUGE':<{bar_w + 2}}{'PEAK':>8}{'VEL':>8}{'POS':>9}{'AGE':>6}  STATUS"
    )
    for motor_id in motor_ids:
        name = motor_label(motor_id)
        limit = torque_limit(motor_id, args.limit)
        state = states.get(motor_id)
        if state is None:
            blank = signed_bar(0.0, None, None, limit, bar_w, colour, unicode_ok)
            out.append(
                f"{motor_id:>3}  {name:<{name_w}}  {'--':>8}  {blank}{'--':>8}{'--':>8}{'--':>9}{'--':>6}  NO DATA"
            )
            continue
        age = now - state["timestamp"]
        tq = state["torque"]
        peak_pos = snap["peak_pos"].get(motor_id, (tq, None))[0]
        peak_neg = snap["peak_neg"].get(motor_id, (tq, None))[0]
        peak_abs = snap["peak_abs"].get(motor_id, (abs(tq), tq, None))
        if state["error_code"] == MotorErrorCode.normal:
            status = "ok"
        else:
            status = paint(f"{state['error_message']} (0x{state['error_code']:X})", ANSI["alarm"], colour)
        if age > args.stale_after:
            status = f"STALE {age:.1f}s ({status})"
        if motor_id in snap["latched"] and state["error_code"] == MotorErrorCode.normal:
            code, when = snap["latched"][motor_id]
            status += paint(
                f"  [had 0x{code:X} {MotorErrorCode.get_error_message(code)} at {time.strftime('%H:%M:%S', time.localtime(when))}]",
                ANSI["red"],
                colour,
            )
        if abs(tq) > limit:
            status = paint(f"OVER {limit:.1f}Nm", ANSI["alarm"], colour) + f"  {status}"
        bar = signed_bar(tq, peak_pos, peak_neg, limit, bar_w, colour, unicode_ok)
        tq_txt = paint(f"{tq:>+8.2f}", torque_colour(tq, limit), colour)
        peak_txt = paint(f"{peak_abs[1]:>+8.2f}", torque_colour(peak_abs[1], limit), colour)
        extra = ""
        if args.meters_per_rad and motor_id == 9:
            force = tq / args.meters_per_rad
            extra = paint(f"  {force:+.0f} N", torque_colour(tq, limit), colour)
        out.append(
            f"{motor_id:>3}  {name:<{name_w}}  {tq_txt}  {bar}{peak_txt}"
            f"{state['velocity']:>+8.2f}{state['position']:>+9.3f}{age:>6.2f}  {status}{extra}"
        )
        if len(motor_ids) > 1:
            hist = bucket_history(snap["history"].get(motor_id, []), max(20, min(48, columns - 20)), now, args.history)
            out.append(f"     {sparkline(hist, limit, colour, unicode_ok)}")

    graph_id = args.graph_id if args.graph_id in motor_ids else (9 if 9 in motor_ids else motor_ids[0])
    graph_limit = torque_limit(graph_id, args.limit)
    graph_w = max(24, min(columns - 10, 88))
    hist = bucket_history(snap["history"].get(graph_id, []), graph_w, now, args.history)
    out.append("")
    title = (
        f"torque vs time  {motor_label(graph_id)} id={graph_id}  "
        f"last {args.history:.0f}s  scale \u00b1{graph_limit:.1f} Nm"
    )
    out.append(paint(title, ANSI["bold"], colour))
    out.extend(plot_history(hist, graph_limit, args.graph_height, colour, unicode_ok))
    out.append(paint(f"      {args.history:.0f}s ago" + " " * max(0, graph_w - 16) + "now", ANSI["grey"], colour))

    if states:
        hot_id, hot_peak = max(snap["peak_abs"].items(), key=lambda kv: kv[1][0])
        live_id, live = max(states.items(), key=lambda kv: abs(kv[1]["torque"]))
        out.append("")
        out.append(
            f"highest |torque| now: {motor_label(live_id)} {live['torque']:+.2f} Nm    "
            f"peak since start: {motor_label(hot_id)} {hot_peak[1]:+.2f} Nm "
            f"({time.strftime('%H:%M:%S', time.localtime(hot_peak[2]))})"
        )
    else:
        expected = ", ".join(f"0x{RECEIVE_MODE.get_receive_id(m):03X}" for m in motor_ids)
        out.append("")
        out.append(f"no feedback frames decoded, expected arbitration ids: {expected}")
        if snap["ids_seen"]:
            seen = ", ".join(f"0x{i:03X}(x{c})" for i, c in sorted(snap["ids_seen"].items()))
            out.append(f"arbitration ids seen: {seen}")
        else:
            out.append("bus is quiet: start flow_base / a recording, or wait for --wait-for-recording")
    for err in snap["errors"][:5]:
        out.append(f"candump: {err}")
    return "\n".join(out)


def wait_for_session(poll=1.0):
    print("waiting for flow_base / recording / teleop session ...")
    spinner = "|/-\\"
    i = 0
    while True:
        procs = session_procs()
        if procs:
            print(f"\nsession detected ({len(procs)} process(es)):")
            for pid, cmd in sorted(procs.items()):
                print(f"  pid {pid}: {cmd[:110]}")
            print()
            return procs
        sys.stdout.write(f"\r  {spinner[i % 4]} no session yet, still waiting ")
        sys.stdout.flush()
        i += 1
        time.sleep(poll)


def run_monitor(args, motor_ids, track_session):
    colour = want_colour(args.color, sys.stdout)
    unicode_ok = want_unicode(sys.stdout)
    sniffer = PassiveSniffer(args.channel, motor_ids, history_s=args.history, use_filter=not args.no_filter)
    sniffer.start()
    period = 1.0 / args.hz if args.hz > 0 else 0.2
    prev_count = 0
    prev_time = time.time()
    missing_session = 0
    try:
        if args.once:
            deadline = time.time() + args.settle
            while time.time() < deadline:
                if all(m in sniffer.snapshot()["states"] for m in motor_ids):
                    break
                time.sleep(0.05)
            count = sniffer.snapshot()["count"]
            rate = count / max(time.time() - prev_time, 1e-6)
            procs = session_procs()
            session = f"LIVE ({len(procs)} process(es))" if procs else "none detected"
            print(render(args, sniffer, motor_ids, rate, session, colour, unicode_ok))
            return True
        start = time.time()
        while True:
            time.sleep(period)
            now = time.time()
            count = sniffer.snapshot()["count"]
            rate = (count - prev_count) / max(now - prev_time, 1e-6)
            prev_count, prev_time = count, now
            procs = session_procs()
            if procs:
                session = f"LIVE ({len(procs)} process(es))"
                missing_session = 0
            else:
                session = "none detected"
                missing_session += 1
            if not args.plain:
                sys.stdout.write(CLEAR)
            print(render(args, sniffer, motor_ids, rate, session, colour, unicode_ok))
            print("\nctrl-c to quit")
            sys.stdout.flush()
            if track_session and missing_session >= 2:
                print("\nsession ended, no flow_base / recording process remains")
                return False
            if args.duration > 0 and now - start >= args.duration:
                return True
    finally:
        sniffer.stop()


def main():
    parser = argparse.ArgumentParser(description="passive DM motor torque monitor, receive only")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="base")
    parser.add_argument("--channel", type=str, default=None, help="defaults to the profile's bus (can_linearbot for base)")
    parser.add_argument("--ids", type=str, default="9", help="motor ids, default 9 = linear rail lift")
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--history", type=float, default=30.0, help="seconds of torque history in the graph")
    parser.add_argument("--graph-height", type=int, default=11)
    parser.add_argument("--graph-id", type=int, default=9, help="motor shown in the large time-series plot")
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="gauge full scale in Nm (default: motor continuous rating, 9 for the rail DM8009)",
    )
    parser.add_argument(
        "--meters-per-rad",
        type=float,
        default=0.0,
        help="if set, also print rail linear force F = torque / meters_per_rad (N)",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wait-for-recording", action="store_true")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--stale-after", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=1.5, help="seconds to collect frames for --once")
    parser.add_argument("--duration", type=float, default=0.0, help="exit after N seconds, 0 means forever")
    parser.add_argument("--no-filter", action="store_true", help="do not apply the candump receive filter")
    parser.add_argument("--plain", action="store_true", help="do not clear the screen between refreshes")
    args = parser.parse_args()

    global MOTOR_TYPES, TORQUE_LIMITS, MOTOR_LABELS
    profile = PROFILES[args.profile]
    MOTOR_TYPES = profile["types"]
    TORQUE_LIMITS = profile["limits"]
    MOTOR_LABELS = profile["labels"]
    if args.channel is None:
        args.channel = profile["channel"]

    motor_ids = parse_ids(args.ids)

    try:
        if args.wait_for_recording:
            while True:
                wait_for_session()
                run_monitor(args, motor_ids, track_session=True)
                if args.once or args.duration > 0:
                    return
                print("re-arming, will resume when the next session starts\n")
        else:
            run_monitor(args, motor_ids, track_session=False)
    except KeyboardInterrupt:
        print("\nstopped")
    except RuntimeError as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
