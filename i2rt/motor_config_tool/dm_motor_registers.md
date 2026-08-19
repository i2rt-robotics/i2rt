# `dm_motor_registers.py`

Read and write a DM motor's configuration registers over CAN.

The protocol is the same one `utils.py` already speaks: broadcast arbitration ID `0x7FF`, read `0x33`,
write `0x55`, save-to-Flash `0xAA`. Registers are addressed by **decimal address**, and a value is a
little-endian `float32` or `uint32` carried in `data[4:8]`. The tool assumes a 1 Mbit/s bus, which is
what every YAM and Flow Base chain runs.

## Quick start

```bash
# Every register this tool knows about, with meanings — needs no CAN bus
python i2rt/motor_config_tool/dm_motor_registers.py list-registers

# One register, then the whole motor
python i2rt/motor_config_tool/dm_motor_registers.py read sw_ver --motor-id 1 --channel can0
python i2rt/motor_config_tool/dm_motor_registers.py read-all --motor-id 1 --channel can0
```

## Command line

```text
python i2rt/motor_config_tool/dm_motor_registers.py <subcommand> [register] [options]
```

| Subcommand | What it does |
| --- | --- |
| `read <register>` | Read one register and print its value |
| `read-all` | Read every register and print one table |
| `write <register> --value V` | Write a register **in RAM** |
| `save <register>` | Persist a register's current value **to Flash** |
| `list-registers` | Print the register table and meanings; opens no bus |

Options: `--motor-id` (required, the motor's CAN ID) and `--channel` (default `can0`; stations with two
chains use their own channel names). `<register>` is a **name** (`sw_ver`) or a **decimal address**
(`14`) — both resolve to the same register.

`--value` is read as an **integer** when typed without a `.` or an exponent, and as a **float**
otherwise. `uint32` registers only accept the integer form, so `--value 5` writes to `MST_ID` but
`--value 5.0` is rejected.

`read-all` reads all 45 registers in about 0.6 s on a healthy chain. A register that fails to read
shows `ERR` and the dump continues — but if the *first* register fails the run stops immediately,
because that means the motor is not answering at all and retrying the rest would spend ~28 s to say so.
The table goes to stdout and per-register warnings to stderr, so `read-all > dump.txt` stays clean.

Every usage error — an unknown register name, a write to a read-only register — is raised **before the
bus is opened**, so a typo never reaches the hardware.

## Safety rules

Read these once before using `write` or `save`.

### Write is RAM, save is Flash

`write` (`0x55`) only changes the value the motor is running on right now. `save` (`0xAA`) is what
survives a power cycle. Every persistent change is therefore **write, then save**, one register at a
time. A read also only ever reports RAM: the only proof a save stuck is to power-cycle and read again.

### Never `save` a register you have not just successfully written

Observed on a DM4340 (`sw_ver` 925970741): a `0x55` write to the read-only `dir` (address 55) was
silently discarded, and the `0xAA` that followed it was followed by **ten unrelated registers reverting**
from their runtime values to what Flash held — `UV_Value` 15→12, `KT_Value` 0→0.0513, `OT_Value`
100→115, `ACC` 2→314.2, `MAX_SPD` 600→31.4, `I_BW` 1000→230, and the four ASR/APR gains. Any register
that had been written but not saved lost its value.

Per-register write-then-save on *writable* registers is unaffected — that verified 19/19 on the same
bus — so the rule is about the pairing, not about `save` itself. Whether `0xAA` is block-wide, or
whether the reload is specific to saving a read-only register, is not established.

This tool refuses to write or save a read-only register outright, which blocks the exact trigger.

### A save reply is a short frame

A `0xAA` reply has **fewer than 8 bytes**, unlike the 8-byte echo a read or write returns. There is no
`data[4:8]` to decode, which is why `save_register_to_flash` returns the raw `can.Message`.

### Stop anything else using the bus

Register access needs an idle bus — one process, one thread. If a control loop, teleop session, or
recording is running, register exchanges fail with `Failed CAN exchange`.

They *fail* rather than return a plausible number, which is the point: a read reply is accepted only
when it echoes the request's motor ID (`data[0]`) and register address (`data[3]`), so another motor's
feedback frame arriving mid-transaction is dropped and retried instead of being decoded as this
register's value. Without that check a bulk read would silently shift every later register by one.

Writes are deliberately **not** echo-checked — see [Changing a motor's ID](#changing-a-motors-id).

### The confirmation prompt

`write` and `save` ask for confirmation on these eight registers:

`ESC_ID`, `MST_ID`, `CTRL_MODE`, `can_br`, `TIMEOUT`, `PMAX`, `VMAX`, `TMAX`

They either make a motor unreachable, change how it is driven, disarm the failsafe, or silently rescale
every MIT command and reading. The prompt names the register, says why it is guarded, and shows the
current value next to the requested one:

```text
MST_ID (@7) changes the id the motor sends feedback with; this library expects ESC_ID + 16.
  motor 4 on can0
  current value: 20  ->  requested: 21
write MST_ID? [y/N]
```

Pipe an answer in to script it: `echo y | python … write MST_ID --value 21 --motor-id 4`. With stdin
closed the command aborts with a message naming the register — it never proceeds unasked.

`read`, `read-all` and `list-registers` are never gated, and neither are the other writable registers:
`write MAX_SPD` / `save MAX_SPD` stays a non-interactive bench loop.

## `nan` from a register read is a reading, not a bug

Every 4-byte pattern is a valid `float32`, so a float register that never held a real value still
decodes to *something*: an unwritten Flash cell reads back `0xFFFFFFFF`, which is NaN. Older DM firmware
answers that for much of the table — on one DM4340, for most addresses above 30. The CLI prints it
verbatim as `nan`, which is distinct from the error a register that did not answer at all produces, and
that distinction is worth keeping: one motor is talking, the other is not.

Two consequences the code depends on:

- **A decode never rejects a value.** The tempting fix — raise on a non-finite decode — would reach past
  the read that provoked it, because the confirmation prompt reads a register *before* writing it. 21 of
  the 24 writable registers are floats, so raising there would make a register holding garbage impossible
  to *repair*, which is the opposite of what this tool is for. `value_fault()` is a predicate instead,
  and each caller decides: a read reports it, a write ignores the old value and proceeds, a verify counts
  it as "did not match" (`isclose(nan, x)` is already `False`).
- **A non-finite value is never *written*.** `nan`/`inf` pack to a legal `float32` and would sail through
  a read-back verify, since `isclose(inf, inf)` is `True` — so the value would be written, reported as
  verified, and then saved to Flash. On the protection thresholds (`OC_Value`, `OT_Value`, `UV_Value`,
  `OV_Value`) that is a trip that may then never fire, so the encoder refuses it.

## Changing a motor's ID

A motor's own CAN ID is **`ESC_ID` (address 8)**. **`MST_ID` (address 7) is the ID the motor uses for its
feedback frames.** This library's driver uses `ReceiveMode.p16` (see `i2rt/motor_drivers/utils.py`), i.e.
**feedback frame ID = motor ID + 16** — so when changing the ID you **must also set `MST_ID` to
`ESC_ID + 16`**, or the host receives no feedback and the motor looks as if it isn't connected.

Both IDs are reported in hex as well as decimal (`2 (0x02)`), because everything that talks about them on
the wire — the DM host tool, `candump`, this driver's own `0x200 + id` arbitration — uses hex, while
`--motor-id` takes decimal. It also makes the `+ 16` rule readable straight off a dump: +16 is +0x10, so
a correct pair is the same low nibble with the high one bumped by exactly 1 — `0x02` → `0x12`.

> **Before you start:** stop anything using the bus, and have **only the target motor connected** so a
> new ID cannot clash with an existing one. A full YAM chain is IDs 1–7.

Example — change motor **4 → 5**, so `MST_ID = 5 + 16 = 21`:

```bash
# 0) Read the current values first to confirm
python i2rt/motor_config_tool/dm_motor_registers.py read ESC_ID --motor-id 4
python i2rt/motor_config_tool/dm_motor_registers.py read MST_ID --motor-id 4

# 1) Change the feedback ID first, and save it — still addressed by the old ID 4
python i2rt/motor_config_tool/dm_motor_registers.py write MST_ID --value 21 --motor-id 4
python i2rt/motor_config_tool/dm_motor_registers.py save  MST_ID --motor-id 4

# 2) Then change the motor ID; it answers as the new ID 5 immediately afterwards
python i2rt/motor_config_tool/dm_motor_registers.py write ESC_ID --value 5 --motor-id 4
#    so the save is addressed by the NEW id
python i2rt/motor_config_tool/dm_motor_registers.py save  ESC_ID --motor-id 5

# 3) Power-cycle the motor, then verify with the new ID
python i2rt/motor_config_tool/dm_motor_registers.py read ESC_ID --motor-id 5   # expect 5
python i2rt/motor_config_tool/dm_motor_registers.py read MST_ID --motor-id 5   # expect 21
```

Key points:

- **Order matters**: change `MST_ID` before `ESC_ID`. The ID changes the instant `ESC_ID` is *written*,
  so do it last. The half-state where `MST_ID` moved but `ESC_ID` did not makes the motor answer at the
  old ID while transmitting at `new + 16` — indistinguishable from a dead motor.
- This is why writes are not echo-checked: an `ESC_ID` write is answered by the motor's *new* ID, so
  rejecting that reply would turn a successful renumber into a spurious failure.
- Always **power-cycle and verify** afterwards.

## MIT scaling: `PMAX`, `VMAX`, `TMAX`

These three registers are the most dangerous ones in the table, because getting them wrong produces no
error anywhere.

MIT frames pack position, velocity and torque into 16/12/12 bits as a fraction of a full-scale range.
`dm_driver` encodes using the **hard-coded** constants in `MotorType.get_motor_constants`
(`i2rt/motor_drivers/utils.py`); the **firmware** decodes using registers 21/22/23. The two are a shared
scale, and nothing checks that they agree.

When they disagree, everything stays self-consistent and is simply wrong by a constant factor in both
directions. A motor whose `PMAX` reads 3.1416 against a host assuming 12.5 moves 0.251 rad when
commanded 1.00 rad — and reports back 1.00 rad. A DM4310 whose `VMAX` reads 45 instead of 30 runs 1.5×
too fast.

Values this library expects:

| Motor type | `PMAX` | `VMAX` | `TMAX` |
| --- | ---: | ---: | ---: |
| DM3507 | 12.5 | 50 | 5 |
| DM4310 | 12.5 | 30 | 10 |
| DM4310V / DMH6215 / DM_FLOW_WHEEL | 3.1415926 | 30 | 10 |
| DM4340 | 12.5 | 10 | 28 |
| DM6248 | 12.5 | 20 | 120 |
| DM8009 | 12.5 | 45 | 54 |
| DMH6215MIT | 12.5 | 45 | 10 |

`read-all` is the quickest way to check a motor against this table; identify which row applies with
`Gr`, below.

## Identifying a motor from `Gr`

`Gr` (address 20) is the **gear reduction ratio**, and it is the one register that identifies a motor.
`PMAX`/`VMAX`/`TMAX` cannot: they are writable, so they can be wrong on a correct motor *and* right on
a wrong one. `Gr` is read-only and describes the physical gearbox.

A DM part number ends in its gear ratio — not in "80 → 80:1", which is the tempting misreading:

| `Gr` | Motor type | Provenance |
| ---: | --- | --- |
| 7 | `DM3507` | DM manual |
| 9 | `DM8009` | bench read, 2026-08-14 and 2026-08-18, flow base motor 9 (two bases) |
| 10 | `DM4310` | DM manual |
| 10 | `DM4310V` | bench read, 2026-08-18, flow base steering motors 1/3/7 |
| 10 | `DM_FLOW_WHEEL` | bench read, 2026-08-18, flow base drive motors 2/4/6/8 |
| 40 | `DM4340` | DM manual |
| 48 | `DM6248` | bench read, 2026-08-18, big_yam motor 2 |

Note that `Gr` maps **many-to-one**: 10 is a `DM4310`, a `DM4310V` *or* a `DM_FLOW_WHEEL`, so it can
distinguish classes of motor but not those three from each other. `DM_FLOW_WHEEL` is also a *role*
rather than a part number — the Flow Base's physical drive motor may differ per unit — so its row is one
station's reading, not a guarantee; see [`flow_base/README.md`](../flow_base/README.md).

`verify_motor_types` in `i2rt/motor_drivers/motor_check.py` reads `Gr` from every motor in a chain and
refuses to go on with one whose declared motor types it does not match. `DMChainCanInterface.__init__`
calls it, before it opens its socket, whenever it is built with `check_motor_types=True` — which arms
and the Flow Base both do. The mix-up it exists for is a `yam_ultra_2` arm
launched as `--arm yam_ultra`, whose configs differ only at joint 4 (`DM4340` vs `DM4310`), which would
otherwise encode that joint's torque 2.8x too large with no error anywhere. Its expected values come
from `MotorType.get_gear_ratio` in `i2rt/motor_drivers/utils.py`; a motor type absent from that table
raises rather than being compared against a guess.

`verify_motor_config`, the other function in that module, checks `CTRL_MODE` and `PMAX`/`VMAX`/`TMAX`;
arms and the Flow Base both run it (`check_motor_config=True`), though an arm is the stricter caller —
being a MIT chain that names no loop-critical subset, all three registers on all of its motors can refuse
the launch, where on the base only `PMAX`/`VMAX` on the four steering motors can. The chain runs the type
check first, and that
verdict is what keeps the config check from advising a `PMAX` rewrite — or writing `CTRL_MODE` to
Flash — on a motor that should be swapped instead.

## Code reference

```python
from i2rt.motor_config_tool.dm_motor_registers import read_register, write_register, save_register_to_flash
```

- `read_register(can_if, motor_id, reg)` — read. `reg` may be a **name**, a **decimal address**, a
  decimal-address **string**, or a `DMRegAddr` member.
- `write_register(can_if, motor_id, reg, value)` — write to RAM; raises `ValueError` on a read-only
  register and `ValueError` on a non-finite float.
- `save_register_to_flash(can_if, motor_id, reg)` — persist; returns the raw `can.Message`.
- `REG_BY_NAME` / `REG_BY_ADDR` — name/address → `RegSpec` (`addr`, `name`, `is_float`, `rw`).
- `DMRegAddr` — decimal-address `IntEnum` for code references.
- `value_fault(spec, value)` — why a reply is not a usable number, or `None`.
- `CTRL_MODE_ID_TO_NAME`, `GUARDED_REGISTERS`, `REG_MEANING`.

`can_if` is a `utils.RawCanInterface`, the same interface the other tools in this directory use.

**The library functions never prompt.** The confirmation gate is CLI-only, so a caller such as
`set_timeout.py` can write `TIMEOUT` as a plain function call without blocking on a console.

## Relationship to `utils.py`

`utils.py` keeps `RawCanInterface`, the four byte helpers, and three legacy helpers
(`get_special_message_response`, `write_special_message`, `save_to_memory`) over a 10-name subset of the
table (`id`, `master_id`, `timeout`, …) that `set_timeout.py` uses. Those helpers now **delegate to this
module**, which owns the authoritative register table — so there is one implementation, and the legacy
names keep working. New code should use this module directly.

## Register table

Every register known to the tool. This mirrors `_DM_TABLE` and `REG_MEANING` in the source, which stay
authoritative — `_check_tables()` fails at import if they drift apart. Pass either the **name** or the
**decimal address** as the register argument. `rw` = read/write, `ro` = read-only. Meanings are
summarized from the DM manual; see it for exact units and semantics. Registers tagged *calibration* or
*advanced tuning* are factory-set — don't change them without good reason.

| Addr | Name | Type | Access | Meaning |
| ---: | --- | --- | --- | --- |
| 0 | `UV_Value` | float | rw | Under-voltage protection threshold (V) |
| 1 | `KT_Value` | float | rw | Torque constant Kt (N·m/A) — the manual says N·m/mA, but a DM4340 reads 0.0513, which is only physical as N·m/A |
| 2 | `OT_Value` | float | rw | Over-temperature protection threshold (°C) |
| 3 | `OC_Value` | float | rw | Over-current protection threshold (A) |
| 4 | `ACC` | float | rw | Acceleration limit (position/speed modes) |
| 5 | `DEC` | float | rw | Deceleration limit (position/speed modes) |
| 6 | `MAX_SPD` | float | rw | Maximum speed limit |
| 7 | `MST_ID` | uint32 | rw | Master/feedback CAN ID — the ID the motor sends feedback frames with; this driver expects `ESC_ID + 16` |
| 8 | `ESC_ID` | uint32 | rw | The motor's own CAN (drive) ID |
| 9 | `TIMEOUT` | uint32 | rw | CAN loss-of-comms timeout (ms); the motor disables itself if idle past it (0 = off, no failsafe) |
| 10 | `CTRL_MODE` | uint32 | rw | Control mode: 1 MIT, 2 pos-speed, 3 speed, 4 torque-pos |
| 11 | `Damp` | float | ro | Damping coefficient |
| 12 | `Inertia` | float | ro | Rotor inertia |
| 13 | `hw_ver` | uint32 | ro | Hardware version |
| 14 | `sw_ver` | uint32 | ro | Firmware (software) version |
| 15 | `SN` | uint32 | ro | Serial number |
| 16 | `NPP` | uint32 | ro | Number of motor pole pairs |
| 17 | `Rs` | float | ro | Phase (stator) resistance (Ω) |
| 18 | `Ls` | float | ro | Phase (stator) inductance (H) |
| 19 | `Flux` | float | ro | Rotor flux linkage (Wb) |
| 20 | `Gr` | float | ro | Gear reduction ratio — the part number's last two digits; see [Identifying a motor from `Gr`](#identifying-a-motor-from-gr) |
| 21 | `PMAX` | float | rw | Max position; the ±range used to encode position in MIT mode (rad). Must match `POSITION_MAX` — see [MIT scaling](#mit-scaling-pmax-vmax-tmax) |
| 22 | `VMAX` | float | rw | Max velocity; the range used to encode velocity in MIT mode (rad/s). Must match `VELOCITY_MAX` |
| 23 | `TMAX` | float | rw | Max torque; the range used to encode torque in MIT mode (N·m). Must match `TORQUE_MAX` |
| 24 | `I_BW` | float | rw | Current-loop bandwidth (Hz) |
| 25 | `KP_ASR` | float | rw | Speed-loop (ASR) proportional gain |
| 26 | `KI_ASR` | float | rw | Speed-loop (ASR) integral gain |
| 27 | `KP_APR` | float | rw | Position-loop (APR) proportional gain |
| 28 | `KI_APR` | float | rw | Position-loop (APR) integral gain |
| 29 | `OV_Value` | float | rw | Over-voltage protection threshold (V) |
| 30 | `GREF` | float | rw | Gear efficiency — gearbox torque de-rating; default 1.0 |
| 31 | `Deta` | float | rw | Advanced/vendor tuning constant |
| 32 | `V_BW` | float | rw | Velocity-loop filter bandwidth (Hz); advanced tuning |
| 33 | `IQ_c1` | float | rw | q-axis current filter coefficient; advanced tuning |
| 34 | `VL_c1` | float | rw | Velocity filter coefficient; advanced tuning |
| 35 | `can_br` | uint32 | rw | CAN bus baud-rate setting |
| 36 | `sub_ver` | uint32 | ro | Firmware sub-version |
| 50 | `u_off` | float | ro | U-phase current-sensor offset (calibration) |
| 51 | `v_off` | float | ro | V-phase current-sensor offset (calibration) |
| 52 | `k1` | float | ro | Encoder linearization coefficient (calibration) |
| 53 | `k2` | float | ro | Encoder linearization coefficient (calibration) |
| 54 | `m_off` | float | ro | Encoder mechanical zero offset (calibration) |
| 55 | `dir` | float | ro | Motor rotation-direction sign |
| 80 | `p_m` | float | ro | Rotor (motor-side) mechanical position — runtime readout |
| 81 | `xout` | float | ro | Output-shaft position after the gearbox — runtime readout |
