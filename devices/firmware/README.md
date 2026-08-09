# Upgrading the teaching-handle encoder firmware

This guide walks you end to end through updating the firmware on the **ioheart** encoder — the
passive joint encoder inside the `yam_teaching_handle` (leader arm). It covers what you need, how to
do the upgrade, how to confirm it worked, and how to recover if any step fails.

Board: **STM32F103C8T6**. Firmware shipped here: **v2.4.0**.

**The short version:** check the version → flash over CAN → confirm the new version → re-check the
zero position. Most upgrades are one command and take under a minute. If anything goes wrong, jump
to [Recovery](#5-recovery-when-a-step-fails) — the encoder is very hard to permanently break, because
the CAN flashing tool never touches the bootloader.

---

## 1. What you need

### Hardware

- The teaching handle, powered, with its CAN bus connected to your computer (a SocketCAN adapter,
  e.g. a USB-CAN device that shows up as `can0`).
- The leader-arm CAN bus normally carries this one encoder and nothing else. That is the ideal
  setup for flashing.
- *Only if you end up needing the SWD path:* a SEGGER J-Link probe and the J-Link software pack.
  Most customers never need this.

### Software

Everything for the CAN path ships with this repo. From the repo root:

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .          # brings in python-can, crcmod-plus, tyro
```

The two commands used throughout:

| Command | Purpose |
| --- | --- |
| `python -m i2rt.utils.can_flash` | flashing over CAN (`--channel`, `--bitrate`, `--device-id`) |
| `python -m i2rt.utils.encoder_manager <subcommand>` | live encoder ops (`--bus`, `--device`, `--bitrate`) |

Note the flag difference: `can_flash` takes `--channel`, `encoder_manager` takes `--bus`. Both
default to `can0` at 1 Mbps. Add `--help` to either for the current options.

### Firmware images in this directory

| File | Size | Contents | Load address | Used by |
| --- | --- | --- | --- | --- |
| `ioheart-f103-v2.4.0.bin` | 30,476 B | Application only | `0x08003000` | CAN upgrade (§3) |
| `ioheart-f103-combined-v2.4.0.bin` | 43,008 B | Bootloader + application | `0x08000000` | J-Link recovery (§6) |

---

## 2. Check what you have now

### 2a. Bring up the CAN interface at 1 Mbps

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
# or, for every CAN interface on the machine:
./scripts/reset_all_can.sh
```

### 2b. Find the encoder and read its version

```bash
uv run python -m i2rt.utils.encoder_manager list-devices --bus can0
uv run python -m i2rt.utils.encoder_manager get-version --bus can0
```

`get-version` prints one line per device:

```text
VersionReply(device=1, major=2, minor=4, patch=0)
```

Write down the **device number** and the **version** — you need both later. If either command finds
nothing, go to [R1](#r1-the-encoder-does-not-appear-on-the-bus).

### 2c. Record the current configuration

Flashing replaces the application. Note your zero position first so you can tell whether it survived
and restore it if not:

```bash
uv run python -m i2rt.utils.encoder_manager read-eeprom-zpos --bus can0
```

### 2d. Choose your path

| Version reported | What to do |
| --- | --- |
| **>= 2.2.20** | [§3 Upgrade over CAN](#3-upgrade-over-can-normal-path) — one command, no disassembly |
| **< 2.2.20** | [§6 Upgrade over SWD](#6-upgrade-over-swd-with-a-j-link) — these boards have no CAN flashing support |
| Nothing reported at all | [R1](#r1-the-encoder-does-not-appear-on-the-bus) |

> Compare the patch number **numerically**: `2.2.9` is *older* than `2.2.20`, so a board reporting
> `2.2.9` still needs the J-Link path.
>
> The tool does not check this for you — there is no version gate in `can_flash.py`. On a board below
> 2.2.20 the CAN upgrade simply fails with `device N never entered bootloader mode within 30s`.
> Once a board has been brought up to >= 2.2.20 once via J-Link, every later upgrade can use CAN.

---

## 3. Upgrade over CAN (normal path)

Run from the repo root:

```bash
uv run python -m i2rt.utils.can_flash devices/firmware/ioheart-f103-v2.4.0.bin \
    --channel can0 --bitrate 1000000
```

You do not need to put the encoder into any special mode — the tool discovers it, restarts it into
the bootloader, and flashes it.

**By default every encoder found on the bus is flashed.** On a normal leader-arm bus that is exactly
one device, which is what you want. To be explicit and protect against flashing the wrong bus, pass
the device number you noted in §2b:

```bash
uv run python -m i2rt.utils.can_flash devices/firmware/ioheart-f103-v2.4.0.bin \
    --channel can0 --device-id 1
```

**What a healthy run looks like:** device discovery, then a line per 1 KiB page ending in `CRC OK`,
then a whole-image CRC-32 check, then:

```text
Device flashed successfully
Device will restart automatically after timeout
```

and a final summary listing which device IDs succeeded and which failed.

Each page is retried up to 10 times, so occasional retry lines are not a problem on their own — only
a final failure is. **Do not power off the handle while pages are being written.** If it does get
interrupted, that is recoverable — see [R3](#r3-the-flash-was-interrupted-partway-through).

The board restarts on its own a few seconds after the flash completes.

---

## 4. Verify the upgrade

Four checks, in order. All four should pass before you call the handle good.

### Check 1 — the new version is running

```bash
uv run python -m i2rt.utils.encoder_manager get-version --bus can0
# expect: VersionReply(device=1, major=2, minor=4, patch=0)
```

If this still shows the old version, the board booted the old app — re-run §3. If it prints
`No version`, the board is probably still sitting in the bootloader; see
[R4](#r4-flash-reported-success-but-the-encoder-does-not-come-back).

### Check 2 — the encoder reports data

```bash
uv run python -m i2rt.utils.encoder_manager get-report --bus can0
```

### Check 3 — the readings move and are sane

Run the same command while slowly moving the handle joint by hand, and confirm the position value
tracks the motion smoothly with no jumps.

### Check 4 — the zero position is still correct

```bash
uv run python -m i2rt.utils.encoder_manager read-eeprom-zpos --bus can0
```

Compare against what you recorded in §2c. If it changed or the handle's home pose is visibly off,
move the joint to its true zero and re-set it:

```bash
uv run python -m i2rt.utils.encoder_manager reset-zero-position --bus can0 --restart
```

---

## 5. Recovery: when a step fails

**Read this first — it is the reason most failures are not serious.** The CAN flashing tool only
ever writes the *application* region of flash (`0x08003000` and up). It cannot damage the bootloader
that lives below it. So even a flash that dies halfway through leaves the bootloader intact and the
board still reachable over CAN. **In almost every case the fix is simply to run the same flash
command again** — flashing always starts over from page 0, so there is no half-finished state to
clean up first.

Only a corrupted *bootloader* requires the J-Link path, and normal CAN flashing cannot cause that.

### R1. The encoder does not appear on the bus

`list-devices` finds nothing, or `can_flash` exits with `No encoder devices found on the bus.`

Work through these in order:

1. **Is the interface up at the right bitrate?** `ip -details link show can0` — it must say `can`
   state `UP` with `bitrate 1000000`. Re-run §2a if not.
2. **Is anything on the wire at all?** `candump can0` should show traffic. Silence points at power,
   wiring, or CAN termination rather than firmware.
3. **Right channel?** If you have several adapters, try `--bus can1` / `--channel can1`.
4. **Power-cycle the handle** and retry.
5. Still nothing → the app may be missing or corrupt. Go to
   [§6 SWD](#6-upgrade-over-swd-with-a-j-link) with the combined image.

### R2. `device N never entered bootloader mode within 30s`

The encoder answered, but would not go into flashing mode.

1. **Check the version — this is the most common cause.** Firmware below **2.2.20** has no CAN
   flashing support at all. Run `get-version`; if it is older, you must use
   [§6 SWD](#6-upgrade-over-swd-with-a-j-link).
2. If the version is >= 2.2.20, ask it to restart and immediately re-run the flash:

   ```bash
   uv run python -m i2rt.utils.encoder_manager restart --bus can0
   ```

3. Otherwise power-cycle the handle and start the flash command right away — the tool retries
   discovery for 30 seconds and will catch the board during its startup window.

### R3. The flash was interrupted partway through

Power loss, unplugged cable, or Ctrl-C during page writes. Symptoms afterwards: the encoder does not
answer `get-version` (prints `No version`), but `list-devices` still shows a device — it is sitting
in the bootloader with a half-written application.

**This is a normal, fully recoverable state, not a brick.** Reconnect, then run the exact same
command from §3 again. It rewrites every page from the start.

### R4. Flash reported success but the encoder does not come back

Wait ~10 seconds first; the board restarts on a bootloader timeout, not instantly.

1. `uv run python -m i2rt.utils.encoder_manager list-devices --bus can0` — if a device is listed but
   `get-version` says `No version`, it is still in the bootloader. Power-cycle the handle.
2. Still no version after a power cycle → re-run §3.
3. Re-runs keep ending the same way → [§6 SWD](#6-upgrade-over-swd-with-a-j-link) with the combined image.

### R5. `Page write failed` after retries, or a CRC error

Seen as repeated page retries, `Page write failed`, or `Bootloader command 3 error #N` at the final
verification step. This is almost always bus quality, not the board.

1. Re-seat the CAN connector and check termination.
2. Confirm both ends really are at 1 Mbps.
3. Disconnect other CAN nodes and flash with only the encoder on the bus.
4. Re-run §3 — a fresh run rewrites and re-verifies everything.

### R6. `Could not connect to device.`

The bootloader was found but the handshake did not complete, usually because other nodes are
flooding the bus. Flash on an isolated bus with only the encoder attached.

### R7. Some devices flashed, others failed

The end-of-run summary lists failed device IDs. Re-run for just those, one at a time:

```bash
uv run python -m i2rt.utils.can_flash devices/firmware/ioheart-f103-v2.4.0.bin \
    --channel can0 --device-id <failed-id>
```

### R8. You flashed the *combined* image over CAN by mistake

The app will not boot. The file is under the tool's size limit so nothing stops you, but the
bootloader writes it starting at the application base, so the bootloader's own bytes land at
`0x08003000` and there is no valid app there.

**The board is fine.** The real bootloader is untouched — just flash the correct app-only image
(`ioheart-f103-v2.4.0.bin`) over CAN as in §3.

### R9. Nothing above worked

Use [§6 SWD](#6-upgrade-over-swd-with-a-j-link) with `ioheart-f103-combined-v2.4.0.bin` at
`0x08000000`. That rewrites bootloader and application together and recovers any state the CAN path
cannot. If SWD also cannot connect, contact I2RT support with: the output of `get-version` and
`list-devices`, the full console output of the failed flash, and which recovery steps you tried.

---

## 6. Upgrade over SWD with a J-Link

Needed only when: the firmware is older than **2.2.20**, the MCU is blank, the bootloader is
corrupted, or the CAN recovery steps above have all failed.

**Wiring:** `SWDIO`, `SWCLK`, `GND`, and `VTref` (3.3 V reference). `nRESET` is optional but makes
connecting more reliable. The target must be powered — a J-Link does not supply power on the
standard 20-pin connector.

### Steps

1. **New project** in J-Flash → *Create a new project*.
2. **Target device:** `STM32F103C8` — the board is an STM32F103C8T6 (64 KiB flash, 1 KiB pages).
   J-Flash lists it under the base part number, without the `T6` package suffix.
3. **Target interface:** `SWD`, speed `4000 kHz` (drop to `1000 kHz` if the connection is flaky).
4. `Target → Connect`. The log should report a Cortex-M3 core and 64 KiB of flash.
5. `File → Open data file…` → pick the image and enter its start address when prompted:

   | Goal | File | Start address |
   | --- | --- | --- |
   | Full program / recover a dead board / firmware < 2.2.20 | `ioheart-f103-combined-v2.4.0.bin` | `0x08000000` |
   | Replace only the app, keep the existing bootloader | `ioheart-f103-v2.4.0.bin` | `0x08003000` |

   **Getting this address wrong is the most common mistake here.** A `.bin` file carries no address
   information, so J-Flash writes to exactly the address you type.
6. `Target → Production Programming` (erase + program + verify).
7. `Target → Disconnect`, power-cycle the board, then verify over CAN using [§4](#4-verify-the-upgrade).

### Command-line equivalent

```text
J-Link> connect
Device> STM32F103C8
TIF> S
Speed> 4000
J-Link> loadfile ioheart-f103-combined-v2.4.0.bin 0x08000000
J-Link> verifybin ioheart-f103-combined-v2.4.0.bin 0x08000000
J-Link> r
J-Link> g
J-Link> exit
```

### SWD troubleshooting

| Symptom | Fix |
| --- | --- |
| Cannot connect / no core found | Check `VTref` and `GND`, confirm the target is powered, lower speed to 1000 kHz, connect with `nRESET` held. |
| Programming refused | Read-out protection is enabled. `unlock STM32` in `JLinkExe` clears it — this mass-erases the chip. |
| Verify fails | Lower the SWD speed and retry; check wiring length and grounding. |
| Programs fine but stays dead on CAN | Wrong start address. Reflash the **combined** image at `0x08000000`. |

---

## Appendix: flash layout

```text
0x08000000  +---------------------------+
            |  CAN bootloader (12 KiB)  |   never written by can_flash.py
0x08003000  +---------------------------+
            |  Application (<= 52 KiB)  |   what can_flash.py writes
0x08010000  +---------------------------+
```

The combined image is exactly the bootloader followed by the application: bytes `0x3000..0xA70C` of
the combined image are byte-identical to `ioheart-f103-v2.4.0.bin`. The 52 KiB application cap is
`MAX_FSIZE` in [can_flash.py](../../i2rt/utils/can_flash.py); pages are 1 KiB.

This layout is why the CAN path is safe: the bootloader at `0x08000000` is never a write target, so
a failed or interrupted application flash always leaves a working bootloader behind.

## Related code

- [i2rt/utils/can_flash.py](../../i2rt/utils/can_flash.py) — CAN bootloader protocol and the `flash` CLI
- [i2rt/utils/encoder_manager.py](../../i2rt/utils/encoder_manager.py) — live encoder ops (version, config, readings)
