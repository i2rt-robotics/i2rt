# ioheart encoder firmware

Firmware images for the **ioheart** board (STM32F103C8T6) — the passive joint encoder used by the
`yam_teaching_handle` (leader arm).

> **Check the installed version first — it decides which method you can use.**
>
> ```bash
> uv run python -m i2rt.utils.encoder_manager get-version --bus can0
> ```
>
> - **< 2.2.0 (or no response at all)** → you **must** use [Option 2 (J-Link / SWD)](#option-2--flash-over-swd-with-j-flash)
>   with the combined image. Flashing over CAN is not available on these boards.
> - **>= 2.2.0** → use [Option 1 (CAN)](#option-1--flash-over-can). No probe or disassembly needed.
>
> Nothing in `can_flash.py` enforces this — there is no version gate in the tool, so it is on you to
> check before reaching for the CAN path. Once a board has been brought up to >= 2.2.0 via SWD, all
> later updates can go over CAN.

## Images in this directory

| File | Size | Contents | Load address | Flash with |
| --- | --- | --- | --- | --- |
| `ioheart-f103-v2.4.0.bin` | 30,476 B | Application only | `0x08003000` | CAN bootloader (Option 1) **or** SWD |
| `ioheart-f103-combined-v2.4.0.bin` | 43,008 B | Bootloader + application | `0x08000000` | SWD only (Option 2) |

The combined image is exactly the bootloader followed by the application: bytes `0x3000..0xA70C`
of the combined image are byte-identical to `ioheart-f103-v2.4.0.bin`.

### Flash memory map

```
0x08000000  +---------------------------+
            |  CAN bootloader (12 KiB)  |   never touched by can_flash.py
0x08003000  +---------------------------+
            |  Application (<= 52 KiB)  |   what can_flash.py writes
0x08010000  +---------------------------+
```

The 52 KiB application cap is enforced by `MAX_FSIZE` in
[i2rt/utils/can_flash.py](../../i2rt/utils/can_flash.py); pages are 1 KiB.

> **Do not flash the combined image over CAN.** It is under the 52 KiB limit so the size check will
> not stop you, but the bootloader would write the *bootloader* bytes into the *application* region
> starting at `0x08003000`, leaving a non-booting app. The board stays recoverable over CAN (the real
> bootloader is untouched) — just re-flash the app-only image.

---

## Option 1 — Flash over CAN

The normal path for a board already running **firmware >= 2.2.0**. Older boards cannot be flashed
this way — see [Option 2](#option-2--flash-over-swd-with-j-flash).

No hardware disassembly required. The tool talks to the on-board CAN bootloader
(cmd ID `0x700`, replies `0x701+`) and restarts a running encoder into bootloader mode via the
encoder REQ channel (`0x50E`) automatically.

### 1. Bring up the CAN interface at 1 Mbps

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
# or, for every CAN interface on the machine:
./scripts/reset_all_can.sh
```

### 2. Confirm the encoder is on the bus

```bash
uv run python -m i2rt.utils.encoder_manager list-devices --bus can0
uv run python -m i2rt.utils.encoder_manager get-version --bus can0
```

### 3. Flash

From the repo root:

```bash
uv run python -m i2rt.utils.can_flash devices/firmware/ioheart-f103-v2.4.0.bin \
    --channel can0 --bitrate 1000000
```

By default **every encoder discovered on the bus is flashed**. The leader-arm bus normally carries
exactly one encoder, so this is what you want. To restrict the flash to a single device as a
safeguard:

```bash
uv run python -m i2rt.utils.can_flash devices/firmware/ioheart-f103-v2.4.0.bin \
    --channel can0 --device-id 1
```

What you should see: discovery, a per-page progress line with `CRC OK`, a whole-image CRC-32
verification, then `Device flashed successfully`. Each page is retried up to 10 times on CRC
mismatch, and the run ends with a summary listing successful and failed device IDs.

### 4. Verify

The device restarts on its own after a bootloader timeout (a few seconds). Then:

```bash
uv run python -m i2rt.utils.encoder_manager get-version --bus can0
# expect major=2 minor=4 patch=0
```

### Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `No encoder devices found on the bus.` | Interface down, wrong `--channel`, bitrate not 1 Mbps, or no power/termination. Check `candump can0` shows traffic. |
| `device N never entered bootloader mode within 30s` | Most likely the board is running firmware **< 2.2.0**, which has no CAN flashing support — check with `get-version` and use Option 2. Otherwise the encoder is not responding to the restart request: power-cycle the board and re-run, and the tool will catch it during the bootloader window. |
| `Could not connect to device.` | Heavy bus traffic from other nodes. Flash on an isolated bus with only the encoder attached. |
| `Page write failed` after retries | Marginal wiring or bitrate mismatch. Re-seat the CAN harness and re-run — the flash is restartable from scratch. |
| App no longer boots after a bad flash | Re-run Option 1 with the app-only image; the bootloader survives. If the bootloader itself is gone, use Option 2. |

---

## Option 2 — Flash over SWD with J-Flash

Required when:

- the board runs **firmware older than 2.2.0** — these predate CAN flashing support, so SWD is the
  only way to update them (once on >= 2.2.0, later updates can use Option 1);
- the MCU is blank (factory programming);
- the bootloader itself has been erased or corrupted — the one case Option 1 cannot fix.

### Requirements

- SEGGER J-Link probe + J-Flash (part of the J-Link Software and Documentation Pack)
- SWD wiring to the board: `SWDIO`, `SWCLK`, `GND`, and `VTref` (target reference voltage, 3.3 V).
  `nRESET` is optional but makes connecting more reliable.
- Target powered — J-Link does **not** supply power on the standard 20-pin connector.

### Steps

1. **New project** in J-Flash → *Create a new project*.
2. **Target device:** `STM32F103C8` — the board uses an **STM32F103C8T6** (64 KiB flash, 1 KiB pages).
   J-Flash lists it by the base part number without the `T6` package/temperature suffix.
3. **Target interface:** `SWD`, speed `4000 kHz` (drop to `1000 kHz` if the connection is flaky).
4. `Target → Connect`. The log should report the Cortex-M3 core ID and the correct flash size.
5. `File → Open data file…` → select the image and enter its start address when prompted:

   | Goal | File | Start address |
   | --- | --- | --- |
   | Full program / recover a bricked board | `ioheart-f103-combined-v2.4.0.bin` | `0x08000000` |
   | Replace only the app, keep the existing bootloader | `ioheart-f103-v2.4.0.bin` | `0x08003000` |

   Getting this address wrong is the single most common mistake here — a `.bin` carries no address
   information, so J-Flash uses exactly what you type.
6. `Target → Production Programming` (erase + program + verify), or
   `Target → Manual Programming → Program & Verify`.
7. `Target → Disconnect`, power-cycle the board, then verify over CAN:

   ```bash
   uv run python -m i2rt.utils.encoder_manager get-version --bus can0
   ```

### Command-line equivalent

The same thing without the GUI, using `JLinkExe`:

```
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

> If the board has read-out protection enabled, J-Flash will refuse to program. `unlock STM32`
> in `JLinkExe` clears it, which performs a full chip mass-erase.

---

## Which option should I use?

| Situation | Method |
| --- | --- |
| Installed firmware **< 2.2.0** | **Option 2** (SWD), combined image — no CAN path exists |
| Blank MCU / factory programming | **Option 2** (SWD), combined image |
| Bootloader erased or corrupted, board never appears on CAN | **Option 2** (SWD), combined image |
| Routine update, installed firmware **>= 2.2.0** | **Option 1** (CAN) — no disassembly, no probe |

## Related code

- [i2rt/utils/can_flash.py](../../i2rt/utils/can_flash.py) — CAN bootloader protocol and the `flash` CLI
- [i2rt/utils/encoder_manager.py](../../i2rt/utils/encoder_manager.py) — live encoder ops (version, config, readings)
