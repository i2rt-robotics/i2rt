# Linear Bot — Hardware Setup

Linear Bot is a Flow Base + vertical linear rail lift + mounted YAM arm.

::: tip Prerequisite
Finish [Flow Base setup](/getting-started/hardware/flow-base) first — the chassis is identical.
:::

## Checklist

### 1. Verify the Flow Base section works

- [ ] Battery installed, E-stop released
- [ ] CAN selector switch **UP** (Pi mode)
- [ ] Successfully SSH into the Pi (`ssh i2rt@172.6.2.20`)
- [ ] Joystick moves the base in XY

### 2. Linear rail homing

The vertical rail homes to its **lower limit switch** on boot.

- [ ] Clear the space below the rail
- [ ] Power-cycle — rail should drive down until it hits the limit, then stop
- [ ] Verify both limit switches with:
  ```bash
  python i2rt/flow_base/flow_base_client.py --command get_linear_rail_state --host 172.6.2.20
  ```

### 3. Mount the YAM arm

- [ ] Bolt the arm to the rail carriage top plate
- [ ] Route the arm's CAN cable through the rail cable chain
- [ ] Connect to the on-board CANable adapter

### 4. Power on the arm

- [ ] Verify all CAN devices are visible from the Pi
- [ ] Test arm floating mode:
  ```bash
  ssh i2rt@172.6.2.20
  python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
  ```

## Done

You're ready to run the [Linear Bot demo](/getting-started/demos/linear-bot).

For full specs and examples — see the [Linear Bot product page](/products/linear-bot).
