# YAM Box Demo — Enclosed Manipulation

YAM Box runs the same SDK as a standalone YAM. After assembly, the demo workflow is identical to the [YAM demo](/getting-started/demos/yam).

::: tip Prerequisite
[YAM Box Hardware Setup](/getting-started/hardware/yam-box) done — fully assembled, arm wired, CAN up.
:::

## Quick test

```bash
python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
```

Arm should float in zero-gravity mode inside the enclosure. Push it through reachable space to verify nothing collides with the walls or top frame.

## Python API

Same as YAM standalone:

```python
from i2rt.robots.get_robot import get_yam_robot
import numpy as np

robot = get_yam_robot(channel="can0")
print(robot.get_observations())
robot.command_joint_pos(np.zeros(7))
robot.close()
```

## Next steps

- Full assembly photo guide and YAM Box features → [YAM Box product page](/products/yam-box)
- For full SDK reference → [YAM product page](/products/yam)
