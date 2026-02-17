import time

from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface, MotorType, ReceiveMode
from i2rt.utils.utils import RateRecorder

motor_id = 0x01
motor_type = MotorType.DM4340
motor_chain = DMSingleMotorCanInterface(channel="can0", receive_mode=ReceiveMode.p16)
motor_chain.motor_on(motor_id, motor_type)

with RateRecorder(name="dm_driver_single_motor_example", report_interval=10.0) as recorder:
    while True:
        motor_chain.set_control(motor_id, motor_type, 0.0, 0.0, 0.0, 0.0, 0.0)
        recorder.track()
        time.sleep(0.0001)
