import time

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorType, ReceiveMode

motor_list = [
    [0x01, MotorType.DM4340],
    [0x02, MotorType.DM4340],
    [0x03, MotorType.DM4340],
    [0x04, MotorType.DM4310],
    [0x05, MotorType.DM4310],
    [0x06, MotorType.DM4310],
    [0x07, MotorType.DM4310],
]

motor_chain = DMChainCanInterface(
    motor_list, [0, 0, 0, 0, 0, 0, 0][:7], [1, 1, 1, 1, 1, 1, 1][:7], channel="can0", receive_mode=ReceiveMode.p16
)


while True:
    time.sleep(1)
