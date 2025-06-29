import time

from i2rt.motor_drivers.dm_driver import CanInterface
from i2rt.robots.get_robot import get_encoder_chain

can_interface = CanInterface(channel="can_right", use_buffered_reader=False)

encoder_chain = get_encoder_chain(can_interface)


while True:
    encoder_states = encoder_chain.read_states()
    print(encoder_states)
    time.sleep(0.01)
