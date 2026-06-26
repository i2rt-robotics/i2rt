import argparse
import sys
import time

from i2rt.motor_drivers.dm_driver import CanInterface
from i2rt.robots.get_robot import get_encoder_chain

parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, default="can0")
args = parser.parse_args()

can_interface = CanInterface(channel=args.channel, use_buffered_reader=False)
encoder_chain = get_encoder_chain(can_interface)

while True:
    encoder_state = encoder_chain.read_states()[0]
    encoder_positions = encoder_state.position
    encoder_button = encoder_state.io_inputs
    sys.stdout.write(f"\rpos: {encoder_positions:.1f} button: {encoder_button}")
    sys.stdout.flush()

    time.sleep(0.01)
