# Short aliases for the i2rt ROS 2 nodes.
#
# Add this line to your ~/.bashrc (use the absolute path to this repo):
#     source /path/to/i2rt/scripts/ros2_aliases.sh
#
# Then you can run, from anywhere:
#     yam teleop --sim
#     yam-teleop --bilateral-kp 0.2
#     yam-dagger --bilateral-kp 0.15
#     yam-wrapper --sim

_I2RT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

alias yam="$_I2RT_REPO/scripts/yam"
alias yam-wrapper="$_I2RT_REPO/scripts/yam wrapper"
alias yam-teleop="$_I2RT_REPO/scripts/yam teleop"
alias yam-dagger="$_I2RT_REPO/scripts/yam dagger"
alias yam-can="$_I2RT_REPO/scripts/yam can"
alias yam-canup="$_I2RT_REPO/scripts/yam canup"
