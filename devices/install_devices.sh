#!/bin/sh

USER_ID="$(id -u)"
USER=$(logname)

if [ "$USER_ID" -ne 0 ]; then
    echo "Please run this as root."
    exit 1
fi

INSTALL_DIR=$(dirname "$0")
RULES_DIR="$INSTALL_DIR/rules/*.rules"

for udev_rule in $RULES_DIR; do
    rule=$(basename "$udev_rule")
    echo "Installing $rule"
    cp $udev_rule /etc/udev/rules.d/$rule
done

echo "Adding $LOGNAME to group plugdev, video"
adduser $USER plugdev
adduser $USER video

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "Finished, please re-insert devices."
