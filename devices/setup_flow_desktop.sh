#!/bin/sh

USER_ID="$(id -u)"
USER=$(logname)

INSTALL_DIR=$(dirname "$0")
cp $INSTALL_DIR/FlowBase.desktop ~/Desktop/

gio set ~/Desktop/FlowBase.desktop metadata::trusted true
chmod +x ~/Desktop/FlowBase.desktop
