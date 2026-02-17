#!/bin/sh

USER_ID="$(id -u)"
USER=$(logname)

INSTALL_DIR=$(dirname "$0")

# Copy FlowBase.desktop (base only, 8 motors)
cp $INSTALL_DIR/FlowBase.desktop ~/Desktop/
gio set ~/Desktop/FlowBase.desktop metadata::trusted true
chmod +x ~/Desktop/FlowBase.desktop

# Copy LinearRailVehicle.desktop (with linear rail, 9 motors)
cp $INSTALL_DIR/LinearRailVehicle.desktop ~/Desktop/
gio set ~/Desktop/LinearRailVehicle.desktop metadata::trusted true
chmod +x ~/Desktop/LinearRailVehicle.desktop
