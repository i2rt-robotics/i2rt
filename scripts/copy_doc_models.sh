#!/usr/bin/env bash
# Copies STL files needed by the docs 3D viewer from their canonical source
# locations into docs/public so VitePress can serve them.
# Run automatically before docs:dev and docs:build — do NOT commit docs/public/models/.

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/docs/public/models/yam/assets"
mkdir -p "$DEST"

# YAM arm meshes
cp "$REPO/i2rt/robot_models/arm/yam/assets/base.stl"  "$DEST/"
cp "$REPO/i2rt/robot_models/arm/yam/assets/link1.stl" "$DEST/"
cp "$REPO/i2rt/robot_models/arm/yam/assets/link2.stl" "$DEST/"
cp "$REPO/i2rt/robot_models/arm/yam/assets/link3.stl" "$DEST/"
cp "$REPO/i2rt/robot_models/arm/yam/assets/link4.stl" "$DEST/"
cp "$REPO/i2rt/robot_models/arm/yam/assets/link5.stl" "$DEST/"

# linear_4310 gripper meshes
cp "$REPO/i2rt/robot_models/gripper/linear_4310/assets/gripper.stl"   "$DEST/"
cp "$REPO/i2rt/robot_models/gripper/linear_4310/assets/tip_left.stl"  "$DEST/"
cp "$REPO/i2rt/robot_models/gripper/linear_4310/assets/tip_right.stl" "$DEST/"

# BIG YAM arm meshes
DEST_BIG="$REPO/docs/public/models/big_yam/assets"
mkdir -p "$DEST_BIG"
cp "$REPO/i2rt/robot_models/arm/big_yam/assets/base.stl"       "$DEST_BIG/"
cp "$REPO/i2rt/robot_models/arm/big_yam/assets/link1.stl"  "$DEST_BIG/"
cp "$REPO/i2rt/robot_models/arm/big_yam/assets/link2.stl"  "$DEST_BIG/"
cp "$REPO/i2rt/robot_models/arm/big_yam/assets/link3.stl"  "$DEST_BIG/"
cp "$REPO/i2rt/robot_models/arm/big_yam/assets/link4.stl"  "$DEST_BIG/"
cp "$REPO/i2rt/robot_models/arm/big_yam/assets/link5.stl"  "$DEST_BIG/"

echo "✓ Doc models copied to $DEST and $DEST_BIG"
