#!/bin/sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info()    { printf '%b\n' "${BLUE}[INFO]${NC} $1"; }
print_success() { printf '%b\n' "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { printf '%b\n' "${YELLOW}[WARNING]${NC} $1"; }

USER_ID="$(id -u)"
USER=$(logname)

INSTALL_DIR=$(dirname "$0")

print_info "=== Flow desktop launcher setup ==="

# Install lxterminal (required by the desktop launchers)
print_info "Checking dependencies..."
if ! command -v lxterminal >/dev/null 2>&1; then
    print_warning "lxterminal not found, installing..."
    if sudo apt-get update && sudo apt-get install -y lxterminal; then
        print_success "lxterminal installed"
    else
        print_warning "Failed to install lxterminal (may require sudo password), continuing anyway"
    fi
else
    print_success "lxterminal already installed"
fi

# Copy FlowBase.desktop (base only, 8 motors)
print_info "Deploying FlowBase.desktop..."
cp $INSTALL_DIR/FlowBase.desktop ~/Desktop/
gio set ~/Desktop/FlowBase.desktop metadata::trusted true
chmod +x ~/Desktop/FlowBase.desktop
print_success "Deployed: FlowBase.desktop"

# Copy FlowBaseGamepad.desktop (base only, 8 motors, gamepad teleop)
print_info "Deploying FlowBaseGamepad.desktop..."
cp $INSTALL_DIR/FlowBaseGamepad.desktop ~/Desktop/
gio set ~/Desktop/FlowBaseGamepad.desktop metadata::trusted true
chmod +x ~/Desktop/FlowBaseGamepad.desktop
print_success "Deployed: FlowBaseGamepad.desktop"

# Copy LinearRailVehicle.desktop (with linear rail, 9 motors)
print_info "Deploying LinearRailVehicle.desktop..."
cp $INSTALL_DIR/LinearRailVehicle.desktop ~/Desktop/
gio set ~/Desktop/LinearRailVehicle.desktop metadata::trusted true
chmod +x ~/Desktop/LinearRailVehicle.desktop
print_success "Deployed: LinearRailVehicle.desktop"

# Copy LinearRailVehicleGamepad.desktop (with linear rail, 9 motors, gamepad teleop)
print_info "Deploying LinearRailVehicleGamepad.desktop..."
cp $INSTALL_DIR/LinearRailVehicleGamepad.desktop ~/Desktop/
gio set ~/Desktop/LinearRailVehicleGamepad.desktop metadata::trusted true
chmod +x ~/Desktop/LinearRailVehicleGamepad.desktop
print_success "Deployed: LinearRailVehicleGamepad.desktop"

print_success "=== Setup completed: 4 desktop launchers deployed ==="
