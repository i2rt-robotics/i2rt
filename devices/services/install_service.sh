#!/bin/sh
set -eu

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info()    { printf '%b\n' "${BLUE}[INFO]${NC} $1"; }
print_success() { printf '%b\n' "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { printf '%b\n' "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { printf '%b\n' "${RED}[ERROR]${NC} $1" >&2; }

REPO_URL="git@github.com:i2rt-robotics/i2rt.git"
CLONE_DIR="/srv/i2rt"
INSTALL_DIR=$(dirname "$0")
RUN_USER=$(id -un)
RUN_GROUP=$(id -gn)

usage() {
    cat <<EOF
Usage: $(basename "$0") [-b <branch>] <flowbase|linearbot>

Deploy an i2rt FlowBase controller as a systemd service.

Clones $REPO_URL into $CLONE_DIR (or updates an existing checkout), builds the
venv with uv, then installs and starts the chosen service while removing the
other one (only one service may own the robot at a time).

Services:
  flowbase    Base only, 8 motors          (flow_base_controller.py --channel can0)
  linearbot   Base + linear rail, 9 motors  (--channel can0 --linear-rail)

Options:
  -b <branch>  Git branch to deploy (default: main)
  -h           Show this help
EOF
}

BRANCH="main"
while getopts "b:h" opt; do
    case "$opt" in
        b) BRANCH="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

SERVICE="${1:-}"
case "$SERVICE" in
    flowbase)  OTHER="linearbot" ;;
    linearbot) OTHER="flowbase" ;;
    "")        print_error "Missing service name."; usage; exit 2 ;;
    *)         print_error "Unknown service: $SERVICE"; usage; exit 2 ;;
esac

# Must run as the operator (not root): git clone over SSH uses the operator's key
# and uv runs in their environment. Privileged steps below are sudo'd individually.
if [ "$(id -u)" -eq 0 ]; then
    print_error "Run this as your normal user (not root). It will sudo for privileged steps."
    exit 1
fi

# Preflight: required tooling.
for cmd in git uv sudo; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        print_error "$cmd not found on PATH. Please install it first."
        [ "$cmd" = uv ] && print_error "  Install uv: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
done

SERVICE_FILE="$INSTALL_DIR/$SERVICE.service"
if [ ! -f "$SERVICE_FILE" ]; then
    print_error "Service unit not found: $SERVICE_FILE"
    exit 1
fi

print_info "=== Deploying $SERVICE.service (branch: $BRANCH) ==="

# The units hardcode User=i2rt; warn if the operator account differs.
if [ "$RUN_USER" != "i2rt" ]; then
    print_warning "Current user is '$RUN_USER', but the service runs as User=i2rt."
    print_warning "Ensure user 'i2rt' can read/write $CLONE_DIR, or the service may fail to start."
fi

# 1. Ensure /srv/i2rt exists and is owned by the current user (so clone + uv need no root).
if [ ! -d "$CLONE_DIR" ]; then
    print_info "Creating $CLONE_DIR (owned by $RUN_USER:$RUN_GROUP)..."
    sudo install -d -o "$RUN_USER" -g "$RUN_GROUP" "$CLONE_DIR"
fi

# 2. Clone or update the repo at the requested branch.
if [ -d "$CLONE_DIR/.git" ]; then
    print_info "Updating existing checkout in $CLONE_DIR to '$BRANCH'..."
    git -C "$CLONE_DIR" fetch origin
    git -C "$CLONE_DIR" checkout "$BRANCH"
    git -C "$CLONE_DIR" pull --ff-only
else
    print_info "Cloning $REPO_URL into $CLONE_DIR (branch: $BRANCH)..."
    git clone -b "$BRANCH" "$REPO_URL" "$CLONE_DIR"
fi
print_success "Repository ready at $CLONE_DIR"

# 3. Build the venv (the service imports the installed i2rt package).
print_info "Building virtualenv with uv (python 3.11)..."
( cd "$CLONE_DIR" && uv venv --python 3.11 && uv sync )
print_success "venv built at $CLONE_DIR/.venv"

# 4. Remove the other service if installed (only one may own the robot).
if [ -f "/etc/systemd/system/$OTHER.service" ]; then
    print_info "Removing other service: $OTHER.service..."
    sudo systemctl disable --now "$OTHER.service" 2>/dev/null || true
    sudo rm -f "/etc/systemd/system/$OTHER.service"
    print_success "Removed $OTHER.service"
fi

# 5. Install and start the chosen service.
print_info "Installing $SERVICE.service..."
sudo cp "$SERVICE_FILE" "/etc/systemd/system/$SERVICE.service"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE.service"
print_success "$SERVICE.service installed and started"

print_info "=== Done ==="
print_info "Status:  systemctl status $SERVICE.service"
print_info "Logs:    journalctl -u $SERVICE.service -f"
