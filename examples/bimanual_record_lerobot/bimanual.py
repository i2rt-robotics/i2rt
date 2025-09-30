#!/usr/bin/env python3

import subprocess
import os
import signal
import time
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))

def check_can_interface(interface):
    """Check if a CAN interface exists and is available"""
    try:
        # Check if interface exists in network interfaces
        result = subprocess.run(['ip', 'link', 'show', interface],
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False

        # Check if interface is UP
        if 'state UP' in result.stdout or 'state UNKNOWN' in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False

    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False

def check_all_can_interfaces():
    """Check if all required CAN interfaces exist"""
    required_interfaces = [
        'can_follower_r',
        'can_leader_r',
        'can_follower_l',
        'can_leader_l'
    ]

    missing_interfaces = []

    for interface in required_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)

    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")

    print("✓ All CAN interfaces are available")
    return True

def launch_gello_process(can_channel, gripper, mode=None, server_port=None, teleop_server_port = None):
    """Launch a single gello process with given parameters"""
    python_path = "python"
    script_path = os.path.join(current_file_path, "..", "..", "scripts", "record_gello.py")

    cmd = [python_path, os.path.expanduser(script_path),
           "--can_channel", can_channel,
           "--gripper", gripper]

    if mode:
        cmd.extend(["--mode", mode])

    if server_port:
        cmd.extend(["--server_port", str(server_port)])

    if teleop_server_port is not None:
        cmd.extend(["--teleop_server_port", str(teleop_server_port)])

    print(f"Starting: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None

def main():
    processes = []

    try:
        # First check if all CAN interfaces exist
        print("Checking CAN interfaces...")
        check_all_can_interfaces()

        # Define the processes to launch
        process_configs = [
            {
                'can_channel': 'can_follower_r',
                'gripper': 'linear_4310',
                'server_port': 1234
            },
            {
                'can_channel': 'can_leader_r',
                'gripper': 'yam_teaching_handle',
                'mode': 'leader',
                'server_port': 1234,
                'teleop_server_port': 2234
            },
            {
                'can_channel': 'can_follower_l',
                'gripper': 'linear_4310',
                'server_port': 1235
            },
            {
                'can_channel': 'can_leader_l',
                'gripper': 'yam_teaching_handle',
                'mode': 'leader',
                'server_port': 1235,
                'teleop_server_port': 2235
            }
        ]

        # Launch all processes
        print("\nLaunching processes...")
        for config in process_configs:
            process = launch_gello_process(**config)
            if process:
                processes.append(process)
                print(f"✓ Started process {process.pid} for {config['can_channel']}")
            else:
                raise RuntimeError(f"Failed to start process for {config['can_channel']}")

        print(f"\n✓ Successfully launched {len(processes)} processes")
        print("Press Ctrl+C to stop all processes")

        # Wait for processes and handle termination
        try:
            while True:
                # Check if any process has died
                for i, process in enumerate(processes):
                    if process.poll() is not None:
                        print(f"Process {process.pid} has terminated")
                        processes.pop(i)
                        break

                if not processes:
                    print("All processes have terminated")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\nReceived Ctrl+C, terminating all processes...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up: terminate all running processes
        for process in processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()

                # Wait up to 5 seconds for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {process.pid}...")
                    process.kill()
                    process.wait()

            except Exception as e:
                print(f"Error terminating process {process.pid}: {e}")

        print("All processes terminated")

if __name__ == "__main__":
    main()
