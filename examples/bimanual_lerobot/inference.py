import torch
import time
import subprocess
import os
import signal
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from i2rt import I2RTRobot
from config import I2RTFollowerConfig

def check_can_interface(interface):
    """Check if a CAN interface exists and is available"""
    try:
        result = subprocess.run(['ip', 'link', 'show', interface],
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False
        if 'state UP' in result.stdout or 'state UNKNOWN' in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False
    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False

def launch_follower_processes():
    """Launch only the two follower processes (no leaders)"""
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    processes = []
    
    # Check required CAN interfaces for followers only
    follower_interfaces = ['can_follower_r', 'can_follower_l']
    missing_interfaces = []
    
    for interface in follower_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)
    
    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")
    
    print("✓ All follower CAN interfaces are available")
    
    # Define only follower processes
    follower_configs = [
        {
            'can_channel': 'can_follower_r',
            'gripper': 'linear_4310',
            'server_port': 1234
        },
        {
            'can_channel': 'can_follower_l',
            'gripper': 'linear_4310',
            'server_port': 1235
        }
    ]
    
    # Launch follower processes
    print("\nLaunching follower processes...")
    for config in follower_configs:
        python_path = "python"
        script_path = os.path.join(current_file_path, "..", "..", "scripts", "record_gello.py")
        
        cmd = [python_path, os.path.expanduser(script_path),
               "--can_channel", config['can_channel'],
               "--gripper", config['gripper'],
               "--server_port", str(config['server_port'])]
        
        print(f"Starting: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(cmd)
            processes.append(process)
            print(f"✓ Started follower process {process.pid} for {config['can_channel']}")
        except Exception as e:
            print(f"Error starting follower process for {config['can_channel']}: {e}")
            # Clean up already started processes
            for p in processes:
                p.terminate()
            raise
    
    print(f"✓ Successfully launched {len(processes)} follower processes")
    return processes

def cleanup_processes(processes):
    """Clean up all launched processes"""
    print("\nCleaning up follower processes...")
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

def test_inference():
    """Test function to deploy trained model"""
    print("Loading model and setting up robot...")
    
    follower_processes = []
    try:
        follower_processes = launch_follower_processes()
        print("Waiting for follower processes to initialize...")
        time.sleep(3)
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model_path = "/root/.cache/huggingface/lerobot/pretrained_model"
        print(f"Loading model from: {model_path}")
        model = ACTPolicy.from_pretrained(pretrained_name_or_path=model_path)
        print("Model loaded successfully!")
        
        # Load dataset metadata for preprocessing
        dataset_id = "zetanschy/train_merged_v2"
        print(f"Loading dataset metadata from: {dataset_id}")
        dataset_metadata = LeRobotDatasetMetadata(dataset_id)
        preprocess, postprocess = make_pre_post_processors(
            model.config, dataset_stats=dataset_metadata.stats
        )
        print("Preprocessing functions loaded!")
        
        # Setup robot
        robot_cfg = I2RTFollowerConfig()
        robot = I2RTRobot(robot_cfg)
        robot.connect()
        print("Robot connected! Starting observation test...")
        print("Press Ctrl+C to stop")
        
        MAX_STEPS_PER_EPISODE = 20

        try:
            for _ in range(MAX_STEPS_PER_EPISODE):
                # Process observation
                raw_obs = robot.get_observation()
                obs_frame = build_inference_frame(observation=raw_obs, device=device, ds_features=dataset_metadata.features)
                preprocessed_obs = preprocess(obs_frame)
                
                # Get action from model
                action = model.select_action(preprocessed_obs)
                postprocessed_action = postprocess(action)
                robot_action = make_robot_action(postprocessed_action, dataset_metadata.features)
                
                # Execute action on robot
                robot.execute_inference_action(robot_action)
                    
        except KeyboardInterrupt:
            print("\nStopping inference test...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Disconnecting robot...")
            robot.disconnect()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always clean up follower processes
        cleanup_processes(follower_processes)

if __name__ == "__main__":
    test_inference()