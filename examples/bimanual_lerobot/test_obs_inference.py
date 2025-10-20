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

def test_observations_with_model():
    """Test function to print raw and preprocessed observations with model loaded"""
    print("Loading model and setting up robot for observation testing...")
    
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
        
        try:
            step = 0
            while True:
                # Get raw observation from robot
                raw_obs = robot.get_observation()
                
                print(f"\n--- Step {step} ---")
                print("=== RAW OBSERVATION ===")
                print("raw_obs: ", raw_obs)
                print(f"Observation type: {type(raw_obs)}")
                
                if isinstance(raw_obs, dict):
                    for key, value in raw_obs.items():
                        if hasattr(value, 'shape'):
                            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                            if value.size < 20:  # Only print small arrays
                                print(f"  values: {value}")
                        else:
                            print(f"{key}: {value}")
                else:
                    print(f"Raw observation: {raw_obs}")
                
                # Preprocess observation
                print("\n=== PREPROCESSING ===")
                try:
                    obs_frame = build_inference_frame(observation=raw_obs, device=device, ds_features=dataset_metadata.features)
                    print("inf_frame: ", obs_frame)
                    print(f"Built inference frame: {type(obs_frame)}")
                    
                    preprocessed_obs = preprocess(obs_frame)
                    print("preprocessed_obs: ", preprocessed_obs)
                    print(f"Preprocessed observation type: {type(preprocessed_obs)}")
                    
                    if isinstance(preprocessed_obs, dict):
                        for key, value in preprocessed_obs.items():
                            if hasattr(value, 'shape'):
                                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                                if value.numel() < 20:  # Only print small tensors
                                    print(f"    values: {value}")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print(f"Preprocessed observation: {preprocessed_obs}")
                        
                except Exception as e:
                    print(f"Preprocessing error: {e}")
                    step += 1
                    time.sleep(0.5)
                    continue
                
                # Model inference with timing
                print("\n=== MODEL INFERENCE ===")
                try:
                    start_time = time.time()
                    
                    # Get action from model
                    action = model.select_action(preprocessed_obs)
                    
                    # Postprocess action
                    postprocessed_action = postprocess(action)
                    
                    # Convert to robot action format
                    robot_action = make_robot_action(postprocessed_action, dataset_metadata.features)
                    
                    inference_time = time.time() - start_time
                    
                    # Print only the final robot action
                    print(f"Robot action type: {type(robot_action)}")
                    if isinstance(robot_action, dict):
                        for key, value in robot_action.items():
                            if hasattr(value, 'shape'):
                                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                                if value.numel() < 20:  # Only print small values
                                    print(f"    values: {value}")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print(f"Robot action: {robot_action}")
                    
                    # Calculate and print inference rate
                    inference_rate = 1.0 / inference_time if inference_time > 0 else 0
                    print(f"\n=== TIMING ===")
                    print(f"Inference time: {inference_time*1000:.2f} ms")
                    print(f"Inference rate: {inference_rate:.2f} Hz")
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                
                step += 1
                time.sleep(0.5)  # Slower for easier reading
                
        except KeyboardInterrupt:
            print("\nStopping observation test...")
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
    test_observations_with_model()