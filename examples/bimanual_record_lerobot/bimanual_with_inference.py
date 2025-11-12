#!/usr/bin/env python3

import subprocess
import os
import signal
import time
import sys
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

import sys
sys.path.insert(0, "/home/i2rt/dev/robot-os/thirdparty/i2rt")

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
from config import I2RTFollowerConfig
from i2rt_robot import I2RTRobot, PortalFollowerClient
import cv2
current_file_path = os.path.dirname(os.path.abspath(__file__))


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


def launch_gello_process(can_channel, gripper, server_port):
    """Launch a single follower gello process"""
    python_path = "python"
    script_path = os.path.join(current_file_path, "..", "..", "scripts", "minimum_gello.py")
    
    cmd = [python_path, os.path.expanduser(script_path),
           "--can_channel", can_channel,
           "--gripper", gripper,
           "--mode", "follower",
           "--server_port", str(server_port)]
    
    print(f"Starting: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


class YAMLeaderRobot:
    """Wrapper for YAM leader robot with button and gripper support"""
    def __init__(self, robot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self):
        """Get joint positions and button states"""
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray):
        """Command joint positions (without gripper)"""
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray):
        """Update PD gains"""
        self._robot.update_kp_kd(kp, kd)


class BimanualTeleopWithInference:
    """
    Manages bimanual teleoperation with inference mode.
    
    Button 0: Toggle sync/unsync (teleoperation)
    Button 1: Toggle inference mode (only when unsynced)
    """
    
    def __init__(self, model_path: str, dataset_id: str, bilateral_kp: float = 0.0):
        self.bilateral_kp = bilateral_kp
        
        # State flags
        self.synchronized = False
        self.inference_mode = False
        self.button0_prev = False
        self.button1_prev = False
        
        # Load inference model
        print("Loading inference model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = ACTPolicy.from_pretrained(pretrained_name_or_path=model_path)
        
        #Faster actions test
        self.model.config.temporal_ensemble_coeff = None
        self.model.config.n_action_steps = 30
        
        #Chunk size test
        # model_inference.config.chunk_size = 30
        # model_inference.config.n_action_steps = 30
        print("Model loaded successfully!")
        
        # Load dataset metadata for preprocessing
        HF_DATASET_ID = "jdgalviss/so101_test" # TODO: verify if needed
        FPS = 30
        EPISODE_TIME_SEC = 60
        TASK_DESCRIPTION = "Put cups on plate"
        dataset_id = "cups_to_plate/20251022-230112"

        print(f"Loading dataset metadata from: {dataset_id}")
        self.dataset_metadata = LeRobotDatasetMetadata(dataset_id)
        self.preprocess, self.postprocess = make_pre_post_processors(self.model.config, dataset_stats=self.dataset_metadata.stats)
        print("Preprocessing functions loaded!")
        
        # Setup robot for inference (reads observations and commands followers)
        robot_cfg = I2RTFollowerConfig()
        self.inference_robot = I2RTRobot(robot_cfg)
        
        # Setup leader robots (read joint positions and buttons)
        gripper_type = GripperType.from_string_name("yam_teaching_handle")
        self.leader_right = YAMLeaderRobot(get_yam_robot(channel="can_leader_r", gripper_type=gripper_type))
        self.leader_left = YAMLeaderRobot(get_yam_robot(channel="can_leader_l", gripper_type=gripper_type))
        
        # Setup follower clients (for direct teleoperation)
        self.follower_right = PortalFollowerClient("127.0.0.1", 1234)
        self.follower_left = PortalFollowerClient("127.0.0.1", 1235)
        
        # Store initial kp values for bilateral control
        self.robot_kp_right = self.leader_right._robot._kp
        self.robot_kp_left = self.leader_left._robot._kp

        # For debugging
        self.time = time.time()
        self.steps_current_episode = 0
        self.is_first_torso_frame_saved = False

    def connect_inference_robot(self):
        """Connect the inference robot to read observations"""
        self.inference_robot.connect()
        print("Inference robot connected!")

    def slow_move_to_leader(self, leader_pos_right, leader_pos_left, duration: float = 1.0):
        """Slowly move followers to match leader positions"""
        current_follower_right = self.follower_right.get_joint_pos()
        current_follower_left = self.follower_left.get_joint_pos()
        
        steps = 100
        for i in range(steps):
            alpha = i / steps
            
            # Interpolate positions
            target_right = leader_pos_right * alpha + current_follower_right * (1 - alpha)
            target_left = leader_pos_left * alpha + current_follower_left * (1 - alpha)
            
            self.follower_right.command_joint_pos(target_right)
            self.follower_left.command_joint_pos(target_left)
            
            time.sleep(0.03)

    def handle_button_events(self, current_button):
        """Handle button press events with debouncing"""
        button0_pressed = current_button[0] > 0.5
        button1_pressed = current_button[1] > 0.5 # check if button 1 is pressed by default

        # print(f"button0_pressed: {button0_pressed}, button1_pressed: {button1_pressed}")
        # print(f"inference_mode: {self.inference_mode}, synchronized: {self.synchronized}")
        # print(f"button0_prev: {self.button0_prev}, button1_prev: {self.button1_prev}")

        # # For debugging wait 3 seconds and activate inference mode
        # if time.time() - self.time > 3:
        #     self.inference_mode = True
        #     self.synchronized = False
        #     print("🤖 INFERENCE mode activated (debugging)")
        #     self.time = time.time()
        
        # Button 0: Toggle sync/unsync
        if button0_pressed and not self.button0_prev:
            if self.inference_mode:
                # If inference is active, just disable it and go to unsync
                print("🤖 Inference mode DISABLED (Button 0 pressed during inference)")
                self.inference_mode = False
                self.synchronized = False
                
                # Clear bilateral PD
                self.leader_right.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                self.leader_left.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
            else:
                # Normal sync toggle
                self.synchronized = not self.synchronized
                
                if self.synchronized:
                    print("🔗 SYNC mode activated")
                    # Get current positions
                    leader_pos_right, _ = self.leader_right.get_info()
                    leader_pos_left, _ = self.leader_left.get_info()
                    
                    # Set bilateral PD
                    self.leader_right.update_kp_kd(
                        kp=self.robot_kp_right * self.bilateral_kp, 
                        kd=np.zeros(6)
                    )
                    self.leader_left.update_kp_kd(
                        kp=self.robot_kp_left * self.bilateral_kp, 
                        kd=np.zeros(6)
                    )
                    
                    # Command leaders to current position
                    self.leader_right.command_joint_pos(leader_pos_right[:6])
                    self.leader_left.command_joint_pos(leader_pos_left[:6])
                    
                    # Slowly move followers to leader
                    self.slow_move_to_leader(leader_pos_right, leader_pos_left)
                else:
                    print("🔓 UNSYNC mode - robot stopped")
                    # Clear bilateral PD
                    self.leader_right.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    self.leader_left.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        
        # Button 1: Toggle inference (only when unsynced)
        if button1_pressed and not self.button1_prev:
            if not self.synchronized:
                self.inference_mode = not self.inference_mode
                
                if self.inference_mode:
                    print("🤖 INFERENCE mode activated")
                else:
                    print("🛑 INFERENCE mode deactivated - robot stopped")
            else:
                print("⚠️  Cannot activate inference while in sync mode")
        
        # Update previous button states
        self.button0_prev = button0_pressed
        self.button1_prev = button1_pressed

    def run_teleoperation(self):
        """Execute one step of teleoperation"""
        # Get leader positions
        print("Running teleoperation...")
        leader_pos_right, _ = self.leader_right.get_info()
        leader_pos_left, _ = self.leader_left.get_info()
        
        # Get follower positions
        follower_pos_right = self.follower_right.get_joint_pos()
        follower_pos_left = self.follower_left.get_joint_pos()


        # Limit follower and leader positions difference to 0.08 radians
        diff_right = leader_pos_right - follower_pos_right
        diff_left = leader_pos_left - follower_pos_left
        diff_right = np.clip(diff_right, -0.08, 0.08)
        diff_left = np.clip(diff_left, -0.08, 0.08)
        follower_pos_right = leader_pos_right - diff_right
        follower_pos_left = leader_pos_left - diff_left
        
        # Command followers to match leaders
        self.follower_right.command_joint_pos(leader_pos_right)
        self.follower_left.command_joint_pos(leader_pos_left)
        
        # Set bilateral force (followers push back on leaders)
        self.leader_right.command_joint_pos(follower_pos_right[:6])
        self.leader_left.command_joint_pos(follower_pos_left[:6])

    def execute_inference_action(self, robot_action):
        """
        Extract joint positions from robot_action dict and command follower arms.
        
        Args:
            robot_action: dict with keys like 'right.j0.pos', 'right.j1.pos', etc.
        """
        # Extract right arm target positions (j0 through j6)
        target_right = np.array([
            robot_action['right.j0.pos'],
            robot_action['right.j1.pos'],
            robot_action['right.j2.pos'],
            robot_action['right.j3.pos'],
            robot_action['right.j4.pos'],
            robot_action['right.j5.pos'],
            robot_action['right.j6.pos']
        ])
        
        # Extract left arm target positions (j0 through j6)
        target_left = np.array([
            robot_action['left.j0.pos'],
            robot_action['left.j1.pos'],
            robot_action['left.j2.pos'],
            robot_action['left.j3.pos'],
            robot_action['left.j4.pos'],
            robot_action['left.j5.pos'],
            robot_action['left.j6.pos']
        ])
        
        # Get current follower positions
        follower_pos_right = self.follower_right.get_joint_pos()
        follower_pos_left = self.follower_left.get_joint_pos()
        
        # Limit target and current positions difference to 0.08 radians
        diff_right = target_right - follower_pos_right
        diff_left = target_left - follower_pos_left
        diff_right = np.clip(diff_right, -0.08, 0.08)
        diff_left = np.clip(diff_left, -0.08, 0.08)
        target_right = follower_pos_right + diff_right
        target_left = follower_pos_left + diff_left
        
        # Command the follower arms
        self.follower_right.command_joint_pos(target_right)
        self.follower_left.command_joint_pos(target_left)

    def run_inference(self):
        """Execute one step of inference"""
        try:
            print("Running inference...")
            # Get observation from robot
            raw_obs = self.inference_robot.get_observation()

            print("raw_obs: ", type(raw_obs))
            print("raw_obs: ", raw_obs.keys())
            print(f"raw_obs left: {raw_obs['teleop_left'].shape}, {np.mean(raw_obs['teleop_left'])}")
            print(f"raw_obs right: {raw_obs['teleop_right'].shape}, {np.mean(raw_obs['teleop_right'])}")
            print(f"raw_obs torso: {raw_obs['torso'].shape}, {np.mean(raw_obs['torso'])}")

            if not self.is_first_torso_frame_saved:
                cv2.imwrite("/workspace/robot-os/outputs/torso_frame.jpg", raw_obs['torso'])
                self.is_first_torso_frame_saved = True

            # Process observation
            obs_frame = build_inference_frame(
                observation=raw_obs, 
                device=self.device, 
                ds_features=self.dataset_metadata.features
            )
            print("\nobs_frame: ", type(obs_frame))
            print("obs_frame: ", obs_frame.keys())
            print(f"obs_frame left: {obs_frame['observation.images.teleop_left'].shape}, {torch.mean(obs_frame['observation.images.teleop_left'])}")
            print(f"obs_frame right: {obs_frame['observation.images.teleop_right'].shape}, {torch.mean(obs_frame['observation.images.teleop_right'])}")
            print(f"obs_frame torso: {obs_frame['observation.images.torso'].shape}, {torch.mean(obs_frame['observation.images.torso'])}")

            preprocessed_obs = self.preprocess(obs_frame)

            print("preprocessed_obs: ", type(preprocessed_obs))
            print("preprocessed_obs: ", preprocessed_obs.keys())
            # Get action from model
            action = self.model.select_action(preprocessed_obs)
            postprocessed_action = self.postprocess(action)
            robot_action = make_robot_action(postprocessed_action, self.dataset_metadata.features)
            
            # Execute action on robot
            print(f"robot_action: {robot_action}")
            self.execute_inference_action(robot_action)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            self.inference_mode = False  # Disable inference on error

    def run(self):
        """Main control loop"""
        print("\n" + "="*60)
        print("  Bimanual Teleoperation with Inference")
        print("="*60)
        print("\n📋 Controls:")
        print("  Button 0 (first button):  Toggle SYNC/UNSYNC")
        print("  Button 1 (second button): Toggle INFERENCE mode (when unsynced)")
        print("\n🤖 Current state: UNSYNC (robot stopped)")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while True:
                # Get button state from right leader (buttons are shared)
                _, current_button = self.leader_right.get_info()
                
                # Handle button events
                self.handle_button_events(current_button)
                
                # Execute appropriate control mode
                if self.inference_mode:
                    self.run_inference()
                    # print("🤖 INFERENCE mode active")
                    # return
                elif self.synchronized:
                    self.run_teleoperation()
                # else: do nothing (unsync mode)
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nStopping control loop...")


def main():
    processes = []
    controller = None
    
    try:
        # # DEBUGGING
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model_path = "/root/.cache/huggingface/lerobot/single_arm_kl_10/pretrained_model"
        # #model_path = "/root/.cache/huggingface/lerobot/single_arm_chunk_30/pretrained_model"

        # model = ACTPolicy.from_pretrained(pretrained_name_or_path=model_path)
        # model.config.temporal_ensemble_coeff = None
        # model.config.n_action_steps = 20
        # print("Model loaded successfully!")

        # Check CAN interfaces
        print("Checking CAN interfaces...")
        check_all_can_interfaces()
        
        # Launch follower processes
        print("\nLaunching follower processes...")
        follower_configs = [
            {'can_channel': 'can_follower_r', 'gripper': 'linear_4310', 'server_port': 1234},
            {'can_channel': 'can_follower_l', 'gripper': 'linear_4310', 'server_port': 1235}
        ]
        
        # Comment to disable follower processes
        for config in follower_configs:
            process = launch_gello_process(**config)
            if process:
                processes.append(process)
                print(f"✓ Started follower process {process.pid} for {config['can_channel']}")
            else:
                raise RuntimeError(f"Failed to start follower process for {config['can_channel']}")
        
        print(f"✓ Successfully launched {len(processes)} follower processes")
        print("Waiting for processes to initialize...")
        time.sleep(3)
        
        # Setup controller with inference
        # model_path = "/root/.cache/huggingface/lerobot/single_arm_kl_10/pretrained_model"
        model_path = "/root/.cache/huggingface/lerobot/single_arm_chunk_30/pretrained_model"

        dataset_id = "cups_to_plate/20251024-160232"
        

        controller = BimanualTeleopWithInference(
            model_path=model_path,
            dataset_id=dataset_id,
            bilateral_kp=0.0  # Set to > 0 for bilateral force feedback
        )
        
        # Connect inference robot
        controller.connect_inference_robot()
        
        # Run main control loop
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up: terminate all follower processes
        print("\nCleaning up processes...")
        for process in processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()
                
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

