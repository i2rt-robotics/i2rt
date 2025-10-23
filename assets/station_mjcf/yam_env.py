import time
from typing import Dict, Optional

import dm_env
import mujoco
import mujoco.viewer
import numpy as np
from dm_env import specs


class YamEnv(dm_env.Environment):
    """A MuJoCo environment for robotic manipulation tasks."""

    # Configuration constants
    CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640
    GRIPPER_SCALE = 0.041
    DUAL_ARM_DOFS, ARM_DOFS, GRIPPER_DOFS = 14, 6, 1
    DEFAULT_PHYSICS_DT, DEFAULT_CONTROL_DT = 0.002, 0.02
    INTEGRATOR_RK4, MODEL_TIMESTEP = 3, 0.0001

    # Gripper action indices
    LEFT_GRIPPER_IDX = 6
    RIGHT_GRIPPER_IDX = 13


    def __init__(
        self,
        seed: Optional[int] = None,
        control_dt: float = DEFAULT_CONTROL_DT,
        physics_dt: float = DEFAULT_PHYSICS_DT,
        time_limit: float = np.inf,
        randomize_scene: bool = True,
    ) -> None:
        """Initialize the YamEnv environment.

        Args:
            seed: Random seed for reproducibility.
            control_dt: Control timestep in seconds.
            physics_dt: Physics simulation timestep in seconds.
            time_limit: Maximum episode length in seconds.
            randomize_scene: Whether to randomize the scene on reset.
        """
        self._model = self._build_model()
        self._data = mujoco.MjData(self._model)
        self._randomize_scene = randomize_scene

        self._model.opt.timestep = physics_dt
        self.control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._terminated_already = False
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._renderer: Optional[mujoco.Renderer] = None

        # Initialize state buffer
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(self._model, state_spec), np.float64)

    def _build_task_spec(self, station_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Override this method to add task-specific objects to the scene.

        Args:
            station_spec: The base station specification to modify.

        Returns:
            The modified station specification with task-specific objects.
        """
        return station_spec

    def _build_model(self) -> mujoco.MjModel:
        """Build the MuJoCo model from the station XML file.

        Returns:
            The compiled MuJoCo model with configured physics and mappings.
        """
        station_spec = mujoco.MjSpec.from_file("station.xml")
        station_spec.copy_during_attach = True
        station_spec = self._build_task_spec(station_spec)

        # Set up camera mappings
        cameras = list(station_spec.cameras)
        self.camera_ids = {}
        for side in ["top", "left", "right"]:
            camera = next(x for x in cameras if side in x.name)
            self.camera_ids[side] = camera.name

        # Compile and configure model
        model = station_spec.compile()
        model.opt.timestep, model.opt.integrator = (
            self.MODEL_TIMESTEP,
            self.INTEGRATOR_RK4,
        )

        # Set up joint and actuator mappings
        self.actuator_names = [x.name for x in station_spec.actuators]
        self.actuator_ids = np.array(
            [model.actuator(name).id for name in self.actuator_names]
        )
        self.joint_names = [x.name for x in station_spec.joints]

        self.left_joint_names = [
            x.name for x in station_spec.joints if x.name.startswith("left_")
        ]
        self.left_joint_ids = np.array(
            [model.joint(name).id for name in self.left_joint_names]
        )

        self.right_joint_names = [
            x.name for x in station_spec.joints if x.name.startswith("right_")
        ]
        self.right_joint_ids = np.array(
            [model.joint(name).id for name in self.right_joint_names]
        )
        return model

    def time_limit_exceeded(self) -> bool:
        """Check if the episode time limit has been exceeded.

        Returns:
            True if the current simulation time exceeds the time limit.
        """
        return self._data.time >= self._time_limit

    def render(self, camera_name: str) -> np.ndarray:
        """Render the scene from the specified camera.

        Args:
            camera_name: Name of the camera to render from.

        Returns:
            RGB image array of shape (height, width, 3).
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, self.CAMERA_HEIGHT, self.CAMERA_WIDTH
            )
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self._data, camera=camera_name)
        return self._renderer.render()

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment to its initial state.

        Returns:
            Initial timestep with observations.
        """
        self._terminated_already = False
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs,
        )

    def action_spec(self) -> specs.Array:
        return specs.Array(shape=(self.DUAL_ARM_DOFS,), dtype=np.float32)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Execute an action and return the resulting timestep.

        Args:
            action: Action array of shape (DUAL_ARM_DOFS,) containing joint commands.

        Returns:
            Timestep containing observations, reward, and episode status.

        Raises:
            ValueError: If environment has already terminated.
        """
        if self._terminated_already:
            raise ValueError("Environment has terminated. Please reset.")

        # Scale gripper actions and execute
        action = action.copy()
        action[self.LEFT_GRIPPER_IDX] *= self.GRIPPER_SCALE
        action[self.RIGHT_GRIPPER_IDX] *= self.GRIPPER_SCALE
        self._data.ctrl[self.actuator_ids] = action

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()
        if terminated:
            step_type = dm_env.StepType.LAST
            self._terminated_already = True
        else:
            step_type = dm_env.StepType.MID

        return dm_env.TimeStep(
            step_type=step_type,
            reward=rew,
            discount=1.0,
            observation=obs,
        )

    def observation_spec(self) -> Dict[str, specs.Array]:
        """Return the observation specification."""
        spec = {
            side: {
                "joint_pos": specs.Array(shape=(self.ARM_DOFS,), dtype=np.float32),
                "joint_vel": specs.Array(shape=(self.ARM_DOFS,), dtype=np.float32),
                "gripper_pos": specs.Array(
                    shape=(self.GRIPPER_DOFS,), dtype=np.float32
                ),
            }
            for side in ["left", "right"]
        }

        for camera in ["top", "left", "right"]:
            spec[f"{camera}_camera"] = {
                "images": {
                    "rgb": specs.Array(
                        (self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 3), np.uint8
                    )
                },
                "timestamp": specs.Array(shape=(1,), dtype=np.float32),
            }

        spec["state"] = specs.Array(shape=self.state.shape, dtype=np.float32)
        return spec

    def get_obs(self) -> Dict[str, any]:
        return self._compute_observation()

    def _compute_observation(self) -> Dict[str, any]:
        """Compute the current observation.

        Returns:
            Dictionary containing joint states, camera images, and full state.
        """
        obs = {}

        # Joint observations for both arms
        arm_joint_mapping = {"left": self.left_joint_ids, "right": self.right_joint_ids}
        for side_name, joint_ids in arm_joint_mapping.items():
            qpos = self._data.qpos[joint_ids].astype(np.float32)
            qvel = self._data.qvel[joint_ids].astype(np.float32)
            obs[side_name] = {
                "joint_pos": qpos[: self.ARM_DOFS],
                "gripper_pos": qpos[self.ARM_DOFS :],
                "joint_vel": qvel[: self.ARM_DOFS],
            }

        # Camera observations
        obs.update(self._get_camera_observations())

        # Full state
        mujoco.mj_getState(
            self._model, self._data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        obs["state"] = self.state.copy()
        return obs

    def _get_camera_observations(self) -> Dict[str, any]:
        """Get observations from all cameras.

        Returns:
            Dictionary containing camera observations with RGB images and timestamps.
        """
        camera_obs = {}
        for camera in ["top", "left", "right"]:
            camera_obs[f"{camera}_camera"] = {
                "images": {"rgb": self.render(self.camera_ids[camera])},
                "timestamp": time.time(),
            }
        return camera_obs

    def _compute_reward(self) -> float:
        """Compute the reward for the current state.

        Returns:
            Reward value (0.0 for base environment).
        """
        return 0.0


class YamEnvPickRedCube(YamEnv):
    """A specific task environment for picking up a red cube."""

    # Task-specific constants
    CUBE_SIZE = 0.015
    CUBE_SPAWN_POS = [0.6, -0.3, 0.753]
    GOAL_REGION_POS = [0.6, -0.3, 0.753 + 0.3]
    GOAL_REGION_SIZE = [0.25, 0.25, 0.1]
    RANDOMIZATION_RANGE = 0.15
    JOINT_NOISE_RANGE = 0.1

    def _build_task_spec(self, station_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Add a red cube and goal region to the station.

        Args:
            station_spec: The base station specification to modify.

        Returns:
            The modified station specification with cube and goal region.
        """
        self._add_cube_to_scene(station_spec)
        self._add_goal_region_to_scene(station_spec)
        return station_spec

    def _add_cube_to_scene(self, station_spec: mujoco.MjSpec) -> None:
        """Add a red cube to the scene.

        Args:
            station_spec: The station specification to modify.
        """
        spawn_pos = self.CUBE_SPAWN_POS.copy()
        spawn_pos[2] += self.CUBE_SIZE / 2  # Adjust for cube height
        cube_spawn_site = station_spec.worldbody.add_site(pos=spawn_pos)

        cube_spec = mujoco.MjSpec()
        body = cube_spec.worldbody.add_body(name="cube_body")
        body.add_geom(
            name="red_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.CUBE_SIZE] * 3,
            rgba=[1, 0, 0, 1],
        )

        cube_body = cube_spawn_site.attach_body(cube_spec.worldbody, "cube_", "")
        self.cube_freejoint_name = "cube_joint"
        cube_body.add_freejoint(name=self.cube_freejoint_name)

    def _add_goal_region_to_scene(self, station_spec: mujoco.MjSpec) -> None:
        """Add a transparent goal region above the cube.

        Args:
            station_spec: The station specification to modify.
        """
        station_spec.worldbody.add_geom(
            pos=self.GOAL_REGION_POS,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.GOAL_REGION_SIZE,
            rgba=[0.5, 1, 0.5, 0.05],
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
        )

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment with optional scene randomization."""
        self._terminated_already = False
        mujoco.mj_resetData(self._model, self._data)

        if self._randomize_scene:
            self._randomize_cube_position()
            self._randomize_robot_joints()

        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs,
        )

    def _randomize_cube_position(self) -> None:
        """Randomize the cube's initial position and orientation."""
        cube_joint = self._data.jnt(self.cube_freejoint_name)

        # Randomize x, y position
        random_offset = self._random.uniform(
            -self.RANDOMIZATION_RANGE, self.RANDOMIZATION_RANGE, size=2
        )
        cube_joint.qpos[:2] += random_offset

        # Randomize rotation around z-axis
        perturb_axis = np.array([0.0, 0.0, 1.0])
        perturb_theta = self._random.uniform(-np.pi, np.pi)
        mujoco.mju_axisAngle2Quat(cube_joint.qpos[3:], perturb_axis, perturb_theta)

    def _randomize_robot_joints(self) -> None:
        """Add noise to robot joint positions."""
        all_joint_names = self.right_joint_names + self.left_joint_names
        arm_joint_names = [j for j in all_joint_names if "finger" not in j]

        for joint_name in arm_joint_names:
            joint = self._data.jnt(joint_name)
            noise = self._random.uniform(
                -self.JOINT_NOISE_RANGE, self.JOINT_NOISE_RANGE, size=joint.qpos.shape
            )
            joint.qpos += noise


class KeyReset:
    """Helper class to handle keyboard input for environment reset."""

    def __init__(self) -> None:
        """Initialize the key reset handler."""
        self.reset = False

    def key_callback(self, keycode: int) -> None:
        """Handle key press events.

        Args:
            keycode: The pressed key code.
        """
        from dm_control.viewer import user_input

        if keycode == user_input.KEY_SPACE:
            self.reset = True


def main() -> None:
    """Run the environment with random actions in the MuJoCo viewer.

    This function demonstrates the environment by running it with random actions
    and allowing the user to reset with the spacebar.
    """
    env = YamEnvPickRedCube()
    t = env.reset()
    reset = KeyReset()  # press space to reset the environment

    def action(t):
        return np.random.uniform(-1, 1, (14,))

    with mujoco.viewer.launch_passive(
        env._model, env._data, key_callback=reset.key_callback
    ) as viewer:
        while viewer.is_running():
            if reset.reset:
                t = env.reset()
                reset.reset = False

            step_start = time.time()
            t = env.step(action(t))
            viewer.sync()

            sleep_time = env.control_dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
