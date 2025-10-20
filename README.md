# I2RT Python API

Please refer to the original repository for detailed instructions on installation and use:
**[https://github.com/i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt)**

## Fork Additions

This fork adds:

- **LeRobot Integration**: Data collection and inference support in `examples/bimanual_record_lerobot/`
  - See: [LeRobot Teleop and Record Guide](https://github.com/robot-com-projects/robot-os/blob/main/docs/04_TeleopAndRecord.md)
  - See: [LeRobot GUI Recording Documentation](https://github.com/robot-com-projects/robot-os/blob/main/docs/Record_README.md)

- **MuJoCo Simulation**: Episode replay in `sim/` folder
  ```bash
  # Basic episode replay
  python sim/mujoco_rerun.py --repo-id RECORDING_NAME_PATH --episode-index EPISODE_IDX
  
  # Inference with trained model
  python sim/inference_mujoco_rerun.py --repo-id RECORDING_NAME_PATH --episode-index EPISODE_IDX --model-path MODEL_PATH
  ```
