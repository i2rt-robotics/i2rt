from datetime import datetime
from pathlib import Path

import pytest
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from i2rt.utils.recording import RobotMcapRecorder


def test_robot_mcap_recorder_writes_ros2_cdr_motor_feedback(tmp_path: Path) -> None:
    joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    recorder = RobotMcapRecorder.create(
        joint_names,
        root=tmp_path,
        now=datetime(2026, 8, 5, 12, 34, 56),
    )
    recorder.add(
        timestamp=100.25,
        position=[1, 2, 3, 4, 5, 6, 0.5],
        velocity=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        effort=[10, 20, 30, 40, 50, 60, 70],
        required_torque=[11, 21, 31, 41, 51, 61, 71],
        temp_mos=[31, 32, 33, 34, 35, 36, 37],
        temp_rotor=[41, 42, 43, 44, 45, 46, 47],
    )
    recorder.close()

    assert recorder.path == tmp_path / "20260805" / "20260805-123456" / "robot.mcap"
    with recorder.path.open("rb") as stream:
        messages = list(make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages())

    assert len(messages) == 16
    assert {channel.message_encoding for _, channel, _, _ in messages} == {"cdr"}
    decoded_by_topic = {
        channel.topic: (schema.name, message.log_time, decoded) for schema, channel, message, decoded in messages
    }

    schema_name, log_time, joint_state = decoded_by_topic["/joint_states"]
    assert schema_name == "sensor_msgs/msg/JointState"
    assert log_time == 100_250_000_000
    assert joint_state.name == joint_names
    assert joint_state.position == pytest.approx([1, 2, 3, 4, 5, 6, 0.5])
    assert joint_state.velocity == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert joint_state.effort == pytest.approx([10, 20, 30, 40, 50, 60, 70])

    required_schema, required_log_time, required_torques = decoded_by_topic["/required_torques"]
    assert required_schema == "sensor_msgs/msg/JointState"
    assert required_log_time == log_time
    assert required_torques.name == joint_names
    assert required_torques.position == []
    assert required_torques.velocity == []
    assert required_torques.effort == pytest.approx([11, 21, 31, 41, 51, 61, 71])

    for index, joint_name in enumerate(joint_names):
        mos_schema, _, mos = decoded_by_topic[f"/{joint_name}/temperature/mos"]
        rotor_schema, _, rotor = decoded_by_topic[f"/{joint_name}/temperature/rotor"]
        assert mos_schema == rotor_schema == "sensor_msgs/msg/Temperature"
        assert mos.header.frame_id == rotor.header.frame_id == joint_name
        assert mos.temperature == 31 + index
        assert rotor.temperature == 41 + index
