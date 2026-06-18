"""Unified rig config: load, control overrides, camera serials, CLI>YAML>default."""

from __future__ import annotations

import argparse

import i2rt.serving.rig_config as rc
from i2rt.serving import control_config as cc
from i2rt.serving.rig_config import Resolver, apply_camera_serials, apply_control_overrides, find_rig, load_rig
from workstation.lerobot_recorder.config import default_cameras


def _isolate(monkeypatch, tmp_path):
    """No env rig, empty repo root, empty HOME -> nothing auto-discovered."""
    monkeypatch.delenv("YAM_RIG", raising=False)
    monkeypatch.setattr(rc, "_repo_root", lambda: str(tmp_path / "norepo"))
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))


def test_load_rig(tmp_path, monkeypatch):
    p = tmp_path / "rig.yaml"
    p.write_text("robot:\n  host: 1.2.3.4\n  port: 9\ntasks:\n  - a\n  - b\n")
    rig = load_rig(str(p))
    assert rig["robot"] == {"host": "1.2.3.4", "port": 9}
    assert rig["tasks"] == ["a", "b"]
    _isolate(monkeypatch, tmp_path)
    assert load_rig(None) == {}


def test_find_rig_discovery(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("x: 1\n")
    env = tmp_path / "env.yaml"
    env.write_text("y: 1\n")
    monkeypatch.setenv("YAM_RIG", str(env))
    assert find_rig() == str(env)  # env when no explicit
    assert find_rig(str(explicit)) == str(explicit)  # explicit beats env

    # repo-root is the fixed global location
    monkeypatch.delenv("YAM_RIG", raising=False)
    rr = tmp_path / "repo"
    rr.mkdir()
    (rr / "rig.yaml").write_text("z: 1\n")
    monkeypatch.setattr(rc, "_repo_root", lambda: str(rr))
    assert find_rig() == str(rr / "rig.yaml")


def test_apply_control_overrides(monkeypatch):
    monkeypatch.setattr(cc, "BILATERAL_KP", 0.0, raising=False)
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", None, raising=False)
    monkeypatch.setattr(cc, "FOLLOWER_JOINT_LIMITS", None, raising=False)
    applied = apply_control_overrides(
        {"control": {"bilateral_kp": 0.2, "follower_effort_limit": 25.0, "follower_joint_limits": [[-1, 1], [-2, 2]]}}
    )
    assert cc.BILATERAL_KP == 0.2
    assert cc.FOLLOWER_EFFORT_LIMIT == 25.0
    assert cc.FOLLOWER_JOINT_LIMITS == [(-1, 1), (-2, 2)]  # lists -> tuples
    assert "BILATERAL_KP" in applied
    assert apply_control_overrides({}) == {}


def test_apply_camera_serials():
    cams = apply_camera_serials(default_cameras(), {"cameras": {"agentview": "AAA", "wrist_left": "BBB"}})
    by = {c.key: c.serial for c in cams}
    assert by["agentview"] == "AAA" and by["wrist_left"] == "BBB"
    assert by["wrist_right"] == ""  # untouched


def test_resolver_precedence():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dflt")

    r_yaml = Resolver(p.parse_args([]), p, {"root": "yaml"})
    assert r_yaml.get("root") == "yaml"  # YAML beats the default

    r_cli = Resolver(p.parse_args(["--root", "cli"]), p, {"root": "yaml"})
    assert r_cli.get("root") == "cli"  # explicit CLI beats YAML

    r_def = Resolver(p.parse_args([]), p, {})
    assert r_def.get("root") == "dflt"  # nothing -> default


def test_resolver_key_alias():
    p = argparse.ArgumentParser()
    p.add_argument("--robot-host", default="127.0.0.1")
    r = Resolver(p.parse_args([]), p, {"host": "10.0.0.5"})
    assert r.get("robot_host", key="host") == "10.0.0.5"
