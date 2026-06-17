# yam-policy — policy serving (openpi-compatible)

A small, **dependency-light** websocket policy layer. The robot/workstation side
holds only this package (`numpy`, `msgpack-numpy`, `websockets`, `pillow`); the
**policy server** runs in its own unrestricted env with whatever the model needs
(torch / JAX / CUDA), on this machine or a remote GPU box.

The wire protocol is **identical to openpi** (`openpi_client`), so a real openpi
checkpoint served by `openpi` works against our client, and a policy written
against `BasePolicy` can be served by openpi.

```
workstation (policy bridge)              policy server (this package)
  WebsocketClientPolicy  ──ws+msgpack──▶  WebsocketPolicyServer(policy)
  + ActionChunkBroker    ◀────actions───  policy.infer(obs) -> {"actions": (H, D)}
```

## Contract

```python
obs = {
    "observation/state":            np.ndarray,   # proprioception, unnormalized
    "observation/images/<cam>":     np.ndarray,   # HxWx3 uint8 (e.g. 224x224)
    "prompt":                       "do the task" # optional, language-conditioned
}
action_chunk = client.infer(obs)["actions"]       # (action_horizon, action_dim)
```

## Install (policy server env — unrestricted)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd policy_serving
uv venv                      # any Python >= 3.10; no ROS, no rclpy ABI constraint
source .venv/bin/activate
uv pip install -e .          # + your model deps, e.g. uv pip install -e /path/to/openpi
```

## Run a server

```bash
# zero-model smoke test (returns a "hold pose" chunk):
python -m yam_policy.serve

# a real LeRobot policy (template — adapt policies/lerobot_policy.py):
python -m yam_policy.serve \
    --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
    --config pretrained_path=/abs/path --config device=cuda

# a real openpi checkpoint (template — needs openpi installed here):
python -m yam_policy.serve \
    --policy yam_policy.policies.openpi_policy:OpenPiPolicy \
    --config config_name=pi0_fast_droid --config checkpoint_dir=/abs/ckpt
```

## Add your own policy

Subclass `BasePolicy` and implement `infer(obs) -> {"actions": (H, D)}`. See
[`yam_policy/policies/dummy.py`](yam_policy/policies/dummy.py) for the simplest
example and [`lerobot_policy.py`](yam_policy/policies/lerobot_policy.py) /
[`openpi_policy.py`](yam_policy/policies/openpi_policy.py) for real-model templates.
Then serve it with `--policy your.module:YourPolicy`.

## Client side (used by the workstation bridge)

```python
from yam_policy import WebsocketClientPolicy, AsyncActionChunkBroker
client = WebsocketClientPolicy(host="policy-host", port=8000)
policy = AsyncActionChunkBroker(client, action_horizon=16)   # prefetches the next chunk
action = policy.infer(obs)["actions"]   # one (action_dim,) step per call, re-queries every 16
```

- **AsyncActionChunkBroker** fetches the next chunk in a background thread so the
  per-chunk inference latency doesn't stall the control loop (use
  `ActionChunkBroker` for the simple synchronous version).
- **Metadata-driven config**: declare `action_horizon` (and optionally an
  `obs_spec` dict with `image_keys` / `image_size`) on your policy; `serve.py`
  puts them in the server metadata and the bridge auto-configures from
  `get_server_metadata()` — no need to hand-match the bridge to the policy.
