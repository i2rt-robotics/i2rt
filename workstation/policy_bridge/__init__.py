"""Policy bridge (workstation): joins the robot (portal) and policy (websocket).

Runs the openpi-style observation→action loop, translating between the robot
server's portal RPC and a websocket policy server. See :mod:`workstation.policy_bridge.bridge`.
"""
