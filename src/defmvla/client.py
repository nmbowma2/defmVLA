"""Client for scripts/serve_policy.py. numpy + stdlib only, so it runs inside the simulator's python env.

    from depthvla.client import PolicyClient
    pc = PolicyClient(port=6040)                    # host="gpu-box" to reach a server on another machine
    info = pc.info()                                # cameras, depth cameras, image size, horizon, ...
    pc.reset(seed=episode_seed)                     # at every episode start
    actions = pc.act(images={"wrist": rgb}, depth={"wrist": depth_m}, state=state, task="pick up the mug")
    # actions: (horizon, action_dim) float32, un-normalised; execute the first info["n_action_steps"], re-query
"""

from __future__ import annotations

from multiprocessing.connection import Client
from typing import Dict, Optional

import numpy as np

DEFAULT_PORT = 6040
DEFAULT_AUTHKEY = "depthvla"


class PolicyClient:
    def __init__(self, port: int = DEFAULT_PORT, host: str = "localhost", authkey: str = DEFAULT_AUTHKEY,
                 square: Optional[int] = None):
        """square: same value as EpisodeWriter(square=...) used for the training data, so live frames get the
        identical centre-crop + resize before they are sent."""
        self.conn = Client((host, port), authkey=authkey.encode())
        self.square = int(square) if square else None

    def _call(self, req: Dict) -> Dict:
        self.conn.send(req)
        rep = self.conn.recv()
        if "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        return rep

    def info(self) -> Dict:
        return self._call({"cmd": "info"})

    def reset(self, seed: Optional[int] = None) -> None:
        self._call({"cmd": "reset", "seed": seed})

    def act(self, images: Dict[str, np.ndarray], state: np.ndarray, task: str,
            depth: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        depth = depth or {}
        if self.square:
            from .episodes import center_square
            images = {k: center_square(v, self.square) for k, v in images.items()}
            depth = {k: center_square(np.asarray(v).squeeze(), self.square, depth=True) for k, v in depth.items()}
        rep = self._call({"cmd": "act", "images": images, "depth": depth, "state": np.asarray(state, np.float32),
                          "task": task})
        self.latency_s = rep.get("latency_s")
        return np.asarray(rep["actions"], np.float32)

    def close(self):
        self.conn.close()
