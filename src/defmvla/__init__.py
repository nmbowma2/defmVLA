"""SmolVLA + frozen DeFM depth encoder (element-wise fusion on the RGB image tokens).

The package root imports nothing heavy: `defmvla.episodes` (numpy + pillow) can be used inside a simulator
python env; `defmvla.policy` needs torch + lerobot + defm.
"""

__version__ = "0.1.0"
