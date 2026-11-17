"""Tests against the real SmolVLA + DeFM weights (downloaded on first run; CPU is enough, a GPU is used if free).

They answer the two design questions directly:
  1. element-wise add: the depth residual lands on exactly the 64 tokens of its own camera, nothing else moves,
     and with the gate at 0 the policy is numerically stock SmolVLA.
  2. token count: the prefix the VLM sees has the same length with and without depth.
plus freezing, LoRA save / load / merge.

    pytest -q -m weights
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("defm")

from depthvla.runtime import make_features, make_processors, preprocess  # noqa: E402

pytestmark = pytest.mark.weights

BASE = "lerobot/smolvla_base"
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10 * 2**30 else "cpu"
CAMS, HW, S, A, H = ["wrist", "front"], (96, 96), 8, 7, 8
DEPTH_MAP = {"observation.images.front": "observation.depth.front"}  # depth on the SECOND camera only
STATS = {"observation.state": {"mean": np.zeros(S, np.float32), "std": np.ones(S, np.float32),
                               "min": -np.ones(S, np.float32), "max": np.ones(S, np.float32)},
         "action": {"mean": np.zeros(A, np.float32), "std": np.ones(A, np.float32),
                    "min": -np.ones(A, np.float32), "max": np.ones(A, np.float32)}}


def raw_batch(bs=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = torch.rand(bs, 1, *HW, generator=g) * 0.8 + 0.3
    depth[:, :, :10, :10] = 0.0  # some invalid pixels
    return {"observation.images.wrist": torch.rand(bs, 3, *HW, generator=g),
            "observation.images.front": torch.rand(bs, 3, *HW, generator=g),
            "observation.depth.front": depth,
            "observation.state": torch.randn(bs, S, generator=g),
            "action": torch.randn(bs, H, A, generator=g),
            "action_is_pad": torch.zeros(bs, H, dtype=torch.bool),
            "task": ["pick up the mug"] * bs}


def build(**kw):
    from depthvla.policy import build_policy

    inputs, outputs = make_features(CAMS, HW, S, A)
    kw.setdefault("use_depth", True)
    if kw["use_depth"]:
        kw.setdefault("depth_map", DEPTH_MAP)
    return build_policy(BASE, inputs, outputs, device=DEV, chunk_size=H, n_action_steps=H, empty_cameras=0,
                        depth_dropout=0.0, **kw)


@pytest.fixture(scope="module")
def policy():
    return build(lora_r=8)


@pytest.fixture(scope="module")
def pre(policy):
    return make_processors(policy.config, STATS)[0]


def chunk(policy, batch, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    noise = torch.randn((batch["observation.state"].shape[0], H, policy.config.max_action_dim), generator=g, device=DEV)
    return policy.predict_action_chunk(dict(batch), noise=noise).float().cpu()


def prefix(policy, batch, with_depth):
    """The prefix embeddings SmolVLA hands to the VLM, with the depth hook active or not."""
    m = policy.model
    images, masks = policy.prepare_images(batch)
    args = (images, masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"])
    state = policy.prepare_state(batch)
    with torch.no_grad():
        if not with_depth:
            return m.embed_prefix(*args, state=state)
        with policy._depth(batch):
            return m.embed_prefix(*args, state=state)


def test_token_count_is_unchanged_and_residual_hits_only_its_camera(policy, pre):
    b = preprocess(pre, raw_batch(), DEV)
    policy.eval()
    gate = policy.model.depth_fusion.gate
    emb0, pad0, att0 = prefix(policy, b, with_depth=False)
    gate.data.fill_(0.5)
    try:
        emb1, pad1, att1 = prefix(policy, b, with_depth=True)
        with torch.no_grad():
            res = policy.depth_residuals(b)
    finally:
        gate.data.zero_()
    n = policy.image_token_grid() ** 2
    assert n == 64
    assert res[0] is None and res[1].shape == (2, n, emb0.shape[-1])  # (B, 64, 960) for the depth camera only
    # 1. same prefix length, same masks: the VLM cannot tell the token layout changed
    assert emb1.shape == emb0.shape and torch.equal(pad0, pad1) and torch.equal(att0, att1)
    # 2. only the second camera's 64 tokens moved
    moved = (emb1.float() - emb0.float()).abs().amax(dim=(0, 2)) > 0
    assert moved.nonzero().flatten().tolist() == list(range(n, 2 * n))
    # and they moved by exactly residual * sqrt(dim) (SmolVLA scales image tokens after the hook)
    want = res[1].float() * emb0.shape[-1] ** 0.5
    got = (emb1.float() - emb0.float())[:, n:2 * n]
    assert torch.allclose(got, want.to(got.device), rtol=1e-3, atol=1e-3 * float(want.abs().max()))


def test_zero_gate_is_stock_smolvla_and_open_gate_uses_depth(policy, pre):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    b = preprocess(pre, raw_batch(), DEV)
    ours = chunk(policy, b)  # gate = 0, LoRA B = 0 -> must equal the stock policy
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.input_features, cfg.output_features = policy.config.input_features, policy.config.output_features
    cfg.chunk_size = cfg.n_action_steps = H
    cfg.empty_cameras, cfg.device = 0, DEV
    stock = SmolVLAPolicy.from_pretrained(BASE, config=cfg)
    stock.model.vlm_with_expert.lm_expert.float()  # same precision as ours (exact cast), see expert_float32
    ref = chunk(stock, {k: v for k, v in b.items() if "depth" not in k})
    del stock
    assert torch.allclose(ours, ref, atol=1e-4), float((ours - ref).abs().max())

    gate = policy.model.depth_fusion.gate
    gate.data.fill_(0.5)
    try:
        opened = chunk(policy, b)
        b2 = dict(b)
        b2["observation.depth.front"] = b["observation.depth.front"] * 0.5  # scene twice as close
        closer = chunk(policy, b2)
    finally:
        gate.data.zero_()
    assert (opened - ours).abs().max() > 1e-3  # depth now changes the actions
    assert (closer - opened).abs().max() > 1e-4  # and different depth gives different actions


def test_only_fusion_lora_and_heads_train(policy, pre):
    counts = policy.configure_trainable("lora", train_heads=True)
    assert counts["vlm_trainable"] == 0 and counts["defm_trainable"] == 0
    assert counts["depth_fusion_trainable"] > 0 and counts["depth_fusion_frozen"] == 0
    assert 0 < counts["expert_trainable"] < 0.1 * counts["expert_frozen"]  # adapters only
    policy.train()
    assert not policy.model.depth_encoder.model.training and not policy.model.vlm_with_expert.vlm.training
    policy.model.depth_fusion.gate.data.fill_(0.1)
    try:
        loss, _ = policy.forward(preprocess(pre, raw_batch(), DEV))
        loss.backward()
        named = dict(policy.named_parameters())
        with_grad = {n for n, p in named.items() if p.grad is not None and p.grad.abs().sum() > 0}
        assert "model.depth_fusion.gate" in with_grad and "model.depth_fusion.fc1.weight" in with_grad
        assert any("lora_" in n for n in with_grad)
        assert not any(n.startswith(("model.vlm_with_expert.vlm.", "model.depth_encoder.")) for n in with_grad)
        assert all(named[n].requires_grad for n in with_grad)
    finally:
        policy.zero_grad(set_to_none=True)
        policy.model.depth_fusion.gate.data.zero_()
        policy.eval()


def test_save_load_merge_round_trip(policy, pre, tmp_path):
    from depthvla.policy import load_policy
    from depthvla.runtime import load_processors, save_checkpoint

    # make the adapters and the gate non-trivial, as after training
    torch.manual_seed(0)
    for n, p in policy.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    policy.model.depth_fusion.gate.data.fill_(0.3)
    try:
        b = preprocess(pre, raw_batch(), DEV)
        before = chunk(policy, b)
        pre_, post_ = make_processors(policy.config, STATS)
        ck = str(tmp_path / "ck")
        save_checkpoint(ck, policy, pre_, post_, {"cameras": CAMS, "depth_cameras": ["front"], "image_hw": list(HW)})
    finally:
        for n, p in policy.named_parameters():
            if "lora_B" in n:
                p.data.zero_()
        policy.model.depth_fusion.gate.data.zero_()

    loaded = load_policy(ck, DEV)
    assert loaded.config.lora_r == 8 and not any(p.requires_grad for p in loaded.parameters())
    pre2, _ = load_processors(ck, loaded.config, DEV)
    b2 = preprocess(pre2, raw_batch(), DEV)
    assert torch.allclose(chunk(loaded, b2), before, atol=1e-4)
    loaded.merge_lora()
    assert loaded.config.lora_r == 0 and not any("lora_" in n for n, _ in loaded.named_parameters())
    assert torch.allclose(chunk(loaded, b2), before, atol=1e-4)
