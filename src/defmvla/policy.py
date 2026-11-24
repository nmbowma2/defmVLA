"""SmolVLA with DeFM depth fused into the RGB image tokens (lerobot 0.6.1).

What changes relative to stock SmolVLA
--------------------------------------
Stock:   image -> SigLIP -> connector -> 64 tokens x 960  -> [img tokens | language | state] -> VLM -> expert
Here:    the same, plus for every camera listed in `depth_map`:
             depth -> frozen DeFM -> DepthFusion -> 64 x 960 residual, ADDED to that camera's 64 image tokens
         (scaled by sqrt(dim) like the image tokens, so it is equivalent to adding at the connector output).

Nothing else in the prefix changes: same number of tokens, same attention masks, same position ids.
The hook is a wrapper around `VLAFlowMatching.embed_prefix`: SmolVLA's own `embed_prefix`, `forward` and
`sample_actions` run unmodified (no copied upstream code to drift), and the residual is added to the finished
prefix embeddings.

Trainable / frozen
------------------
frozen:     SigLIP, DeFM, the early VLM text layers (and, unless switched on, everything else of the VLM)
trainable:  DepthFusion (MLP + gate), the action expert (LoRA adapters or all of it), optionally the small
            state / action / time projections, optionally the connector (image projector, all of it) and LoRA
            adapters on the attention of the last `vlm_lora_layers` VLM text layers
With the gate at its initial 0 the policy is numerically the RGB policy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .fusion import DeFMEncoder, DepthFusion, resize_depth_like_rgb

log = logging.getLogger(__name__)

HEADS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")
TRAIN_MODES = ("lora", "expert", "none")


@PreTrainedConfig.register_subclass("smolvla_depth")
@dataclass
class DepthSmolVLAConfig(SmolVLAConfig):
    # --- depth branch -------------------------------------------------------------------------------
    use_depth: bool = True  # False = stock SmolVLA architecture (the RGB-only baseline, same code path)
    # RGB feature -> depth batch key, e.g. {"observation.images.wrist": "observation.depth.wrist"}.
    # The depth map must be pixel-aligned with that RGB image (same camera, same resolution).
    depth_map: dict[str, str] = field(default_factory=dict)
    defm_variant: str = "defm_vit_s14"
    defm_input_size: int = 448  # 448 / 14 = 32 x 32 DeFM patches = 4 x 4 per SmolVLA image token (8 x 8)
    defm_weights_path: str | None = None  # local .pth; None = Hugging Face Hub (leggedrobotics/defm)
    defm_pretrained: bool = True  # set False by load_policy: the checkpoint already holds the DeFM weights
    depth_pool: str = "unshuffle"  # "unshuffle" | "avg"
    depth_hidden: int = 1024
    depth_max_m: float = 0.0  # > 0: clamp depth to this range before DeFM (keeps DeFM's per-image channel stable)
    depth_dropout: float = 0.25  # training: drop the whole depth residual of a sample with this probability
    depth_noise_m: float = 0.0  # training: Gaussian noise (std, metres) on valid depth pixels

    # --- LoRA on the action expert (0 = no adapters) -------------------------------------------------
    lora_r: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # smolvla_base stores most of the action expert in bfloat16 and lerobot keeps (and trains) it that way.
    # Here the expert's weights are held in float32 (the cast is exact): optimizer updates and LoRA deltas
    # smaller than bf16's 8-bit mantissa are not rounded away, and merging LoRA for deployment is exact.
    # Under autocast the forward pass still runs in bf16. The frozen VLM stays bfloat16.
    expert_float32: bool = True

    # --- LoRA on the VLM backbone: attention of the last `vlm_lora_layers` text layers (0 = no adapters) ---
    # The depth residual joins after SigLIP and the connector, so only the text layers process fused tokens.
    vlm_lora_r: int = 0
    vlm_lora_alpha: int = 32
    vlm_lora_dropout: float = 0.0
    vlm_lora_layers: int = 8
    vlm_lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Same reason as expert_float32. Both stay true after merging: a checkpoint must be rebuilt in the precision
    # its weights were saved in, or the merged / trained weights would be rounded to bf16 on load.
    vlm_float32: bool = False  # the VLM layers that carry (or carried) adapters
    connector_float32: bool = False  # the connector, when it is (or was) trained

    def __post_init__(self):
        super().__post_init__()
        if self.depth_pool not in ("unshuffle", "avg"):
            raise ValueError(f"depth_pool must be 'unshuffle' or 'avg', got {self.depth_pool!r}")
        if not 0.0 <= self.depth_dropout < 1.0:
            raise ValueError("depth_dropout must be in [0, 1)")


class DepthSmolVLAPolicy(SmolVLAPolicy):
    config_class = DepthSmolVLAConfig
    name = "smolvla_depth"

    def __init__(self, config: DepthSmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        m = self.model
        m._depth_res = None  # per-forward list of residuals, aligned with the image list
        m._depth_i = 0
        m._depth_ratio = None
        if config.use_depth:
            unknown = set(config.depth_map) - set(config.image_features)
            if not config.depth_map or unknown:
                raise ValueError(f"depth_map must map RGB features of the policy to depth keys; unknown RGB keys "
                                 f"{sorted(unknown)}, image features {list(config.image_features)}")
            m.depth_encoder = DeFMEncoder(config.defm_variant, config.defm_pretrained, config.defm_weights_path,
                                          config.defm_input_size)
            m.depth_fusion = DepthFusion(m.depth_encoder.out_channels, m.depth_encoder.grid,
                                         self.image_token_grid(), m.vlm_with_expert.config.text_config.hidden_size,
                                         hidden=config.depth_hidden, pool=config.depth_pool)
            self._install_prefix_hook()
        if config.expert_float32:
            self.expert.float()
        if config.vlm_float32:
            for layer in {i for i, _ in self._vlm_lora_sites()}:
                self.text_model.layers[layer].float()  # whole layer, so it also runs without autocast
        if config.connector_float32:
            connector = m.vlm_with_expert.get_vlm_model().connector
            connector.float()  # SigLIP hands over bf16; without autocast the dtypes would not match
            connector.register_forward_pre_hook(lambda _m, args: (args[0].float(),))
        if config.lora_r > 0:
            self._inject_lora()
        if config.vlm_lora_r > 0:
            self._inject_vlm_lora()

    def train(self, mode: bool = True):
        """lerobot puts the whole VLM in eval mode when train_expert_only is set, which would silently switch off
        the dropout of the VLM adapters."""
        super().train(mode)
        for n, mod in self.text_model.named_modules():
            if n.endswith("lora_dropout.default"):
                mod.train(mode)
        return self

    # ------------------------------------------------------------------ token-count bookkeeping
    def image_token_grid(self) -> int:
        """Side of the square token grid one image becomes (8 for SmolVLM2-500M at 512 px: 32 x 32 SigLIP
        patches, pixel-shuffled 4 x 4 -> 8 x 8 = 64 tokens)."""
        vcfg = self.model.vlm_with_expert.config
        size = self.config.resize_imgs_with_padding
        if size is None:
            side = vcfg.vision_config.image_size
        else:
            if size[0] != size[1]:
                raise ValueError(f"depth fusion needs a square SmolVLA input, got resize_imgs_with_padding={size}")
            side = size[0]
        patches = side // vcfg.vision_config.patch_size
        if patches % vcfg.scale_factor:
            raise ValueError(f"{patches} patches per side is not divisible by the connector scale {vcfg.scale_factor}")
        return patches // vcfg.scale_factor

    def _install_prefix_hook(self):
        """Wrap SmolVLA's `embed_prefix`: it runs unmodified, then each depth residual is added to the block of
        image tokens of its camera inside the finished prefix. Layout of the prefix (stock):
            [ image 0 | image 1 | ... | language | state ]     each image = (2 special) + n tokens + (1 special)
        The special tokens only exist with `add_image_special_tokens`.

        Stock scales image tokens by sqrt(dim) before concatenating, so the residual is scaled the same way:
        this equals adding it to the connector output. The add is done in float32: the VLM weights are
        bfloat16 (8 bits of mantissa), where a residual below ~1/256 of the RGB token would be rounded away,
        i.e. everything while the gate is still small. A zero residual leaves the prefix bit-identical."""
        m = self.model
        stock_embed_prefix = m.embed_prefix
        n = self.image_token_grid() ** 2

        def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
            embs, pad_masks, att_masks = stock_embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
            res = m._depth_res
            if res is None:
                return embs, pad_masks, att_masks
            m._depth_i += 1
            lead = m.global_image_start_token.numel() if m.add_image_special_tokens else 0
            tail = m.image_end_token.numel() if m.add_image_special_tokens else 0
            block = lead + n + tail
            bsize, length, dim = embs.shape
            if len(res) > len(images) or length < len(images) * block:
                raise RuntimeError(f"prefix of {length} tokens cannot hold {len(images)} images of {block} tokens: "
                                   "the depth hook is out of sync with this lerobot version")
            add, ratios = None, []
            for i, r in enumerate(res):
                if r is None:
                    continue
                if r.shape != (bsize, n, dim):
                    raise RuntimeError(f"depth residual {tuple(r.shape)} does not match the image-token block "
                                       f"{(bsize, n, dim)} of image {i}")
                if add is None:
                    add = torch.zeros(embs.shape, dtype=torch.float32, device=embs.device)
                start = i * block + lead
                add[:, start:start + n] = r.float() * dim**0.5
                with torch.no_grad():  # size of the depth signal relative to the RGB tokens it is added to
                    size = add[:, start:start + n].norm(dim=-1).mean(1)
                    kept = r.detach().flatten(1).any(1)  # samples whose depth was not dropped
                    rel = size / embs[:, start:start + n].float().norm(dim=-1).mean(1)
                    ratios.append((rel * kept).sum() / kept.sum().clamp(min=1))
            if ratios:
                m._depth_ratio = torch.stack(ratios).mean()
            if add is not None:
                embs = embs.float() + add
            return embs, pad_masks, att_masks

        m.embed_prefix = embed_prefix

    # ------------------------------------------------------------------ depth -> residual tokens
    def depth_residuals(self, batch: dict) -> list[Tensor | None]:
        """One entry per image SmolVLA will embed (same order as `prepare_images`): a (B, 64, 960) residual
        for cameras with depth, None for the others."""
        cfg, m = self.config, self.model
        out: list[Tensor | None] = []
        for key in (k for k in cfg.image_features if k in batch):
            dkey = cfg.depth_map.get(key)
            if dkey is None:
                out.append(None)
                continue
            if dkey not in batch:
                raise KeyError(f"batch has {key!r} but not its depth {dkey!r}")
            d = batch[dkey]
            if d.ndim == 5:  # (B, T, 1, H, W) -> last step
                d = d[:, -1]
            if d.ndim == 3:  # (B, H, W)
                d = d[:, None]
            if d.ndim != 4 or d.shape[1] != 1:
                raise ValueError(f"{dkey} must be (B, 1, H, W) metric depth, got {tuple(d.shape)}")
            rgb = batch[key]
            if d.shape[-2:] != rgb.shape[-2:]:
                raise ValueError(f"{dkey} {tuple(d.shape[-2:])} is not pixel-aligned with {key} {tuple(rgb.shape[-2:])}")
            d = torch.nan_to_num(d.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if cfg.depth_max_m > 0:
                d = d.clamp(max=cfg.depth_max_m)
            if self.training and cfg.depth_noise_m > 0:
                d = torch.where(d > 0, (d + cfg.depth_noise_m * torch.randn_like(d)).clamp(min=1e-3), d)
            if cfg.resize_imgs_with_padding is not None:
                d = resize_depth_like_rgb(d, *cfg.resize_imgs_with_padding)
            res = m.depth_fusion(m.depth_encoder(d))
            if self.training and cfg.depth_dropout > 0:
                keep = torch.rand(res.shape[0], 1, 1, device=res.device) >= cfg.depth_dropout
                res = res * keep.to(res.dtype)
            out.append(res)
        return out

    @contextlib.contextmanager
    def _depth(self, batch: dict):
        m = self.model
        if not self.config.use_depth:
            yield
            return
        m._depth_res, m._depth_i = self.depth_residuals(batch), 0
        try:
            yield
            if m._depth_i < 1:
                raise RuntimeError("embed_prefix never ran: the depth hook is out of sync with this lerobot version")
        finally:
            m._depth_res = None

    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        with self._depth(batch):
            return super().forward(batch, noise=noise, time=time, reduction=reduction)

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        with self._depth(batch):
            return super()._get_action_chunk(batch, noise=noise, **kwargs)

    # ------------------------------------------------------------------ LoRA on the action expert
    @property
    def expert(self) -> nn.Module:
        return self.model.vlm_with_expert.lm_expert

    def _inject_lora(self):
        from peft import LoraConfig, inject_adapter_in_model

        c = self.config
        inject_adapter_in_model(LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
                                           target_modules=list(c.lora_targets), bias="none"), self.expert)

    # ------------------------------------------------------------------ LoRA on the VLM backbone
    @property
    def text_model(self) -> nn.Module:
        return self.model.vlm_with_expert.get_vlm_model().text_model

    def _vlm_lora_sites(self) -> list[tuple[int, str]]:
        """(layer, projection) pairs that get an adapter. In the very last layer only k / v reach the action
        expert; its q / o feed an output nobody reads and would never receive a gradient."""
        total = len(self.text_model.layers)
        first = total - self.config.vlm_lora_layers
        if not 0 <= first < total:
            raise ValueError(f"vlm_lora_layers must be in 1..{total}, got {self.config.vlm_lora_layers}")
        return [(i, t) for i in range(first, total) for t in self.config.vlm_lora_targets
                if not (i == total - 1 and t in ("q_proj", "o_proj"))]

    def _inject_vlm_lora(self):
        from peft import LoraConfig, inject_adapter_in_model

        c = self.config
        pattern = "|".join(rf"layers\.{i}\.self_attn\.{t}" for i, t in self._vlm_lora_sites())
        inject_adapter_in_model(LoraConfig(r=c.vlm_lora_r, lora_alpha=c.vlm_lora_alpha, lora_dropout=c.vlm_lora_dropout,
                                           target_modules=pattern, bias="none"), self.text_model)

    def add_vlm_lora(self, r: int):
        """Add adapters to the VLM of a policy that has none; layers / targets / alpha / dropout come from the
        config. Needs vlm_float32 (set it when building), otherwise the adapters would live in bf16."""
        if self.config.vlm_lora_r > 0:
            raise RuntimeError("this policy already has VLM LoRA adapters")
        if not self.config.vlm_float32:
            raise RuntimeError("build the policy with vlm_float32=True before adding VLM LoRA")
        self.config.vlm_lora_r = int(r)
        self._inject_vlm_lora()
        self.to(next(self.model.action_out_proj.parameters()).device)

    def add_lora(self, r: int, alpha: int = 32, dropout: float = 0.0, targets: list[str] | None = None):
        """Add LoRA adapters to the action expert of a policy that has none (recorded in the config, so a
        saved checkpoint rebuilds them on load)."""
        if self.config.lora_r > 0:
            raise RuntimeError("this policy already has LoRA adapters")
        c = self.config
        c.lora_r, c.lora_alpha, c.lora_dropout = int(r), int(alpha), float(dropout)
        if targets:
            c.lora_targets = list(targets)
        self._inject_lora()
        self.to(next(self.model.action_out_proj.parameters()).device)

    def merge_lora(self):
        """Fold the expert's and the VLM's adapters into their weights and remove them: a plain frozen network for
        deployment (same outputs, no adapter overhead). The config then says lora_r = vlm_lora_r = 0."""
        from peft.tuners.lora import LoraLayer

        for root in (self.expert, self.text_model):
            for parent in list(root.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, LoraLayer):
                        child.merge()
                        setattr(parent, name, child.get_base_layer())
        self.config.lora_r = self.config.vlm_lora_r = 0

    # ------------------------------------------------------------------ freezing
    def configure_trainable(self, mode: str = "lora", train_heads: bool = True,
                            train_connector: bool = False) -> dict[str, int]:
        """Freeze everything, then enable: the expert's LoRA adapters ("lora") or the whole expert ("expert"),
        the depth fusion MLP + gate, and (train_heads) the state / action / time projections.
        Of the VLM only its LoRA adapters (if it has any) and, with train_connector, the connector train.
        SigLIP, DeFM and the VLM's own text weights always stay frozen."""
        if mode not in TRAIN_MODES:
            raise ValueError(f"mode must be one of {TRAIN_MODES}")
        if mode == "lora" and self.config.lora_r <= 0:
            raise ValueError("mode='lora' needs LoRA adapters (lora_r > 0)")
        self.requires_grad_(False)
        m = self.model
        if mode == "lora":
            for n, p in self.expert.named_parameters():
                p.requires_grad = "lora_" in n
        elif mode == "expert":
            for n, p in self.expert.named_parameters():
                p.requires_grad = "lm_head" not in n
        for n, p in self.text_model.named_parameters():
            p.requires_grad = "lora_" in n
        if train_connector:
            if not self.config.connector_float32:
                raise ValueError("train_connector needs a policy built with connector_float32=True")
            m.vlm_with_expert.get_vlm_model().connector.requires_grad_(True)
        if train_heads:
            for h in HEADS:
                getattr(m, h).requires_grad_(True)
        if self.config.use_depth:
            m.depth_fusion.requires_grad_(True)
        return self.count_parameters()

    def count_parameters(self) -> dict[str, int]:
        groups = {"depth_fusion": "model.depth_fusion.", "defm": "model.depth_encoder.",
                  "expert": "model.vlm_with_expert.lm_expert.",
                  "connector": "model.vlm_with_expert.vlm.model.connector.", "vlm": "model.vlm_with_expert.vlm."}
        out = {f"{g}_{k}": 0 for g in (*groups, "heads") for k in ("trainable", "frozen")}
        for n, p in self.named_parameters():
            g = next((g for g, pre in groups.items() if n.startswith(pre)), "heads")
            out[f"{g}_{'trainable' if p.requires_grad else 'frozen'}"] += p.numel()
        out["total_trainable"] = sum(v for k, v in out.items() if k.endswith("_trainable"))
        return out

    def param_groups(self, lr: float, depth_lr_mult: float = 4.0, vlm_lr_mult: float = 0.5) -> list[dict]:
        """Optimizer groups: the new depth fusion weights get lr * depth_lr_mult (they start from scratch and
        behind a zero gate), the pretrained VLM parts (connector, LoRA) lr * vlm_lr_mult, everything else lr."""
        fusion, vlm = "model.depth_fusion.", "model.vlm_with_expert.vlm."
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        depth = [p for n, p in named if n.startswith(fusion)]
        backbone = [p for n, p in named if n.startswith(vlm)]
        rest = [p for n, p in named if not n.startswith((fusion, vlm))]
        groups = [{"params": rest, "lr": lr, "lr_mult": 1.0, "name": "policy"}]
        if depth:
            groups.append({"params": depth, "lr": lr * depth_lr_mult, "lr_mult": depth_lr_mult, "name": "depth_fusion"})
        if backbone:
            groups.append({"params": backbone, "lr": lr * vlm_lr_mult, "lr_mult": vlm_lr_mult, "name": "vlm"})
        return [g for g in groups if g["params"]]

    def gate_abs_mean(self) -> float:
        return float(self.model.depth_fusion.gate.detach().abs().mean()) if self.config.use_depth else 0.0

    def depth_ratio(self) -> float:
        """|depth residual| / |RGB image token| in the last forward pass (mean over depth cameras). The gate value
        alone says little: it multiplies an MLP output of arbitrary size."""
        r = self.model._depth_ratio
        return float(r) if r is not None else 0.0


# ---------------------------------------------------------------------- build / load
def _checkpoint_keys(init_from: str) -> set[str]:
    import os

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    init_from = str(init_from)
    f = (os.path.join(init_from, "model.safetensors") if os.path.isdir(init_from)
         else hf_hub_download(repo_id=init_from, filename="model.safetensors"))
    with safe_open(f, framework="pt") as sf:
        return set(sf.keys())


def build_policy(init_from: str, input_features: dict, output_features: dict, *, device: str = "cuda",
                 lora_r: int = 0, lora_alpha: int = 32, lora_dropout: float = 0.0,
                 lora_targets: list[str] | None = None, vlm_lora_r: int = 0, train_connector: bool = False,
                 **overrides) -> DepthSmolVLAPolicy:
    """New training run. `init_from` is `lerobot/smolvla_base` (or any stock SmolVLA checkpoint) or one of this
    project's checkpoints (e.g. the RGB-only run, to warm-start the RGB-D run from it).

    Weights present in `init_from` are loaded; the only weights allowed to be new are the depth branch (DeFM
    comes from its own pretrained weights, the fusion MLP is fresh with a zero gate) and LoRA adapters."""
    src = PreTrainedConfig.from_pretrained(init_from)
    base_fields = {f.name for f in dataclasses.fields(DepthSmolVLAConfig) if f.init}
    kw = {f.name: getattr(src, f.name) for f in dataclasses.fields(src) if f.init and f.name in base_fields}
    src_lora = int(getattr(src, "lora_r", 0))
    if src_lora and lora_r and lora_r != src_lora:
        raise ValueError(f"{init_from} already has LoRA r={src_lora}; cannot re-initialise with r={lora_r} "
                         "(merge it first with scripts/merge_for_deploy.py)")
    src_vlm_lora = int(getattr(src, "vlm_lora_r", 0))
    if src_vlm_lora and vlm_lora_r and vlm_lora_r != src_vlm_lora:
        raise ValueError(f"{init_from} already has VLM LoRA r={src_vlm_lora}; cannot re-initialise with r={vlm_lora_r} "
                         "(merge it first with scripts/merge_for_deploy.py)")
    kw.update(overrides)
    kw["vlm_float32"] = bool(kw.get("vlm_float32")) or vlm_lora_r > 0
    kw["connector_float32"] = bool(kw.get("connector_float32")) or train_connector
    kw.update(input_features=input_features, output_features=output_features, device=device, defm_pretrained=True)
    if not src_lora:
        kw["lora_r"] = 0  # build without adapters so the plain expert weights load, add them afterwards
    cfg = DepthSmolVLAConfig(**kw)
    policy = DepthSmolVLAPolicy.from_pretrained(init_from, config=cfg, strict=False)

    have, want = _checkpoint_keys(init_from), set(policy.state_dict())
    missing = sorted(k for k in want - have if not k.startswith(("model.depth_encoder.", "model.depth_fusion.")))
    if missing:
        raise RuntimeError(f"{len(missing)} weights of the policy are not in {init_from} and would stay randomly "
                           f"initialised, e.g. {missing[:5]}")
    unexpected = sorted(have - want)
    if unexpected:
        log.warning("%d weights in %s were not used, e.g. %s", len(unexpected), init_from, unexpected[:5])
    if not src_lora and lora_r > 0:
        policy.add_lora(lora_r, lora_alpha, lora_dropout, lora_targets)
    if not src_vlm_lora and vlm_lora_r > 0:
        policy.add_vlm_lora(vlm_lora_r)
    return policy


def load_policy(checkpoint: str, device: str = "cuda") -> DepthSmolVLAPolicy:
    """Load one of this project's checkpoints for evaluation / deployment: eval mode, every parameter frozen.
    All weights (including DeFM) come from the checkpoint; nothing is fetched for DeFM."""
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(cfg, DepthSmolVLAConfig):
        raise TypeError(f"{checkpoint} is a {type(cfg).__name__} checkpoint, not smolvla_depth")
    cfg.device = device
    cfg.defm_pretrained, cfg.defm_weights_path = False, None
    policy = DepthSmolVLAPolicy.from_pretrained(checkpoint, config=cfg, strict=True)
    policy.requires_grad_(False)
    return policy.eval()
