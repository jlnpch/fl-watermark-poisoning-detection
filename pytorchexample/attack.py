"""Attack registry for FL poisoning experiments.

Each attack implements one or both hooks:
  - poison_data: modify trainloader before training (data poisoning)
  - poison_model: modify model after training (model poisoning)
"""

import torch
from torch.utils.data import DataLoader


class Attack:
    type = "none"

    def __init__(self, config=None):
        self.config = config or {}

    def poison_data(self, trainloader, partition_id, is_attacker):
        return trainloader

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        return model


class NoAttack(Attack):
    type = "none"


class NoiseAttack(Attack):
    """Replace attacker's model update with N(0, noise_scale) on all float params."""

    type = "noise"

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        if not is_attacker:
            return model
        noise_scale = self.config.get("attacker-noise-scale", 1.0)
        state_dict = model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                if state_dict[key].dtype.is_floating_point:
                    state_dict[key] = initial_state[key] + torch.randn_like(state_dict[key]) * noise_scale
        model.load_state_dict(state_dict)
        return model


class _BackdoorDataset(torch.utils.data.Dataset):
    """Replaces source-class labels with target-class label (backdoor poisoning)."""

    def __init__(self, base_dataset, source_class, target_class):
        self.base = base_dataset
        self.source = source_class
        self.target = target_class

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, dict):
            item = {k: v for k, v in item.items()}
            if "label" in item and item["label"] == self.source:
                item["label"] = self.target
            return item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            if y == self.source:
                y = self.target
            return (x, y) + item[2:]
        return item


class SignFlipAttack(Attack):
    """Flip the sign of the gradient update (optionally scaled).

    Sends:  initial_state - scale * (trained_state - initial_state)
    """

    type = "sign_flip"

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        if not is_attacker:
            return model
        scale = self.config.get("sign-flip-scale", 1.0)
        state_dict = model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                residual = state_dict[key] - initial_state[key]
                state_dict[key] = initial_state[key] - residual * scale
        model.load_state_dict(state_dict)
        return model


class FreezeFCAttack(Attack):
    """Kerckhoffs-aware sign-flip: freeze the watermarked FC layer(s) and
    poison only the rest of the network.

    The attacker keeps every layer named in ``frozen-layers`` (default: the
    single ``watermark-layer``) at its initial (watermarked) state — residual
    zero — and sign-flips the residual of every other layer. This keeps the
    attacker's watermark BER low (looks honest) while poisoning the feature
    extractor.
    """

    type = "freeze_fc"

    def _frozen_layers(self):
        raw = self.config.get("frozen-layers", "")
        layers = [l.strip() for l in raw.split(",") if l.strip()]
        if not layers:
            layers = [self.config.get("watermark-layer", "backbone.fc.weight")]
        return set(layers)

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        if not is_attacker:
            return model
        scale = self.config.get("sign-flip-scale", 1.0)
        frozen = self._frozen_layers()
        state_dict = model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                if key in frozen:
                    state_dict[key] = initial_state[key]
                else:
                    residual = state_dict[key] - initial_state[key]
                    state_dict[key] = initial_state[key] - residual * scale
        model.load_state_dict(state_dict)
        return model


class LabelFlipAttack(Attack):
    """Backdoor via source→target label flip at the Dataset level.

    The attacker trains normally (standard cross-entropy loss) but every
    sample with ground-truth label == source_class has its label overwritten
    to target_class. This forces the model to map source-class features to
    the target-class decision boundary.
    """

    type = "label_flip"

    def poison_data(self, trainloader, partition_id, is_attacker):
        if not is_attacker:
            return trainloader
        source = self.config.get("label-flip-source", 9)
        target = self.config.get("label-flip-target", 2)
        poisoned_dataset = _BackdoorDataset(trainloader.dataset, source, target)
        return DataLoader(
            poisoned_dataset,
            batch_size=trainloader.batch_size,
            shuffle=trainloader.sampler is None and not isinstance(trainloader.sampler, torch.utils.data.SequentialSampler),
            num_workers=trainloader.num_workers,
            pin_memory=trainloader.pin_memory,
            drop_last=trainloader.drop_last,
        )

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        if not is_attacker:
            return model
        scale = self.config.get("label-flip-scale", 1.0)
        if scale == 1.0:
            return model
        state_dict = model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                residual = state_dict[key] - initial_state[key]
                state_dict[key] = initial_state[key] + residual * scale
        model.load_state_dict(state_dict)
        return model


def apply_trigger(img, size=4, value=1.0, pos="bottom-right"):
    """Overlay a fixed pixel-pattern trigger on a ``[..., H, W]`` tensor (returns a copy).

    Works on a single image ``[C, H, W]`` or a batch ``[N, C, H, W]``. ``value`` is in the
    normalized ``[-1, 1]`` space (1.0 == white). Used identically for poisoning (train)
    and for trigger-ASR evaluation (test) so the trigger matches.
    """
    img = img.clone()
    h, w = img.shape[-2], img.shape[-1]
    s = size
    if pos == "bottom-right":
        img[..., h - s:, w - s:] = value
    elif pos == "top-left":
        img[..., :s, :s] = value
    elif pos == "center":
        img[..., h // 2 - s // 2: h // 2 + s // 2, w // 2 - s // 2: w // 2 + s // 2] = value
    else:
        raise ValueError(f"Unknown trigger position '{pos}'")
    return img


class _TriggerDataset(torch.utils.data.Dataset):
    """BadNets data poisoning: overlay a fixed trigger patch on a fraction of images
    and relabel them to the target class (deterministic per ``seed``)."""

    def __init__(self, base, target_class, size=4, value=1.0, pos="bottom-right", frac=0.5, seed=42):
        self.base = base
        self.target = target_class
        self.size = size
        self.value = value
        self.pos = pos
        n = len(base)
        k = max(1, int(frac * n))
        gen = torch.Generator().manual_seed(seed)
        self.poisoned = set(torch.randperm(n, generator=gen)[:k].tolist())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, dict):
            item = {k: v for k, v in item.items()}
            if idx in self.poisoned:
                item["img"] = apply_trigger(item["img"], self.size, self.value, self.pos)
                item["label"] = self.target
            return item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            if idx in self.poisoned:
                x = apply_trigger(x, self.size, self.value, self.pos)
                y = self.target
            return (x, y) + item[2:]
        return item


class BadNetsAttack(Attack):
    """Trigger-based backdoor (BadNets).

    ``poison_data`` overlays a fixed trigger patch on ``badnets-poison-frac`` of the
    attacker's training images and relabels them to ``badnets-target``. ``poison_model``
    amplifies the update residual by ``badnets-scale`` (model replacement: ``1.0`` =
    stealthy, ``~10`` = aggregate ≈ attacker model).

    Constrain-and-scale (Bagdasaryan et al.): when ``badnets-constrain > 0``, the scaled
    update's L2 norm is clipped to ``badnets-constrain * ||raw_update||`` — i.e. the sent
    update is projected onto the benign-magnitude ball after scaling, so it keeps the
    backdoor *direction* but looks benign in magnitude. Effective scale = ``min(scale,
    constrain)``.
    """

    type = "badnets"

    @staticmethod
    def _l2_norm(state, initial):
        s = 0.0
        for k, v in state.items():
            if v.dtype.is_floating_point:
                s += float((v - initial[k]).pow(2).sum())
        return s ** 0.5

    def poison_data(self, trainloader, partition_id, is_attacker):
        if not is_attacker:
            return trainloader
        target = self.config.get("badnets-target", 0)
        size = self.config.get("badnets-trigger-size", 4)
        value = self.config.get("badnets-trigger-value", 1.0)
        pos = self.config.get("badnets-trigger-pos", "bottom-right")
        frac = self.config.get("badnets-poison-frac", 0.5)
        seed = self.config.get("seed", 42)
        poisoned = _TriggerDataset(trainloader.dataset, target, size, value, pos, frac, seed + partition_id)
        return DataLoader(
            poisoned,
            batch_size=trainloader.batch_size,
            shuffle=trainloader.sampler is None and not isinstance(trainloader.sampler, torch.utils.data.SequentialSampler),
            num_workers=trainloader.num_workers,
            pin_memory=trainloader.pin_memory,
            drop_last=trainloader.drop_last,
        )

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        if not is_attacker:
            return model
        scale = self.config.get("badnets-scale", 1.0)
        constrain = self.config.get("badnets-constrain", 0.0)
        if scale == 1.0 and constrain <= 0:
            return model
        state_dict = model.state_dict()
        with torch.no_grad():
            if constrain > 0:
                raw_norm = self._l2_norm(state_dict, initial_state)
                for key in state_dict:
                    residual = state_dict[key] - initial_state[key]
                    state_dict[key] = initial_state[key] + residual * scale
                scaled_norm = self._l2_norm(state_dict, initial_state)
                budget = constrain * raw_norm
                if scaled_norm > budget and scaled_norm > 0:
                    factor = budget / scaled_norm
                    for key in state_dict:
                        if state_dict[key].dtype.is_floating_point:
                            residual = state_dict[key] - initial_state[key]
                            state_dict[key] = initial_state[key] + residual * factor
            else:
                for key in state_dict:
                    residual = state_dict[key] - initial_state[key]
                    state_dict[key] = initial_state[key] + residual * scale
        model.load_state_dict(state_dict)
        return model


_ATTACK_REGISTRY = {
    "none": NoAttack,
    "noise": NoiseAttack,
    "sign_flip": SignFlipAttack,
    "label_flip": LabelFlipAttack,
    "freeze_fc": FreezeFCAttack,
    "badnets": BadNetsAttack,
}


def create_attack(config):
    """Factory: create Attack instance from run_config dict."""
    attack_type = config.get("attacker-type", "none")
    cls = _ATTACK_REGISTRY.get(attack_type)
    if cls is None:
        raise ValueError(f"Unknown attacker-type '{attack_type}'. Available: {list(_ATTACK_REGISTRY)}")
    return cls(config)
