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


_ATTACK_REGISTRY = {
    "none": NoAttack,
    "noise": NoiseAttack,
    "sign_flip": SignFlipAttack,
    "label_flip": LabelFlipAttack,
}


def create_attack(config):
    """Factory: create Attack instance from run_config dict."""
    attack_type = config.get("attacker-type", "none")
    cls = _ATTACK_REGISTRY.get(attack_type)
    if cls is None:
        raise ValueError(f"Unknown attacker-type '{attack_type}'. Available: {list(_ATTACK_REGISTRY)}")
    return cls(config)
