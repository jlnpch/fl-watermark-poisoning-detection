"""Attack registry for FL poisoning experiments.

Each attack implements one or both hooks:
  - poison_data: modify trainloader before training (data poisoning)
  - poison_model: modify model after training (model poisoning)
"""

import torch


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


class _LabelShiftedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and shifts labels."""

    def __init__(self, base_dataset, offset, num_classes):
        self.base = base_dataset
        self.offset = offset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, dict):
            item = {k: v for k, v in item.items()}
            if "label" in item:
                item["label"] = (item["label"] + self.offset) % self.num_classes
            return item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            y = (y + self.offset) % self.num_classes
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
    """Label flip + optional gradient scaling."""

    type = "label_flip"

    def poison_data(self, trainloader, partition_id, is_attacker):
        if not is_attacker:
            return trainloader
        offset = self.config.get("label-flip-offset", 1)
        num_classes = self.config.get("num-classes", 10)
        original_dataset = trainloader.dataset
        shifted_dataset = _LabelShiftedDataset(original_dataset, offset, num_classes)
        trainloader.dataset = shifted_dataset
        return trainloader

    def poison_model(self, model, initial_state, partition_id, is_attacker):
        if not is_attacker:
            return model
        scale = self.config.get("label-flip-scale", 1.0)
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
