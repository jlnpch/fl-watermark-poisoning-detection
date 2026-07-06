import hashlib

import torch


class UchidaWatermark:
    """Uchida-style watermark: embed bits via random projection into a chosen layer."""

    def __init__(self, message="uchida", num_bits=64, layer_name="backbone.fc.weight", seed=42):
        self.layer_name = layer_name
        self.num_bits = num_bits
        self.seed = seed

        hash_bytes = hashlib.sha256(message.encode()).digest()
        bits = []
        for byte in hash_bytes[: (num_bits + 7) // 8]:
            for i in range(8):
                bits.append((byte >> i) & 1)
        self.b = torch.tensor(bits[:num_bits], dtype=torch.float32)
        self.b_target = 2.0 * self.b - 1.0

        self.P = None
        self.num_params = None

    def _init_projection(self, model):
        layer = dict(model.named_parameters())[self.layer_name]
        self.num_params = layer.data.numel()

        rng = torch.Generator()
        rng.manual_seed(self.seed)
        self.P = torch.randn((self.num_bits, self.num_params), generator=rng)
        self.P = self.P.to(layer.device)
        self.b = self.b.to(layer.device)
        self.b_target = self.b_target.to(layer.device)

    def embed(self, model, strength=1.0):
        if self.P is None:
            self._init_projection(model)

        layer = dict(model.named_parameters())[self.layer_name]
        W = layer.data.flatten()

        z = self.P @ W
        P_Pt = self.P @ self.P.T
        x = torch.linalg.solve(P_Pt, self.b_target - z)
        delta_W = self.P.T @ x

        layer.data += strength * delta_W.reshape(layer.data.shape)

    def regularization_loss(self, model):
        if self.P is None:
            self._init_projection(model)

        layer = dict(model.named_parameters())[self.layer_name]
        W = layer.flatten()
        z = self.P @ W
        return torch.mean((z - self.b_target) ** 2)

    def compute_ber(self, model):
        if self.P is None:
            self._init_projection(model)

        layer = dict(model.named_parameters())[self.layer_name]
        W = layer.data.flatten()

        z = self.P @ W
        decoded = (z > 0).float()

        mismatches = (decoded != self.b).float()
        return mismatches.mean().item()


_WATERMARK_REGISTRY = {
    "uchida": UchidaWatermark,
}


def create_watermark(config):
    """Factory: create watermark instance from run_config dict."""
    wm_type = config.get("watermark-type", "uchida")
    cls = _WATERMARK_REGISTRY.get(wm_type)
    if cls is None:
        raise ValueError(f"Unknown watermark-type '{wm_type}'. Available: {list(_WATERMARK_REGISTRY)}")
    return cls(
        message=config.get("watermark-message", "uchida"),
        num_bits=config.get("watermark-num-bits", 64),
        layer_name=config.get("watermark-layer", "backbone.fc.weight"),
    )
