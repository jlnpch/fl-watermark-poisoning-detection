import hashlib

import torch


class UchidaWatermark:
    """Uchida-style watermark: embed bits via random projection into one or more layers.

    When ``layer_names`` is provided, the 128-bit (or ``num_bits``) message is embedded
    independently into every listed layer, each with its own deterministic projection
    matrix ``P`` (seeded by ``seed + layer_index``). Detection BER is the **maximum**
    across layers (most conservative); ``compute_ber_mean`` reports the layer mean for
    the FP-trade-off ablation, and ``compute_ber_by_layer`` exposes per-layer BERs.
    """

    def __init__(self, message="uchida", num_bits=64, layer_name="backbone.fc.weight", seed=42, layer_names=None):
        self.layer_names = sorted(layer_names) if layer_names else [layer_name]
        self.num_bits = num_bits
        self.seed = seed

        hash_bytes = hashlib.sha256(message.encode()).digest()
        bits = []
        for byte in hash_bytes[: (num_bits + 7) // 8]:
            for i in range(8):
                bits.append((byte >> i) & 1)
        self.b = torch.tensor(bits[:num_bits], dtype=torch.float32)
        self.b_target = 2.0 * self.b - 1.0

        self._P = {}  # layer_name -> projection matrix
        self._num_params = {}  # layer_name -> num params

    def _init_projection(self, model, layer_name):
        layer = dict(model.named_parameters())[layer_name]
        num_params = layer.data.numel()

        rng = torch.Generator()
        rng.manual_seed(self.seed + self.layer_names.index(layer_name))
        P = torch.randn((self.num_bits, num_params), generator=rng).to(layer.device)
        self._P[layer_name] = P
        self._num_params[layer_name] = num_params

        if self.b.device != layer.device:
            self.b = self.b.to(layer.device)
            self.b_target = self.b_target.to(layer.device)
        return P

    def _projection(self, model, layer_name):
        P = self._P.get(layer_name)
        return P if P is not None else self._init_projection(model, layer_name)

    def embed(self, model, strength=1.0):
        named = dict(model.named_parameters())
        for name in self.layer_names:
            P = self._projection(model, name)
            W = named[name].data.flatten()
            z = P @ W
            P_Pt = P @ P.T
            x = torch.linalg.solve(P_Pt, self.b_target - z)
            delta_W = P.T @ x
            named[name].data += strength * delta_W.reshape(named[name].data.shape)

    def regularization_loss(self, model):
        named = dict(model.named_parameters())
        losses = []
        for name in self.layer_names:
            P = self._projection(model, name)
            z = P @ named[name].flatten()
            losses.append(torch.mean((z - self.b_target) ** 2))
        return torch.stack(losses).mean()

    def _ber_layer(self, model, layer_name):
        P = self._projection(model, layer_name)
        W = dict(model.named_parameters())[layer_name].data.flatten()
        z = P @ W
        decoded = (z > 0).float()
        mismatches = (decoded != self.b).float()
        return mismatches.mean().item()

    def compute_ber_by_layer(self, model):
        """Per-layer BER as a dict {layer_name: ber}."""
        return {name: self._ber_layer(model, name) for name in self.layer_names}

    def compute_ber(self, model):
        """Aggregate BER = max across layers (most conservative)."""
        return max(self.compute_ber_by_layer(model).values())

    def compute_ber_mean(self, model):
        """Aggregate BER = mean across layers (ablation for the FP trade-off)."""
        bers = self.compute_ber_by_layer(model)
        return sum(bers.values()) / len(bers)


_WATERMARK_REGISTRY = {
    "uchida": UchidaWatermark,
}


def create_watermark(config):
    """Factory: create watermark instance from run_config dict."""
    wm_type = config.get("watermark-type", "uchida")
    cls = _WATERMARK_REGISTRY.get(wm_type)
    if cls is None:
        raise ValueError(f"Unknown watermark-type '{wm_type}'. Available: {list(_WATERMARK_REGISTRY)}")

    layer_names = None
    wl = config.get("watermark-layers", "")
    if wl:
        layer_names = [l.strip() for l in wl.split(",") if l.strip()]

    return cls(
        message=config.get("watermark-message", "uchida"),
        num_bits=config.get("watermark-num-bits", 64),
        layer_name=config.get("watermark-layer", "backbone.fc.weight"),
        seed=config.get("seed", 42),
        layer_names=layer_names,
    )
