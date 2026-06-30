import hashlib

import torch


class UchidaWatermark:
    def __init__(self, message="uchida", num_bits=64, layer_name="conv2.weight", seed=42):
        self.layer_name = layer_name
        self.num_bits = num_bits
        self.seed = seed

        # Generate watermark bits from message hash
        hash_bytes = hashlib.sha256(message.encode()).digest()
        bits = []
        for byte in hash_bytes[: (num_bits + 7) // 8]:
            for i in range(8):
                bits.append((byte >> i) & 1)
        self.b = torch.tensor(bits[:num_bits], dtype=torch.float32)
        self.b_target = 2.0 * self.b - 1.0  # map {0,1} → {-1,+1}

        self.P = None
        self.num_params = None

    def _init_projection(self, model):
        layer = dict(model.named_parameters())[self.layer_name]
        self.num_params = layer.data.numel()

        rng = torch.Generator()
        rng.manual_seed(self.seed)
        self.P = torch.randn((self.num_bits, self.num_params), generator=rng)

    def embed(self, model):
        if self.P is None:
            self._init_projection(model)

        layer = dict(model.named_parameters())[self.layer_name]
        W = layer.data.flatten()

        z = self.P @ W
        P_Pt = self.P @ self.P.T
        x = torch.linalg.solve(P_Pt, self.b_target - z)
        delta_W = self.P.T @ x

        layer.data += delta_W.reshape(layer.data.shape)

    def compute_ber(self, model):
        if self.P is None:
            self._init_projection(model)

        layer = dict(model.named_parameters())[self.layer_name]
        W = layer.data.flatten()

        z = self.P @ W
        decoded = (z > 0).float()

        mismatches = (decoded != self.b).float()
        return mismatches.mean().item()
