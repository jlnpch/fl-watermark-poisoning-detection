from logging import INFO

from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg


class WatermarkedFedAvg(FedAvg):
    def aggregate_train(self, server_round, replies):
        for i, msg in enumerate(replies):
            if not msg.has_error():
                metrics = msg.content["metrics"]
                ber = metrics.get("watermark_ber", None)
                if ber is not None:
                    log(INFO, "  └─ Client %d: watermark_ber = %.4f", i, ber)

        return super().aggregate_train(server_round, replies)
