"""pytorchexample: A Flower / PyTorch app."""

from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from pytorchexample.task import Net, load_centralized_dataset, load_server_pretrain_data
from pytorchexample.task import pretrain_with_watermark as server_pretrain
from pytorchexample.task import test
from pytorchexample.watermark import create_watermark
from pytorchexample.watermarked_strategy import WatermarkedFedAvg

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    cfg = context.run_config

    fraction_evaluate = cfg["fraction-evaluate"]
    num_rounds = cfg["num-server-rounds"]
    lr = cfg["learning-rate"]
    wm_lambda = cfg["watermark-lambda"]
    pt_fraction = cfg["pretrain-fraction"]
    pt_epochs = cfg["pretrain-epochs"]
    es_patience = cfg["early-stopping-patience"]
    es_delta = cfg["early-stopping-delta"]
    max_ber = cfg["max-trusted-ber"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    watermark = create_watermark(cfg)
    global_model = Net()
    global_model.to(device)

    log(INFO, "Loading %.0f%% of CIFAR-10 training set for server pretraining...", pt_fraction * 100)
    pretrain_loader = load_server_pretrain_data(fraction=pt_fraction, batch_size=64)

    wd = cfg["weight-decay"]
    log(INFO, "Pretraining with watermark regularization (λ=%.4f, %d epochs)...", wm_lambda, pt_epochs)
    server_pretrain(global_model, pretrain_loader, watermark, pt_epochs, lr, device, lambda_reg=wm_lambda, weight_decay=wd)

    log(INFO, "Pretrain done — BER = %.4f, Loss = %.4f", watermark.compute_ber(global_model), 0.0)

    attacker_frac = cfg.get("attacker-fraction", 0.0)
    log(INFO, "Attack config: type=%s, fraction=%.1f, max_trusted_ber=%.2f",
        cfg.get("attacker-type", "none"), attacker_frac, max_ber)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize WatermarkedFedAvg strategy
    strategy = WatermarkedFedAvg(
        fraction_evaluate=fraction_evaluate,
        early_stopping_patience=es_patience,
        early_stopping_delta=es_delta,
        max_trusted_ber=max_ber,
        attacker_fraction=attacker_frac,
    )

    # Define global evaluation function (captures context via closure)
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        test_dataloader = load_centralized_dataset()

        # Standard accuracy (true labels)
        test_loss, test_acc = test(model, test_dataloader, device)

        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
