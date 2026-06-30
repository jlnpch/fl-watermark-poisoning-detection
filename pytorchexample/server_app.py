"""pytorchexample: A Flower / PyTorch app."""

from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from pytorchexample.task import Net, load_centralized_dataset, load_server_pretrain_data
from pytorchexample.task import pretrain_with_watermark as server_pretrain
from pytorchexample.task import test
from pytorchexample.watermark import UchidaWatermark
from pytorchexample.watermarked_strategy import WatermarkedFedAvg

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    wm_message: str = context.run_config["watermark-message"]
    wm_bits: int = context.run_config["watermark-num-bits"]
    wm_lambda: float = context.run_config["watermark-lambda"]
    pt_fraction: float = context.run_config["pretrain-fraction"]
    pt_epochs: int = context.run_config["pretrain-epochs"]

    # Server-side pretraining with watermark regularization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    watermark = UchidaWatermark(message=wm_message, num_bits=wm_bits)
    global_model = Net()
    global_model.to(device)

    log(INFO, "Loading %.0f%% of CIFAR-10 training set for server pretraining...", pt_fraction * 100)
    pretrain_loader = load_server_pretrain_data(fraction=pt_fraction, batch_size=64)

    log(INFO, "Pretraining with watermark regularization (λ=%.4f, %d epochs)...", wm_lambda, pt_epochs)
    server_pretrain(global_model, pretrain_loader, watermark, pt_epochs, lr, device, lambda_reg=wm_lambda)

    log(INFO, "Pretrain done — BER = %.4f, Loss = %.4f", watermark.compute_ber(global_model), 0.0)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize WatermarkedFedAvg strategy
    strategy = WatermarkedFedAvg(fraction_evaluate=fraction_evaluate)

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


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
