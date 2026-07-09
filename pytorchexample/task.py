"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

class Net(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 inputs)."""

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.backbone = resnet18(num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


fds = None  # Cached client dataset (remaining after server's pretrain slice)

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to a partition of CIFAR-10."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, pretrain_fraction: float = 0.1):
    """Load partition CIFAR10 data. Server takes first pretrain_fraction exclusively."""
    from datasets import load_dataset
    global fds
    if fds is None:
        pct = int(pretrain_fraction * 100)
        dataset = load_dataset("uoft-cs/cifar10", split=f"train[{pct}%:]")
        dataset = dataset.shuffle(seed=42)
        fds = dataset
    partition = fds.shard(num_partitions, partition_id, contiguous=True)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_server_pretrain_data(fraction=0.1, batch_size=64):
    """Load a fraction of the CIFAR-10 training set for server-side pretraining."""
    from datasets import load_dataset
    pct = int(fraction * 100)
    dataset = load_dataset("uoft-cs/cifar10", split=f"train[:{pct}%]")
    dataset = dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_centralized_dataset(batch_size=128):
    """Load test set and return dataloader."""
    from datasets import load_dataset
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=batch_size)


def train(net, trainloader, epochs, lr, device, weight_decay=0.0):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def compute_asr(net, testloader, source_class, target_class, device):
    """Fraction of source-class images predicted as target-class."""
    net.to(device)
    net.eval()
    total = 0
    success = 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            preds = outputs.argmax(dim=1)
            source_mask = labels == source_class
            total += source_mask.sum().item()
            success += (preds[source_mask] == target_class).sum().item()
    return success / total if total > 0 else 0.0


def pretrain_with_watermark(net, trainloader, watermark, epochs, lr, device, lambda_reg=0.01, weight_decay=0.0):
    """Pretrain the model with task loss + watermark regularization."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            task_loss = criterion(outputs, labels)

            wm_loss = watermark.regularization_loss(net)
            loss = task_loss + lambda_reg * wm_loss

            loss.backward()
            optimizer.step()
