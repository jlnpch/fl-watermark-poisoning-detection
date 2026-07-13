"""pytorchexample: A Flower / PyTorch app."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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


fds = None  # Cached client dataset (remaining after server's private slice)
_partition_indices = None  # Cached Dirichlet partition assignments

CIFAR10_TRAIN_SIZE = 50000

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to a partition of CIFAR-10."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def _build_dirichlet_partitions(dataset, num_partitions, alpha, num_classes=10, seed=42):
    """Assign each sample to a partition via Dirichlet(alpha) per class."""
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    partitions = [[] for _ in range(num_partitions)]
    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_partitions)
        proportions = np.cumsum(proportions) * len(idxs)
        split_idxs = np.round(proportions).astype(int)[:-1]
        for i, (start, end) in enumerate(zip([0] + split_idxs.tolist(), split_idxs.tolist() + [len(idxs)])):
            partitions[i].extend(idxs[start:end].tolist())
    return partitions


def load_data(partition_id: int, num_partitions: int, batch_size: int,
              server_private_samples: int = 5000, client_samples: int = 4000,
              partition_type: str = "iid", partition_alpha: float = 0.5):
    """Load partition CIFAR10 data. Server takes first server_private_samples exclusively."""
    from datasets import load_dataset
    import numpy as np
    global fds, _partition_indices
    if fds is None:
        full = load_dataset("uoft-cs/cifar10", split="train")
        full = full.shuffle(seed=42)
        # Skip server's private portion, take the rest for clients
        dataset = full.select(range(server_private_samples, len(full)))
        # Cap total client samples to client_samples * num_partitions
        max_client_total = client_samples * num_partitions
        if len(dataset) > max_client_total:
            dataset = dataset.select(range(max_client_total))
        fds = dataset
        if partition_type == "dirichlet":
            _partition_indices = _build_dirichlet_partitions(
                fds, num_partitions, partition_alpha, seed=42
            )

    if partition_type == "dirichlet":
        idxs = _partition_indices[partition_id]
        partition_subset = Subset(fds, idxs)
        # manual train/test split on indices
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(idxs))
        split = int(0.8 * len(idxs))
        train_idxs = [idxs[i] for i in perm[:split]]
        test_idxs = [idxs[i] for i in perm[split:]]
        train_dataset = Subset(fds, train_idxs)
        test_dataset = Subset(fds, test_idxs)
        # Apply transforms manually via a wrapper
        class _TransformedSubset(torch.utils.data.Dataset):
            def __init__(self, subset):
                self.subset = subset
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                item = self.subset[idx]
                img = pytorch_transforms(item["img"])
                return {"img": img, "label": item["label"]}
        trainloader = DataLoader(_TransformedSubset(train_dataset), batch_size=batch_size, shuffle=True)
        testloader = DataLoader(_TransformedSubset(test_dataset), batch_size=batch_size)
        return trainloader, testloader
    else:
        partition = fds.shard(num_partitions, partition_id, contiguous=True)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader


def load_server_pretrain_data(num_samples=5000, batch_size=64):
    """Load the first num_samples of CIFAR-10 training set for server-side pretraining."""
    from datasets import load_dataset
    dataset = load_dataset("uoft-cs/cifar10", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(num_samples))
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
