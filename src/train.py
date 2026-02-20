import os
import time
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


print("TRAIN STARTED", flush=True)
print("RANK:", os.environ.get("RANK"), flush=True)
print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"), flush=True)
print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"), flush=True)


@dataclass
class Config:
    epochs: int = int(os.getenv("EPOCHS", "2"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    lr: float = float(os.getenv("LR", "0.01"))
    data_dir: str = os.getenv("DATA_DIR", "/tmp/mnist")
    out_dir: str = os.getenv("OUT_DIR", "/outputs")
    backend: str = os.getenv("DDP_BACKEND", "gloo")


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp(cfg):
    if not is_distributed():
        return None, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(
        backend=cfg.backend,
        rank=rank,
        world_size=world_size
    )

    return dist.group.WORLD, rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate(model, dataloader, device, rank):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted"
    )

    if rank == 0:
        print(f"[EVAL] accuracy={accuracy:.4f}")
        print(f"[EVAL] precision={precision:.4f}")
        print(f"[EVAL] recall={recall:.4f}")
        print(f"[EVAL] f1_score={f1:.4f}")

        print("\n[EVAL] Classification Report:")
        print(classification_report(all_labels, all_preds))

    return accuracy, precision, recall, f1


def main():

    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    group, rank, world_size, local_rank = setup_ddp(cfg)

    device = torch.device("cpu")

    host = os.getenv("HOSTNAME", "unknown")

    if rank == 0:
        print(f"[INIT] world_size={world_size} backend={cfg.backend}")

    print(f"[RANK {rank}] host={host}")


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    train_ds = datasets.MNIST(
        cfg.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_ds = datasets.MNIST(
        cfg.data_dir,
        train=False,
        download=True,
        transform=transform
    )


    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None


    test_sampler = DistributedSampler(
        test_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None


    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        shuffle=False
    )


    model = SmallCNN().to(device)

    if world_size > 1:
        model = DDP(model)


    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9
    )

    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(cfg.epochs):

        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()

        t0 = time.time()
        total_loss = 0
        total_seen = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            logits = model(x)

            loss = loss_fn(logits, y)

            loss.backward()

            opt.step()

            bs = x.size(0)

            total_loss += loss.item() * bs
            total_seen += bs


        if world_size > 1:

            tensor = torch.tensor([total_loss, total_seen])

            dist.all_reduce(tensor)

            total_loss = tensor[0].item()
            total_seen = int(tensor[1].item())


        avg_loss = total_loss / total_seen

        dt = time.time() - t0

        samples_sec = total_seen / dt


        if rank == 0:

            print(f"[EPOCH {epoch+1}/{cfg.epochs}] loss={avg_loss:.4f} samples/sec={samples_sec:.1f}")


    if rank == 0:

        evaluate(model, test_loader, device, rank)

        path = os.path.join(cfg.out_dir, "mnist_cnn_ddp.pt")

        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        torch.save(state, path)

        print(f"[DONE] Saved checkpoint: {path}")


    cleanup_ddp()


if __name__ == "__main__":
    main()