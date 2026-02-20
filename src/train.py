import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


@dataclass
class Config:
    epochs: int = int(os.getenv("EPOCHS", "2"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    lr: float = float(os.getenv("LR", "0.01"))
    data_dir: str = os.getenv("DATA_DIR", "/tmp/mnist")
    out_dir: str = os.getenv("OUT_DIR", "/outputs")
    backend: str = os.getenv("DDP_BACKEND", "gloo")  # gloo for CPU


class SmallCNN(nn.Module):
    def __init__(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp(cfg: Config):
    if not is_distributed():
        return None, 0, 1, 0

    # torchrun sets these:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend=cfg.backend, rank=rank, world_size=world_size)
    return dist.group.WORLD, rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    group, rank, world_size, local_rank = setup_ddp(cfg)

    device = torch.device("cpu")

    # Log identity for debugging
    host = os.getenv("HOSTNAME", "unknown-host")
    if rank == 0:
        print(f"[INIT] world_size={world_size} backend={cfg.backend} data_dir={cfg.data_dir} out_dir={cfg.out_dir}")
    print(f"[RANK {rank}] host={host} local_rank={local_rank}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    model = SmallCNN().to(device)

    if world_size > 1:
        model = DDP(model)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_seen += bs

        # Aggregate metrics across ranks (optional but useful)
        if world_size > 1:
            loss_tensor = torch.tensor([total_loss, total_seen], dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = float(loss_tensor[0].item())
            total_seen = int(loss_tensor[1].item())

        dt = time.time() - t0
        avg_loss = total_loss / max(total_seen, 1)
        samples_per_sec = total_seen / max(dt, 1e-9)

        if rank == 0:
            print(f"[EPOCH {epoch+1}/{cfg.epochs}] loss={avg_loss:.4f} samples/sec={samples_per_sec:.1f} dt={dt:.2f}s")

    # Save checkpoint from rank 0 only
    if rank == 0:
        ckpt_path = os.path.join(cfg.out_dir, "mnist_cnn_ddp.pt")
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, ckpt_path)
        print(f"[DONE] Saved checkpoint: {ckpt_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()