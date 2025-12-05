import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP CIFAR10 example")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed():
    # torchrun sets these environment variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class SimpleCNN(nn.Module):
    """Simple CNN model for CIFAR10."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1_sequence = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2_sequence = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.ffn_sequence = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10)
        )

    def forward(self, x):
        x = self.conv1_sequence(x)
        x = self.conv2_sequence(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.ffn_sequence(x)
        return x


def build_dataloader(args, world_size, rank):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    # Only rank 0 downloads; others wait at barrier
    if rank == 0:
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    dist.barrier()
    if rank != 0:
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return dataloader, sampler


def train(rank, world_size, local_rank, device, args):
    # Log device info on rank 0
    if rank == 0:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"可用 GPU 数量: {device_count}")
            for i in range(device_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("当前环境无 GPU，将使用 CPU 演示（无法体现并行加速）。")

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataloader, sampler = build_dataloader(args, world_size, rank)

    # Training
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        start = time.time()
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

        # Aggregate metrics across ranks
        loss_tensor = torch.tensor([running_loss, total, correct], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        agg_loss, agg_total, agg_correct = loss_tensor.tolist()
        epoch_loss = agg_loss / agg_total
        epoch_acc = agg_correct / agg_total

        if rank == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {elapsed:.2f}s")

    # Save checkpoint only on rank 0
    # if rank == 0:
    #     state = {
    #         "model": model.module.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "epochs": args.epochs,
    #         "seed": args.seed,
    #     }
    #     os.makedirs("checkpoints", exist_ok=True)
    #     torch.save(state, "checkpoints/cifar10_ddp_simple.pth")
    #     print("已保存模型到 checkpoints/cifar10_ddp_simple.pth")


def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    try:
        train(rank, world_size, local_rank, device, args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

# 启动方式（Linux, 使用 4 个进程，按需修改）:
# torchrun --nproc_per_node=4 /home/cseadmin/zsh/handonDL/scripts/torch_ddp.py --epochs 10 --batch-size 128 --lr 1e-3