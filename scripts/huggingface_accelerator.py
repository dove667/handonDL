import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from accelerate import Accelerator, DistributedType


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Accelerate CIFAR-10 example")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    return parser.parse_args()


def build_dataloader(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])
    # Let Accelerate handle distributed sampling internally
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Accelerate replaces with a distributed-aware sampler when preparing
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return dataloader


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    # Seed for reproducibility across processes
    torch.manual_seed(args.seed)

    # Info logs only on main process
    if accelerator.is_main_process:
        if accelerator.state.distributed_type in (DistributedType.MULTI_GPU, DistributedType.FSDP):
            device_count = torch.cuda.device_count()
            print(f"可用 GPU 数量: {device_count}")
            for i in range(device_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        elif accelerator.state.distributed_type == DistributedType.MULTI_CPU:
            print("使用 CPU 分布式（Gloo）。")
        else:
            print("单进程训练。")

    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataloader = build_dataloader(args.data_dir, args.batch_size, args.num_workers)

    # Let Accelerate wrap everything (device placement, DDP/FSDP, sampler, fp16/bf16, etc.)
    model, optimizer, dataloader, criterion = accelerator.prepare(model, optimizer, dataloader, criterion)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        start = time.time()
        for images, targets in dataloader:
            with accelerator.accumulate(model):
                outputs = model(images)
                loss = criterion(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()

            # Detach and gather metrics across processes
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size

        # Aggregate metrics
        agg_loss = accelerator.gather(torch.tensor(running_loss, device=accelerator.device)).sum().item()
        agg_total = accelerator.gather(torch.tensor(total, device=accelerator.device)).sum().item()
        agg_correct = accelerator.gather(torch.tensor(correct, device=accelerator.device)).sum().item()

        epoch_loss = agg_loss / agg_total
        epoch_acc = agg_correct / agg_total

        if accelerator.is_main_process:
            elapsed = time.time() - start
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {elapsed:.2f}s")

    # Save checkpoint only on main process
    if accelerator.is_main_process:
        os.makedirs("checkpoints", exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        state = {
            "model": unwrapped_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epochs": args.epochs,
            "seed": args.seed,
        }
        torch.save(state, "checkpoints/cifar10_accelerate_simple.pth")
        print("已保存模型到 checkpoints/cifar10_accelerate_simple.pth")


if __name__ == "__main__":
    main()