import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm # 导入 tqdm

# 1. 设备设置
def get_device():
    """Gets the appropriate device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# 2. 数据转换
def get_transform():
    """Defines the data transformations for CIFAR10."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

# 3. 加载和分割数据集
def load_datasets(transform, data_root='../data'):
    """Loads CIFAR10 datasets and splits the training set."""
    full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

# 4. 创建 DataLoader
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128, val_batch_size=1000):
    """Creates DataLoaders for training, validation, and testing."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# 5. 模型定义
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

# 6. 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, patience=5):
    """Trains the model with early stopping and progress bars."""
    best_val_loss = float('inf')
    trigger_times = 0

    print('Starting training...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # 使用 tqdm 包装 train_loader
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (data, target) in enumerate(train_loop):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=running_loss/(i+1))

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # 使用 tqdm 包装 val_loader
            val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_loop.set_postfix(loss=val_loss/(len(val_loader)))

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Optionally save the best model
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    print('Finished Training')

# 7. 评估模型
def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set with a progress bar."""
    model.eval()
    correct = 0
    total = 0
    print('Starting evaluation...')
    with torch.no_grad():
        # 使用 tqdm 包装 test_loader
        test_loop = tqdm(test_loader, leave=True, desc="Evaluating")
        for images, labels in test_loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_loop.set_postfix(accuracy=100. * correct / total)

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f} %')
    print("Evaluation finished!")

# 8. 主执行块
if __name__ == "__main__":
    device = get_device()

    transform = get_transform()

    train_dataset, val_dataset, test_dataset = load_datasets(transform)

    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    evaluate_model(model, test_loader, device)
