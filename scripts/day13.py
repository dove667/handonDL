import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter # 连接 PyTorch 和 TensorBoard 的桥梁
from datetime import datetime # 动态生成时间戳
import argparse # 解析命令行参数
import os

log_dir = os.path.expanduser(f"runs/SimpleCNN/SimpleCNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
writer = SummaryWriter(log_dir=log_dir) # 创建一个 SummaryWriter 对象，用于记录训练过程中的信息

def get_device():
    """Gets the appropriate device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def get_transform():
    """Defines the data transformations for CIFAR10."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def load_datasets(transform, data_root='../data'):
    """Loads CIFAR10 datasets and splits the training set."""
    full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128, val_batch_size=1000):
    """Creates DataLoaders for training, validation, and testing."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch=0, epochs=50, patience=10, best_val_loss=float('inf')):
    """Trains the model with early stopping and progress bars."""
    trigger_times = 0

    print('Starting training...')
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

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

        train_loss = running_loss / len(train_loader)

        # 记录标量 (Scalar) - 例如损失、准确率
        # tag:  数据的名称，用于  在sorBoard 中显示，建议使用 / 分隔符组织结构
        # scalar_value: 要记录的数值
        # global_step: 当前的训练步数或 epoch 数，作为 X 轴
        writer.add_scalar('Loss/train', train_loss , epoch)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
       
        model.eval()
        val_loss = 0
        correct = 0  # 新增：统计正确预测数
        total = 0    # 新增：统计总样本数
        with torch.no_grad():
            
            val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                 # 新增：统计准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_loop.set_postfix(loss=val_loss/(len(val_loader)))
                
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total  # 新增：计算验证集准确率
        # 记录验证损失、准确率
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)  # 新增：记录验证集准确率
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }
            torch.save(checkpoint, os.path.expanduser(f'~/DLModel/SimpleCNN/checkpoint_{epoch}.pth'))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break
    # 可视化参数和梯度直方图
    for name, param in model.named_parameters():
        if param.grad is not None: 
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
    print('Finished Training')

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set with a progress bar."""
    model.eval()
    correct = 0
    total = 0
    print('Starting evaluation...')
    with torch.no_grad():
        
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

    writer.add_scalar('Accuracy/test', accuracy, 0)  # 记录测试准确率
    print(f'Accuracy of the model on the test images: {accuracy:.2f} %')
    print("Evaluation finished!")

# 8. 主执行块
if __name__ == "__main__":
    # 在命令行传入希望加载的checkpoint版本，默认不加载
    parser = argparse.ArgumentParser(description='SimpleCNN for CIFAR10')
    parser.add_argument("--checkpoint", '-c', type=str, help="Checkpoint version to load, ignore if not needed", default=None)
    parser.add_argument("--mode", '-m', type=str, help="train or test", default='test')
    parser.add_argument("--epochs", '-e', type=int, default=50, help="training epochs, useless if mode is test")
    args = parser.parse_args()
    checkpoint_v = args.checkpoint
    mode = args.mode
    epochs = args.epochs
    
    device = get_device()
    transform = get_transform()
    train_dataset, val_dataset, test_dataset = load_datasets(transform)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    #把图像数据可视化到 TensorBoard
    iter = iter(train_loader)
    images, labels = next(iter)
    img_grid = torchvision.utils.make_grid(images[:25], nrow=5)
    writer.add_image('CIFAR10-images', img_grid)
    #把模型可视化到 TensorBoard（传入单个样本）
    model = SimpleCNN().to(device)
    writer.add_graph(model, images[0].unsqueeze(0).to(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    start_epoch = 0
    best_val_loss = float('inf')

    # 2. 加载 checkpoint 
    if mode not in ['train', 'test']:
        raise ValueError("mode must be 'train' or 'test'")
    elif mode == 'test' and checkpoint_v is None:
        raise ValueError("checkpoint must be specified in test mode")
    elif mode == 'test' and checkpoint_v is not None:
        print(f"Loading checkpoint_{checkpoint_v} for testing")
        checkpoint = torch.load(os.path.expanduser(f'~/DLModel/SimpleCNN/checkpoint_{checkpoint_v}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate_model(model, test_loader, device)
    elif mode == 'train' and checkpoint_v is not None:
        print(f"Loading checkpoint_{checkpoint_v} for training")
        checkpoint = torch.load(os.path.expanduser(f'~/DLModel/SimpleCNN/checkpoint_{checkpoint_v}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']  
        train_model(model, train_loader, val_loader, criterion, optimizer, 
                    scheduler, device, start_epoch=start_epoch,epochs=epochs,
                    patience=10, best_val_loss=best_val_loss)
        evaluate_model(model, test_loader, device)
    elif mode == 'train' and checkpoint_v is None:
        print("Starting training from scratch...")
        train_model(model, train_loader, val_loader, criterion, optimizer, 
                    scheduler, device, start_epoch=start_epoch, epochs=epochs,
                    patience=10, best_val_loss=best_val_loss)
        evaluate_model(model, test_loader, device)
    
    writer.close()
