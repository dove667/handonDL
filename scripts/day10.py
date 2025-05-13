import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import numpy as np

# 设置随机种子，保证可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # mean, std
])

# Define data directory
data_dir = 'data'

def get_data_loaders(data_dir, transform, batch_size_train=64, batch_size_test=1000, num_workers=2):
    print('Loading data...')
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# Define the MLP model
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = Mlp().to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.AdamW(model.parameters(), lr=lr) # AdamW optimizer with weight decay

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    print('Starting training...')
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}')
    print("Training finished!")

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    print('Starting evaluation...')
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    print(f'Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f} %')
    print("Evaluation finished!")
    return avg_loss, accuracy

# Main execution block
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(data_dir, transform)
    epochs = 50
    train_model(model, train_loader, criterion, optimizer, epochs, device)
    evaluate_model(model, test_loader, criterion, device)

# Avg Loss: 0.1115 | Accuracy: 97.74 % AdamW, lr=0.001
# Avg Loss: 0.0838 | Accuracy: 97.61 % SGD, lr=0.01