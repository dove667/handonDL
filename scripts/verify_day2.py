import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def verify_linear():
    print("Verifying nn.Linear...")
    linear = nn.Linear(in_features=10, out_features=5)
    input_tensor = torch.randn(2, 10)
    output = linear(input_tensor)
    print("Linear output shape:", output.shape)

def verify_conv_pool():
    print("Verifying Conv & Pool...")
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    input_img = torch.randn(1, 3, 32, 32)
    output_img = conv(input_img)
    print("Conv output shape:", output_img.shape)
    
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled_output = pool(output_img)
    print("Pool output shape:", pooled_output.shape)
    return pooled_output

def verify_norm_dropout(pooled_output):
    print("Verifying Norm & Dropout...")
    bn = nn.BatchNorm2d(num_features=16)
    bn_output = bn(pooled_output)
    print("BN output shape:", bn_output.shape)

    dropout = nn.Dropout(p=0.5)
    x = torch.randn(5, 10)
    dropout.train()
    _ = dropout(x)
    dropout.eval()
    _ = dropout(x)

def verify_sequential():
    print("Verifying Sequential...")
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    input_img = torch.randn(1, 3, 32, 32)
    output = model(input_img)
    print("Sequential output shape:", output.shape)
    return model

def verify_loss():
    print("Verifying Loss...")
    criterion = nn.CrossEntropyLoss()
    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 2])
    loss = criterion(logits, targets)
    print("Loss value:", loss.item())

def verify_optim(model):
    print("Verifying Optim...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    # Dummy loss for backward
    input_img = torch.randn(1, 3, 32, 32)
    output = model(input_img)
    loss = output.sum()
    loss.backward()
    optimizer.step()

def verify_functional():
    print("Verifying Functional...")
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            return x
    
    net = MyNet()
    x = torch.randn(1, 1, 32, 32)
    out = net(x)
    print("Functional net output:", out.shape)

if __name__ == "__main__":
    verify_linear()
    pooled = verify_conv_pool()
    verify_norm_dropout(pooled)
    model = verify_sequential()
    verify_loss()
    verify_optim(model)
    verify_functional()
    print("All verifications passed!")
