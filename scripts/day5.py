import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # 连接 PyTorch 和 TensorBoard 的桥梁
from datetime import datetime # 动态生成时间戳
import os

log_dir = os.path.expanduser(f"~/runs/SimpleLSTM/SimpleLSTM_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
writer = SummaryWriter(log_dir=log_dir) # 创建一个 SummaryWriter 对象，用于记录训练过程中的信息

# 参数设置
maxlen = 200
batch_size = 64
embedding_dim = 128
hidden_dim = 128
output_dim = 1
epochs = 10

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def preprocess_dataset(tokenizer, maxlen=200):
    dataset = load_dataset('imdb')
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=maxlen)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets

def get_dataloaders(tokenized_datasets, batch_size=64, valid_ratio=0.2):
    full_train_dataset = tokenized_datasets["train"]
    n_total = len(full_train_dataset)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_dataset, valid_dataset = random_split(full_train_dataset, [n_train, n_valid])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)
    return train_dataloader, valid_dataloader, test_dataloader

class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        prediction = self.fc(hidden)
        return prediction

def binary_accuracy(preds, y):
    with torch.no_grad():
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
    return acc

def train(model, train_dataloader, valid_dataloader, optimizer, criterion, device, start_epoch=0, epochs=epochs):
    best_valid_acc = 0.0
    best_valid_loss = float('inf')
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        epoch_acc = 0
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            predictions = model(input_ids).squeeze(1)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_acc += acc.item()
        train_loss = running_loss / len(train_dataloader)
        train_acc = 100*epoch_acc / len(train_dataloader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy%/train', train_acc, epoch)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}%")

        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy%/valid', valid_acc, epoch)
        print(f"Epoch {epoch+1}, Valid Loss: {valid_loss:.4f}")
        print(f"Epoch {epoch+1}, Valid Acc: {valid_acc:.4f}%")
        if (valid_acc > best_valid_acc) or (valid_acc == best_valid_acc and valid_loss < best_valid_loss):
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'acc': valid_acc
            }
            torch.save(checkpoint, os.path.expanduser(f'~/DLModel/SimpleLSTM/checkpoint_{epoch+1}.pth'))
    return best_epoch

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            predictions = model(input_ids).squeeze(1)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    eval_loss = epoch_loss / len(iterator)
    eval_acc = 100.*epoch_acc / len(iterator)
    return eval_loss, eval_acc

def predict_sentiment(model, tokenizer, sentence, device, maxlen=200):
    model.eval()
    tokens = tokenizer(sentence, padding='max_length', truncation=True, max_length=maxlen, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    with torch.no_grad():
        output = model(input_ids)
        prob = torch.sigmoid(output.squeeze(1)).item()
    return prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SimpleLSTM for IMDB')
    parser.add_argument("--checkpoint", '-c', type=str, help="Checkpoint version to load", default=None)
    parser.add_argument("--mode", '-m', type=str, help="train or test", default='test')
    parser.add_argument("--epochs", '-e', type=int, default=10, help="training epochs")
    parser.add_argument("--text", '-t', type=str, help="Text for inference (test mode only)", default=None)
    args = parser.parse_args()
    checkpoint_v = args.checkpoint
    mode = args.mode
    epochs = args.epochs
    infer_text = args.text

    device = get_device()
    tokenizer = get_tokenizer()
    tokenized_datasets = preprocess_dataset(tokenizer, maxlen=maxlen)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(tokenized_datasets, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size
    model = SimpleLSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters())

    dummy_input = torch.randint(0, vocab_size, (1, maxlen)).to(device)
    writer.add_graph(model, dummy_input) # 记录模型图

    # 训练模型
    start_epoch = 0
    best_valid_acc = 0.0
    best_valid_loss = float('inf')
    best_epoch = 0

    if checkpoint_v is not None:
        print(f"Loading checkpoint_{checkpoint_v}")
        checkpoint = torch.load(os.path.expanduser(f'~/DLModel/SimpleLSTM/checkpoint_{checkpoint_v}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint['loss']
        best_valid_acc = checkpoint['acc']
        best_epoch = checkpoint['epoch']

    if mode == 'train':
        print("start training...")
        best_epoch = train(model, train_dataloader, valid_dataloader, optimizer, criterion, device, start_epoch, epochs)
        # 加载最优模型并评估
        checkpoint = torch.load(os.path.expanduser(f'~/DLModel/SimpleLSTM/checkpoint_{checkpoint_v}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)       
        writer.add_scalar('Loss/test', test_loss, best_epoch)
        writer.add_scalar('Accuracy%/test', test_acc, best_epoch)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    elif mode == 'test':
        if checkpoint_v is None:
            raise ValueError("Test mode requires --checkpoint argument.")
        if infer_text is not None:
            prob = predict_sentiment(model, tokenizer, infer_text, device)
            print(f'正面情感概率: {prob:.4f}')
            print('预测标签:', 'positive' if prob > 0.5 else 'negative')
        else:
            test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)
            writer.add_scalar('Loss/test', test_loss, best_epoch)
            writer.add_scalar('Accuracy%/test', test_acc, best_epoch)
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

    writer.close()
