import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def main():
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    data_save_dir = os.path.join(args.save_path, "data")
    model_save_path = os.path.join(args.save_path, "models", "trained_model.pt")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    
    print("Preparing Dataset...", flush=True)
    train_dataset = torchvision.datasets.MNIST(root=data_save_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_save_dir, train=False, transform=transform, download=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Model Initialize..", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Start Training..", flush=True)
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        accuracy = eval(model, test_dataloader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}", flush=True)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}", flush=True)

def train(model, train_dataloader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

def eval(model, test_dataloader, device):
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="/storage")
    
    args = parser.parse_args()
    
    main()