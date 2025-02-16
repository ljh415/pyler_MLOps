import os
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def main():
    ## init arguments, path
    epochs = int(os.environ.get("EPOCHS", 10))
    batch_size = int(os.environ.get("BATCH_SIZE", 64))
    lr = float(os.environ.get("LR", 1e-3))
    root_save_dir = os.environ.get("SAVE_PATH", "/storage")
    
    data_save_dir = os.path.join(root_save_dir, "data")
        
    ## timestamp
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
    
    ## logging
    log_file = os.path.join(root_save_dir, "logs", timestamp, f"train.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    
    logger.info("Preparing Dataset...")
    train_dataset = torchvision.datasets.MNIST(root=data_save_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_save_dir, train=False, transform=transform, download=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("Model Initialize..")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    logger.info("Start Training..")
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        accuracy = eval(model, test_dataloader, device)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    ## test
    final_test_acc = eval(model, test_dataloader, device)
    logger.info(f"Final Test Accuracy: {final_test_acc:.4f}")
    
    ## save
    model_filename = f"model_{timestamp}_{final_test_acc:.4f}.pth"
    model_save_path = os.path.join(root_save_dir, "torch_models", model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model, model_save_path)
    logger.info(f"Model saved at {model_save_path}")
    
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
    main()