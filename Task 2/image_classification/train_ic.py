import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def parse_args():
    """
    Parses command-line arguments. Uses argparse to handle inputs like
    dataset path, epochs, and learning rate dynamically without hardcoding.
    """
    parser = argparse.ArgumentParser(description="Train Animal Image Classifier (ResNet)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset (folders with class names)")
    parser.add_argument("--output_model", type=str, default="animal_model.pth", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()


def train_model():
    """
    Main function to train the ResNet18 classifier. Automatically selects GPU/CPU,
    applies data augmentation, uses transfer learning by replacing the final FC layer,
    and saves the trained model weights alongside class names for future inference.
    """
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Directory {args.data_dir} not found!")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading data...")
    dataset = datasets.ImageFolder(args.data_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Classes found ({num_classes}): {class_names}")

    if num_classes < 2:
        raise ValueError("Need at least 2 classes to train!")

    print("Loading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    save_data = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }
    torch.save(save_data, args.output_model)
    print(f"Model saved to '{args.output_model}'")


if __name__ == "__main__":
    train_model()