import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split

from data_gen import FamSpectDataset
from data_augmentation import Y_special

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 64
learning_rate = 0.0001

# Normalizes image values
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset for PHASE signals
phase_dataset = FamSpectDataset(
    folder_path='spectrograms/phase',
    labels=Y_special['phase'],
    transform=transform
)

# Partition into training and validation
val_split = 0.2
val_size = int(len(phase_dataset) * val_split)
train_size = len(phase_dataset) - val_size

train_dataset, val_dataset = random_split(phase_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Model Definition
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)            # (batch, channels, 1, 1)
        x = torch.flatten(x, 1)    # Flatten to (batch, channels)
        x = self.fc(x)
        return x


# Training setup
num_phase_classes = Y_special['phase'].shape[1]
model = ConvNet(num_classes=num_phase_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

for epoch in range(num_epochs):

    # Training Phase
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_train_loss:.4f}')

    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            prob = torch.sigmoid(outputs)
            predicted_label = (prob > 0.5).float()
            total += labels.numel()
            correct += (predicted_label == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.2f}%\n")


# Save model checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'final_val_accuracy': val_accuracy,
}, "checkpoints/phase_final_checkpoint.pth")

print("Final checkpoint saved at checkpoints/phase_final_checkpoint.pth")


# Testing / Prediction function
def predict(model, input_patch):
    model.eval()
    with torch.no_grad():
        input_patch = input_patch.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_patch)
        probs = torch.sigmoid(output)
        predicted_labels = (probs > 0.5).int().squeeze()
    return predicted_labels.cpu().tolist()
