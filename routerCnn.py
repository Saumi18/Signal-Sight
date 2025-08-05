import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import os

# Import your prepared data variables
from full_data_extract import family_X, family_Y, len_family
from generate_input_data import X, Y_special, Y_router

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (adjust as needed)
num_epochs = 100
batch_size = 64
learning_rate = 1e-4

# Normalization transform for spectrograms (single-channel)
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset class to wrap the preloaded numpy arrays into torch tensors
class RouterDataset(Dataset):
    def __init__(self, X_data, Y_data, transform=None):
        self.X_data = X_data
        self.Y_data = Y_data
        self.transform = transform

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        spec = self.X_data[idx].astype(np.float32)
        spec = np.expand_dims(spec, axis=0)  # Add channel dim for CNN: [C,H,W]
        spec = torch.from_numpy(spec)
        if self.transform:
            spec = self.transform(spec)
        label = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return spec, label

family_names = ['analog', 'phase', 'qam', 'apsk']
num_classes = len(family_names)

# For example, train on the 'analog' family data - customize as needed!
X_data = X['analog']   # Mixed signals array for 'analog'
Y_data = Y_special['analog']  # Corresponding multi-labels

dataset = RouterDataset(X_data, Y_data, transform=transform)

val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class RouterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Adjust fc input size based on input spectrogram size after conv+pooling
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 1, 256),  # Example: adjust 16x1 for your input size
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = RouterCNN(num_classes=num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            total += labels.numel()
            correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")


def predict(model, input_spec):
    model.eval()
    with torch.no_grad():
        input_spec = np.expand_dims(input_spec, axis=0)  # add channel dim if needed
        input_spec = torch.tensor(input_spec, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(input_spec)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().squeeze().tolist()
    return preds
