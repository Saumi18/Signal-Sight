import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 64
learning_rate = 0.0001

# Normalizes image values.
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class Patchnormalize(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.patches = []
        self.labels = []

        self.label_map = {folder: idx for idx, folder in enumerate(sorted(os.listdir(folder_path)))}
        for folder in os.listdir(folder_path):
            path = os.path.join(folder_path, folder)
            for file in os.listdir(path):
                self.patches.append(os.path.join(path, file))
                self.labels.append(self.label_map[folder])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = np.load(self.patches[index]).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        if self.transform:
            patch = self.transform(torch.from_numpy(patch))
        label_idx = self.labels[index]
        label = torch.zeros(len(self.label_map), dtype=torch.float32)
        label[label_idx] = 1.0
        return patch, label


# Phase dataset
phase_dataset = Patchnormalize(folder_path='spectrograms/phase', transform=transform)

val_split = 0.2
val_size = int(len(phase_dataset) * val_split)
train_size = len(phase_dataset) - val_size

train_dataset, val_dataset = random_split(phase_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class ConvNet(nn.Module):
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
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


num_phase_classes = len(os.listdir('spectrograms/phase'))
model = ConvNet(num_classes=num_phase_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

for epoch in range(num_epochs):
    # Training phase
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
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_train_loss:.4f}')

    # Validation phase
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


# Prediction function
def predict(model, input_patch):
    model.eval()
    with torch.no_grad():
        input_patch = input_patch.unsqueeze(0).to(device)
        output = model(input_patch)
        probs = torch.sigmoid(output)
        predicted_labels = (probs > 0.5).int().squeeze()
    return predicted_labels.cpu().tolist()
