import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
from full_data_extract import mod_map 

class SpectrogramDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data['X_spectrograms']
        self.Y = data['Y_labels']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spectrogram = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.Y[idx], dtype=torch.long)
        return spectrogram, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels); self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels); self.relu = nn.ReLU(inplace=True); self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)) if stride != 1 or in_channels != out_channels else None
    def forward(self, x):
        identity = x; out = self.relu(self.bn1(self.conv1(x))); out = self.dropout(out); out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; return self.relu(out)

class ResNet34ForSpectrograms(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(64, 3, 1, dropout)
        self.layer2 = self._make_layer(128, 4, 2, dropout)
        self.layer3 = self._make_layer(256, 6, 2, dropout)
        self.layer4 = self._make_layer(512, 3, 2, dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = [ResidualBlock(self.in_channels, out_channels, stride, dropout)]; self.in_channels = out_channels
        for _ in range(1, blocks): layers.append(ResidualBlock(self.in_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x); x = torch.flatten(x, 1); return self.fc(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step(); running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion, device):
    model.eval(); all_labels, all_preds = [], []; running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0); _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy())
    loss = running_loss / len(loader.dataset)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return loss, accuracy, all_labels, all_preds

def main():
    NPZ_PATH = 'processed_data/pure_only_spectrograms.npz'
    CHECKPOINT_DIR, RESULTS_DIR = "pure_cnn_checkpoints", "pure_cnn_results"
    NUM_CLASSES = 24
    CLASS_NAMES = list(mod_map.keys())
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DROPOUT = 64, 40, 1e-3, 0.25

    os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_pure_cnn_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpectrogramDataset(NPZ_PATH)
    
    test_split_ratio = 0.2
    test_size = int(test_split_ratio * len(dataset))
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()

    model = ResNet34ForSpectrograms(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_test_acc = 0.0
    print("--- Starting Training of Custom ResNet-34 ---")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        # Save the model if it has the best test accuracy so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f" -> New best model saved with accuracy: {best_test_acc:.4f}")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    _, _, test_labels, test_preds = evaluate_model(model, test_loader, criterion, device)
    report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, labels=range(NUM_CLASSES), zero_division=0)
    print("Classification Report:\n", report)
    pd.DataFrame(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, labels=range(NUM_CLASSES), zero_division=0, output_dict=True)).transpose().to_csv(os.path.join(RESULTS_DIR, "pure_cnn_scratch_report.csv"))

if __name__ == '__main__':
    main()