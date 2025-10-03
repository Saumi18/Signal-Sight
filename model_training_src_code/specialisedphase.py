import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class PhaseSpecialistDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        print(f"Loading and preparing data for Phase Specialist from {npz_path}...")
        data = np.load(npz_path)
        
        self.base_mod_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', 
                                 '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', 
                                 '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 
                                 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        self.phase_mods = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK']
        phase_indices_in_special = {self.base_mod_classes.index(m): i for i, m in enumerate(self.phase_mods)}
        
        Y_router = data['Y_router']
        phase_samples_mask = Y_router[:, 1] == 1 
        X_spectrograms_filtered = data['X_spectrograms'][phase_samples_mask]
        Y_special_filtered = data['Y_special'][phase_samples_mask]
        
        clean_spectrograms, clean_labels = [], []
        for i in range(len(X_spectrograms_filtered)):
            spectrogram, special_vec = X_spectrograms_filtered[i], Y_special_filtered[i]
            found_label = False
            for global_idx, class_idx in phase_indices_in_special.items():
                if special_vec[global_idx] == 1:
                    clean_spectrograms.append(spectrogram)
                    clean_labels.append(class_idx)
                    found_label = True
                    break
        
        self.X_spectrograms = np.array(clean_spectrograms)
        self.Y_labels = np.array(clean_labels)
        self.transform = transform
        print(f"Data preparation complete. Found {len(self.X_spectrograms)} valid samples.")

    def __len__(self):
        return len(self.X_spectrograms)

    def __getitem__(self, idx):
        spectrogram, label = self.X_spectrograms[idx], self.Y_labels[idx]
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.transform:
            spectrogram_tensor = self.transform(spectrogram_tensor)
        return spectrogram_tensor, label_tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x; out = self.relu(self.bn1(self.conv1(x))); out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class SpectrogramResNet(nn.Module):
    def __init__(self, num_classes, dropout=0.15):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(64, 2, 1, dropout); self.layer2 = self._make_layer(128, 2, 2, dropout)
        self.layer3 = self._make_layer(256, 2, 2, dropout); self.layer4 = self._make_layer(512, 2, 2, dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = [ResidualBlock(self.in_channels, out_channels, stride, dropout)]; self.in_channels = out_channels
        for _ in range(1, blocks): layers.append(ResidualBlock(self.in_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train(); running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step(); running_loss += loss.item() * inputs.size(0)
    return running_loss / len(data_loader.dataset)

def evaluate_model(model, data_loader, criterion, device):
    model.eval(); all_labels, all_preds = [], []; running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0); _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy())
    loss = running_loss / len(data_loader.dataset)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return loss, accuracy, all_labels, all_preds

def main():
    NPZ_DATA_PATH = 'processed_data/spectrograms_and_labels.npz'
    CHECKPOINT_DIR, RESULTS_DIR = "specialist_checkpoints", "specialist_results"
    
    FAMILY_NAME = "phase"
    CLASS_NAMES = ['BPSK', 'QPSK', '8PSK', '16PSK'] #, '32PSK'
    NUM_CLASSES = len(CLASS_NAMES)
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DROPOUT = 32, 25, 1e-4, 0.2

    os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_{FAMILY_NAME}_specialist_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

    dataset = PhaseSpecialistDataset(npz_path=NPZ_DATA_PATH)
    
    test_split = 0.15
    val_split = 0.15
    test_size = int(test_split * len(dataset))
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size - test_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Simple Train/Val/Test split: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    criterion = nn.CrossEntropyLoss()

    model = SpectrogramResNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(" -> New best model saved.")

    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss, test_acc, test_labels, test_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    try:
        report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, labels=np.arange(NUM_CLASSES), output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(RESULTS_DIR, f"{FAMILY_NAME}_specialist_report.csv"))
        print("\nClassification Report:\n", report_df)
    except ValueError as e:
        print(f"\nCould not generate classification report: {e}")
        print(pd.DataFrame({'accuracy': [test_acc]}))

    try:
        cm = confusion_matrix(test_labels, test_preds, labels=np.arange(NUM_CLASSES))
        plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix for {FAMILY_NAME.title()} Specialist'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        plt.tight_layout(); cm_path = os.path.join(RESULTS_DIR, f"{FAMILY_NAME}_specialist_confusion_matrix.png"); plt.savefig(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"\nCould not generate confusion matrix: {e}")

if __name__ == '__main__':
    main()