import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, f1_score, hamming_loss, multilabel_confusion_matrix

class SpectrogramDataset(Dataset):

    def __init__(self, npz_path, transform=None):
        print(f"Loading data from {npz_path}...")
        data = np.load(npz_path)
        
        self.X_spectrograms = data['X_spectrograms']
        self.Y_router = data['Y_router']
        
        self.transform = transform
        print("Data loaded successfully.")

    def __len__(self):
        return len(self.X_spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.X_spectrograms[idx]
        router_label = self.Y_router[idx]

        # Add a channel dimension for the CNN (C, H, W)
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        router_label_tensor = torch.tensor(router_label, dtype=torch.float32)

        if self.transform:
            spectrogram_tensor = self.transform(spectrogram_tensor)

        return spectrogram_tensor, router_label_tensor

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
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SpectrogramResNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.15):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, blocks=2, stride=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, router_labels in data_loader:
        inputs, router_labels = inputs.to(device), router_labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, router_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(data_loader.dataset)

def evaluate_model(model, data_loader, criterion, device, thresholds=None):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, router_labels in data_loader:
            inputs, router_labels = inputs.to(device), router_labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, router_labels)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            if thresholds is None:
                preds = (probs >= 0.515).astype(float)
            else:
                preds = (probs >= thresholds).astype(float)
            all_labels.append(router_labels.cpu().numpy())
            all_preds.append(preds)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    return running_loss / len(data_loader.dataset), all_labels, all_preds

def optimize_thresholds(model, data_loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, router_labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(router_labels.numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    thresholds = []
    for i in range(all_probs.shape[1]):
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(all_labels[:, i], (all_probs[:, i] >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        thresholds.append(best_thresh)
    return np.array(thresholds)

def main():
    NPZ_DATA_PATH = 'processed_data/spectrograms_and_labels.npz'
    
    CHECKPOINT_DIR = "router_checkpoints"
    RESULTS_DIR = "router_results"
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.3
    DROPOUT = 0.15

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_router_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SpectrogramDataset(npz_path=NPZ_DATA_PATH)
    
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight = torch.tensor([1.0, 1.2, 1.0]).to(device) # Weight 'phase' class higher
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = SpectrogramResNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_labels, val_preds = evaluate_model(model, val_loader, criterion, device)
        f1 = f1_score(val_labels, val_preds, average='samples')
        hamming = hamming_loss(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Hamming: {hamming:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(" -> New best model saved.")

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    thresholds = optimize_thresholds(model, val_loader, device)
    print("Optimized thresholds:", thresholds)

    _, all_labels, all_preds = evaluate_model(model, val_loader, criterion, device, thresholds=None)
    family_names = ['analog', 'phase', 'qam']

    report_df = pd.DataFrame(classification_report(all_labels, all_preds, target_names=family_names, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(RESULTS_DIR, "router_classification_report.csv"))
    print("\nClassification Report:")
    print(report_df)

    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    conf_dict = {}
    for i, fam in enumerate(family_names):
        tn, fp, fn, tp = conf_matrices[i].ravel()
        conf_dict[fam] = [tn, fp, fn, tp]
    conf_df = pd.DataFrame.from_dict(conf_dict, orient='index', columns=['TN', 'FP', 'FN', 'TP'])
    conf_df.to_csv(os.path.join(RESULTS_DIR, "router_confusion_matrix.csv"))
    print("\nConfusion Matrices (TN, FP, FN, TP):")
    print(conf_df)

    print("\nHamming Loss:", hamming_loss(all_labels, all_preds))
    print("F1 Score (samples average):", f1_score(all_labels, all_preds, average='samples'))

if __name__ == '__main__':
    main()
