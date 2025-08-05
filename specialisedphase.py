import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from generate_input_data import Y_special

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

family_name = "phase"
phase_modulations = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK']
num_classes = len(phase_modulations)

def load_phase_spectrograms_and_labels():
    spectrograms_dir = "spectrograms"
    phase_dir = os.path.join(spectrograms_dir, family_name)

    all_spectrograms = []
    all_filenames = []

    for filename in sorted(os.listdir(phase_dir)):
        if filename.endswith('.npy'):
            spec_path = os.path.join(phase_dir, filename)
            spec = np.load(spec_path)
            all_spectrograms.append(spec)
            all_filenames.append(filename)
    
    return np.array(all_spectrograms), all_filenames


print(f"Loading {family_name} spectrograms...")
all_X, filenames = load_phase_spectrograms_and_labels()
all_Y = Y_special[family_name]  

print(f"Loaded {len(all_X)} {family_name} spectrograms with shape {all_X[0].shape}")
print(f"Labels shape: {all_Y.shape}")
print(f"Sample filenames: {filenames[:5]}")
print(f"Sample labels: {all_Y[:5]}")

normalize = transforms.Normalize(mean=[0.5], std=[0.5])

class PhaseSpectrogramDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spec = self.X[idx]
        if spec.ndim == 2:  
            spec = np.expand_dims(spec, axis=0)
        
        spec = torch.from_numpy(spec.astype(np.float32))
        if self.transform:
            spec = self.transform(spec)
        
        label = torch.tensor(self.Y[idx], dtype=torch.float32)
        return spec, label


dataset = PhaseSpectrogramDataset(all_X, all_Y, transform=normalize)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

print(f"Training samples: {train_size}, Validation samples: {val_size}")

class PhaseSpecializedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )       
          
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = PhaseSpecializedCNN(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print(f"Phase specialized model created and moved to {device}")

num_epochs = 100

print("Starting Phase specialized CNN training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_train_loss = train_loss / len(train_loader)
    
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")


torch.save(model.state_dict(), 'phase_specialized_cnn_model.pth')
print("Phase specialized model saved as phase_specialized_cnn_model.pth")

def predict_phase_modulations(spectrogram_path):
    model.eval()
    
    
    spec = np.load(spectrogram_path).astype(np.float32)
    if spec.ndim == 2:
        spec = np.expand_dims(spec, axis=0)  
    
    
    spec_tensor = torch.from_numpy(spec).unsqueeze(0) 
    spec_tensor = normalize(spec_tensor).to(device)
    
    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).cpu().numpy().flatten()


    predicted_modulations = [phase_modulations[i] for i, pred in enumerate(preds) if pred == 1]
    return predicted_modulations

def predict_phase_modulations_from_array(spectrogram_array):
    model.eval()
    
    spec = spectrogram_array.astype(np.float32)
    if spec.ndim == 2:
        spec = np.expand_dims(spec, axis=0)  
    
    spec_tensor = torch.from_numpy(spec).unsqueeze(0)  
    spec_tensor = normalize(spec_tensor).to(device)
    
    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).cpu().numpy().flatten()

    predicted_modulations = [phase_modulations[i] for i, pred in enumerate(preds) if pred == 1]
    return predicted_modulations

print("Phase specialized CNN training complete.")

