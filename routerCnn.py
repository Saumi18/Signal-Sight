import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import numpy as np

from data_gen import FamSpectDataset        
from data_augmentation import Y_router     
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 64
learning_rate = 0.0001

# Normalization transform
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Prepare datasets list for all families concatenated
datasets = []
for family in Y_router:
    folder_path = f'final_norm_spectrograms/{family}'  
    labels = Y_router[family]
    dataset = FamSpectDataset(folder_path=folder_path, labels=labels, transform=transform)
    datasets.append(dataset)

# Concatenate all family datasets
full_dataset = ConcatDataset(datasets)

# Split into train, val and test
val_split = 0.2
test_split = 0.2
val_size = int(len(full_dataset) * val_split)
test_size = int(len(full_dataset) * test_split)
train_size = len(full_dataset) - val_size - test_size

train_dataset, val_dataset, test_dataset= random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Number of classes (multi-hot for 4 families)
num_classes = len(Y_router[next(iter(Y_router))][0])  
family_names = ['analog', 'phase', 'qam', 'apsk']

# Define model
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2),
            # nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256),   
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = ConvNet(num_classes=num_classes).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 
                              mode='min',      # Look for a decrease in the metric
                              factor=0.1,      # Reduce LR by a factor of 10
                              patience=5,      # Wait 5 epochs for improvement before reducing LR
                              )    
# TRAINING LOOP
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
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_train_loss:.4f}')

    # VALIDATION
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.525).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

            total += labels.numel()
            correct += (predicted == labels).sum().item()
        
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Confusion Matrices per family 
    os.makedirs("conf_matrixnew1/routernew1", exist_ok=True)
    if (epoch+1) % 10 == 0:  
        for fam_idx, fam_name in enumerate(family_names):
            true_binary = all_labels[:, fam_idx].astype(int)
            pred_binary = all_preds[:, fam_idx].astype(int)

            # 2x2 confusion matrix: [[TN, FP], [FN, TP]]
            tn = np.sum((true_binary == 0) & (pred_binary == 0))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            cm = np.array([[tn, fp], [fn, tp]])
            cm_sum = cm.sum()
            cm_percent = (cm / cm_sum) * 100 if cm_sum > 0 else cm  # safe division

            csv_path = f"conf_matrixnew1/routernew1/{fam_name}new1.csv"
            with open(csv_path, "a") as f:
                f.write(f"Epoch {epoch+1}\n")
                f.write(" ,Pred_0,Pred_1\n")
                f.write(f"True_0,{cm_percent[0,0]:.2f}%,{cm_percent[0,1]:.2f}%\n")
                f.write(f"True_1,{cm_percent[1,0]:.2f}%,{cm_percent[1,1]:.2f}%\n\n")


    avg_val_loss = val_loss / len(val_loader)

    val_accuracy = 100 * correct / total

    scheduler.step(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n")


# SAVE FINAL CHECKPOINT
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'final_val_accuracy': val_accuracy,
}, "checkpoints/rcnn_final_checkpoint.pth")

print("Final checkpoint saved at checkpoints/rcnn_final_checkpoint.pth")


# PREDICT FUNCTION
def predict(model, input_patch):
    model.eval()
    with torch.no_grad():
        input_patch = input_patch.unsqueeze(0).to(device)
        output = model(input_patch)
        probs = torch.sigmoid(output)
        predicted_labels = (probs > 0.525).int().squeeze()
    return predicted_labels.cpu().tolist()
