import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split

from data_gen import FamSpectDataset
from data_augmentation import Y_special

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs=100
batch_size=64
learning_rate=0.0001

#normalizes image values.
transform=transforms.Compose([
    transforms.Normalize(mean=[0.5],std=[0.5])
])


#Partitions the dataset into training and validation
apsk_dataset = FamSpectDataset(folder_path='spectrograms/apsk', labels=Y_special['apsk'],transform=transform)
val_split = 0.2 
val_size = int(len(apsk_dataset) * val_split)
train_size = len(apsk_dataset) - val_size

train_dataset, val_dataset = random_split(apsk_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



import torch
import torch.nn as nn

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


  
num_analog_classes = Y_special['apsk'].shape[1]
model = ConvNet(num_classes=num_apsk_classes).to(device)            #num_classes is the total classes we will get as outputs after softmax
criterion = nn.BCEWithLogitsLoss()                                   #uses softmax loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

for epoch in range(num_epochs):

    #Training Phase

    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               #resets gradient from previous batches.
        outputs = model(inputs)             
        loss = criterion(outputs, labels)   #calculates the loss.
        loss.backward()                     #backpropagates to check how weights affect the previous layers.
        optimizer.step()                    #updates weights 

        running_loss += loss.item()         
        if (i+1) % 100 == 0:  #added + 1 to make it easier to understand steps. Every 100 steps this will print the loss.
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {running_loss/len(train_loader):.4f}')


    avg_train_loss = running_loss / len(train_loader)




    #Validation phase
   
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

            # predicted_label = torch.argmax(outputs, dim=1).item() #outputs the predicted label or the most appropriate index
            prob=torch.sigmoid(outputs)
            predicted_label=(prob>0.5).float()
            total += labels.numel()
            correct += (predicted_label == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total


    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.2f}%\n")


os.makedirs("checkpoints", exist_ok=True)
# Save full checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'final_val_accuracy': val_accuracy,
}, "checkpoints/apsk_final_checkpoint.pth")

print("Final checkpoint saved at checkpoints/apsk_final_checkpoint.pth")


# Testing Phase       

def predict(model, input_patch):
    model.eval()
    with torch.no_grad():
        input_patch = input_patch.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_patch)
        probs = torch.sigmoid(output)                      
        predicted_labels = (probs > 0.5).int().squeeze()
    return predicted_labels.cpu().tolist()
        # predicted_label = torch.argmax(output, dim=1).item()
    #return predicted_label
