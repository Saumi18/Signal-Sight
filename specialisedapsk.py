import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs=100
batch_size=64
learning_rate=0.0001

#normalizes image values.
transform=transforms.Compose([
    transforms.Normalize(mean=[0.5],std=[0.5])
])

import os

# edit Dataset/ Dataloader later.
# a different method can be used if we do not plan on assiging labels to classes within a family.


class Patchnormalize(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform #stores the transformation defined above
        self.patches = []          #stores the paths to the patches required 
        self.labels = []           #stores the corresponding labels

        self.label_map = {folder: idx for idx, folder in enumerate(sorted(os.listdir(folder_path)))}
        #this function sorts the subfolders and then assigns indices to them using enumerate
        for folder in os.listdir(folder_path):                  #accesses the spectrogram folder for this family
            path = os.path.join(folder_path, folder)            #navigates to the subfolders in this family ie apsk folder will have subfolders like 8apsk allat.
            for file in os.listdir(path):                       #from these subfolders it accesses the files and then appends them to patches[].
                self.patches.append(os.path.join(path, file))   #then we assign labels.
                self.labels.append(self.label_map[folder])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = np.load(self.patches[index]).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        if self.transform:
            patch = self.transform(torch.from_numpy(patch))
        label_idx = self.labels[index]
        label=torch.zeros(len(self.label_map),dtype=torch.float32)
        label[label_idx]=1.0
        return patch, label


#Partitions the dataset into training and validation
apsk_dataset = Patchnormalize(folder_path='spectrograms', transform=transform)
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


#forward describes the flow of the input data.
#self refers to the instance of convnet
#x is the batch of inputs

  
num_apsk_classes = len(os.listdir('spectrograms'))
model = ConvNet(num_classes=num_apsk_classes).to(device)            #num_classes is the total classes we will get as outputs after softmax
criterion = nn.BCEwithLogitsLoss()                                   #uses softmax loss
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

            # predicted_label = torch.argmax(outputs, dim=1).item() #outputs the predicted label or the most appropriate index.
            # total += labels.size(0)
            # correct += (predicted_label == labels).sum().item()
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
    #     predicted_label = torch.argmax(output, dim=1).item()
    # return predicted_label
