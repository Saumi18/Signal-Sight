
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs=100
batch_size=64
learning_rate=0.001

transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

from torch.utils.data import Dataset,DataLoader
import os
class Patchnormalize(Dataset):
  def __init__(self,folder_path,label=0,transform=None):
      self.folder_path=folder_path#analog directory is passed
      self.transform=transform
      self.label=label
      self.patches=[]
      for family in os.listdir(self.folder_path):#family is for each signal inside analog
        family_path=os.path.join(self.folder_path,family)
        for patch in os.listdir(family_path):#going through each patch
            self.patches.append(os.path.join(family_path,patch))


  def __len__(self):
    return len(self.patches)

  def __getitem__(self,index):
      patch=np.load(self.patches[index]).astype(np.float32)
      # patch=patch/(np.max(patch)+1e-8)
      patch=np.expand_dims(patch,axis=0)

      # label=0#for analog,will change for diff cnns
      # return patch,self.label
      if self.transform:
        patch=self.transform(torch.from_numpy(patch))
      return patch,self.label

analog_dataset=Patchnormalize(folder_path='folder where analog patch are stored',label=0,transform=transform)
train_loader=DataLoader(analog_dataset,batch_size=batch_size,shuffle=True)

class ConvNet(nn.Module):
  def __init__(self,num_classes=5):
    super().__init__()
    self.conv=nn.Sequential(
        nn.Conv2d(1,16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.fc=nn.Sequential(
        nn.Linear(32*16*16,128),
        nn.ReLU(),
        nn.Linear(128,num_classes)
    )

  def forward(self,x):
    x=self.conv(x)
    x=x.view(x.size(0),-1)
    x=self.fc(x)
    return x

model=ConvNet(num_classes=4).to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

