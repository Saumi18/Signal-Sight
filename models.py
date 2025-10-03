import torch
import torch.nn as nn
import os

# ResNet-18 
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

class SpectrogramResNet(nn.Module):
    def __init__(self, num_classes=1, dropout=0.15):
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
        x = self.gap(x); x = torch.flatten(x, 1); return self.fc(x)

# ResNet-34
class ResNet34ForSpectrograms(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(64, 3, 1, dropout); self.layer2 = self._make_layer(128, 4, 2, dropout)
        self.layer3 = self._make_layer(256, 6, 2, dropout); self.layer4 = self._make_layer(512, 3, 2, dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = [ResidualBlock(self.in_channels, out_channels, stride, dropout)]; self.in_channels = out_channels
        for _ in range(1, blocks): layers.append(ResidualBlock(self.in_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x); x = torch.flatten(x, 1); return self.fc(x)


def load_all_models(checkpoint_dir, device):
    models = {}

    models['jamming'] = SpectrogramResNet(num_classes=1).to(device)
    models['jamming'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_jamming_classifier_model.pth'), map_location=device))

    models['pure_mixed'] = SpectrogramResNet(num_classes=1).to(device)
    models['pure_mixed'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_pure_mixed_model.pth'), map_location=device))
    
    models['router'] = SpectrogramResNet(num_classes=3).to(device)
    models['router'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_router_model.pth'), map_location=device))
    
    models['analog'] = SpectrogramResNet(num_classes=5).to(device)
    models['analog'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_analog_specialist_model.pth'), map_location=device))
    models['phase'] = SpectrogramResNet(num_classes=4).to(device)
    models['phase'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_phase_specialist_model.pth'), map_location=device))
    models['qam'] = SpectrogramResNet(num_classes=7).to(device)
    models['qam'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_qam_specialist_model.pth'), map_location=device))
    
    models['pure_cnn'] = ResNet34ForSpectrograms(num_classes=24).to(device)
    models['pure_cnn'].load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_pure_cnn_model.pth'), map_location=device))
    
    for model in models.values():
        model.eval()
        
    print("All models loaded successfully from flat checkpoint directory.")
    return models

