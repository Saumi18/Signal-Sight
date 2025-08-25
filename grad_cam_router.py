import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add path to your RouterCNN implementation
sys.path.append("/content/Signal-Sight")  # adjust as needed
from routerCnn import RouterCNN  # import your existing model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
num_classes = 4  # adjust based on your router CNN
model = RouterCNN(num_classes=num_classes).to(device)
#assumes that parameters are stored after model saved
model.load_state_dict(torch.load("routercnn.pth", map_location=device))
model.eval()

#Grad-CAM hooks
activations = None
grads = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0].detach()

# Attach hooks to the last convolutional layer
model.conv[-1].register_forward_hook(forward_hook)
model.conv[-1].register_backward_hook(backward_hook)

# -------------------- Input spectrogram --------------------
spec = np.load("/content/Signal-Sight/spec.npy")  #spectrogram input
spec = np.expand_dims(spec, axis=0)  # [C=1, H, W]
spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device) 

# Forward pass
logits = model(spec_tensor) 
probs = torch.sigmoid(logits)
predicted = (probs > 0.5).int().squeeze() 

#Grad-CAM for active classes only
cams = {}

for target_class in range(num_classes):
    if predicted[target_class] == 0:
        continue

    model.zero_grad()
    score = logits[0, target_class]
    score.backward(retain_graph=True)

    # Compute importance weights
    weights = grads.mean(dim=(2, 3))[0]  # [C]

    # Weighted combination of feature maps
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
    for c, w in enumerate(weights):
        cam += w * activations[0, c, :, :]

    cam = torch.relu(cam)   # ReLU
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # Upsample to match spectrogram size
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=spec_tensor.shape[2:],
        mode='bilinear',
        align_corners=False
    )

    cams[target_class] = cam.squeeze().cpu().numpy()

#Plot Grad-CAM
for class_idx, cam in cams.items():
    plt.figure()
    plt.title(f"Grad-CAM for active class {class_idx}")
    plt.imshow(spec.squeeze(), cmap='gray')          # base spectrogram
    plt.imshow(cam, cmap='jet', alpha=0.4)          # overlay Grad-CAM
    plt.colorbar()
    plt.show()
