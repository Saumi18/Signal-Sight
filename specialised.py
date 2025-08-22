import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

activations = None
grads = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0].detach()

target_layer = model.conv[12]   # ReLU after last conv in Block 3
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

input_spec = torch.tensor(input_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) 

logits = model(input_spec)   #shape(1,24)
probs = torch.sigmoid(logits).squeeze(0)

# ---------- 5. Pick active classes ----------
active_classes = (probs > 0.5).nonzero(as_tuple=True)[0]

# ---------- 6. Grad-CAM per active class ----------
for cls in active_classes:
    model.zero_grad()
    score = logits[0, cls]
    score.backward(retain_graph=True)

    # grads: (C,H,W), activations: (C,H,W)
    pooled_grads = torch.mean(grads, dim=(1,2))          # shape: (C,)
    weighted_activations = (pooled_grads[:, None, None] * activations).sum(dim=0)  
    cam = F.relu(weighted_activations)

    # normalize to 0–1
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # resize CAM to input size
    cam_resized = cv2.resize(cam, (input_spec.shape[3], input_spec.shape[2]))

    # ---------- 7. Overlay heatmap ----------
    plt.figure(figsize=(8,6))
    plt.imshow(input_spec.squeeze().cpu(), cmap="gray")  # original spectrogram
    plt.imshow(cam_resized, cmap="jet", alpha=0.5)       # overlay
    plt.title(f"Grad-CAM for class {cls.item()}")
    plt.axis("off")
    plt.show()