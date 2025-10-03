import numpy as np
from scipy.signal import spectrogram
import torch
import cv2 
import base64
from io import BytesIO
from PIL import Image

GLOBAL_MIN = 0.00000000
GLOBAL_MAX = 6.73748350

FS = 1000
NPERSEG = 64
NOVERLAP = 48
EPSILON = 1e-8

class GradCAM:
    """
    Grad-CAM implementation to visualize model decisions.
    We capture the gradients and activations from a target layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to intercept the forward and backward passes
        self.target_layer.register_full_backward_hook(self._backward_hook)
        self.target_layer.register_forward_hook(self._forward_hook)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _forward_hook(self, module, input, output):
        self.activations = output

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Forward pass to get model output
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # 2. Backward pass to get gradients
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)
        
        # 3. Get captured activations and gradients
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 4. Weight the channels by the mean gradient (neuron importance)
        weights = np.mean(gradients, axis=(1, 2))
        
        # 5. Build the heatmap by weighting the activation channels
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i, :, :]
            
        # 6. Apply ReLU to keep only positive influences and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + EPSILON)
        return heatmap

def create_spectrogram_tensor(signal_data):
    """
    Takes raw signal data and converts it into a normalized 1-channel
    spectrogram tensor, which is the base input for all models.
    """
    if signal_data.ndim == 2 and signal_data.shape[1] == 2:
        signal_data = signal_data[:, 0] + 1j * signal_data[:, 1]
    
    _, _, Sxx_complex = spectrogram(signal_data, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx_complex))
    Sxx_norm = (Sxx_log - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + EPSILON)
    spectrogram_tensor = torch.tensor(Sxx_norm, dtype=torch.float32).unsqueeze(0)
    return spectrogram_tensor

def generate_gradcam_overlay(heatmap, base_spectrogram_tensor):
    """
    Overlays the Grad-CAM heatmap on the original spectrogram and returns
    a base64 encoded image string.
    """
    # Convert base spectrogram to a BGR image
    spec_img = base_spectrogram_tensor.squeeze().cpu().numpy()
    spec_img = np.flipud(spec_img) # Flip for correct orientation
    spec_img = np.clip(spec_img, 0, 1)
    spec_img = (255.0 * spec_img).astype(np.uint8)
    spec_img_bgr = cv2.cvtColor(spec_img, cv2.COLOR_GRAY2BGR)

    # Resize heatmap to match spectrogram and apply a colormap
    heatmap_resized = cv2.resize(heatmap, (spec_img.shape[1], spec_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Blend the heatmap and the spectrogram image
    overlay = cv2.addWeighted(spec_img_bgr, 0.6, heatmap_color, 0.4, 0)
    
    # Encode the final image to a base64 string for web display
    _, buffer = cv2.imencode('.png', overlay)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_b64
