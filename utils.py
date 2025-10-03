import numpy as np
from scipy.signal import spectrogram
import torch
import cv2
import base64
from io import BytesIO

# --- UNIFIED CONFIGURATION ---
# Replace these placeholder values with your universal constants
GLOBAL_MIN = 0.00000000
GLOBAL_MAX = 6.73748350

FS = 1000
NPERSEG = 64
NOVERLAP = 48
EPSILON = 1e-8

class GradCAM:
    """
    Grad-CAM implementation to visualize model decisions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture the gradients and activations
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))
        self.hook_handles.append(self.target_layer.register_forward_hook(self._forward_hook))

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _forward_hook(self, module, input, output):
        self.activations = output

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval() # Ensure model is in eval mode
        
        # 1. Forward pass to get the prediction
        # Enable grad for the input tensor for the backward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # 2. Backward pass to get gradients
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True) # Retain graph for potential multiple uses
        
        # 3. Get activations and gradients from hooks
        if self.gradients is None or self.activations is None:
             raise RuntimeError("Hooks did not capture gradients or activations.")
             
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 4. Calculate neuron importance weights (Global Average Pooling)
        weights = np.mean(gradients, axis=(1, 2))
        
        # 5. Create heatmap by weighting the activation channels
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i, :, :]
            
        # 6. Apply ReLU (to keep only positive influences) and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + EPSILON)
        
        self.remove_hooks() # Clean up hooks after use
        return heatmap

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def create_spectrogram_tensor(signal_data):
    """
    Takes raw signal data and converts it into a normalized 1-channel
    spectrogram tensor. This is the base input for all models.
    """
    if signal_data.ndim == 2 and signal_data.shape[1] == 2:
        signal_data = signal_data[:, 0] + 1j * signal_data[:, 1]
    
    _, _, Sxx_complex = spectrogram(signal_data, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx_complex))
    Sxx_norm = (Sxx_log - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + EPSILON)
    spectrogram_tensor = torch.tensor(Sxx_norm, dtype=torch.float32).unsqueeze(0)
    return spectrogram_tensor

def tensor_to_base64_image(tensor, colormap=cv2.COLORMAP_INFERNO):
    """
    Converts a 1-channel spectrogram tensor to a vibrant, colorized 
    base64 image string for clear visualization.
    """
    # Squeeze tensor, convert to numpy, and normalize locally for visualization
    img_array = tensor.squeeze().cpu().numpy()
    local_min, local_max = img_array.min(), img_array.max()
    if local_max > local_min:
        img_array = (img_array - local_min) / (local_max - local_min)
    
    # Scale to 0-255 and apply a high-contrast colormap
    img_array_u8 = (img_array * 255).astype(np.uint8)
    img_array_color = cv2.applyColorMap(img_array_u8, colormap)

    # Invert y-axis for standard spectrogram display (low freqs at bottom)
    img_array_color = np.flipud(img_array_color)

    # Convert to base64 string
    _, buffer = cv2.imencode('.png', img_array_color)
    return base64.b64encode(buffer).decode('utf-8')

def generate_gradcam_overlay(heatmap, base_spectrogram_tensor):
    """
    Overlays the Grad-CAM heatmap on a colorized spectrogram and returns
    a base64 encoded image string.
    """
    # Create a color version of the base spectrogram using the same colormap
    spec_img_u8 = (np.clip(base_spectrogram_tensor.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    spec_img_u8 = np.flipud(spec_img_u8) # Flip for correct orientation
    spec_img_bgr = cv2.applyColorMap(spec_img_u8, cv2.COLORMAP_INFERNO)


    # Resize and colorize the heatmap
    heatmap_resized = cv2.resize(heatmap, (spec_img_u8.shape[1], spec_img_u8.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Blend the two images
    overlay = cv2.addWeighted(spec_img_bgr, 0.6, heatmap_color, 0.4, 0)
    
    # Convert to base64 string
    _, buffer = cv2.imencode('.png', overlay)
    return base64.b64encode(buffer).decode('utf-8')

