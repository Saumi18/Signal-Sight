from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import traceback

from utils import create_spectrogram_tensor, GradCAM, generate_gradcam_overlay
from models import load_all_models

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = 'checkpoints'
MODELS = load_all_models(CHECKPOINT_DIR, device)

PURE_CLASS_NAMES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
ROUTER_CLASS_NAMES = ['analog', 'phase', 'qam']
ANALOG_CLASS_NAMES = ['AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM']
PHASE_CLASS_NAMES = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK']
QAM_CLASS_NAMES = ['4ASK', '8ASK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']

SPECIALIST_MAPPINGS = {
    'analog': (MODELS['analog'], ANALOG_CLASS_NAMES),
    'phase': (MODELS['phase'], PHASE_CLASS_NAMES),
    'qam': (MODELS['qam'], QAM_CLASS_NAMES),
}

def tensor_to_base64(tensor):
    """Converts a 1-channel spectrogram tensor to a base64 image string for visualization."""
    img_array = tensor.squeeze().cpu().numpy()
    img_array = np.flipud(img_array)
    img_array = np.clip(img_array, 0, 1)
    img_array = (255.0 * img_array).astype(np.uint8)
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request using the full pipeline with Grad-CAM."""
    if not request.json or 'signal' not in request.json:
        return jsonify({'error': 'Missing signal data'}), 400

    try:
        # 1. Create the base 1-channel spectrogram tensor
        signal_data = np.array(request.json['signal'], dtype=np.float32)
        base_spectrogram = create_spectrogram_tensor(signal_data).to(device)
        spectrogram_img_b64 = tensor_to_base64(base_spectrogram)
        model_input = base_spectrogram.unsqueeze(0) # Add batch dimension for model

        with torch.no_grad():
            jamming_output = MODELS['jamming'](model_input)
            is_jammed = torch.sigmoid(jamming_output).item() > 0.5

        if is_jammed:
            return jsonify({
                'prediction_type': 'Jammed Signal',
                'modulation': 'N/A',
                'spectrogram_image': spectrogram_img_b64,
                'gradcam_image': None
            })

        with torch.no_grad():
            pure_mixed_output = MODELS['pure_mixed'](model_input)
            is_mixed = torch.sigmoid(pure_mixed_output).item() > 0.5
        
        # --- Grad-CAM and Prediction Logic ---
        if not is_mixed:
            # PURE SIGNAL PATH
            model_to_explain = MODELS['pure_cnn']
            target_layer = model_to_explain.layer4[-1] # Last block of the final layer is a good choice

            with torch.no_grad():
                pure_cnn_output = model_to_explain(model_input)
                pred_idx = torch.argmax(pure_cnn_output, dim=1).item()
            
            modulation = PURE_CLASS_NAMES[pred_idx]
            
            # Generate Grad-CAM for the pure prediction
            grad_cam = GradCAM(model_to_explain, target_layer)
            heatmap = grad_cam.generate_heatmap(model_input, class_idx=pred_idx)
            gradcam_img_b64 = generate_gradcam_overlay(heatmap, base_spectrogram)
            
            response = {
                'prediction_type': 'Pure',
                'modulation': modulation,
                'spectrogram_image': spectrogram_img_b64,
                'gradcam_image': gradcam_img_b64
            }
        else:
            # MIXED SIGNAL PATH
            with torch.no_grad():
                router_output = MODELS['router'](model_input)
                router_preds = torch.sigmoid(router_output).squeeze() > 0.5
                detected_families = [ROUTER_CLASS_NAMES[i] for i, pred in enumerate(router_preds) if pred]
            
            final_modulations = []
            gradcam_img_b64 = None 
            
            if detected_families:
                # For simplicity, generate Grad-CAM for the first detected family's specialist
                family_to_explain = detected_families[0]
                model_to_explain, _ = SPECIALIST_MAPPINGS[family_to_explain]
                target_layer = model_to_explain.layer4[-1]

                with torch.no_grad():
                    specialist_output = model_to_explain(model_input)
                    pred_idx = torch.argmax(specialist_output, dim=1).item()
                
                # Generate Grad-CAM for the specialist's prediction
                grad_cam = GradCAM(model_to_explain, target_layer)
                heatmap = grad_cam.generate_heatmap(model_input, class_idx=pred_idx)
                gradcam_img_b64 = generate_gradcam_overlay(heatmap, base_spectrogram)
                
                # Get all specialist predictions
                for family in detected_families:
                    specialist_model, specialist_classes = SPECIALIST_MAPPINGS[family]
                    output = specialist_model(model_input)
                    idx = torch.argmax(output, dim=1).item()
                    final_modulations.append(specialist_classes[idx])

            response = {
                'prediction_type': 'Mixed',
                'modulations': final_modulations if final_modulations else ["Unknown"],
                'spectrogram_image': spectrogram_img_b64,
                'gradcam_image': gradcam_img_b64
            }
        
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

