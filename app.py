from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Make sure these point to your utility and model loading scripts
from utils import create_spectrogram_tensor, GradCAM, generate_gradcam_overlay, tensor_to_base64_image
from models import load_all_models

# --- Initialize Flask App and Load Models ---
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load all models into memory once when the app starts
CHECKPOINT_DIR = 'checkpoints'
MODELS = load_all_models(CHECKPOINT_DIR, device)

# --- Define Class Name Mappings ---
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

# --- Define Routes ---
@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'signal' not in request.json:
        return jsonify({'error': 'Missing signal data'}), 400

    try:
        # 1. Create the base 1-channel spectrogram tensor
        signal_data = np.array(request.json['signal'], dtype=np.float32)
        base_spectrogram = create_spectrogram_tensor(signal_data).to(device)

        # 2. Generate the colorized visualization for the frontend
        spectrogram_img_b64 = tensor_to_base64_image(base_spectrogram)

        # Add a batch dimension for model input
        model_input = base_spectrogram.unsqueeze(0)

        # 3. Pipeline Step 1: Jamming Classification
        with torch.no_grad():
            jamming_output = MODELS['jamming'](model_input)
            is_jammed = torch.sigmoid(jamming_output).item() > 0.5

        if is_jammed:
            response = {
                'prediction_type': 'Jammed',
                'modulation': 'N/A - Signal is Jammed',
                'spectrogram_image': spectrogram_img_b64,
                'gradcam_image': None # No Grad-CAM for this path
            }
            return jsonify(response)

        # 4. Pipeline Step 2: Pure vs. Mixed Classification
        with torch.no_grad():
            pure_mixed_output = MODELS['pure_mixed'](model_input)
            is_mixed = torch.sigmoid(pure_mixed_output).item() > 0.5
        
        gradcam_img_b64 = None # Default value

        if not is_mixed:
            # 5a. Pure Signal Path
            model_to_explain = MODELS['pure_cnn']
            # Target the last residual block of the last layer of the ResNet-34
            target_layer = model_to_explain.layer4[-1] 
            
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
            # 5b. Mixed Signal Path
            with torch.no_grad():
                router_output = MODELS['router'](model_input)
                router_preds = torch.sigmoid(router_output).squeeze() > 0.5
                detected_families = [ROUTER_CLASS_NAMES[i] for i, pred in enumerate(router_preds) if pred]
            
            final_modulations = []
            if detected_families:
                # Explain the decision of the first detected specialist
                family_to_explain = detected_families[0]
                model_to_explain, _ = SPECIALIST_MAPPINGS[family_to_explain]
                target_layer = model_to_explain.layer4[-1]

                # Generate Grad-CAM for that specialist's top prediction
                grad_cam = GradCAM(model_to_explain, target_layer)
                heatmap = grad_cam.generate_heatmap(model_input) # Let it pick the top class
                gradcam_img_b64 = generate_gradcam_overlay(heatmap, base_spectrogram)
                
                # Get predictions from all relevant specialists
                for family in detected_families:
                    specialist_model, specialist_classes = SPECIALIST_MAPPINGS[family]
                    with torch.no_grad():
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

