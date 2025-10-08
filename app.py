from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import os

# Make sure these point to your utility and model loading scripts
from utils import create_spectrogram_tensor, GradCAM, generate_gradcam_overlay, tensor_to_base64_image
from models import load_all_models

# --- Initialize Flask App and Load Models ---
app = Flask(__name__)

# Security: Set max content length to prevent large payload attacks
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

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

# --- Helper Function to Process Signal Data ---
def process_signal_input(signal_data):
    """
    Process signal data - handles both flat arrays and I/Q pairs
    Returns: 1D numpy array
    """
    # Check if it's a 2D array (I/Q pairs like [[I,Q], [I,Q], ...])
    if isinstance(signal_data, list) and len(signal_data) > 0:
        if isinstance(signal_data[0], (list, tuple)):
            # It's I/Q pairs - flatten to interleaved I,Q,I,Q,...
            flattened = []
            for pair in signal_data:
                if len(pair) != 2:
                    raise ValueError("Each I/Q pair must have exactly 2 values")
                flattened.extend(pair)
            return np.array(flattened, dtype=np.float32)
        else:
            # Already a flat array
            return np.array(signal_data, dtype=np.float32)
    else:
        raise ValueError("Signal must be a non-empty array")

# --- Define Routes ---
@app.route('/')
def index():
    """Serve the main input page"""
    return render_template('index.html')

@app.route('/results')
def results():
    """Serve the results/prediction page"""
    return render_template('predict.html')

@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({'status': 'healthy', 'device': str(device)}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - processes signal data"""
    
    # Validate request has JSON data
    if not request.json or 'signal' not in request.json:
        return jsonify({'error': 'Missing signal data'}), 400

    try:
        # Parse and validate signal data
        signal_data = request.json['signal']
        
        # Validate it's a list/array
        if not isinstance(signal_data, (list, tuple)):
            return jsonify({'error': 'Signal must be an array'}), 400
        
        # Process the signal (handles both flat arrays and I/Q pairs)
        signal_array = process_signal_input(signal_data)
        
        # Validate array length (prevent extremely large inputs)
        if len(signal_array) < 10:
            return jsonify({'error': 'Signal too short (minimum 10 samples)'}), 400
        if len(signal_array) > 1000000:
            return jsonify({'error': 'Signal too long (maximum 1M samples)'}), 400
        
        # 1. Create the base 1-channel spectrogram tensor
        base_spectrogram = create_spectrogram_tensor(signal_array).to(device)

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
                'gradcam_image': None
            }
            return jsonify(response)

        # 4. Pipeline Step 2: Pure vs. Mixed Classification
        with torch.no_grad():
            pure_mixed_output = MODELS['pure_mixed'](model_input)
            is_mixed = torch.sigmoid(pure_mixed_output).item() > 0.5
        
        gradcam_img_b64 = None

        if not is_mixed:
            # 5a. Pure Signal Path
            model_to_explain = MODELS['pure_cnn']
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
                heatmap = grad_cam.generate_heatmap(model_input)
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

    except ValueError as e:
        return jsonify({'error': f'Invalid signal data format: {str(e)}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # IMPORTANT: Set debug=False for production!
    # Use debug=True only for local development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
