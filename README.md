📡 SignalSight
🎓 Signal Classification System

What It Does
SignalSight is an end-to-end intelligent signal classification system built to identify and interpret digital modulation types using deep learning and real-time visualization. It integrates a trained hierarchical model with a user-friendly Flask web interface that allows anyone to upload wireless signal data and view classification results with interpretability.

It automates complex modulation recognition by converting 1D I/Q signals into 2D spectrograms and passing them through a multi-stage classifier pipeline. The system distinguishes between pure, mixed, and jammed/spoofed signals while visualizing critical spectrogram regions that influenced each decision.

✨ Key Features

🎯 Converts 1D signal arrays into 2D spectrograms for model inference
🔁 Hierarchical classification pipeline (pure, mixed, and spoofed signals)
🧠 Specialized CNN trained for various SNR levels
🖥️ Flask-based web UI for easy real-time testing and visualization
🔍 Interpretable AI via Grad-CAM — highlights decisive spectrogram regions
⚙️ Supports classification across 24 digital modulation types (QAM, PSK, AM, FM, etc.)

🧩 Tech Stack

Programming: Python
Framework: Flask (for web-based UI and backend inference)
Deep Learning: PyTorch (trained CNNs)
Signal Processing: Matplotlib
Visualization: Grad-CAM for interpretability
Dataset: RadioML 2018.01A (SNR −10,10,30)
UI Styling: HTML/CSS

🧠 Trained Models

All model checkpoints are pre-trained and available in the checkpoints/ directory:
• best_pure_cnn_model.pth – Pure signal classifier
• best_pure_mixed_model.pth – Mixed signal classifier
• best_qam_specialist_model.pth – QAM modulation specialist
• best_phase_specialist_model.pth – Phase modulation specialist
• best_analog_specialist_model.pth – Analog modulation specialist
• best_jamming_classifier_model.pth – Spoofing and jamming detector
• best_router_model.pth – SNR routing model

These models collectively enable hierarchical inference for signals under different SNR conditions.

🚀 How to Run the Project

1. Clone the Repository
git clone https://github.com/your-username/Signal-Sight.git
cd Signal-Sight

2. Install Dependencies
pip install -r requirements.txt

3. Verify Pre-trained Models
Ensure all .pth model files are present in the checkpoints/ directory.

4. Start the Flask App
python app.py

5. Access the Web Interface
Open your browser and go to:
http://127.0.0.1:5500/Signal-Sight/templates/index.html

6. Classify a Signal
• Paste your signal data in JSON format (e.g., [0.1, 0.2, 0.3, ...])
• Click “Classify Signal”
• View predicted modulation type and Grad-CAM visualization of spectrogram features

📈 Project Architecture

1️⃣ Input 1D I/Q signal data
2️⃣ Convert to spectrogram representation
3️⃣ Route through SNR classifier
4️⃣ Process via hierarchical CNN pipeline
5️⃣ Output predicted modulation class
6️⃣ Generate Grad-CAM heatmap for interpretability
7️⃣ Display results in Flask UI

🏭 Industrial Applications

🚗 Automotive – Secure V2X communications
🏦 Finance – Detection of spoofed wireless signals in trading systems
🏬 Retail – IoT device authentication and interference detection
🛰️ Defense – Real-time RF signal intelligence and jamming analysis
📡 Telecom – Network monitoring and intrusion detection

🌐 User Interface Highlights

• Built with Flask
• Gradient background and card layout for modern look
• Responsive input box for JSON signal array
• One-click signal classification button with smooth gradient
• Automatic visualization of spectrogram and model interpretability results
