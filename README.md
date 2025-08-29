# Signal-Sight
A Deep Learning model for real-time wireless signal classification

## OVERVIEW
- We use the RadioML dataset for training our models, which contains a wide range of modulated signals recorded under different SNR conditions.
- The system uses Convolutional Neural Networks (CNNs) to learn patterns from spectrograms of signals.
- To improve explainability, we apply Grad-CAM visualizations, which highlight the parts of the spectrogram that influenced the model’s decision.

## WORKFLOW
- Raw IQ data from the RadioML dataset is converted into spectrogram images, capturing the time-frequency representation of signals.
- The Router CNN identifies the broad signal family from the spectrogram.
- Corresponding Specialized CNN refines the classification within the identified family.
- Grad-CAM visualizations are applied to highlight important spectrogram regions influencing the model’s decisions.

## PROJECT STRUCTURE
- `full_data_extract.py` – Load and organize raw IQ signals and labels from RadioML into family-wise datasets.
- `generate_input_data.py` – Simulate pure and mixed wireless signals, prepare multi-hot labels, and create training inputs.
- `generate_jammed_dataset.py` – Add jamming interference to pure and mixed signals, creating realistic jammed datasets.
- `jamming_functions.py` – Implements different jamming techniques for dataset augmentation.
- `spectrogram_extract.py` – Convert IQ signals to spectrogram images for CNN model inputs.
- `routerCnn.py` - Main CNN for signal family classification.
- `grad_cam_router.py` - Generates Grad-CAM visualizations.
- `specialised*.py` - Specialist CNNs for fine-grained modulation classification.

## FEATURES
- Converts wireless signals into spectrogram images for visual pattern recognition.
- Uses CNNs for high accuracy in signal classification.
- Handles noisy and jamming/spoofing signals with robustness.
- Real-time signal processing capabilities.

## Tech Stack

- [Python]
- [PyTorch]
- [NumPy]
- [Matplotlib]
- GitHub
- GradCam
- MATLAB

## Applications
- Real-time wireless spectrum monitoring for managing congested frequency bands.
- Cognitive radio systems enabling adaptive interference avoidance.
- Defense and signal intelligence for identifying unknown transmissions.
- Supporting efficient communication in IoT and 5G networks.
- Modulation recognition and switching in software-defined radios.


