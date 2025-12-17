AGMSCRA-Net: Pneumonia Detection with Cross-Dataset Generalization
Official implementation of "Anatomically Guided Multi-Scale Cross-Regional Attention Network for Cross-Dataset Pneumonia Detection in Chest X-Rays"
📊 Key Results

Same-Domain Accuracy: 97.36% (Pneumonia) | 97.52% (COVID-19)
Cross-Domain Performance Drop: Only 0.16%
Statistically Validated: McNemar's test (p=0.71, 0.85) | DeLong's test (p=0.72, 0.68)

🚀 Quick Start
Run Locally
bash# Clone repository
git clone [https://github.com/yourusername/AGMSCRA-Net](https://github.com/mohammedalani1991/Multi-Scale-Feature-Learning-of-Domain-Robust-Pneumonia-Detection-)
cd AGMSCRA-Net

# Install requirements
pip install torch torchvision timm scikit-learn matplotlib pillow numpy

# Run the notebook
jupyter notebook agmscra-net-pneumonia.ipynb

📁 Repository Contents
AGMSCRA-Net/
├── README.md
├── agmscra-net-pneumonia.ipynb    # Complete Kaggle notebook
├── requirements.txt                # Python dependencies
└── splits/                         # Dataset split manifests
    ├── pneumonia_splits.csv
    └── covid_splits.csv



📦 Datasets
The notebook uses two publicly available datasets from Kaggle:

Chest X-Ray Pneumonia

Kaggle Link
5,216 training + 624 test images
Pediatric patients (1-5 years)


COVID-19 Radiography Database

Kaggle Link
4,684 training + 1,288 test images
Adult patients (18-85 years)



Both datasets are automatically loaded when running the notebook on Kaggle.

🔬 Reproducibility
All experiments use fixed parameters for reproducibility:
ParameterValueRandom Seed42Decision Threshold0.5 (fixed)Image Size224×224Batch Size32Learning Rate0.001OptimizerAdamEpochs100
Dataset Splits
Exact train/test splits are provided in the splits/ directory:

pneumonia_splits.csv - Train/val/test assignments for pneumonia dataset
covid_splits.csv - Train/test assignments for COVID-19 dataset

These CSV files ensure you use the identical data splits from our paper.
🎯 Model Architecture
AGMSCRA-Net consists of:

EfficientNet-B3 Backbone (10.7M parameters)
Cross-Regional Attention - Models relationships between 7 lung regions
Multi-Scale Feature Pyramid - Captures features at 3 scales
Curriculum Learning - 4-phase progressive training
Anatomy Consistency Loss - Enforces clinical coherence

📈 Performance Metrics
Same-Domain Evaluation
DatasetAccPrecisionRecallF1AUC-ROCPneumonia97.36%97.62%97.77%97.69%99.42%COVID-1997.52%97.16%97.12%97.14%99.31%
Cross-Domain Evaluation
Training → TestingAccuracyDropPneumonia → COVID-1997.12%0.24%COVID-19 → Pneumonia97.44%0.08%Average97.28%0.16%
💻 Usage
Training
The notebook contains all code to train from scratch:
python# The notebook includes:
# 1. Data loading and preprocessing
# 2. Model architecture definition
# 3. Four-phase curriculum training
# 4. Evaluation on same-domain and cross-domain
# 5. Statistical validation (McNemar's & DeLong's tests)
Inference
python# Load trained model
model = AGMSCRANet(num_classes=2)
model.load_state_dict(torch.load('agmscra_best.pth'))
model.eval()

# Predict on new image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('chest_xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output).item()
    
print(f"Pneumonia probability: {prediction:.2%}")
🔧 Requirements
torch>=2.0.1
torchvision>=0.15.2
timm>=0.9.2
scikit-learn>=1.3.0
numpy>=1.24.3
pillow>=10.0.0
matplotlib>=3.7.2
Install all dependencies:
bashpip install -r requirements.txt
