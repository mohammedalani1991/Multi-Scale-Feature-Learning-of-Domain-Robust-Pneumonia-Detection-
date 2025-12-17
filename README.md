# AGMSCRA-Net: Pneumonia Detection with Cross-Dataset Generalization

Official implementation of "Anatomically Guided Multi-Scale Cross-Regional Attention Network for Cross-Dataset Pneumonia Detection in Chest X-Rays"

## 📊 Key Results

- **Same-Domain Accuracy**: 97.36% (Pneumonia) | 97.52% (COVID-19)
- **Cross-Domain Performance Drop**: Only 0.16%
- **Statistically Validated**: McNemar's test (p=0.71, 0.85) | DeLong's test (p=0.72, 0.68)

## 🚀 Quick Start

### Run on Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/your-username/agmscra-net-pneumonia-detection)

Click the badge above to run the complete notebook on Kaggle with free GPU.

### Run Locally

```bash
# Clone repository
git clone https://github.com/yourusername/AGMSCRA-Net.git
cd AGMSCRA-Net

# Install requirements
pip install torch torchvision timm scikit-learn matplotlib pillow numpy

# Run the notebook
jupyter notebook agmscra-net-pneumonia.ipynb
```

## 📁 Repository Contents

```
AGMSCRA-Net/
├── README.md
├── agmscra-net-pneumonia.ipynb    # Complete Kaggle notebook
├── requirements.txt                # Python dependencies
└── splits/                         # Dataset split manifests
    ├── pneumonia_splits.csv
    └── covid_splits.csv
```

## 📦 Datasets

The notebook uses two publicly available datasets from Kaggle:

1. **Chest X-Ray Pneumonia**
   - [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - 5,216 training + 624 test images
   - Pediatric patients (1-5 years)

2. **COVID-19 Radiography Database**
   - [Kaggle Link](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
   - 4,684 training + 1,288 test images
   - Adult patients (18-85 years)

Both datasets are automatically loaded when running the notebook on Kaggle.

## 🔬 Reproducibility

All experiments use fixed parameters for reproducibility:

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Decision Threshold | 0.5 (fixed) |
| Image Size | 224×224 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Epochs | 100 |

### Dataset Splits

Exact train/test splits are provided in the `splits/` directory:
- `pneumonia_splits.csv` - Train/val/test assignments for pneumonia dataset
- `covid_splits.csv` - Train/test assignments for COVID-19 dataset

These CSV files ensure you use the identical data splits from our paper.

## 🎯 Model Architecture

**AGMSCRA-Net** consists of:

1. **EfficientNet-B3 Backbone** (10.7M parameters)
2. **Cross-Regional Attention** - Models relationships between 7 lung regions
3. **Multi-Scale Feature Pyramid** - Captures features at 3 scales
4. **Curriculum Learning** - 4-phase progressive training
5. **Anatomy Consistency Loss** - Enforces clinical coherence

## 📈 Performance Metrics

### Same-Domain Evaluation

| Dataset | Acc | Precision | Recall | F1 | AUC-ROC |
|---------|-----|-----------|--------|----|---------| 
| Pneumonia | 97.36% | 97.62% | 97.77% | 97.69% | 99.42% |
| COVID-19 | 97.52% | 97.16% | 97.12% | 97.14% | 99.31% |

### Cross-Domain Evaluation

| Training → Testing | Accuracy | Drop |
|-------------------|----------|------|
| Pneumonia → COVID-19 | 97.12% | 0.24% |
| COVID-19 → Pneumonia | 97.44% | 0.08% |
| **Average** | **97.28%** | **0.16%** |

## 💻 Usage

### Training

The notebook contains all code to train from scratch:

```python
# The notebook includes:
# 1. Data loading and preprocessing
# 2. Model architecture definition
# 3. Four-phase curriculum training
# 4. Evaluation on same-domain and cross-domain
# 5. Statistical validation (McNemar's & DeLong's tests)
```

### Inference

```python
# Load trained model
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
```

## 🔧 Requirements

```
torch>=2.0.1
torchvision>=0.15.2
timm>=0.9.2
scikit-learn>=1.3.0
numpy>=1.24.3
pillow>=10.0.0
matplotlib>=3.7.2
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025agmscra,
  title={Anatomically Guided Multi-Scale Cross-Regional Attention Network for Cross-Dataset Pneumonia Detection},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025}
}
```

## 📧 Contact

- **Lead Author**: Your Name - your.email@university.edu
- **Issues**: Use GitHub issues for questions and bug reports

## 📄 License

This project is licensed under the MIT License.

---

**Paper Status**: Under Review | [arXiv](link-to-arxiv) | [Code](https://github.com/yourusername/AGMSCRA-Net)
