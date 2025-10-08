<div align="center">

# 🎤 Speech Emotion Recognition Using Deep Learning

### *Advanced Audio Analysis for Human Emotion Classification*

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio-00599C?style=for-the-badge)](https://librosa.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

**Authors:** Zahra Boucheta • Islam Roubache



</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

This project presents a **comprehensive Speech Emotion Recognition (SER) system** that leverages advanced machine learning and deep learning techniques to classify human emotions from audio recordings. The system analyzes acoustic features to identify seven distinct emotional states with high accuracy.

### Research Objectives

- Implement and compare multiple ML approaches for emotion recognition
- Develop an optimized CNN architecture for audio classification
- Evaluate model performance using comprehensive metrics
- Identify key challenges and future research directions

### Applications

| Domain | Use Case |
|--------|----------|
| 🏥 **Healthcare** | Psychological assessment and therapy monitoring |
| 📞 **Customer Service** | Automated sentiment analysis and quality assurance |
| 🔒 **Security** | Surveillance and threat detection systems |
| 🎮 **Entertainment** | Adaptive gaming experiences and interactive media |
| 🤖 **HCI** | Enhanced human-computer interaction systems |

---

## ✨ Key Features

### 🎵 Advanced Audio Processing
- **Multi-feature extraction** using Librosa (MFCCs, Spectrograms, ZCR, RMSE)
- **Mel-Frequency Cepstral Coefficients (MFCCs)** for human-like perception
- **Zero Crossing Rate (ZCR)** for temporal analysis
- **Root Mean Square Energy (RMSE)** for amplitude tracking

### 🔄 Data Augmentation Pipeline
Four sophisticated augmentation techniques for improved robustness:

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Noise Injection** | Adds Gaussian noise (0.005× amplitude) | Improves noise resilience |
| **Time Shifting** | Random shift (±5ms equivalent) | Handles temporal variations |
| **Time Stretching** | Speed modification (0.8-1.2×) | Adapts to speech rate changes |
| **Pitch Shifting** | Pitch adjustment (±2 semitones) | Speaker-independent recognition |

### 🧠 Model Comparison
- **Decision Tree Classifier** - Baseline traditional ML approach
- **K-Nearest Neighbors (KNN)** - Distance-based classification
- **Convolutional Neural Network (CNN)** - Deep learning solution

---

## 📊 Dataset

### RAVDESS Dataset Specifications

The **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** provides high-quality emotional audio recordings.

| Parameter | Specification |
|-----------|---------------|
| **Total Recordings** | 2,452 (1,440 speech + 1,012 audio) |
| **Emotional States** | 7 + neutral baseline |
| **Performers** | 24 (12 male, 12 female) |
| **Sampling Rate** | 48 kHz |
| **Bit Depth** | 16-bit |
| **Format** | WAV (uncompressed) |

### Emotion Classes

\`\`\`
🔴 Angry    🟢 Happy      🔵 Sad       🟡 Surprise
🟠 Disgust  🟣 Fear       ⚪ Neutral
\`\`\`

### Distribution Analysis

**Training Set:**
- **Angry:** 3,995 samples (13.92%)
- **Disgust:** 436 samples (1.52%) ⚠️ *Minority class*
- **Fear:** 4,097 samples (14.27%)
- **Happy:** 7,215 samples (25.13%) ✓ *Majority class*
- **Neutral:** 4,965 samples (17.29%)
- **Sad:** 4,830 samples (16.82%)
- **Surprise:** 3,171 samples (11.05%)

**Total:** 28,709 training samples

**Test Set:** 7,178 samples with similar distribution

---

## 🔬 Methodology

### 1️⃣ Audio Preprocessing Pipeline

\`\`\`python
# Feature Extraction Process
1. Load audio files (48 kHz, 16-bit WAV)
2. Apply data augmentation (4 techniques)
3. Extract features:
   - MFCCs (40 coefficients)
   - Zero Crossing Rate
   - Root Mean Square Energy
4. Standardize feature lengths (padding)
5. Normalize using StandardScaler
\`\`\`

### 2️⃣ Feature Engineering

**MFCC Advantages:**
- ✓ Human-like auditory perception (Mel scale)
- ✓ Compact representation (40 coefficients)
- ✓ Noise resilience
- ✓ Computational efficiency
- ✓ Speaker independence

### 3️⃣ Data Normalization

- **Missing Values:** Replaced with zeros
- **Label Encoding:** One-hot encoding for 7 emotion classes
- **Data Splitting:** 80% training, 20% testing (stratified)
- **Feature Standardization:** StandardScaler for zero mean, unit variance

---

## 🏗️ Model Architecture

### Baseline Models

#### Decision Tree Classifier
\`\`\`
Configuration:
├── Criterion: Gini impurity
├── Max Depth: Unlimited
└── Random State: Fixed

Performance:
├── Training Accuracy: 100.0% ⚠️ Overfitting
└── Test Accuracy: 41.5%
\`\`\`

#### K-Nearest Neighbors (KNN)
\`\`\`
Configuration:
├── Neighbors: k=4
└── Distance: Euclidean

Performance:
├── Training Accuracy: 67.3%
└── Test Accuracy: 49.4%
\`\`\`

### Proposed CNN Architecture

\`\`\`
Input Layer (Audio Features)
        ↓
┌─────────────────────┐
│  Conv1D (64 filters)│  ← Kernel size: 3, ReLU activation
└─────────────────────┘
        ↓
┌─────────────────────┐
│  MaxPooling1D       │  ← Pool size: 2
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Flatten            │
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Dense (64 units)   │  ← ReLU activation
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Dropout (0.3)      │  ← Regularization
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Dense (7 units)    │  ← Softmax activation
└─────────────────────┘
        ↓
   Output (7 Emotions)
\`\`\`

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rates |
| **Learning Rate** | 0.001 | Default, proven effective |
| **Loss Function** | Categorical Crossentropy | Multi-class classification |
| **Batch Size** | 32 | Memory-gradient balance |
| **Epochs** | 50 | With early stopping |
| **Validation Split** | 20% | Hyperparameter tuning |

---

## 📈 Results

### Performance Comparison

| Model | Training Accuracy | Test Accuracy | Improvement |
|-------|------------------|---------------|-------------|
| **Decision Tree** | 100.0% | 41.5% | Baseline |
| **KNN (k=4)** | 67.3% | 49.4% | +7.9% |
| **CNN (Proposed)** | **97.3%** | **72.5%** | **+23.1%** 🏆 |

### Detailed Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Angry** | 0.75 | **0.88** | 0.81 | 183 |
| **Disgust** | **0.79** | 0.73 | 0.76 | 193 |
| **Fear** | 0.72 | 0.73 | 0.72 | 177 |
| **Happy** | 0.69 | 0.68 | 0.68 | 212 |
| **Neutral** | **0.80** | **0.84** | **0.82** | 270 |
| **Sad** | 0.73 | 0.63 | 0.68 | 226 |
| **Surprise** | 0.74 | 0.74 | 0.74 | 179 |
| | | | | |
| **Accuracy** | | | **0.75** | 1,440 |
| **Macro Avg** | 0.75 | 0.75 | 0.74 | 1,440 |
| **Weighted Avg** | 0.75 | 0.75 | 0.75 | 1,440 |

### Key Findings

✅ **Best Performing Emotions:**
- **Neutral** (F1: 0.82) - Most distinct features
- **Angry** (Recall: 0.88) - High detection rate

⚠️ **Challenging Emotions:**
- **Happy** (F1: 0.68) - Confused with Surprise
- **Sad** (Recall: 0.63) - Often misclassified as Neutral

### Confusion Matrix Insights

- **Neutral** and **Angry** achieved highest classification accuracy
- **Disgust** frequently confused with **Angry** (22 misclassifications)
- **Fear** confused with **Sad** (20) and **Happy** (16)
- **Surprise** sometimes misclassified as **Happy** (23 instances)
- Positive emotions (happy, surprise) better classified than negative ones

### Processing Performance

- **Inference Speed:** 6ms per step
- **Batch Processing:** 45 batches (32 samples each)
- **Total Test Samples:** 1,440
- ✓ Suitable for real-time applications

---

## 🚀 Installation

### Prerequisites

\`\`\`bash
Python 3.7+
CUDA 10.1+ (optional, for GPU acceleration)
\`\`\`

### Setup Instructions

1. **Clone the repository**
\`\`\`bash
git clone https://github.com/Islamroubache/Speech_Emotion_Recogination_Using_Deep_Learning_Approaches
.git
cd speech-emotion-recognition
\`\`\`

2. **Create virtual environment**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. **Install dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Required Libraries

\`\`\`txt
tensorflow>=2.8.0
keras>=2.8.0
librosa>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
\`\`\`

---

## 💻 Usage

### 1. Data Preparation

\`\`\`python
# Load and preprocess RAVDESS dataset
from preprocessing import load_dataset, extract_features

# Load audio files
audio_data = load_dataset('path/to/ravdess')

# Extract features with augmentation
features, labels = extract_features(audio_data, augment=True)
\`\`\`

### 2. Model Training

\`\`\`python
# Train CNN model
from models import build_cnn_model

# Build and compile model
model = build_cnn_model(input_shape=(features.shape[1], 1))
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=50,
                   validation_split=0.2)
\`\`\`

### 3. Emotion Prediction

\`\`\`python
# Predict emotion from audio file
from inference import predict_emotion

emotion = predict_emotion('path/to/audio.wav', model)
print(f"Detected Emotion: {emotion}")
\`\`\`

### 4. Visualization

\`\`\`python
# Generate visualizations
from visualization import plot_waveform, plot_spectrogram

plot_waveform('audio.wav')
plot_spectrogram('audio.wav')
\`\`\`

---


---

## 🛠️ Technologies

### Core Frameworks

<div align="center">

| Technology | Purpose | Version |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Programming Language | 3.7+ |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep Learning Framework | 2.8+ |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | Neural Network API | 2.8+ |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine Learning | 1.0+ |

</div>

### Audio Processing

- **Librosa** - Audio analysis and feature extraction
- **NumPy** - Numerical computations
- **SciPy** - Signal processing

### Data Science

- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical graphics

### Utilities

- **Joblib** - Parallel processing
- **Jupyter** - Interactive development

---

## 🔮 Future Work

### Planned Enhancements

1. **Multimodal Approaches**
   - Combine audio with text transcriptions
   - Integrate visual facial expression data
   - Fusion of multiple modalities for improved accuracy

2. **Advanced Architectures**
   - Explore Transformer-based models (Audio Transformers)
   - Implement attention mechanisms
   - Test pre-trained models (Wav2Vec 2.0, HuBERT)

3. **Dataset Improvements**
   - Address class imbalance (especially Disgust)
   - Expand to multilingual datasets
   - Include real-world noisy environments

4. **Deployment**
   - Real-time emotion recognition system
   - Mobile application development
   - Cloud-based API service

5. **Optimization**
   - Model compression and quantization
   - Edge device deployment
   - Reduce inference latency

---
<div align="center">

<table>
<tr>
<td align="center">
<img src="https://github.com/Islamroubache.png" width="100px;" alt="Islam Roubache"/><br>
<sub><b>Islam Roubache</b></sub><br>
🎓 Master's Student in AI & Data Science<br>
📍 Higher School of Computer Science 08 May 1945<br>
Sidi Bel Abbes, Algeria
</td>
</tr>
</table>

</div>
<div align="center">


[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

</div>

---
## 📚 References

### Key Publications

1. **Xu, M., Zhang, F., & Zhang, W. (2021)**  
   *Head Fusion: Improving Speech Emotion Recognition*  
   IEEE Transactions on Affective Computing

2. **Huang, J.-T., Li, J., & Gong, Y. (2015)**  
   *CNN Analysis for Speech Recognition*  
   IEEE/ACM Transactions on Audio, Speech, and Language Processing

3. **Muda, L., Begam, M., & Elamvazuthi, I. (2010)**  
   *Voice Recognition Using MFCC-DTW*  
   Journal of Signal Processing

### Dataset Citation

```bibtex
@misc{livingstone2018ravdess,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  year={2018},
  publisher={Zenodo}
}
