# 🌿 PlantAI – AI-Powered Plant Disease Detection System

# 🌿 PlantAI - Plant Disease Detection

## 🚀 Live Demo

Try the working application here:

https://plantai-plant-disease-detection-utxqtxjmddmbauqsvpfadx.streamlit.app/

## 💻 Source Code

https://github.com/pj879013-ctrl/PlantAI-Plant-Disease-Detection

## Project Overview

PlantAI is a deep learning-based web application designed to classify plant leaf images into 15 plant health categories using a MobileNetV2-based image classification model.

The application provides image-based disease classification, confidence scores, disease information, prevention guidance, and permanent prediction history.

## Key Features

- AI-based plant disease classification
- 15 plant health categories
- Image upload and prediction
- Prediction confidence scores
- Top 3 model predictions
- Disease descriptions and common symptoms
- General prevention information
- SQLite-based prediction history
- CSV export
- Interactive Streamlit dashboard

## Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pandas
- Pillow
- Streamlit
- SQLite

## Dataset

The project uses the PlantVillage image dataset.

- Training images: 16,511
- Validation images: 4,127
- Total images: 20,638
- Classification categories: 15

## Model Performance

Validation Accuracy: 86.65%

The model was trained using transfer learning with a MobileNetV2-based architecture.

## Project Structure

```text
PlantAI/
│
├── app.py
├── predict.py
├── train_model.py
├── prepare_data.py
├── database.py
├── class_names.json
│
├── models/
│   ├── plant_disease_model.keras
│   └── class_names.json
│
├── data/
│   └── raw/
│
├── plantai_history.db
│
├── requirements.txt
└── README.md
```

## How to Run

1. Clone or download the project.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
streamlit run app.py
```

## Limitations

- Predictions are limited to the 15 trained categories.
- Dataset validation performance does not guarantee real-world accuracy.
- Image quality and lighting may affect predictions.
- Disease information is educational and is not a substitute for agricultural expert advice.

## Future Scope

- Expand the dataset with more crops and diseases.
- Improve model performance using additional training and evaluation.
- Add multilingual support.
- Explore mobile application integration.
- Add more detailed agricultural guidance.

## Project Objective

To demonstrate how artificial intelligence, deep learning, and computer vision can support plant health monitoring and disease classification.
