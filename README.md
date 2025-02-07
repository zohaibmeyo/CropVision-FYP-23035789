# 🌿 CropVision: A System to Detect and Classify Plant Diseases Using Deep Learning

## 📌 Project Overview
CropVision is a deep learning-based system designed to detect and classify plant diseases using leaf images. The project leverages convolutional neural networks (CNNs) to enable early and precise disease identification, which is crucial for effective crop health management.

## 📊 Dataset
- **Dataset Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- **Collected By:** Penn State University & collaborators (2015-2016)
- **Description:** 
  - 54,305 images of diseased and healthy plant leaves
  - Covers 14 crop species
  - 17 basic diseases + 4 bacterial + 2 mold (oomycete) + 2 viral + 1 mite-related disease
  - 12 crop species also have healthy leaf images

## 🏗️ Project Structure
- `data/` → Dataset & preprocessing scripts
- `models/` → Saved models & checkpoints
- `notebooks/` → Experimentation and model training notebooks
- `webapp/` → Streamlit web app for real-time disease detection

## 🔧 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/zohaibmeyo/CropVision-FYP-23035789
   cd CropVision-FYP-23035789
