# Messy Mashup – Robust Music Genre Classification  
Deep Learning & Generative AI Project  
BS in Data Science and Applications (Diploma Level)  

## Student Details
- Name: Dhruv Bansal  
- Roll No: 24f1001707
- Email: 24f1001707@ds.study.iitm.ac.in

## Links
- W&B Project: https://wandb.ai/24f1001707-dl-genai-project/24f1001707-t12026
- Kaggle Competition: https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project/

## Project Overview

This project focuses on robust music genre classification under noisy and realistic mashup conditions.  
The goal is to build models that can generalize across distribution shifts and predict the correct genre label.

## Environment Setup
1. Clone the repository and navigate to the project directory.
2. Create a conda environment and install dependencies:
```bash
conda env create -f environment.yaml
conda activate messy_mashup
```

## Training the Model
To train the model, run the following command:
```bash
python -m src.train
```

## Building the Dataset
To build the dataset by mixing the stems and adding noise, run:
```bash
python -m scripts.build_audio_dataset
```