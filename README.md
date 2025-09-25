# Deep-Learning-based-Image-Caption-Generator
# 🖼️ Image Caption Generator (CNN + LSTM)

This project implements an **Image Caption Generator** using **Deep Learning** techniques.  
It combines a **Convolutional Neural Network (CNN)** for image feature extraction with a **Recurrent Neural Network (LSTM)** for generating descriptive captions in natural language.  

---

## Project Overview
- **Dataset:** Flickr8k (8,000 images with 5 captions each)  
- **Model:** CNN (InceptionV3) + LSTM  
- **Frameworks:** Python, TensorFlow/Keras  
- **Goal:** Generate accurate captions for unseen images by learning the mapping between image features and natural language.

---

## Features
- Image preprocessing and feature extraction using pre-trained **InceptionV3**  
- Caption preprocessing with tokenization and padding  
- Sequence modeling with **LSTM**  
- Evaluation using **BLEU scores**  
- Inference script to generate captions for custom images  

---

## Folder Structure

data/ # Dataset: images and captions
src/ # Scripts: utils, feature extraction, model, training, inference
notebooks/ # Interactive notebooks for exploration, training, inference
requirements.txt # Python dependencies

---

## Dataset
- **Images:** 8,000 images of everyday scenes.
- **Captions:** 5 captions per image (`Flickr8k.token.txt`).

**Instructions:**
1. Download Flickr8k dataset.
2. Place images in `data/Images/` and captions in `data/Flickr8k.token.txt`.

---

## Setup
1. Clone the repo:
git clone <your_repo_link>
cd Image-Caption-Generator

2. Install dependencies:
   ## Training
- Run `notebooks/03_model_training.ipynb` to train the model.
- Training and validation loss will be plotted automatically.
- Model weights will be saved to `models/model_weights.h5`.

---

## Inference / Demo
- Run `notebooks/04_inference_demo.ipynb` to generate captions for sample images.
- Visualize results inline or save them in `examples/`.

---
## References
- [Flickr8k Dataset](https://github.com/jbrownlee/Datasets)
