# 🧠 Explainable Brain Tumor Classification from MRI Images

A deep learning system that classifies brain tumors from MRI scans and explains its predictions using Grad-CAM visualizations.

## Overview

This project implements an explainable AI framework for multi-class brain tumor classification using MRI images. The model can distinguish between four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

## Results

- **Training Accuracy:** 98.1%
- **Test Accuracy:** 93.6%
- **Dataset:** 5,600 training images, 1,600 test images

## Features

- ResNet18 CNN fine-tuned on brain MRI data
- Grad-CAM heatmaps showing which regions influenced the prediction
- Interactive Gradio dashboard for real-time inference
- Confidence scores for all 4 classes

## Project Structure
