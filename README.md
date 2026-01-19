**Eye Health AI Companion**

This project focuses on multi-eye disease classification using lens-assisted retinal images captured through a smartphone and handheld 20D lens. The system integrates CNN and Transformer-based models to provide accurate, real-time disease predictions.

*Overview*

The model uses EfficientNet-B5, ResNet-RS-50, Swin Transformer, and ViT-B/16 for hybrid feature extraction. Attention U-Net is used for segmentation and LightGBM is used for final classification. A Flask backend and a simple web frontend support real-time inference and explainability.

*Features*

=>Multi-disease detection: CSR, Myopia, Ocular Toxoplasmosis, Retinitis Pigmentosa and Healthy cases

=>Hybrid CNN + Transformer feature fusion

=>Flask-based real-time prediction

=>Grad-CAM explainable heatmaps

=>Low-cost image acquisition workflow

*Tech Stack* 

Backend: Python, Flask, TensorFlow/PyTorch, LightGBM

Frontend: HTML, CSS, JavaScript

Image Processing: OpenCV, Grad-CAM

*Modules*

  =>Image Acquisition
  
  =>Preprocessing
  
  =>Feature Extraction
  
  =>Feature Fusion
  
  =>Classification
  
  =>Prediction and Explainability

*Objective*

   To develop an affordable and deployable AI system for early retinal disease detection, suitable for rural and low-resource medical environments.
