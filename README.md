# Steel Defect Detection (Computer Vision & XAI)

An automated industrial inspection system that leverages Convolutional Neural Networks (CNNs) and Transfer Learning to detect micro-level surface cracks in steel in real-time.

## Project Overview

In industrial manufacturing, identifying surface defects on steel components is critical for structural integrity and quality control. Traditional manual inspection is slow and error-prone. This project automates the inspection process by using deep learning to achieve near-perfect classification of defective versus non-defective steel surfaces.

To ensure transparency and trust in the AI's decision-making process (crucial for industrial applications), the system integrates **Explainable AI (XAI)** techniques to visually highlight the exact regions of the image that lead to a "Defect" prediction.

## Technical Architecture & Methodology

*   **Core Model:** MobileNetV2 (Transfer Learning)
*   **Data Augmentation:** Applied robust augmentation pipelines (rotation, zoom, flips, brightness shifts) to prevent overfitting and simulate real-world factory lighting conditions.
*   **Feature Extraction Optimization:** Unfroze the final layers of the MobileNetV2 base model to fine-tune the feature extraction specifically for metallic textures and micro-cracks.
*   **Explainable AI (Grad-CAM):** Implemented Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps over the original images, proving visual accountability for quality control teams by highlighting the specific crack locations.

## Performance Metrics

The model was trained and evaluated on a balanced dataset of steel surface images, achieving exceptional performance metrics:

*   **Overall Accuracy:** 99.7%
*   **Precision (Defect Detection):** 100.0%
*   **Recall:** 99.5%
*   **F1-Score:** 0.99

*Note: A precision score of 100% guarantees that the system produces zero false positives for defects, ensuring that no perfectly good steel is incorrectly scrapped.*

## Key Features

1.  **High-Speed Inference:** Designed to be lightweight (via MobileNetV2) to support real-time sliding-window inference on industrial camera feeds.
2.  **Visual Accountability:** Grad-CAM heatmaps allow human operators to verify the model's logic before taking corrective action.
3.  **Robustness:** Trained to handle variations in lighting, scale, and orientation common in manufacturing environments.

## Repository Contents

*   `Steel_Crack_Detection.ipynb`: The complete end-to-end Jupyter Notebook containing data preprocessing, model architecture definition, training loops, evaluation, and Grad-CAM implementation.
*   `Steel_Crack_Detection_1.pdf`: An exported run of the notebook showcasing the training history graphs, classification reports, and Grad-CAM visualizations.

*(Note: The raw image dataset is omitted from this repository due to size constraints).*
