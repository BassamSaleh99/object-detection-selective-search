# Car Detection using Selective Search and VGG16

This project implements a car detection pipeline using Selective Search for region proposal and a fine-tuned VGG16 CNN model for classification. It processes images to identify bounding boxes most likely to contain a car.

🔧 Requirements

Python 3.8+

TensorFlow

OpenCV

Matplotlib

scikit-learn

pandas

numpy


🧠 Model Details

Backbone: VGG16 (pretrained on ImageNet, top layers removed)

Classification: Binary (car vs. no car)

Training split: 70% training, 30% validation

🚀 How it Works

Preprocessing:

Load and resize image regions to 224x224

Label them based on bounding boxes from CSV

Training:

Train a VGG16-based binary classifier

Region Proposals:

Use Selective Search to generate candidate bounding boxes

Prediction:

Classify each region; draw the one with the highest confidence

🖼️ Output Example

When applied to an image, the model draws a bounding box around the area most likely to contain a car, with the confidence score displayed.

📌 Notes

The notebook uses .pkl files to store and reload processed data for faster reuse.

The Selective Search algorithm may produce many region proposals; only the one with highest probability is used.