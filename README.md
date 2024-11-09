---
title: Simple Resnet 50 Classifier
emoji: 🏆
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# Simple ResNet-50 Classifier 🏆

A Streamlit web application that uses the pre-trained ResNet-50 model for image classification.

## Features

- Image upload through drag-and-drop or file browser
- Real-time image classification
- Top 5 predictions with confidence scores
- Support for JPG, JPEG, and PNG formats
- User-friendly interface
- Built on PyTorch and Streamlit

## Technical Details

- **Framework**: Streamlit v1.40.0
- **Model**: Pre-trained ResNet-50 
- **Image Processing**: PIL and torchvision
- **Classification**: 1000 ImageNet classes

## Requirements

The project requires the following dependencies:
```sh
streamlit==1.24.0
torch==2.0.1
torchvision==0.15.2
Pillow==9.5.0
```

## Usage

1. Clone the repository
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   streamlit run app.py
   ```

## License

This project is licensed under the MIT License.