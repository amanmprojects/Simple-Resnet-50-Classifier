import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import io

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image):
    input_tensor = process_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

def handle_image(image):
    st.image(image, caption='Processed Image', use_container_width=True)
    probabilities = get_prediction(image)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    st.write("Top 5 Predictions:")
    for i in range(5):
        st.write(f"{class_labels[str(top5_idx[i].item())]}: {top5_prob[i].item()*100:.2f}%")

# Load ImageNet class labels
with open('imagenet_classes.json', 'r') as f:
    class_labels = json.load(f)

# Streamlit UI
st.title("Image Classification with ResNet50")
st.write("Upload an image or paste from clipboard (Ctrl+V/Cmd+V) and the model will classify it!")

# Add clipboard paste functionality
clipboard_container = st.container()
with clipboard_container:
    st.write("Click here and press Ctrl+V/Cmd+V to paste an image from clipboard")
    
    # Handle clipboard paste
    clipboard_data = st.text_area("", "", key="clipboard", label_visibility="collapsed")
    if clipboard_data and clipboard_data.startswith(('data:image', 'iVBORw0KGgo')):
        try:
            # Handle base64 encoded image data
            import base64
            image_data = base64.b64decode(clipboard_data.split(',')[1] if ',' in clipboard_data else clipboard_data)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            handle_image(image)
        except Exception as e:
            st.error(f"Error processing pasted image: {str(e)}")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    handle_image(image)
