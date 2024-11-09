import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json

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

# Load ImageNet class labels
with open('imagenet_classes.json', 'r') as f:
    class_labels = json.load(f)

# Streamlit UI
st.title("Image Classification with ResNet50")
st.write("Upload an image and the model will classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction
    input_tensor = process_image(image)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Display results
    st.write("Top 5 Predictions:")
    for i in range(5):
        st.write(f"{class_labels[str(top5_idx[i].item())]}: {top5_prob[i].item()*100:.2f}%")
